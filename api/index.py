import os
import numpy as np
import cv2
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client
from insightface.app import FaceAnalysis
from datetime import datetime, timedelta

import supabase

# Flask App Setup
app = Flask(__name__)
CORS(app)

# Supabase setup
SUPABASE_URL = "https://arlexrfzqvahegtolcjp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFybGV4cmZ6cXZhaGVndG9sY2pwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg2ODE4MjcsImV4cCI6MjA1NDI1NzgyN30.ksThqyqHmQt16ZmlYM7hrutQVmBOcYt-0xap6a7QlhQ"
supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)

# Load InsightFace Model
face_analysis = FaceAnalysis(name="buffalo_l")
face_analysis.prepare(ctx_id=0)
print("InsightFace model loaded successfully.")

# Global Variables
ids, names, stored_embeddings = [], [], []
knn_model = None  # Initialize empty model

# Function to extract embeddings using InsightFace
def get_embedding(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_analysis.get(image)

    if not faces:
        print("No face detected!")
        return None

    # Pick the face with the highest detection score
    best_face = max(faces, key=lambda f: f.det_score)
    return best_face.normed_embedding

# Load stored embeddings from Supabase

def load_embeddings():
    global ids, names, stored_embeddings
    response = supabase_client.table("users").select("id, name, embedding").execute()
    
    if not response.data:  # Ensure data exists
        print("No embeddings found in the database.")
        ids, names, stored_embeddings = [], [], np.array([])  # Reset if empty
        return

    ids, names, stored_embeddings = [], [], []
    for user in response.data:
        try:
            embedding = np.array(user["embedding"], dtype=np.float32)
            if embedding.shape != (512,):  # Ensure correct embedding size
                print(f"Skipping invalid embedding for {user['name']}")
                continue  
            ids.append(user["id"])
            names.append(user["name"])
            stored_embeddings.append(embedding)
            #print(f"Loaded embedding for user: {user['name']}")
        except Exception as e:
            print(f"Error loading embedding for {user['name']}: {str(e)}")

    if stored_embeddings:
        stored_embeddings = np.array(stored_embeddings)
    else:
        stored_embeddings = np.array([])  # Avoid NoneType errors

load_embeddings()
print("Embeddings loaded successfully")

# Train API Endpoint
@app.route('/train', methods=['POST'])
def train():
    try:
        global ids, names, stored_embeddings  # Add global declaration
        name = request.form['name']
        user_id = request.form['id']
        phone = request.form['phone']
        images = request.files.getlist("images")
        embeddings = []
        print(f"Training user: {name} with ID: {user_id}")

        for image in images:
            temp_path = os.path.join(tempfile.gettempdir(), image.filename)
            image.save(temp_path)
            print(f"Image saved to temporary path: {temp_path}")
            embedding = get_embedding(temp_path)
            os.remove(temp_path)

            if embedding is not None:
                embeddings.append(embedding)
                print(f"Valid embedding added. Total embeddings: {len(embeddings)}")
            else:
                print(f"No embedding found for image: {temp_path}")

        if not embeddings:
            print("No valid embeddings generated.")
            return jsonify({"error": "No valid embeddings generated.", "status": 400})

        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        # Use `upsert` to avoid duplicate ID issues
        data = {"id": user_id, "name": name, "phone": phone, "embedding": avg_embedding.tolist()}
        supabase_client.table("users").upsert(data).execute()
        print(f"User data upserted to Supabase for ID: {user_id}")

        ids.append(user_id)
        names.append(name)

        # Ensure stored_embeddings is properly initialized before using np.vstack
        if len(stored_embeddings) == 0:
            stored_embeddings = np.array([avg_embedding])  # Initialize with first embedding
        else:
            stored_embeddings = np.vstack([stored_embeddings, avg_embedding])  # Append new embedding

        print("User trained successfully!")
        return jsonify({"message": "User trained successfully!", "status": 200})
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return jsonify({"error": str(e), "status": 500})


# Function to mark attendance
from datetime import datetime

def mark_attendance(user_id, user_name, timestamp):
    try:
        # Parse the provided timestamp
        timestamp_str = timestamp.rstrip("Z")  # Remove 'Z' if present
        attendance_time = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")

        # Get today's date in YYYY-MM-DD format
        today_date = attendance_time.strftime("%Y-%m-%d")

        # Fetch user's phone number and freeze range from users table
        user_response = supabase_client.table("users")\
            .select("phone, freeze_start, freeze_end")\
            .eq("id", user_id)\
            .execute()
        
        if not user_response.data:
            print(f"User with ID {user_id} not found in users table")
            return

        user_data = user_response.data[0]
        phone = user_data["phone"]
        freeze_start = user_data.get("freeze_start")
        freeze_end = user_data.get("freeze_end")

        # Check if attendance is frozen
        if freeze_start and freeze_end:
            freeze_start_dt = datetime.fromisoformat(freeze_start.rstrip("Z"))
            freeze_end_dt = datetime.fromisoformat(freeze_end.rstrip("Z"))
            
            if freeze_start_dt <= attendance_time <= freeze_end_dt:
                print(f"Attendance is frozen for {user_name} between {freeze_start_dt} and {freeze_end_dt}. No entry made.")
                return

        # Check if the user already has a Time In for today
        response = supabase_client.table("attendance2")\
            .select("id, time_in, time_out")\
            .eq("user_id", user_id)\
            .gte("time_in", f"{today_date}T00:00:00.000Z")\
            .lte("time_in", f"{today_date}T23:59:59.999Z")\
            .order("time_in", desc=True)\
            .limit(1)\
            .execute()

        if response.data:
            attendance_record = response.data[0]
            attendance_id = attendance_record["id"]

            if attendance_record["time_out"] is None:
                # Update Time Out if not marked
                update_data = {"time_out": attendance_time.isoformat() + "Z"}
                supabase_client.table("attendance2").update(update_data).eq("id", attendance_id).execute()
                print(f"Time Out marked for {user_name} at {attendance_time}")
                return  
        else:
            # Insert Time In if no entry exists for today
            insert_data = {
                "user_id": user_id,
                "user_name": user_name,
                "Phone": phone,
                "time_in": attendance_time.isoformat() + "Z",
                "time_out": None
            }
            supabase_client.table("attendance2").insert(insert_data).execute()
            print(f"Time In marked for {user_name} at {attendance_time}")
            return  

    except Exception as e:
        print(f"Error marking attendance: {str(e)}")




# Recognition API Endpoint
@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        if 'image' not in request.files:
            print("Debug: No image provided in request.")
            return jsonify({"error": "No image provided", "status": 400})

        if len(stored_embeddings) == 0:
            print("Debug: Reloading embeddings...")
            load_embeddings()  # Load embeddings if none are available

        if len(stored_embeddings) == 0:
            print("Debug: No stored embeddings available.")
            return jsonify({"error": "No stored embeddings available.", "status": 500})

        image = request.files['image']
        timestamp = request.form.get('timestamp')
        #print(f"Debug: Received timestamp: {timestamp}")

        if not timestamp:
            print("Debug: No timestamp provided in request.")
            return jsonify({"error": "No timestamp provided", "status": 400})
        
        temp_path = os.path.join(tempfile.gettempdir(), "temp_recognition_image.jpg")
        image.save(temp_path)
        print(f"Debug: Image saved to temporary path: {temp_path}")

        embedding = get_embedding(temp_path)
        os.remove(temp_path)

        if embedding is None:
            print("Debug: No face detected in the image.")
            return jsonify({"error": "No face detected!", "status": 400})

        # Ensure embedding is in correct shape
        embedding = embedding.reshape(1, -1)  
        print(f"Debug: Embedding shape after reshaping: {embedding.shape}")

        # Directly compare embeddings using np.dot (Cosine Similarity)
        similarities = np.dot(stored_embeddings, embedding.T).flatten()
        recognized_index = np.argmax(similarities)
        recognized_name = names[recognized_index]
        max_similarity = similarities[recognized_index]

        print(f"Debug: Recognized name: {recognized_name}, Similarity: {max_similarity}")

        # Dynamic Threshold Calculation
        threshold = max(0.4, min(0.7, np.mean(similarities) + 0.1))  
        print(f"Debug: Calculated threshold: {threshold}")

        if max_similarity < threshold:
            print("Debug: Similarity below threshold, returning 'Unknown'.")
            return jsonify({"recognized_name": "Unknown", "similarity": float(max_similarity), "status": 200})

        recognized_user_id = ids[recognized_index]
        # Mark attendance
        mark_attendance(recognized_user_id, recognized_name, timestamp)
        
        return jsonify({"recognized_name": recognized_name, "similarity": float(max_similarity), "attendance_marked": True, "status": 200})

    except Exception as e:
        print(f"Debug: Exception occurred: {str(e)}")
        return jsonify({"error": str(e), "status": 500})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
