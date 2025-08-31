import pickle
import os
import numpy as np
from deepface import DeepFace

STUDENT_DATA_PATH = "ai_service/data/student_faces.pkl"

def initialize_student_faces():
    data_dir = os.path.dirname(STUDENT_DATA_PATH)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if os.path.exists(STUDENT_DATA_PATH):
        try:
            with open(STUDENT_DATA_PATH, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            return {}, {}
    return {}, {}

def enroll_student(image, student_id, name):
    try:
        embedding_objs = DeepFace.represent(image, model_name='VGG-Face', enforce_detection=True)
        if not embedding_objs:
            return False
        embedding = embedding_objs[0]["embedding"]
        
        student_embeddings, student_metadata = initialize_student_faces()
        
        if student_id in student_embeddings:
            student_embeddings[student_id].append(embedding)
        else:
            student_embeddings[student_id] = [embedding]
            
        student_metadata[student_id] = {"name": name}
        
        with open(STUDENT_DATA_PATH, 'wb') as f:
            pickle.dump((student_embeddings, student_metadata), f)
        return True
    except Exception as e:
        print(f"Error during student enrollment: {e}")
        return False

def recognize_student(face_embedding, student_embeddings, threshold=0.40):
    if not student_embeddings: return None
    min_distance = float('inf')
    identity = None

    face_embedding_norm = np.array(face_embedding) / np.linalg.norm(face_embedding)
    
    for student_id, embeddings in student_embeddings.items():
        for known_embedding in embeddings:
            known_embedding_norm = np.array(known_embedding) / np.linalg.norm(known_embedding)
            distance = 1 - np.dot(face_embedding_norm, known_embedding_norm)
            if distance < threshold and distance < min_distance:
                min_distance = distance
                identity = student_id
    return identity