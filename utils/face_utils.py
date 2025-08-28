import pickle
import os
import numpy as np
from deepface import DeepFace

# Define the path for storing registered face embeddings
DATA_PATH = "ai_service/data/registered_faces.pkl"

def initialize_known_faces():
    """
    Loads known face embeddings and metadata from the data file.
    If the file doesn't exist, it creates the directory and returns empty data structures.
    """
    data_dir = os.path.dirname(DATA_PATH)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, 'rb') as f:
                known_face_embeddings, known_face_metadata = pickle.load(f)
            print(f"Loaded {len(known_face_metadata)} known faces.")
            return known_face_embeddings, known_face_metadata
        except (EOFError, pickle.UnpicklingError):
            print("Could not read face data file. Starting fresh.")
            return {}, {}
    return {}, {}

def enroll_face(image, name):
    """
    Generates a facial embedding for a given image and saves it with the associated name.
    """
    try:
        embedding_objs = DeepFace.represent(image, model_name='VGG-Face', enforce_detection=True)
        if not embedding_objs:
            print("Enrollment failed: No face detected.")
            return False
        embedding = embedding_objs[0]["embedding"]
        known_face_embeddings, known_face_metadata = initialize_known_faces()
        if name in known_face_embeddings:
            known_face_embeddings[name].append(embedding)
        else:
            known_face_embeddings[name] = [embedding]
        known_face_metadata[name] = {"name": name}
        with open(DATA_PATH, 'wb') as f:
            pickle.dump((known_face_embeddings, known_face_metadata), f)
        print(f"Successfully enrolled face for: {name}")
        return True
    except Exception as e:
        print(f"Error during face enrollment: {e}")
        return False

def recognize_face(face_embedding, known_face_embeddings, threshold=0.40):
    """
    Compares a detected face embedding against a dictionary of known embeddings.
    """
    if not known_face_embeddings:
        return "Unknown"
    min_distance = float('inf')
    identity = "Unknown"
    face_embedding_norm = np.array(face_embedding) / np.linalg.norm(face_embedding)
    for name, embeddings in known_face_embeddings.items():
        for known_embedding in embeddings:
            known_embedding_norm = np.array(known_embedding) / np.linalg.norm(known_embedding)
            distance = 1 - np.dot(face_embedding_norm, known_embedding_norm)
            if distance < threshold and distance < min_distance:
                min_distance = distance
                identity = name
    return identity