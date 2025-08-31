import cv2
from ultralytics import YOLO
from deepface import DeepFace
from .face_utils import recognize_student, initialize_student_faces
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import os

# --- Model Initialization (Happens once at startup) ---
print("Initializing AI models...")
person_detector = YOLO('yolov8n.pt') 
student_embeddings, student_metadata = initialize_student_faces()

# --- Custom Anomaly Detection Model Loading ---
anomaly_model_path = 'ai_service/custom_models/anomaly_detector.pth'
anomaly_model = None
if os.path.exists(anomaly_model_path):
    print(f"Loading custom anomaly model from {anomaly_model_path}")
    try:
        anomaly_model = models.resnet18(weights=None)
        num_ftrs = anomaly_model.fc.in_features
        anomaly_model.fc = nn.Linear(num_ftrs, 2) # 0:Fight, 1:NoFight
        # Load the model onto the CPU
        anomaly_model.load_state_dict(torch.load(anomaly_model_path, map_location=torch.device('cpu')))
        anomaly_model.eval()
        print("Custom anomaly model loaded successfully.")
    except Exception as e:
        print(f"Error loading custom anomaly model: {e}")
        anomaly_model = None
else:
    print("Custom anomaly model not found. Anomaly detection will be disabled.")

# Transformation pipeline for anomaly detection model input
anomaly_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
anomaly_classes = ['Fight', 'NoFight']

def detect_anomaly(frame):
    """Uses the custom-trained model to classify a frame."""
    if anomaly_model is None: return "Not Available"
    try:
        input_tensor = anomaly_transform(frame).unsqueeze(0)
        with torch.no_grad():
            outputs = anomaly_model(input_tensor)
            _, preds = torch.max(outputs, 1)
        return anomaly_classes[preds[0]]
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        return "Error"

def process_frame_for_analysis(frame):
    """Main function for the Campus Safety Module."""
    analysis_results = {
        "crowd_count": 0,
        "alerts": [],
        "anomaly_status": "Not Available"
    }
    
    # 1. Anomaly Detection
    analysis_results["anomaly_status"] = detect_anomaly(frame)
    if analysis_results["anomaly_status"] == "Fight":
        analysis_results["alerts"].append("High Priority Alert: Potential fight or violence detected.")

    # 2. Crowd Counting
    person_detections = person_detector(frame, classes=[0], verbose=False)
    detected_persons_boxes = person_detections[0].boxes.data.cpu().numpy()
    analysis_results["crowd_count"] = len(detected_persons_boxes)

    if analysis_results["crowd_count"] > 30: # Campus safety threshold
        analysis_results["alerts"].append(f"Crowd Alert: High crowd density of {analysis_results['crowd_count']} people detected.")
        
    return analysis_results

def process_frame_for_attendance(frame):
    """Main function for the Attendance Module."""
    attendance_results = {
        "present_students": [],
        "unrecognized_faces": 0
    }
    present_student_ids = set()

    try:
        # Use DeepFace to find all faces in the frame
        extracted_faces = DeepFace.extract_faces(frame, enforce_detection=False, detector_backend='opencv')
        
        for face_obj in extracted_faces:
            if face_obj['confidence'] > 0.90: # Process only high-confidence faces
                face_roi = face_obj['face']
                
                # Get embedding for the detected face
                embedding_objs = DeepFace.represent(face_roi, model_name='VGG-Face', enforce_detection=False)
                face_embedding = embedding_objs[0]["embedding"]
                
                # Recognize student from the embedding
                student_id = recognize_student(face_embedding, student_embeddings)
                
                if student_id and student_id not in present_student_ids:
                    present_student_ids.add(student_id)
                    attendance_results["present_students"].append({
                        "student_id": student_id,
                        "name": student_metadata.get(student_id, {}).get("name", "Unknown Student")
                    })
                elif not student_id:
                    attendance_results["unrecognized_faces"] += 1
    except Exception as e:
        print(f"An error occurred during attendance face processing: {e}")

    return attendance_results
