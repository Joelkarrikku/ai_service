# File: ai_service/utils/video_processing.py

import cv2
from ultralytics import YOLO
from deepface import DeepFace
from .face_utils import recognize_face, initialize_known_faces
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# --- Model Initialization ---
print("Initializing AI models...")
person_detector = YOLO('yolov8n.pt') 
known_face_embeddings, known_face_metadata = initialize_known_faces()

# --- NEW: Load the custom-trained anomaly detection model ---
ANOMALY_MODEL_PATH = "ai_service/custom_models/anomaly_detector.pth"
anomaly_num_classes = 2
anomaly_model = models.resnet18()
num_ftrs = anomaly_model.fc.in_features
anomaly_model.fc = nn.Linear(num_ftrs, anomaly_num_classes)
anomaly_model.load_state_dict(torch.load(ANOMALY_MODEL_PATH, map_location=torch.device('cpu')))
anomaly_model.eval()
print("Custom anomaly model loaded.")

# --- NEW: Define transforms for the anomaly model ---
anomaly_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
anomaly_class_names = ['NoFight', 'Fight']

# --- NEW: Function to predict anomaly ---
def predict_anomaly(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = anomaly_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = anomaly_model(image)
        _, preds = torch.max(outputs, 1)
        return anomaly_class_names[preds[0]]

# --- Main Processing Function (Updated) ---
def process_frame_for_analysis(frame):
    analysis_results = {
        "crowd_count": 0,
        "demographics": {"male": 0, "female": 0, "unknown": 0},
        "detected_faces": [],
        "anomaly_status": "Normal", # New field
        "alerts": []
    }

    # --- 1. Anomaly Detection ---
    anomaly_status = predict_anomaly(frame)
    analysis_results["anomaly_status"] = anomaly_status
    if anomaly_status == 'Fight':
        analysis_results["alerts"].append("Anomaly Detected: Potential fight in progress.")

    # --- 2. Person Detection ---
    person_detections = person_detector(frame, classes=[0], verbose=False)
    detected_persons_boxes = person_detections[0].boxes.data.cpu().numpy()
    analysis_results["crowd_count"] = len(detected_persons_boxes)

    # --- 3. Face Analysis ---
    for i, person_box in enumerate(detected_persons_boxes):
        x1, y1, x2, y2, conf, cls = person_box
        person_roi = frame[int(y1):int(y2), int(x1):int(x2)]
        if person_roi.size == 0:
            continue
        try:
            face_objs = DeepFace.analyze(person_roi, actions=['gender'], enforce_detection=False, silent=True)
            if isinstance(face_objs, list) and len(face_objs) > 0:
                face_info = face_objs[0]
                gender = face_info.get('dominant_gender', 'Unknown').lower()
                if gender == 'man': analysis_results["demographics"]["male"] += 1
                elif gender == 'woman': analysis_results["demographics"]["female"] += 1
                else: analysis_results["demographics"]["unknown"] += 1
        except Exception:
            analysis_results["demographics"]["unknown"] += 1
    
    if analysis_results["crowd_count"] > 20:
        analysis_results["alerts"].append(f"High crowd density alert: {analysis_results['crowd_count']} people detected.")
    
    return analysis_results