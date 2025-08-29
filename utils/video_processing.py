# ==============================================================================
# File: utils/video_processing.py (UPGRADED FOR DENSE CROWDS)
# ==============================================================================
# Action: Replace the entire content of your video_processing.py file with this.
# ------------------------------------------------------------------------------
import cv2
from ultralytics import YOLO
from deepface import DeepFace
from utils.face_utils import recognize_face, initialize_known_faces
import numpy as np

# --- Model Initialization ---
print("Initializing AI models...")
person_detector = YOLO('yolov8n.pt') 
known_face_embeddings, known_face_metadata = initialize_known_faces()
print("AI models initialized.")

# --- NEW: Define a threshold for switching to the dense crowd model ---
# If YOLO detects more than this many people, we'll use the regression approach.
DENSE_CROWD_THRESHOLD = 30 

def estimate_dense_crowd_count(frame):
    """
    This is a placeholder for a dedicated dense crowd counting model (like CSRNet).
    For this project, we'll simulate its behavior by creating a simple density-based estimate.
    A real implementation would generate and sum a density map here.
    """
    # Convert to grayscale for feature detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use corner detection as a proxy for density. More corners often mean more people.
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    
    if corners is not None:
        # Return a count based on the number of detected corners, with a scaling factor.
        # This simulates a regression model's output.
        return int(corners.shape[0] * 1.2) 
    return 0


def process_frame_for_analysis(frame):
    """
    Processes a single video frame for all AI analyses, now with a hybrid approach
    for handling both sparse and dense crowds.
    """
    analysis_results = {
        "crowd_count": 0,
        "demographics": {"male": 0, "female": 0, "unknown": 0},
        "detected_faces": [],
        "alerts": []
    }

    # --- 1. Initial Person Detection (for density check) ---
    person_detections = person_detector(frame, classes=[0], verbose=False)
    detected_persons_boxes = person_detections[0].boxes.data.cpu().numpy()
    initial_detection_count = len(detected_persons_boxes)

    # --- 2. Intelligent Model Switching ---
    if initial_detection_count >= DENSE_CROWD_THRESHOLD:
        # --- HIGH-DENSITY CROWD LOGIC ---
        print("High density detected. Switching to regression model.")
        analysis_results["crowd_count"] = estimate_dense_crowd_count(frame)
        # In a dense crowd, individual face analysis is unreliable, so we skip it.
        analysis_results["demographics"]["unknown"] = analysis_results["crowd_count"]
        analysis_results["alerts"].append(f"High crowd density alert: Estimated {analysis_results['crowd_count']} people.")

    else:
        # --- LOW-DENSITY CROWD LOGIC (Original Method) ---
        print("Low/Medium density detected. Using detection model.")
        analysis_results["crowd_count"] = initial_detection_count

        for i, person_box in enumerate(detected_persons_boxes):
            x1, y1, x2, y2, conf, cls = person_box
            person_roi = frame[int(y1):int(y2), int(x1):int(x2)]
            if person_roi.size == 0:
                continue
            try:
                # Analyze face for gender and other attributes
                face_objs = DeepFace.analyze(person_roi, actions=['gender', 'emotion'], enforce_detection=False, silent=True)
                if isinstance(face_objs, list) and len(face_objs) > 0:
                    face_info = face_objs[0]
                    gender = face_info.get('dominant_gender', 'Unknown').lower()
                    if gender == 'man': analysis_results["demographics"]["male"] += 1
                    elif gender == 'woman': analysis_results["demographics"]["female"] += 1
                    else: analysis_results["demographics"]["unknown"] += 1
                    
                    # Perform facial recognition
                    embedding_objs = DeepFace.represent(person_roi, model_name='VGG-Face', enforce_detection=False)
                    face_embedding = embedding_objs[0]["embedding"]
                    recognition_status = recognize_face(face_embedding, known_face_embeddings)
                    
                    face_box_region = face_info['region']
                    face_data = {
                        "id": f"person_{i+1}",
                        "box_person": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                        "box_face": [int(x1) + face_box_region['x'], int(y1) + face_box_region['y'], face_box_region['w'], face_box_region['h']],
                        "gender": gender.capitalize(),
                        "emotion": face_info.get('dominant_emotion', 'Unknown').capitalize(),
                        "recognition_status": "Verified: " + recognition_status if recognition_status != "Unknown" else "Unknown"
                    }
                    analysis_results["detected_faces"].append(face_data)
                else:
                    analysis_results["demographics"]["unknown"] += 1
            except Exception as e:
                analysis_results["demographics"]["unknown"] += 1
                pass
        
        unauthorized_persons = [face for face in analysis_results["detected_faces"] if face["recognition_status"] == "Unknown"]
        if unauthorized_persons:
            analysis_results["alerts"].append(f"Unauthorized person alert: {len(unauthorized_persons)} unknown face(s) detected.")

    return analysis_results
