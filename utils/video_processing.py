# ==============================================================================
# File: utils/video_processing.py (CLEANED)
# ==============================================================================
# Action: Replace the entire content of your video_processing.py file with this.
# ------------------------------------------------------------------------------
import cv2
from ultralytics import YOLO
from deepface import DeepFace
from utils.face_utils import recognize_face, initialize_known_faces

# --- Model Initialization ---
print("Initializing AI models...")
person_detector = YOLO('yolov8n.pt') 
known_face_embeddings, known_face_metadata = initialize_known_faces()
print("AI models initialized.")

def process_frame_for_analysis(frame):
    """
    Processes a single video frame for all AI analyses.
    """
    analysis_results = {
        "crowd_count": 0,
        "demographics": {"male": 0, "female": 0, "unknown": 0},
        "detected_faces": [],
        "alerts": []
    }
    person_detections = person_detector(frame, classes=[0], verbose=False)
    detected_persons_boxes = person_detections[0].boxes.data.cpu().numpy()
    analysis_results["crowd_count"] = len(detected_persons_boxes)

    for i, person_box in enumerate(detected_persons_boxes):
        x1, y1, x2, y2, conf, cls = person_box
        person_roi = frame[int(y1):int(y2), int(x1):int(x2)]
        if person_roi.size == 0:
            continue
        try:
            face_objs = DeepFace.analyze(person_roi, actions=['gender', 'emotion'], enforce_detection=False, silent=True)
            if isinstance(face_objs, list) and len(face_objs) > 0:
                face_info = face_objs[0]
                gender = face_info.get('dominant_gender', 'Unknown').lower()
                if gender == 'man': analysis_results["demographics"]["male"] += 1
                elif gender == 'woman': analysis_results["demographics"]["female"] += 1
                else: analysis_results["demographics"]["unknown"] += 1
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
        except Exception as e:
            analysis_results["demographics"]["unknown"] += 1
            pass

    if analysis_results["crowd_count"] > 20:
        analysis_results["alerts"].append(f"High crowd density alert: {analysis_results['crowd_count']} people detected.")
    unauthorized_persons = [face for face in analysis_results["detected_faces"] if face["recognition_status"] == "Unknown"]
    if unauthorized_persons:
        analysis_results["alerts"].append(f"Unauthorized person alert: {len(unauthorized_persons)} unknown face(s) detected.")
    return analysis_results
