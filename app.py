from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import asyncio
from utils.video_processing import process_frame_for_analysis, process_frame_for_attendance

app = FastAPI(
    title="Intelligent Campus AI Service",
    description="An API for real-time attendance and safety analysis.",
    version="1.0.0"
)

# --- Pydantic Models for API Data Validation ---
class ImageRequest(BaseModel):
    image_data: str # Base64 encoded image string

class SafetyResponse(BaseModel):
    crowd_count: int
    anomaly_status: str
    alerts: list

class AttendanceResponse(BaseModel):
    present_students: list
    unrecognized_faces: int

# --- Helper Function ---
def _decode_image(image_data: str):
    """Decodes a base64 string into an OpenCV image."""
    try:
        img_data = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data provided.")
        return frame
    except Exception:
        raise HTTPException(status_code=400, detail="Error decoding base64 image string.")

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Intelligent Campus AI Service is running."}

@app.post("/analyze_safety", response_model=SafetyResponse)
async def analyze_safety_frame(request: ImageRequest):
    """Endpoint for the Campus Safety Module."""
    frame = _decode_image(request.image_data)
    try:
        loop = asyncio.get_event_loop()
        # Run synchronous, CPU-bound AI code in a separate thread pool
        analysis_results = await loop.run_in_executor(None, process_frame_for_analysis, frame)
        return analysis_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during safety analysis: {str(e)}")

@app.post("/mark_attendance", response_model=AttendanceResponse)
async def mark_attendance_frame(request: ImageRequest):
    """Endpoint for the Automated Attendance Module."""
    frame = _decode_image(request.image_data)
    try:
        loop = asyncio.get_event_loop()
        attendance_results = await loop.run_in_executor(None, process_frame_for_attendance, frame)
        return attendance_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during attendance marking: {str(e)}")
