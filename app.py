# ==============================================================================
# File: app.py (CLEANED)
# ==============================================================================
# Action: Replace the entire content of your app.py file with this code.
# ------------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import asyncio
from utils.video_processing import process_frame_for_analysis

app = FastAPI(
    title="AI Crowd Monitoring Service",
    description="An API for real-time crowd analysis.",
    version="1.0.0"
)

class AnalysisRequest(BaseModel):
    image_data: str

class AnalysisResponse(BaseModel):
    crowd_count: int
    demographics: dict
    detected_faces: list
    alerts: list

@app.get("/")
def read_root():
    return {"status": "AI Service is running and ready."}

@app.post("/analyze_frame", response_model=AnalysisResponse)
async def analyze_video_frame(request: AnalysisRequest):
    try:
        img_data = base64.b64decode(request.image_data)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {e}")

    try:
        loop = asyncio.get_event_loop()
        analysis_results = await loop.run_in_executor(None, process_frame_for_analysis, frame)
        return analysis_results
    except Exception as e:
        print(f"FATAL: Error during processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during AI analysis.")