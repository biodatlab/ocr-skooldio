from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any

import shutil
import os
import json
import uvicorn
from datetime import datetime
import uuid

# Import your detector class
from webapp_modules import CarbookFieldsDetector

app = FastAPI(
    title="Carbook Fields Detector API",
    description="API for detecting and recognizing text fields in vehicle images",
    version="1.0.0"
)

# Initialize detector
detector = CarbookFieldsDetector(detector_model_path="models/best.pt")

class DetectionResponse(BaseModel):
    image_id: str
    detected_fields: Dict[str, Dict[str, Any]]
    processed_image_path: str

@app.post("/detect/", response_model=DetectionResponse)
async def detect_fields(file: UploadFile = File(...)):
    """
    Detect and recognize text fields in an uploaded image.
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON response with detection results
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create unique ID for this detection
        image_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create temporary directory for this detection
        temp_dir = f"temp/detection_{timestamp}_{image_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded file
        temp_image_path = os.path.join(temp_dir, file.filename)
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process image
        drawn_image_path, field_texts, json_path = detector.detect(
            image_path=temp_image_path,
            crop_images=True
        )
        
        # Read detection results
        with open(json_path, 'r', encoding='utf-8') as f:
            detection_results = json.load(f)

        return DetectionResponse(
            image_id=image_id,
            detected_fields=detection_results,
            processed_image_path=drawn_image_path
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary files after some time (you might want to implement this)
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)