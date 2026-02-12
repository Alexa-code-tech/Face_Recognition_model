"""
FastAPI Application for Face Recognition

REST API for face recognition system
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2
import io
from PIL import Image
import logging

from models.recognition_pipeline import RecognitionPipeline
from src.utils import setup_logging, load_embeddings

logger = logging.getLogger(__name__)

# Инициализировать приложение
app = FastAPI(
    title="Face Recognition System",
    description="Advanced face recognition API",
    version="1.0.0"
)

# Глобальная переменная для pipeline
pipeline = None


class RecognitionResponse(BaseModel):
    """Модель ответа для распознавания"""
    faces_detected: int
    recognized_people: List[dict]


class VerificationResponse(BaseModel):
    """Модель ответа для верификации"""
    verified: bool
    confidence: float
    message: str


@app.on_event("startup")
async def startup_event():
    """Инициализировать pipeline при запуске"""
    global pipeline
    
    setup_logging()
    
    pipeline = RecognitionPipeline(
        detector_name="mtcnn",
        embedder_name="facenet",
        distance_metric="euclidean",
        tolerance=0.6,
        device="cuda"
    )
    
    # Загрузить эмбеддинги
    embeddings_dict = load_embeddings("data/embeddings/face_embeddings.pkl")
    for name, embeddings in embeddings_dict.items():
        pipeline.matcher.register_face(name, embeddings)
    
    logger.info(f"Loaded {len(embeddings_dict)} people")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Recognition System API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/status")
async def status():
    """Get system status"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    stats = pipeline.get_stats()
    return {
        "status": "ok",
        "system": stats
    }


@app.post("/recognize", response_model=RecognitionResponse)
async def recognize(file: UploadFile = File(...), tolerance: float = Form(0.6)):
    """
    Recognize faces in image
    
    Args:
        file: Image file (jpg, png)
        tolerance: Matching tolerance (0.0-1.0)
        
    Returns:
        RecognitionResponse with detected faces
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Читать файл
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Распознать лица
        pipeline.matcher.tolerance = tolerance
        results = pipeline.recognize(image)
        
        return RecognitionResponse(**results)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recognition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify", response_model=VerificationResponse)
async def verify(file: UploadFile = File(...),
                name: str = Form(...),
                tolerance: float = Form(0.6)):
    """
    Verify if a specific person is in the image
    
    Args:
        file: Image file
        name: Person name to verify
        tolerance: Matching tolerance
        
    Returns:
        VerificationResponse
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Читать файл
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Проверить если человек существует
        known_people = pipeline.matcher.get_known_people()
        if name not in known_people:
            raise HTTPException(status_code=400, detail=f"Unknown person: {name}")
        
        # Верифицировать
        pipeline.matcher.tolerance = tolerance
        result = pipeline.verify(image, name)
        
        return VerificationResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in verification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/people")
async def get_people():
    """Get list of known people"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    known_people = pipeline.matcher.get_known_people()
    
    return {
        "count": len(known_people),
        "people": known_people
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)