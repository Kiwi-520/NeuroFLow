from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional
import json
import logging
from app.ml.emotion_classifier import predict_emotion, analyze_batch_emotions, calculate_emotional_insights

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NeuroFlow Emotion Analysis API")

# Configure CORS properly for GitHub Codespace environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# Error handler for generic exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"An error occurred: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred", "message": str(exc)},
    )

class TextAnalysisRequest(BaseModel):
    text: str

class BatchAnalysisRequest(BaseModel):
    texts: List[str]

class TimeRange(BaseModel):
    start_date: datetime
    end_date: datetime

class InsightsRequest(BaseModel):
    texts: Optional[List[str]] = []
    time_range: TimeRange

@app.get("/")
async def root():
    return {"message": "Emotion Analysis API is running"}

@app.post("/api/emotions/analyze")
async def analyze_emotion(request: TextAnalysisRequest):
    logger.info(f"Received analysis request for text: {request.text[:50]}...")
    try:
        result = predict_emotion(request.text)
        logger.info(f"Analysis successful: {result['emotion']}")
        return result
    except Exception as e:
        logger.error(f"Error analyzing emotion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"message": "Failed to analyze emotion", "error": str(e)}
        )

@app.post("/api/emotions/analyze-batch")
async def analyze_batch(request: BatchAnalysisRequest):
    try:
        results = analyze_batch_emotions(request.texts)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emotions/insights")
async def get_emotional_insights(request: InsightsRequest):
    try:
        return calculate_emotional_insights(
            request.texts,
            request.time_range.start_date.isoformat(),
            request.time_range.end_date.isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))