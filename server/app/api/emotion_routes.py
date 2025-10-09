"""
API endpoints for emotion recognition
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import io
import base64
from PIL import Image
import numpy as np
from ..ml.image_emotion_model import ImageEmotionRecognizer
from ..ml.text_emotion_model import TextEmotionAnalyzer
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/emotion", tags=["emotion_recognition"])

# Initialize models (these will be loaded when the server starts)
image_analyzer = None
text_analyzer = None

class TextEmotionRequest(BaseModel):
    text: str
    analysis_type: Optional[str] = "comprehensive"  # "basic", "comprehensive", "batch"

class TextEmotionResponse(BaseModel):
    predicted_emotion: str
    confidence: float
    emotion_probabilities: dict
    intensity_analysis: Optional[dict] = None
    text_length: Optional[int] = None
    word_count: Optional[int] = None

class ImageEmotionResponse(BaseModel):
    predicted_emotion: str
    confidence: float
    emotion_probabilities: dict
    face_coordinates: Optional[tuple] = None
    processing_time: Optional[float] = None

@router.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global image_analyzer, text_analyzer
    
    try:
        # Initialize text analyzer (doesn't require pre-trained weights)
        text_analyzer = TextEmotionAnalyzer()
        logger.info("Text emotion analyzer initialized successfully")
        
        # Initialize image analyzer (will use pre-trained model if available)
        model_path = "app/models/emotion_cnn_fer2013.pth"
        image_analyzer = ImageEmotionRecognizer(model_path)
        logger.info("Image emotion analyzer initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing emotion analyzers: {str(e)}")
        # Initialize without pre-trained models for development
        text_analyzer = TextEmotionAnalyzer()
        image_analyzer = ImageEmotionRecognizer()

@router.post("/analyze-text", response_model=TextEmotionResponse)
async def analyze_text_emotion(request: TextEmotionRequest):
    """
    Analyze emotion from text input
    """
    try:
        if not text_analyzer:
            raise HTTPException(status_code=500, detail="Text analyzer not initialized")
        
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        # Perform emotion analysis
        if request.analysis_type == "comprehensive":
            result = text_analyzer.comprehensive_emotion_analysis(request.text)
        else:
            result = text_analyzer.analyze_emotion_transformer(request.text)
        
        return TextEmotionResponse(**result)
    
    except Exception as e:
        logger.error(f"Error in text emotion analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing text emotion: {str(e)}")

@router.post("/analyze-text-batch")
async def analyze_text_emotion_batch(texts: List[str]):
    """
    Analyze emotions for multiple texts
    """
    try:
        if not text_analyzer:
            raise HTTPException(status_code=500, detail="Text analyzer not initialized")
        
        if not texts or len(texts) == 0:
            raise HTTPException(status_code=400, detail="Text list cannot be empty")
        
        if len(texts) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
        
        results = text_analyzer.batch_analyze_emotions(texts)
        
        return {
            "batch_size": len(texts),
            "results": results,
            "average_confidence": sum(r['confidence'] for r in results) / len(results)
        }
    
    except Exception as e:
        logger.error(f"Error in batch text emotion analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing batch text emotions: {str(e)}")

@router.post("/analyze-image", response_model=List[ImageEmotionResponse])
async def analyze_image_emotion(file: UploadFile = File(...)):
    """
    Analyze emotion from uploaded image
    """
    try:
        if not image_analyzer:
            raise HTTPException(status_code=500, detail="Image analyzer not initialized")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Analyze emotion (includes face detection)
        import time
        start_time = time.time()
        
        results = image_analyzer.detect_face_and_predict(np.array(image))
        
        processing_time = time.time() - start_time
        
        # Add processing time to results
        for result in results:
            result['processing_time'] = processing_time
        
        return results
    
    except Exception as e:
        logger.error(f"Error in image emotion analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image emotion: {str(e)}")

@router.post("/analyze-image-base64")
async def analyze_image_emotion_base64(
    image_data: str = Form(...),
    detect_faces: bool = Form(True)
):
    """
    Analyze emotion from base64 encoded image
    """
    try:
        if not image_analyzer:
            raise HTTPException(status_code=500, detail="Image analyzer not initialized")
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # Analyze emotion
        import time
        start_time = time.time()
        
        if detect_faces:
            results = image_analyzer.detect_face_and_predict(np.array(image))
        else:
            result = image_analyzer.predict_emotion(image)
            result['face_coordinates'] = None
            results = [result]
        
        processing_time = time.time() - start_time
        
        # Add processing time to results
        for result in results:
            result['processing_time'] = processing_time
        
        return {
            "results": results,
            "faces_detected": len([r for r in results if r.get('face_coordinates')]),
            "processing_time": processing_time
        }
    
    except Exception as e:
        logger.error(f"Error in base64 image emotion analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image emotion: {str(e)}")

@router.get("/emotions/list")
async def get_supported_emotions():
    """
    Get list of supported emotions
    """
    return {
        "image_emotions": ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        "text_emotions": ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'joy'],
        "emotion_categories": {
            "positive": ['happy', 'joy', 'surprise'],
            "negative": ['angry', 'sad', 'fear', 'disgust'],
            "neutral": ['neutral']
        }
    }

@router.get("/model/status")
async def get_model_status():
    """
    Get status of emotion recognition models
    """
    return {
        "text_analyzer_ready": text_analyzer is not None,
        "image_analyzer_ready": image_analyzer is not None,
        "models_loaded": {
            "text_transformer": text_analyzer is not None,
            "image_cnn": image_analyzer is not None and hasattr(image_analyzer.model, 'state_dict')
        }
    }

@router.post("/realtime/analyze")
async def realtime_emotion_analysis(
    text: Optional[str] = Form(None),
    image_data: Optional[str] = Form(None)
):
    """
    Real-time emotion analysis for both text and image
    """
    results = {}
    
    try:
        # Analyze text if provided
        if text and text.strip():
            if not text_analyzer:
                results['text_error'] = "Text analyzer not available"
            else:
                text_result = text_analyzer.comprehensive_emotion_analysis(text)
                results['text_analysis'] = text_result
        
        # Analyze image if provided
        if image_data:
            if not image_analyzer:
                results['image_error'] = "Image analyzer not available"
            else:
                try:
                    # Decode base64 image
                    if image_data.startswith('data:image'):
                        image_data = image_data.split(',')[1]
                    
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Analyze emotion
                    image_results = image_analyzer.detect_face_and_predict(np.array(image))
                    results['image_analysis'] = image_results
                    
                except Exception as e:
                    results['image_error'] = f"Error processing image: {str(e)}"
        
        if not text and not image_data:
            raise HTTPException(status_code=400, detail="Either text or image_data must be provided")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in realtime emotion analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in realtime analysis: {str(e)}")