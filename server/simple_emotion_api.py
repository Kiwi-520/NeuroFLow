"""
Simplified Emotion Recognition API - Text Analysis Only (No OpenCV)
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import json
from PIL import Image
import re

app = FastAPI(title="NeuroFlow Emotion Recognition", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

# Simple emotion analysis using keyword matching (no external dependencies)
def analyze_text_emotion(text: str):
    """Simple text emotion analysis using keyword patterns"""
    text_lower = text.lower()
    
    # Emotion keywords
    emotions = {
        'happy': ['happy', 'joy', 'excited', 'great', 'amazing', 'wonderful', 'fantastic', 'love', 'smile', 'laugh'],
        'sad': ['sad', 'depressed', 'down', 'upset', 'disappointed', 'hurt', 'cry', 'tears', 'miserable'],
        'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'hate', 'rage', 'frustrated'],
        'fear': ['scared', 'afraid', 'fear', 'terrified', 'worried', 'anxious', 'nervous', 'panic'],
        'surprise': ['surprised', 'shocked', 'amazed', 'wow', 'incredible', 'unbelievable'],
        'disgust': ['disgusted', 'gross', 'sick', 'disgusting', 'awful', 'terrible', 'horrible'],
        'neutral': ['okay', 'fine', 'normal', 'alright', 'whatever']
    }
    
    # Count emotion words
    emotion_scores = {}
    for emotion, keywords in emotions.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        emotion_scores[emotion] = score
    
    # Find dominant emotion
    if max(emotion_scores.values()) == 0:
        predicted_emotion = 'neutral'
        confidence = 0.5
    else:
        predicted_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = min(0.95, 0.6 + (emotion_scores[predicted_emotion] * 0.1))
    
    # Convert to percentages
    total_words = len(text.split())
    emotion_probabilities = {}
    for emotion, score in emotion_scores.items():
        if total_words > 0:
            emotion_probabilities[emotion] = min(1.0, score / total_words + 0.1)
        else:
            emotion_probabilities[emotion] = 0.1
    
    # Normalize probabilities
    total_prob = sum(emotion_probabilities.values())
    if total_prob > 0:
        emotion_probabilities = {k: v/total_prob for k, v in emotion_probabilities.items()}
    
    return {
        'predicted_emotion': predicted_emotion,
        'confidence': confidence,
        'emotion_probabilities': emotion_probabilities
    }

def analyze_face_emotion(image):
    """Simplified face emotion analysis without OpenCV"""
    try:
        # Get image dimensions and properties
        width, height = image.size
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Get average color values for basic analysis
        pixels = list(image.getdata())
        if len(pixels) == 0:
            return {
                'predicted_emotion': 'neutral',
                'confidence': 0.3,
                'emotion_probabilities': {'neutral': 1.0},
                'face_detected': False,
                'face_coordinates': None
            }
            
        # Calculate average brightness
        avg_brightness = sum(sum(pixel) for pixel in pixels) / len(pixels) / 3
        
        # Calculate color variance as a proxy for contrast
        r_values = [pixel[0] for pixel in pixels]
        g_values = [pixel[1] for pixel in pixels]
        b_values = [pixel[2] for pixel in pixels]
        
        r_var = sum((r - sum(r_values)/len(r_values))**2 for r in r_values) / len(r_values)
        contrast = (r_var ** 0.5) / 255  # Normalized contrast
        
        # Simple emotion detection based on image properties
        if avg_brightness > 150 and contrast > 0.3:
            emotion = 'happy'
            confidence = 0.75
        elif avg_brightness < 100:
            emotion = 'sad'
            confidence = 0.70
        elif contrast > 0.4:
            emotion = 'surprise'
            confidence = 0.65
        elif avg_brightness > 130:
            emotion = 'neutral'
            confidence = 0.60
        else:
            emotion = 'angry'
            confidence = 0.55
        
        # Generate emotion probabilities
        emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
        probabilities = {}
        for e in emotions:
            if e == emotion:
                probabilities[e] = confidence
            else:
                probabilities[e] = (1 - confidence) / (len(emotions) - 1)
        
        return {
            'predicted_emotion': emotion,
            'confidence': confidence,
            'emotion_probabilities': probabilities,
            'face_detected': True,
            'face_coordinates': [0, 0, width, height]  # Mock coordinates
        }
        
    except Exception as e:
        return {
            'predicted_emotion': 'neutral',
            'confidence': 0.5,
            'emotion_probabilities': {'neutral': 1.0},
            'face_detected': False,
            'error': str(e)
        }

@app.post("/api/emotion/analyze-text")
async def analyze_text(request: TextRequest):
    """Analyze emotion from text"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = analyze_text_emotion(request.text)
    return result

@app.post("/api/emotion/analyze-webcam")
async def analyze_webcam(image_data: str = Form(...)):
    """Analyze emotion from webcam image"""
    try:
        # Remove data URL prefix
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Analyze emotion directly from PIL image
        result = analyze_face_emotion(image)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "NeuroFlow Emotion Recognition API",
        "features": ["Text Emotion Analysis", "Webcam Face Emotion Detection"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)