from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Union
import joblib
import numpy as np
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

class EmotionResponse(BaseModel):
    emotion: str
    probability: float
    emoji: str

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

class EmotionDetector:
    def __init__(self):
        self.model_path = Path(__file__).parent / "models" / "emotion_classifier_pipe_lr.pkl"
        self.emotions_emoji_dict = {
            "anger": "😠", "disgust": "🤮", "fear": "😨😱", 
            "happy": "🤗", "joy": "😂", "neutral": "😐", 
            "sad": "😔", "sadness": "😔", "shame": "😳", 
            "surprise": "😮"
        }
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=4)  # For parallel processing
    
    def load_model(self):
        """Lazy load the model only when needed"""
        if self.model is None:
            self.model = joblib.load(open(self.model_path, "rb"))
        return self.model

    @lru_cache(maxsize=1000)
    def predict_emotion(self, text: str) -> Dict[str, Union[str, float]]:
        """Predict emotion for a single text input with caching"""
        model = self.load_model()
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        max_prob = np.max(probabilities)
        
        return {
            "emotion": prediction,
            "probability": float(max_prob),
            "emoji": self.emotions_emoji_dict.get(prediction, "")
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """Predict emotions for a batch of texts using parallel processing"""
        # Process texts in parallel using the thread pool
        futures = [self.executor.submit(self.predict_emotion, text) for text in texts]
        return [future.result() for future in futures]

# Initialize the detector
detector = EmotionDetector()

@app.post("/api/emotions/analyze", response_model=EmotionResponse)
async def analyze_emotion(input_data: TextInput):
    """Analyze emotion in a single text"""
    try:
        result = detector.predict_emotion(input_data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emotions/analyze-batch", response_model=List[EmotionResponse])
async def analyze_emotions_batch(input_data: BatchTextInput):
    """Analyze emotions in multiple texts"""
    try:
        results = detector.predict_batch(input_data.texts)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))