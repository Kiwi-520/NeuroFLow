import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the current directory
current_dir = Path(__file__).parent

# Load the model
try:
    model_path = current_dir / "models" / "emotion_classifier_pipe_lr.pkl"
    pipe_lr = joblib.load(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮", 
    "fear": "😨",
    "joy": "😂",
    "neutral": "😐",
    "sadness": "😔",
    "surprise": "😮",
    "trust": "🤝"
}

def predict_emotion(text: str) -> Dict:
    # Predict emotion
    emotion = pipe_lr.predict([text])[0]
    probabilities = pipe_lr.predict_proba([text])[0]
    
    # Get probability distribution
    emotion_probs = {
        emotion: float(prob)
        for emotion, prob in zip(pipe_lr.classes_, probabilities)
    }
    
    return {
        "emotion": emotion,
        "probability": float(max(probabilities)),
        "emoji": emotions_emoji_dict[emotion],
        "emotion_spectrum": emotion_probs
    }

def analyze_batch_emotions(texts: List[str]) -> List[Dict]:
    return [predict_emotion(text) for text in texts]

def calculate_emotional_insights(texts: List[str], start_date: str, end_date: str) -> Dict:
    # Analyze all texts
    emotions = [predict_emotion(text) for text in texts]
    
    # Calculate emotion distribution
    emotion_counts = {emotion: 0 for emotion in emotions_emoji_dict.keys()}
    emotion_intensities = {emotion: [] for emotion in emotions_emoji_dict.keys()}
    
    for result in emotions:
        emotion = result["emotion"]
        emotion_counts[emotion] += 1
        for e, prob in result["emotion_spectrum"].items():
            emotion_intensities[e].append(prob)
    
    total = len(emotions)
    distribution = {}
    
    for emotion in emotions_emoji_dict.keys():
        count = emotion_counts[emotion]
        intensities = emotion_intensities[emotion]
        distribution[emotion] = {
            "count": count,
            "percentage": (count / total * 100) if total > 0 else 0,
            "average_intensity": np.mean(intensities) if intensities else 0
        }
    
    # Find dominant emotion
    dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    # Calculate emotional volatility
    volatility = np.std([result["probability"] for result in emotions]) if emotions else 0
    
    # Create temporal patterns (mock data for now)
    temporal_patterns = {
        start_date: [0.7 for _ in range(8)],
        end_date: [0.8 for _ in range(8)]
    }
    
    return {
        "total_entries": total,
        "emotion_distribution": distribution,
        "dominant_emotion": dominant_emotion,
        "emotional_volatility": float(volatility),
        "temporal_patterns": temporal_patterns
    }