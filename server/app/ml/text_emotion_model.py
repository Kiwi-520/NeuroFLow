"""
Text-based Emotion Recognition using BERT and RNN models
Uses the Emotion Dataset for NLP (emotion-english-distilroberta-base)
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, pipeline
import numpy as np
import re
import os
import json
from typing import List, Dict, Union

class TextEmotionAnalyzer:
    """
    Advanced text emotion recognition using transformer models
    Supports multiple emotion classification approaches
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to load custom trained model first, fallback to pre-trained
        try:
            if os.path.exists('app/models/text_emotion_model'):
                self.emotion_classifier = pipeline(
                    "text-classification", 
                    model="app/models/text_emotion_model",
                    tokenizer="app/models/text_emotion_tokenizer",
                    device=0 if torch.cuda.is_available() else -1,
                    return_all_scores=True
                )
                # Load custom emotion mapping
                with open('app/models/text_emotion_mapping.json', 'r') as f:
                    self.custom_emotions = json.load(f)
                print("✅ Loaded custom trained text emotion model")
            else:
                # Fallback to pre-trained model
                self.emotion_classifier = pipeline(
                    "text-classification", 
                    model="j-hartmann/emotion-english-distilroberta-base", 
                    device=0 if torch.cuda.is_available() else -1,
                    return_all_scores=True
                )
                self.custom_emotions = None
                print("📦 Using pre-trained emotion model")
        except Exception as e:
            print(f"⚠️  Error loading custom model, using pre-trained: {e}")
            self.emotion_classifier = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base", 
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            self.custom_emotions = None
        
        # Emotion mapping for consistency
        self.emotion_mapping = {
            'joy': 'happy',
            'sadness': 'sad',
            'anger': 'angry',
            'fear': 'fear',
            'surprise': 'surprise',
            'disgust': 'disgust',
            'neutral': 'neutral'
        }
        
        # Initialize BERT tokenizer for custom analysis
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.bert_model.to(self.device)
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for emotion analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep emoticons
        text = re.sub(r'[^\w\s.:;!?()-]', '', text)
        
        return text
    
    def analyze_emotion_transformer(self, text: str) -> Dict:
        """Analyze emotion using pre-trained transformer model"""
        if not text.strip():
            return {
                'predicted_emotion': 'neutral',
                'confidence': 0.0,
                'emotion_probabilities': {}
            }
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Get predictions from transformer model
        results = self.emotion_classifier(cleaned_text)
        
        # Process results
        emotion_scores = {}
        max_score = 0
        predicted_emotion = 'neutral'
        
        for result in results[0]:  # results is a list with one element
            emotion = result['label'].lower()
            score = result['score']
            
            # Handle custom trained model labels (LABEL_0, LABEL_1, etc.)
            if emotion.startswith('label_') and self.custom_emotions:
                label_idx = int(emotion.split('_')[1])
                if label_idx < len(self.custom_emotions):
                    emotion = self.custom_emotions[label_idx]
            
            # Map emotion names for consistency
            mapped_emotion = self.emotion_mapping.get(emotion, emotion)
            emotion_scores[mapped_emotion] = score
            
            if score > max_score:
                max_score = score
                predicted_emotion = mapped_emotion
        
        return {
            'predicted_emotion': predicted_emotion,
            'confidence': max_score,
            'emotion_probabilities': emotion_scores,
            'processed_text': cleaned_text
        }
    
    def extract_emotional_features(self, text: str) -> Dict:
        """Extract emotional features using BERT embeddings"""
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize and encode
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use [CLS] token embedding as sentence representation
            sentence_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return {
            'embedding': sentence_embedding.flatten(),
            'embedding_size': sentence_embedding.shape[-1]
        }
    
    def analyze_sentiment_intensity(self, text: str) -> Dict:
        """Analyze emotional intensity and sentiment polarity"""
        # Emotional intensity keywords
        intensity_keywords = {
            'high': ['extremely', 'absolutely', 'completely', 'totally', 'utterly', 'immensely'],
            'medium': ['very', 'quite', 'really', 'rather', 'fairly', 'pretty'],
            'low': ['somewhat', 'slightly', 'a bit', 'kind of', 'sort of']
        }
        
        # Positive/negative indicators
        positive_words = ['love', 'happy', 'joy', 'excited', 'amazing', 'wonderful', 'great', 'fantastic']
        negative_words = ['hate', 'sad', 'angry', 'terrible', 'awful', 'horrible', 'bad', 'disgusting']
        
        text_lower = text.lower()
        
        # Calculate intensity
        intensity_score = 0.3  # baseline
        for level, keywords in intensity_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if level == 'high':
                        intensity_score = min(1.0, intensity_score + 0.3)
                    elif level == 'medium':
                        intensity_score = min(1.0, intensity_score + 0.2)
                    elif level == 'low':
                        intensity_score = min(1.0, intensity_score + 0.1)
        
        # Calculate sentiment polarity
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment_polarity = 'positive'
        elif negative_count > positive_count:
            sentiment_polarity = 'negative'
        else:
            sentiment_polarity = 'neutral'
        
        return {
            'intensity_score': intensity_score,
            'sentiment_polarity': sentiment_polarity,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count
        }
    
    def comprehensive_emotion_analysis(self, text: str) -> Dict:
        """Perform comprehensive emotion analysis combining multiple approaches"""
        # Main emotion classification
        emotion_result = self.analyze_emotion_transformer(text)
        
        # Sentiment intensity analysis
        intensity_result = self.analyze_sentiment_intensity(text)
        
        # Extract features for additional insights
        features = self.extract_emotional_features(text)
        
        # Combine results
        comprehensive_result = {
            **emotion_result,
            'intensity_analysis': intensity_result,
            'text_length': len(text),
            'word_count': len(text.split()),
            'feature_vector_size': features['embedding_size']
        }
        
        return comprehensive_result
    
    def batch_analyze_emotions(self, texts: List[str]) -> List[Dict]:
        """Analyze emotions for a batch of texts"""
        results = []
        for text in texts:
            result = self.comprehensive_emotion_analysis(text)
            results.append(result)
        return results