import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Union
import json
from pathlib import Path

class EmotionTransformer(nn.Module):
    def __init__(self, num_emotions: int = 8):
        super().__init__()
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_emotions)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # Take [CLS] token output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class DeepEmotionAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = EmotionTransformer()
        self.model.to(self.device)
        
        # Load model weights
        model_path = Path(__file__).parent / "weights" / "emotion_transformer.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 'trust']
        self.emoji_map = {
            'anger': '😠', 'disgust': '🤮', 'fear': '😨',
            'joy': '😊', 'neutral': '😐', 'sadness': '😔',
            'surprise': '😮', 'trust': '🤝'
        }
        
    def preprocess(self, text: str) -> Dict:
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].to(self.device),
            'attention_mask': inputs['attention_mask'].to(self.device)
        }
        
    def predict(self, text: str) -> Dict[str, Union[str, float, str]]:
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess(text)
            outputs = self.model(**inputs)
            probs = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1)
            confidence = float(probs[0][prediction])
            emotion = self.emotions[prediction]
            
        return {
            'emotion': emotion,
            'probability': confidence,
            'emoji': self.emoji_map.get(emotion, ''),
            'emotion_spectrum': {
                emotion: float(prob) 
                for emotion, prob in zip(self.emotions, probs[0].tolist())
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float, str]]]:
        self.model.eval()
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
            
        return results

    def get_emotional_insights(self, texts: List[str]) -> Dict:
        """Generate comprehensive emotional insights from a list of texts"""
        predictions = self.predict_batch(texts)
        
        # Aggregate emotions
        emotion_counts = {emotion: 0 for emotion in self.emotions}
        emotion_scores = {emotion: [] for emotion in self.emotions}
        
        for pred in predictions:
            emotion = pred['emotion']
            emotion_counts[emotion] += 1
            for emotion, score in pred['emotion_spectrum'].items():
                emotion_scores[emotion].append(score)
        
        # Calculate statistics
        total = len(predictions)
        emotion_stats = {}
        for emotion in self.emotions:
            avg_score = np.mean(emotion_scores[emotion]) if emotion_scores[emotion] else 0
            emotion_stats[emotion] = {
                'count': emotion_counts[emotion],
                'percentage': (emotion_counts[emotion] / total) * 100 if total > 0 else 0,
                'average_intensity': avg_score
            }
        
        return {
            'total_entries': total,
            'emotion_distribution': emotion_stats,
            'dominant_emotion': max(emotion_counts.items(), key=lambda x: x[1])[0],
            'emotional_volatility': np.std([pred['probability'] for pred in predictions])
        }

# Initialize the analyzer
analyzer = DeepEmotionAnalyzer()