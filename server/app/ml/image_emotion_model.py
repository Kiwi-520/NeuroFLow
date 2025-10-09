"""
Image-based Emotion Recognition using CNN on FER2013 dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
from pathlib import Path

class EmotionCNN(nn.Module):
    """
    Deep CNN for emotion recognition from facial expressions
    Based on FER2013 dataset (7 emotions: angry, disgust, fear, happy, sad, surprise, neutral)
    """
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second Convolutional Block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third Convolutional Block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn8 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten and fully connected
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout4(x)
        x = F.relu(self.bn8(self.fc2(x)))
        x = self.dropout5(x)
        x = self.fc3(x)
        
        return x

class ImageEmotionRecognizer:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EmotionCNN()
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No pre-trained model found. Please train the model first.")
    
    def load_model(self, model_path):
        """Load pre-trained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path}")
    
    def preprocess_image(self, image):
        """Preprocess image for emotion recognition"""
        if isinstance(image, str):
            # Load from file path
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # Convert from numpy array (OpenCV format)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict_emotion(self, image):
        """Predict emotion from image"""
        if not hasattr(self.model, 'state_dict'):
            raise ValueError("Model not loaded. Please train or load a pre-trained model.")
        
        with torch.no_grad():
            image_tensor = self.preprocess_image(image)
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            emotion = self.emotions[predicted.item()]
            confidence_score = confidence.item()
            
            # Get all emotion probabilities
            emotion_scores = {}
            for i, emotion_name in enumerate(self.emotions):
                emotion_scores[emotion_name] = probabilities[0][i].item()
        
        return {
            'predicted_emotion': emotion,
            'confidence': confidence_score,
            'emotion_probabilities': emotion_scores
        }
    
    def detect_face_and_predict(self, image):
        """Detect face in image and predict emotion"""
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            emotion_result = self.predict_emotion(face_roi)
            emotion_result['face_coordinates'] = (x, y, w, h)
            results.append(emotion_result)
        
        if not results:
            # No face detected, try to predict on whole image
            emotion_result = self.predict_emotion(img)
            emotion_result['face_coordinates'] = None
            results.append(emotion_result)
        
        return results