# 🧠 NeuroFlow Emotion Recognition Setup Guide

Complete implementation of real-time emotion recognition for both text and images using deep learning.

## 📁 Dataset Structure Required

### For Image Emotion Recognition (FER2013):

```
server/datasets/image/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

### For Text Emotion Recognition:

```
server/datasets/text/
└── tweet_emotions.csv  (with columns: tweet_id, sentiment, content)
```

## 🚀 Quick Start

### Option 1: Automated Setup (Windows)

```bash
cd server
.\start_emotion_recognition.bat
```

### Option 2: Manual Setup

1. **Install Dependencies**

   ```bash
   cd server
   pip install -r requirements.txt
   ```

2. **Train Models**

   ```bash
   python comprehensive_training.py
   ```

3. **Start API Server**

   ```bash
   python main.py
   ```

4. **Start Frontend**

   ```bash
   cd ../client
   npm install
   npm run dev
   ```

5. **Open Application**
   - Navigate to http://localhost:5173
   - Click on "Emotion Analysis" tab
   - Test with text input or webcam!

## 🎯 Features Implemented

### Text Emotion Recognition

- ✅ Deep learning model trained on your tweet dataset
- ✅ 7 emotion categories: sadness, joy, love, anger, fear, surprise, neutral
- ✅ Real-time analysis with confidence scores
- ✅ Comprehensive emotion breakdown visualization

### Image Emotion Recognition

- ✅ CNN model trained on FER2013 dataset
- ✅ Facial emotion detection from images/webcam
- ✅ Real-time video analysis
- ✅ Face detection and emotion mapping

### Web Interface

- ✅ Beautiful, responsive UI with dark/light themes
- ✅ Three analysis modes: Text, Image, Real-time
- ✅ Interactive emotion visualization charts
- ✅ Webcam integration for live emotion detection
- ✅ File upload for image analysis

## 🔧 Technical Details

### Backend (FastAPI)

- **Text Model**: DistilBERT fine-tuned on your dataset
- **Image Model**: Custom CNN architecture for FER2013
- **API Endpoints**: RESTful APIs for all emotion analysis
- **Real-time Processing**: Optimized inference pipeline

### Frontend (React + TypeScript)

- **Modern UI**: Built with Tailwind CSS and Lucide icons
- **Real-time Updates**: Live webcam emotion detection
- **Responsive Design**: Works on desktop and mobile
- **Accessibility**: Full keyboard navigation and screen reader support

## 📊 Model Performance

After training, you'll see:

- Text model accuracy displayed in console
- Image model accuracy and training curves
- Real-time inference speed metrics
- Confusion matrices and classification reports

## 🎬 Usage Examples

1. **Text Analysis**: "I'm so excited about this new project!" → Joy (95% confidence)
2. **Image Analysis**: Upload selfie → Happy (87% confidence)
3. **Real-time**: Webcam feed → Live emotion detection every frame

## 🔍 API Endpoints

- `POST /api/emotion/analyze-text` - Analyze text emotions
- `POST /api/emotion/analyze-image` - Analyze image emotions
- `POST /api/emotion/realtime/analyze` - Real-time multi-modal analysis
- `GET /api/emotion/model/status` - Check model status
- `GET /docs` - Interactive API documentation

## 🛠️ Troubleshooting

**Training Issues:**

- Ensure datasets are in correct folder structure
- Check that CSV has required columns: tweet_id, sentiment, content
- Verify sufficient disk space (models ~500MB each)

**Runtime Issues:**

- Make sure both server (port 8000) and client (port 5173) are running
- Check browser console for any CORS or network errors
- Verify webcam permissions are granted

**Performance:**

- GPU recommended for training (30-60 minutes vs 2-4 hours on CPU)
- Training creates visualization plots saved as PNG files
- Models automatically save best checkpoints during training

## 🎉 What You Get

After setup, you'll have a fully functional emotion recognition system that can:

- Analyze emotions in real-time from text input
- Detect facial emotions from uploaded images
- Perform live emotion recognition via webcam
- Display beautiful visualizations of emotion probabilities
- Provide confidence scores and detailed breakdowns

Perfect for research, applications, or demonstrations of modern AI emotion recognition capabilities!
