#!/bin/bash

# NeuroFlow Emotion Recognition Quick Start Script
echo "🚀 NeuroFlow Emotion Recognition System Setup"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "comprehensive_training.py" ]; then
    echo "❌ Please run this script from the server directory"
    exit 1
fi

# Check if datasets exist
echo "📂 Checking datasets..."
if [ ! -f "datasets/text/tweet_emotions.csv" ]; then
    echo "❌ Text dataset not found at datasets/text/tweet_emotions.csv"
    echo "   Please place your tweet emotion CSV file there"
    exit 1
fi

if [ ! -d "datasets/image/train" ]; then
    echo "❌ Image dataset not found at datasets/image/train/"
    echo "   Please place your FER2013 images in the correct folder structure"
    exit 1
fi

echo "✅ Datasets found!"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Run training
echo "🎯 Starting model training..."
echo "This may take 30-60 minutes depending on your hardware..."
python comprehensive_training.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo ""
    echo "🚀 Starting the API server..."
    echo "The server will run on http://localhost:8000"
    echo "API docs available at http://localhost:8000/docs"
    echo ""
    echo "Next steps:"
    echo "1. Keep this server running"
    echo "2. Open a new terminal and navigate to the client folder"
    echo "3. Run: npm install && npm run dev"
    echo "4. Open http://localhost:5173 in your browser"
    echo "5. Go to the 'Emotion Analysis' tab"
    echo ""
    
    # Start the server
    python main.py
else
    echo "❌ Training failed. Please check the logs above."
    exit 1
fi