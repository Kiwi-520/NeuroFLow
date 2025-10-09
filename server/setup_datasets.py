"""
Script to download and prepare datasets for emotion recognition training
"""
import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import kaggle
from datasets import load_dataset
import torch
from sklearn.model_selection import train_test_split

def setup_directories():
    """Create necessary directories"""
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("app/models", exist_ok=True)
    print("Directories created successfully!")

def download_text_emotion_dataset():
    """Download and prepare text emotion dataset"""
    print("Downloading text emotion dataset...")
    
    try:
        # Use Hugging Face datasets - emotion dataset
        dataset = load_dataset("emotion")
        
        # Convert to pandas DataFrames
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        val_df = pd.DataFrame(dataset['validation'])
        
        # Map emotion labels to names
        emotion_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        train_df['emotion_name'] = train_df['label'].map(lambda x: emotion_names[x])
        test_df['emotion_name'] = test_df['label'].map(lambda x: emotion_names[x])
        val_df['emotion_name'] = val_df['label'].map(lambda x: emotion_names[x])
        
        # Save datasets
        train_df.to_csv('datasets/emotion_train.csv', index=False)
        test_df.to_csv('datasets/emotion_test.csv', index=False)
        val_df.to_csv('datasets/emotion_val.csv', index=False)
        
        print(f"Text emotion dataset downloaded successfully!")
        print(f"Train samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Emotions: {emotion_names}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading text emotion dataset: {str(e)}")
        return False

def download_fer2013_instructions():
    """Provide instructions for FER2013 dataset download"""
    print("\n" + "="*60)
    print("FER2013 DATASET DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("""
To use the FER2013 dataset for image emotion recognition:

1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Click 'Download' (you'll need a Kaggle account)
3. Extract the downloaded zip file
4. Copy 'fer2013.csv' to the 'datasets' folder
5. Re-run the training script

Alternative using Kaggle API:
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials (kaggle.json)
3. Run: kaggle datasets download -d msambare/fer2013
4. Extract and move fer2013.csv to datasets folder

The FER2013 dataset contains:
- 35,887 48x48 grayscale face images
- 7 emotion categories: angry, disgust, fear, happy, sad, surprise, neutral
- Pre-split into train/test sets
    """)
    print("="*60)

def create_sample_fer2013():
    """Create a small sample FER2013 dataset for testing"""
    print("Creating sample FER2013 dataset for testing...")
    
    # Create a small sample dataset
    np.random.seed(42)
    num_samples = 1000
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    data = []
    for i in range(num_samples):
        # Generate random 48x48 grayscale image (as pixel string)
        pixels = np.random.randint(0, 256, 48*48)
        pixel_string = ' '.join(map(str, pixels))
        
        emotion_label = i % 7  # Cycle through emotions
        usage = 'Training' if i < 800 else 'PublicTest'
        
        data.append({
            'emotion': emotion_label,
            'pixels': pixel_string,
            'Usage': usage
        })
    
    df = pd.DataFrame(data)
    df.to_csv('datasets/fer2013_sample.csv', index=False)
    
    print(f"Sample FER2013 dataset created with {num_samples} images")
    print("Note: This is synthetic data for testing. Use real FER2013 for actual training.")
    
    return 'datasets/fer2013_sample.csv'

def verify_datasets():
    """Verify that datasets are properly downloaded and formatted"""
    print("\nVerifying datasets...")
    
    # Check text emotion dataset
    text_files = ['datasets/emotion_train.csv', 'datasets/emotion_test.csv', 'datasets/emotion_val.csv']
    text_available = all(os.path.exists(f) for f in text_files)
    
    if text_available:
        train_df = pd.read_csv('datasets/emotion_train.csv')
        print(f"✓ Text emotion dataset: {len(train_df)} training samples")
        print(f"  Columns: {list(train_df.columns)}")
        print(f"  Sample text: '{train_df.iloc[0]['text']}'")
    else:
        print("✗ Text emotion dataset not found")
    
    # Check FER2013 dataset
    fer2013_paths = ['datasets/fer2013.csv', 'datasets/fer2013_sample.csv']
    fer2013_available = any(os.path.exists(f) for f in fer2013_paths)
    
    if fer2013_available:
        fer_path = 'datasets/fer2013.csv' if os.path.exists('datasets/fer2013.csv') else 'datasets/fer2013_sample.csv'
        fer_df = pd.read_csv(fer_path)
        print(f"✓ FER2013 dataset: {len(fer_df)} image samples")
        print(f"  Columns: {list(fer_df.columns)}")
        if 'fer2013_sample' in fer_path:
            print("  Note: Using sample dataset. Download real FER2013 for better results.")
    else:
        print("✗ FER2013 dataset not found")
    
    return text_available, fer2013_available

def download_pre_trained_models():
    """Download pre-trained emotion recognition models"""
    print("\nChecking for pre-trained models...")
    
    # For now, we'll rely on training from scratch
    # In a production environment, you could download pre-trained weights
    
    model_path = 'app/models/emotion_cnn_fer2013.pth'
    if os.path.exists(model_path):
        print(f"✓ Pre-trained image model found: {model_path}")
    else:
        print("✗ No pre-trained image model found. Will train from scratch.")
    
    print("✓ Text model uses pre-trained transformers (downloaded automatically)")
    
    return os.path.exists(model_path)

if __name__ == "__main__":
    print("Setting up NeuroFlow Emotion Recognition System")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Download text emotion dataset
    text_success = download_text_emotion_dataset()
    
    # Handle FER2013 dataset
    print("\nSetting up FER2013 dataset...")
    if not os.path.exists('datasets/fer2013.csv'):
        download_fer2013_instructions()
        create_sample_fer2013()
    else:
        print("✓ FER2013 dataset already exists")
    
    # Verify datasets
    text_ok, image_ok = verify_datasets()
    
    # Check models
    model_exists = download_pre_trained_models()
    
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    print(f"Text emotion dataset: {'✓' if text_ok else '✗'}")
    print(f"Image emotion dataset: {'✓' if image_ok else '✗'}")
    print(f"Pre-trained models: {'✓' if model_exists else '✗'}")
    
    if text_ok and image_ok:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the server: python main.py")
        print("2. Start the client: npm start (in client folder)")
        print("3. Open http://localhost:3000 and navigate to Emotion Analysis")
        
        if not model_exists:
            print("\nOptional: Train image emotion model for better accuracy:")
            print("python app/ml/train_fer2013.py")
    else:
        print("\n⚠️  Setup completed with issues. Check the errors above.")
    
    print("="*50)