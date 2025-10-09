"""
Comprehensive Training Script for NeuroFlow Emotion Recognition
Trains models on your FER2013 image dataset and tweet emotion CSV dataset
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import logging
import json
import time
from pathlib import Path

# Import our custom models
from app.ml.image_emotion_model import EmotionCNN

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TweetEmotionDataset(Dataset):
    """Custom dataset for tweet emotion data"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_text_dataset():
    """Load and prepare text emotion dataset"""
    logger.info("Loading text emotion dataset...")
    
    # Load CSV file
    df = pd.read_csv('datasets/text/tweet_emotions.csv')
    logger.info(f"Loaded {len(df)} text samples")
    
    # Create emotion mapping (map various emotions to 7 main categories)
    emotion_mapping = {
        'sadness': 0,
        'joy': 1, 
        'love': 2,
        'anger': 3,
        'fear': 4,
        'surprise': 5,
        'neutral': 6,
        # Additional mappings for various emotions in your dataset
        'enthusiasm': 1,  # -> joy
        'happiness': 1,   # -> joy
        'worry': 4,       # -> fear
        'empty': 6,       # -> neutral
        'hate': 3,        # -> anger
        'fun': 1,         # -> joy
        'relief': 1,      # -> joy
        'boredom': 6,     # -> neutral
        'excitement': 1,  # -> joy
        'annoyance': 3,   # -> anger
        'optimism': 1     # -> joy
    }
    
    # Filter and map emotions
    df_filtered = df[df['sentiment'].isin(emotion_mapping.keys())].copy()
    df_filtered['emotion_label'] = df_filtered['sentiment'].map(emotion_mapping)
    
    # Clean and prepare text
    df_filtered['content'] = df_filtered['content'].fillna('').astype(str)
    df_filtered = df_filtered[df_filtered['content'].str.len() > 5]  # Remove very short texts
    
    logger.info(f"After filtering: {len(df_filtered)} samples")
    logger.info(f"Emotion distribution:\n{df_filtered['sentiment'].value_counts()}")
    
    return df_filtered

def train_text_emotion_model():
    """Train text emotion recognition model using transformers"""
    logger.info("=" * 50)
    logger.info("TRAINING TEXT EMOTION MODEL")
    logger.info("=" * 50)
    
    # Prepare dataset
    df = prepare_text_dataset()
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['content'].values,
        df['emotion_label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['emotion_label'].values
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Initialize model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=7  # 7 emotion categories
    )
    
    # Create datasets
    train_dataset = TweetEmotionDataset(X_train, y_train, tokenizer)
    test_dataset = TweetEmotionDataset(X_test, y_test, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./text_training_results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./text_logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # Train the model
    logger.info("Starting text model training...")
    trainer.train()
    
    # Save model and tokenizer
    os.makedirs('app/models', exist_ok=True)
    model.save_pretrained('app/models/text_emotion_model')
    tokenizer.save_pretrained('app/models/text_emotion_tokenizer')
    
    # Evaluate model
    logger.info("Evaluating text model...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Text Model Final Accuracy: {accuracy:.4f}")
    
    # Save class mapping
    emotion_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise', 'neutral']
    with open('app/models/text_emotion_mapping.json', 'w') as f:
        json.dump(emotion_names, f)
    
    logger.info("Text model training completed!")
    return accuracy

def train_image_emotion_model():
    """Train image emotion recognition model using CNN"""
    logger.info("=" * 50)
    logger.info("TRAINING IMAGE EMOTION MODEL")
    logger.info("=" * 50)
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation((-15, 15)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load datasets using ImageFolder
    train_dataset = ImageFolder('datasets/image/train', transform=train_transform)
    test_dataset = ImageFolder('datasets/image/test', transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Emotion classes: {train_dataset.classes}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = EmotionCNN(num_classes=len(train_dataset.classes)).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training variables
    num_epochs = 30
    best_test_acc = 0.0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    logger.info("Starting image model training...")
    
    for epoch in range(num_epochs):
        # ============ TRAINING PHASE ============
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # ============ TESTING PHASE ============
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100 * correct_test / total_test
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Log progress
        logger.info(f'Epoch [{epoch+1}/{num_epochs}] Summary:')
        logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        logger.info(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        logger.info('-' * 60)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_test_acc': best_test_acc,
                'class_names': train_dataset.classes,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies
            }, 'app/models/emotion_cnn_fer2013.pth')
            logger.info(f'🎉 New best model saved! Test Accuracy: {best_test_acc:.2f}%')
    
    # Save class mapping
    with open('app/models/image_emotion_mapping.json', 'w') as f:
        json.dump(train_dataset.classes, f)
    
    # Plot training history
    plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies)
    
    logger.info("Image model training completed!")
    return best_test_acc

def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, test_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    logger.info("Training curves saved as 'training_curves.png'")

def test_trained_models():
    """Test both trained models with sample inputs"""
    logger.info("=" * 50)
    logger.info("TESTING TRAINED MODELS")
    logger.info("=" * 50)
    
    # Test text model
    try:
        from transformers import pipeline
        text_classifier = pipeline(
            "text-classification",
            model="app/models/text_emotion_model",
            tokenizer="app/models/text_emotion_tokenizer",
            device=0 if torch.cuda.is_available() else -1
        )
        
        test_texts = [
            "I'm so happy and excited about this!",
            "This makes me really sad and disappointed",
            "I'm feeling quite angry about what happened",
            "That was really scary and terrifying",
            "What a pleasant surprise this is!",
            "I absolutely love this amazing experience",
            "I feel pretty neutral about this situation"
        ]
        
        logger.info("Text Model Test Results:")
        for text in test_texts:
            result = text_classifier(text)
            emotion = result[0]['label'] if isinstance(result, list) else result['label']
            confidence = result[0]['score'] if isinstance(result, list) else result['score']
            logger.info(f"  '{text}' -> {emotion} (confidence: {confidence:.3f})")
            
    except Exception as e:
        logger.error(f"Error testing text model: {e}")
    
    # Test image model
    try:
        from app.ml.image_emotion_model import ImageEmotionRecognizer
        image_recognizer = ImageEmotionRecognizer('app/models/emotion_cnn_fer2013.pth')
        logger.info("✅ Image model loaded successfully and ready for inference!")
        
    except Exception as e:
        logger.error(f"❌ Error loading image model: {e}")

def main():
    """Main training function"""
    logger.info("🚀 Starting NeuroFlow Emotion Recognition Training Pipeline")
    logger.info("=" * 70)
    
    # Check if datasets exist
    if not os.path.exists('datasets/text/tweet_emotions.csv'):
        logger.error("❌ Text dataset not found! Please place tweet_emotions.csv in datasets/text/")
        return
    
    if not os.path.exists('datasets/image/train'):
        logger.error("❌ Image dataset not found! Please place FER2013 images in datasets/image/train/ and datasets/image/test/")
        return
    
    # Create models directory
    os.makedirs('app/models', exist_ok=True)
    
    # Start training
    start_time = time.time()
    
    try:
        # Phase 1: Train text emotion model
        logger.info("🎯 PHASE 1: Training Text Emotion Recognition Model")
        text_accuracy = train_text_emotion_model()
        
        # Phase 2: Train image emotion model  
        logger.info("🎯 PHASE 2: Training Image Emotion Recognition Model")
        image_accuracy = train_image_emotion_model()
        
        # Phase 3: Test models
        logger.info("🎯 PHASE 3: Testing Trained Models")
        test_trained_models()
        
        # Training summary
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"📊 Text Model Accuracy: {text_accuracy:.4f}")
        logger.info(f"📊 Image Model Accuracy: {image_accuracy:.2f}%") 
        logger.info(f"⏱️  Total Training Time: {total_time/60:.2f} minutes")
        logger.info(f"💾 Models saved in: app/models/")
        logger.info("\n🚀 Next Steps:")
        logger.info("1. Start the FastAPI server: python main.py")
        logger.info("2. Start the React client: npm run dev (in client folder)")
        logger.info("3. Open http://localhost:5173 and go to 'Emotion Analysis' tab")
        logger.info("4. Test real-time emotion recognition with text and webcam!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()