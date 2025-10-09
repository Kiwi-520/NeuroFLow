"""
Complete training pipeline for both text and image emotion recognition models
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
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our models
from app.ml.image_emotion_model import EmotionCNN, ImageEmotionRecognizer
from app.ml.text_emotion_model import TextEmotionAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FER2013Dataset(Dataset):
    """Dataset class for FER2013 emotion recognition"""
    
    def __init__(self, csv_file, transform=None, usage_filter=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations
            transform (callable, optional): Optional transform to be applied on a sample
            usage_filter (str, optional): Filter by Usage column ('Training', 'PublicTest', 'PrivateTest')
        """
        self.data = pd.read_csv(csv_file)
        
        # Filter by usage if specified
        if usage_filter:
            self.data = self.data[self.data['Usage'] == usage_filter].reset_index(drop=True)
        
        self.transform = transform
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        print(f"Loaded {len(self.data)} images for {usage_filter or 'all'} set")
        print(f"Emotion distribution:")
        emotion_counts = self.data['emotion'].value_counts().sort_index()
        for idx, count in emotion_counts.items():
            print(f"  {self.emotions[idx]}: {count}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Extract pixel data and reshape to 48x48
        pixels = self.data.iloc[idx]['pixels']
        image = np.array([int(pixel) for pixel in pixels.split()], dtype=np.uint8)
        image = image.reshape(48, 48)
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        emotion_label = self.data.iloc[idx]['emotion']
        
        return image, emotion_label

def verify_dataset_format():
    """Verify that datasets are in the correct format"""
    print("🔍 Verifying dataset formats...")
    
    # Check FER2013 format
    fer2013_path = "datasets/fer2013.csv"
    if os.path.exists(fer2013_path):
        print("✅ Found FER2013 dataset")
        df = pd.read_csv(fer2013_path)
        required_cols = ['emotion', 'pixels', 'Usage']
        
        if all(col in df.columns for col in required_cols):
            print(f"✅ FER2013 format correct. Columns: {list(df.columns)}")
            print(f"   Total images: {len(df)}")
            print(f"   Training images: {len(df[df['Usage'] == 'Training'])}")
            print(f"   Test images: {len(df[df['Usage'] == 'PublicTest'])}")
        else:
            print(f"❌ FER2013 missing columns. Found: {list(df.columns)}, Expected: {required_cols}")
            return False
    else:
        print("❌ FER2013 dataset not found at datasets/fer2013.csv")
        return False
    
    return True

def train_image_model(data_path, model_save_path, epochs=30, batch_size=32, learning_rate=0.001):
    """Train the CNN model for image emotion recognition"""
    print("\n🖼️  Starting Image Emotion Model Training...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load datasets
    train_dataset = FER2013Dataset(data_path, transform=train_transform, usage_filter='Training')
    val_dataset = FER2013Dataset(data_path, transform=val_transform, usage_filter='PublicTest')
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model, loss, optimizer
    model = EmotionCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training history
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }, model_save_path)
            print(f'✅ New best model saved! Validation accuracy: {best_val_acc:.2f}%')
    
    print(f'\n🎉 Image model training completed! Best validation accuracy: {best_val_acc:.2f}%')
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, 'image')
    
    return model, best_val_acc

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, model_type):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title(f'{model_type.title()} Model - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title(f'{model_type.title()} Model - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_type}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Training plots saved as {model_type}_training_history.png")

def test_trained_models():
    """Test both trained models"""
    print("\n🧪 Testing Trained Models...")
    
    # Test image model
    print("\n📸 Testing Image Emotion Model...")
    try:
        image_model = ImageEmotionRecognizer('app/models/emotion_cnn_fer2013.pth')
        
        # Test with sample from dataset if available
        if os.path.exists('datasets/fer2013.csv'):
            test_df = pd.read_csv('datasets/fer2013.csv')
            test_sample = test_df[test_df['Usage'] == 'PublicTest'].iloc[0]
            
            # Convert pixels to image
            pixels = np.array([int(p) for p in test_sample['pixels'].split()], dtype=np.uint8)
            image = pixels.reshape(48, 48)
            
            result = image_model.predict_emotion(Image.fromarray(image))
            print(f"✅ Image model test successful:")
            print(f"   Predicted: {result['predicted_emotion']}")
            print(f"   Confidence: {result['confidence']:.3f}")
        
    except Exception as e:
        print(f"❌ Image model test failed: {str(e)}")
    
    # Test text model
    print("\n📝 Testing Text Emotion Model...")
    try:
        text_model = TextEmotionAnalyzer()
        
        test_texts = [
            "I am so happy and excited about this amazing project!",
            "This is making me really sad and disappointed.",
            "I'm feeling quite angry about this situation.",
            "What a neutral statement about the weather today."
        ]
        
        for text in test_texts:
            result = text_model.comprehensive_emotion_analysis(text)
            print(f"✅ Text: '{text[:50]}...'")
            print(f"   Predicted: {result['predicted_emotion']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print()
            
    except Exception as e:
        print(f"❌ Text model test failed: {str(e)}")

def evaluate_text_model():
    """Evaluate text emotion model (uses pre-trained transformers)"""
    print("Evaluating text emotion model...")
    
    try:
        # Load test dataset
        test_data_path = 'datasets/emotion_test.csv'
        if not os.path.exists(test_data_path):
            print("Text test dataset not found. Run setup_datasets.py first.")
            return False
        
        test_df = pd.read_csv(test_data_path)
        
        # Initialize text analyzer
        analyzer = TextEmotionAnalyzer()
        
        # Sample evaluation on subset for speed
        sample_size = min(500, len(test_df))
        test_sample = test_df.sample(n=sample_size, random_state=42)
        
        print(f"Evaluating on {sample_size} samples...")
        
        predictions = []
        true_labels = []
        
        # Map dataset emotions to model emotions
        emotion_mapping = {
            'sadness': 'sad',
            'joy': 'happy',
            'love': 'happy',
            'anger': 'angry',
            'fear': 'fear',
            'surprise': 'surprise'
        }
        
        for idx, row in tqdm(test_sample.iterrows(), total=len(test_sample)):
            text = row['text']
            true_emotion = emotion_mapping.get(row['emotion_name'], row['emotion_name'])
            
            try:
                result = analyzer.analyze_emotion_transformer(text)
                predicted_emotion = result['predicted_emotion']
                
                predictions.append(predicted_emotion)
                true_labels.append(true_emotion)
                
            except Exception as e:
                print(f"Error analyzing text: {str(e)}")
                continue
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Text model accuracy: {accuracy:.3f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, zero_division=0))
        
        return True
        
    except Exception as e:
        print(f"Error evaluating text model: {str(e)}")
        return False

def create_model_performance_report():
    """Create a comprehensive performance report"""
    print("Creating model performance report...")
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'models': {},
        'datasets': {},
    }
    
    # Check dataset availability
    datasets_info = {
        'text_emotion': {
            'train': 'datasets/emotion_train.csv',
            'test': 'datasets/emotion_test.csv',
            'val': 'datasets/emotion_val.csv'
        },
        'fer2013': {
            'data': 'datasets/fer2013.csv',
            'sample': 'datasets/fer2013_sample.csv'
        }
    }
    
    for dataset_name, paths in datasets_info.items():
        available_files = {name: os.path.exists(path) for name, path in paths.items()}
        
        if dataset_name == 'text_emotion' and available_files['train']:
            train_df = pd.read_csv(paths['train'])
            report['datasets'][dataset_name] = {
                'available': True,
                'train_samples': len(train_df),
                'emotions': train_df['emotion_name'].unique().tolist()
            }
        elif dataset_name == 'fer2013':
            fer_path = paths['data'] if available_files['data'] else paths['sample']
            if os.path.exists(fer_path):
                fer_df = pd.read_csv(fer_path)
                report['datasets'][dataset_name] = {
                    'available': True,
                    'total_samples': len(fer_df),
                    'is_sample': 'sample' in fer_path,
                    'emotions': list(range(7))  # 0-6 for FER2013
                }
    
    # Check model availability
    model_paths = {
        'image_cnn': 'app/models/emotion_cnn_fer2013.pth',
        'text_transformer': 'j-hartmann/emotion-english-distilroberta-base'
    }
    
    for model_name, path in model_paths.items():
        if model_name == 'image_cnn':
            report['models'][model_name] = {
                'available': os.path.exists(path),
                'path': path,
                'type': 'CNN',
                'framework': 'PyTorch'
            }
        else:
            report['models'][model_name] = {
                'available': True,  # Always available via Hugging Face
                'path': path,
                'type': 'Transformer',
                'framework': 'Transformers'
            }
    
    # Save report
    report_path = 'model_performance_report.json'
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Performance report saved to: {report_path}")
    return report

def main():
    """Main training pipeline"""
    print("🚀 NeuroFlow Emotion Recognition Training Pipeline")
    print("=" * 60)
    
    # Step 1: Verify datasets
    if not verify_dataset_format():
        print("\n❌ Dataset verification failed. Please check your dataset placement and format.")
        print("\n📋 Expected dataset structure:")
        print("datasets/")
        print("├── fer2013.csv              # Image emotion data")
        print("├── emotion_train.csv        # Text emotion training data (optional)")
        print("├── emotion_val.csv          # Text emotion validation data (optional)")
        print("└── emotion_test.csv         # Text emotion test data (optional)")
        print("\n📝 FER2013 CSV should have columns: ['emotion', 'pixels', 'Usage']")
        print("📝 Text CSV should have columns: ['text', 'label'] or ['text', 'emotion_name']")
        return
    
    print("✅ Dataset verified successfully!")
    
    # Step 2: Create models directory
    os.makedirs('app/models', exist_ok=True)
    
    # Step 3: Train image model
    try:
        print(f"\n{'='*60}")
        fer_path = 'datasets/fer2013.csv'
        model_path = 'app/models/emotion_cnn_fer2013.pth'
        
        model, image_acc = train_image_model(fer_path, model_path, epochs=30, batch_size=32)
        print(f"✅ Image model training completed with {image_acc:.2f}% accuracy")
    except Exception as e:
        print(f"❌ Image model training failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Step 4: Test models
    test_trained_models()
    
    print(f"\n{'='*60}")
    print("🎉 Training pipeline completed!")
    print("✅ Image model: app/models/emotion_cnn_fer2013.pth")
    print("✅ Text model: Uses pre-trained transformers (ready to use)")
    print("\n🚀 Next steps:")
    print("1. Start the server: python main.py")
    print("2. Start the client: npm run dev (in client folder)")
    print("3. Navigate to Emotion Analysis tab")
    print("=" * 60)
    
    print("=" * 50)

if __name__ == "__main__":
    main()