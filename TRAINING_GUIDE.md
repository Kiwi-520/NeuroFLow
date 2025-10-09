# 🧠 NeuroFlow Emotion Recognition Setup Guide

## 📁 Dataset Placement Instructions

### Step 1: Prepare Your Datasets

#### For Image Emotion Recognition (FER2013):

1. **Download FER2013 dataset** from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
2. **Place the file** in the correct location:
   ```
   NeuroFLow/server/datasets/fer2013.csv
   ```
3. **Required format**: The CSV should have these columns:
   - `emotion` (0-6: angry, disgust, fear, happy, sad, surprise, neutral)
   - `pixels` (space-separated pixel values for 48x48 grayscale image)
   - `Usage` (Training, PublicTest, or PrivateTest)

#### For Text Emotion Recognition (Optional):

If you have text emotion datasets, place them as:

```
NeuroFLow/server/datasets/
├── emotion_train.csv    # Training data
├── emotion_val.csv      # Validation data
└── emotion_test.csv     # Test data
```

**Required format**: Each CSV should have:

- `text` (the text content)
- `label` (emotion index) OR `emotion_name` (emotion name)

### Step 2: Verify Dataset Structure

Your `datasets` folder should look like this:

```
NeuroFLow/server/datasets/
└── fer2013.csv          # ← YOUR DOWNLOADED FER2013 FILE
```

## 🚀 Training Commands

### Quick Start (Recommended):

```bash
cd "NeuroFLow/server"
python train_models.py
```

This will:

- ✅ Verify your dataset format
- 🏋️ Train the CNN model on your FER2013 data
- 📊 Generate training plots and metrics
- 🧪 Test both image and text models
- 💾 Save the trained model to `app/models/emotion_cnn_fer2013.pth`

### Training Progress:

You'll see output like:

```
🖼️  Starting Image Emotion Model Training...
Using device: cuda  # or cpu
Loaded 28709 images for Training set
Loaded 3589 images for PublicTest set

Epoch [1/30]
Train Loss: 1.8234, Train Acc: 23.45%
Val Loss: 1.7123, Val Acc: 28.67%
✅ New best model saved! Validation accuracy: 28.67%
```

### Expected Training Time:

- **CPU**: ~2-4 hours for 30 epochs
- **GPU**: ~30-60 minutes for 30 epochs

## 🎯 What Gets Trained

### Image Model (CNN):

- **Architecture**: Deep CNN with 6 conv layers + 3 FC layers
- **Input**: 48x48 grayscale face images
- **Output**: 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)
- **Training**: On your FER2013 dataset
- **Saved to**: `app/models/emotion_cnn_fer2013.pth`

### Text Model (Transformer):

- **Architecture**: Pre-trained DistilRoBERTa
- **Model**: `j-hartmann/emotion-english-distilroberta-base`
- **Training**: Already pre-trained (no additional training needed)
- **Output**: 7 emotions with confidence scores

## 🔧 Advanced Training Options

### Custom Training Parameters:

```bash
# Train with custom epochs and batch size
python train_models.py --epochs 50 --batch-size 64

# Different learning rates (edit the script)
# Change learning_rate=0.001 to your preferred value
```

### Monitor Training:

The script will:

- 📈 Show real-time loss and accuracy
- 💾 Auto-save the best model
- 📊 Generate training plots (`image_training_history.png`)
- 🧪 Test the final model

## 🚀 Running the Complete System

### 1. Start the Backend:

```bash
cd "NeuroFLow/server"
python main.py
```

### 2. Start the Frontend:

```bash
cd "NeuroFLow/client"
npm run dev
```

### 3. Access the Application:

- Open: http://localhost:5173
- Navigate to: **Emotion Analysis** tab
- Try both text and image emotion recognition!

## 🛠️ Troubleshooting

### Common Issues:

#### "Dataset not found":

- Ensure `fer2013.csv` is in `NeuroFLow/server/datasets/`
- Check file permissions

#### "CUDA out of memory":

- Reduce batch size: use `--batch-size 16` or `--batch-size 8`
- Use CPU training if needed

#### "Module not found":

```bash
cd "NeuroFLow/server"
pip install -r requirements.txt
```

#### Poor accuracy:

- Train for more epochs: `--epochs 50`
- Ensure you're using the real FER2013 dataset (not sample)
- Check dataset quality and format

### GPU vs CPU Training:

- **GPU**: Much faster, better for experimentation
- **CPU**: Slower but works fine, good for final training

## 📊 Expected Results

### Image Model Performance:

- **Training Accuracy**: 60-75%
- **Validation Accuracy**: 55-70%
- **Training Time**: 30 mins (GPU) to 3 hours (CPU)

### Text Model Performance:

- **Accuracy**: 80-90% (pre-trained)
- **Real-time**: Very fast inference
- **No training needed**: Ready to use!

## 🎉 Success Indicators

You'll know everything worked when you see:

```
🎉 Training pipeline completed!
✅ Image model: app/models/emotion_cnn_fer2013.pth
✅ Text model: Uses pre-trained transformers (ready to use)

🚀 Next steps:
1. Start the server: python main.py
2. Start the client: npm run dev (in client folder)
3. Navigate to Emotion Analysis tab
```

## 📝 Notes

- **Dataset Size**: FER2013 has ~35k images, perfect for training
- **Memory Usage**: ~2-4GB RAM during training
- **Model Size**: Final model ~50MB
- **Inference Speed**: Real-time for both text and images

Happy training! 🚀🧠💙
