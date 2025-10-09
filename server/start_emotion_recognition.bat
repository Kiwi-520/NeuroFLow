@echo off
REM NeuroFlow Emotion Recognition Quick Start Script for Windows

echo 🚀 NeuroFlow Emotion Recognition System Setup
echo =============================================

REM Check if we're in the right directory
if not exist "comprehensive_training.py" (
    echo ❌ Please run this script from the server directory
    pause
    exit /b 1
)

REM Check if datasets exist
echo 📂 Checking datasets...
if not exist "datasets\text\tweet_emotions.csv" (
    echo ❌ Text dataset not found at datasets\text\tweet_emotions.csv
    echo    Please place your tweet emotion CSV file there
    pause
    exit /b 1
)

if not exist "datasets\image\train" (
    echo ❌ Image dataset not found at datasets\image\train\
    echo    Please place your FER2013 images in the correct folder structure
    pause
    exit /b 1
)

echo ✅ Datasets found!

REM Install Python dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Run training
echo 🎯 Starting model training...
echo This may take 30-60 minutes depending on your hardware...
python comprehensive_training.py

REM Check if training was successful
if errorlevel 1 (
    echo ❌ Training failed. Please check the logs above.
    pause
    exit /b 1
)

echo.
echo 🎉 Training completed successfully!
echo.
echo 🚀 Starting the API server...
echo The server will run on http://localhost:8000
echo API docs available at http://localhost:8000/docs
echo.
echo Next steps:
echo 1. Keep this server running
echo 2. Open a new terminal and navigate to the client folder
echo 3. Run: npm install ^&^& npm run dev
echo 4. Open http://localhost:5173 in your browser
echo 5. Go to the 'Emotion Analysis' tab
echo.

REM Start the server
python main.py

pause