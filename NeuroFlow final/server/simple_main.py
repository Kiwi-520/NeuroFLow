from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NeuroFlow Emotion Analysis API")

# Configure CORS to allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextAnalysisRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Emotion Analysis API is running"}

@app.post("/api/emotions/analyze")
async def analyze_emotion(request: TextAnalysisRequest):
    try:
        # Simple mock response for testing
        result = {
            "emotion": "joy",
            "probability": 0.85,
            "emoji": "😂",
            "emotion_spectrum": {
                "anger": 0.02,
                "disgust": 0.01,
                "fear": 0.03,
                "joy": 0.85,
                "neutral": 0.04,
                "sadness": 0.03,
                "surprise": 0.01,
                "trust": 0.01
            }
        }
        logger.info(f"Analyzed text: {request.text}")
        return result
    except Exception as e:
        logger.error(f"Error in analyze_emotion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)