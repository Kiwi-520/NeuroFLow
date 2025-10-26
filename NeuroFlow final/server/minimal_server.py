#!/usr/bin/env python3

import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Server is running"}

@app.post("/api/emotions/analyze")
def analyze_emotion(request: TextRequest):
    return {
        "emotion": "joy",
        "probability": 0.9,
        "emoji": "😂",
        "emotion_spectrum": {
            "anger": 0.01,
            "disgust": 0.01,
            "fear": 0.02,
            "joy": 0.9,
            "neutral": 0.02,
            "sadness": 0.02,
            "surprise": 0.01,
            "trust": 0.01
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)