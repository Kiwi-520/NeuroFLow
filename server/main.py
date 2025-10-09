"""
Main FastAPI application for NeuroFlow backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
import logging
from pathlib import Path

# Import API routes
from app.api.emotion_routes import router as emotion_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting NeuroFlow API server...")
    
    # Create necessary directories
    os.makedirs("app/models", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down NeuroFlow API server...")

# Create FastAPI app
app = FastAPI(
    title="NeuroFlow API",
    description="AI-powered cognitive healthcare support system with emotion recognition",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(emotion_router)

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NeuroFlow API - Cognitive Healthcare Support System",
        "version": "2.0.0",
        "features": [
            "Real-time emotion recognition",
            "Text-based emotion analysis",
            "Image-based facial emotion detection",
            "Multi-modal emotion insights"
        ],
        "endpoints": {
            "emotion_analysis": "/api/emotion/",
            "documentation": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NeuroFlow API",
        "version": "2.0.0"
    }

# Static files (for serving model files, documentation, etc.)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
