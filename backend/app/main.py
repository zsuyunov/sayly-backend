from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.me import router as me_router
from app.api.sessions import router as sessions_router
from app.api.voice import router as voice_router
from app.api.reports import router as reports_router
from app.api.rewards import router as rewards_router
from app.api.privacy import router as privacy_router
from app.api.audio import router as audio_router
from app.api.analysis import router as analysis_router
from app.api.notes import router as notes_router

app = FastAPI(
    title="Gossip Detector API",
    description="Backend API for the Gossip Detector application",
    version="1.0.0",
)

# Configure CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Register API routers
app.include_router(auth_router)
app.include_router(me_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(voice_router, prefix="/api")
app.include_router(reports_router, prefix="/api")
app.include_router(rewards_router, prefix="/api")
app.include_router(privacy_router, prefix="/api")
app.include_router(audio_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(notes_router, prefix="/api")


@app.get(
    "/health",
    tags=["health"],
    summary="Health check endpoint",
    description="Returns the health status of the API",
)
def health_check():
    """Basic health check endpoint to verify the API is running.
    
    Returns:
        dict: Health status information
    """
    return {
        "status": "healthy",
        "service": "Gossip Detector API",
    }

@app.get("/")
def root():
    return {"status": "ok", "service": "Sayly backend"}