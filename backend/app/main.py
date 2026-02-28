from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from dotenv import load_dotenv
import os
import requests

# Load environment variables from .env file
load_dotenv()

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
from app.api.stats import router as stats_router
from app.services.cleanup_service import run_cleanup_job

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
app.include_router(stats_router, prefix="/api")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException):
    """Add X-Upload-Retry: false on 403 so clients stop retrying forbidden uploads."""
    if exc.status_code == 403:
        return JSONResponse(
            status_code=403,
            content={"detail": exc.detail},
            headers={"X-Upload-Retry": "false"},
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


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

@app.get("/debug/env-check")
def env_check():
    """Debug endpoint to check if environment variables are loaded (without exposing sensitive data)."""
    hf_key = os.getenv("HF_API_KEY")
    return {
        "HF_API_KEY_set": bool(hf_key),
        "HF_API_KEY_preview": hf_key[:10] + "..." if hf_key else None,
        "HF_API_KEY_length": len(hf_key) if hf_key else 0,
    }


@app.get("/debug/hf-whoami")
def hf_whoami():
    """Debug endpoint to verify the Hugging Face token is valid (does not call inference)."""
    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        return {"ok": False, "error": "HF_API_KEY is not set"}

    try:
        r = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {hf_key}", "Accept": "application/json"},
            timeout=15,
        )
        return {
            "ok": r.status_code == 200,
            "status": r.status_code,
            "contentType": r.headers.get("content-type"),
            "bodyPreview": (r.text or "")[:300],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post(
    "/admin/cleanup",
    tags=["admin"],
    summary="Run cleanup job",
    description="Manually trigger cleanup of old audio files and failed sessions. Typically run as a scheduled job.",
)
def cleanup_endpoint():
    """Manually trigger cleanup job.
    
    This endpoint should typically be called by a scheduled task (cron job, etc.)
    rather than manually. It cleans up:
    - Audio files older than 30 days after analysis completion
    - Orphaned audio files
    - Failed analysis sessions older than 7 days
    
    Returns:
        Dict with cleanup statistics
    """
    import os
    audio_storage_dir = os.getenv("AUDIO_STORAGE_DIR", "./audio_storage")
    return run_cleanup_job(audio_storage_dir)