from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router

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