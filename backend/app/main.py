"""
PlantHealth AI Backend - FastAPI Application
==============================================
Main entry point for the FastAPI backend server.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .routers import prediction
from .services.classifier import get_classifier
from .schemas.prediction import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting PlantHealth AI Backend...")
    
    # Load model on startup
    classifier = get_classifier()
    if classifier.is_loaded:
        logger.info(f"Model loaded successfully on {classifier.device}")
    else:
        logger.warning("Model failed to load - running in demo mode")
    
    yield
    
    # Shutdown
    logger.info("Shutting down PlantHealth AI Backend...")


# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    PlantHealth AI - Plant Disease Classification API
    
    This API allows you to upload images of plant leaves and get predictions
    for plant diseases using a deep learning model trained on the PlantVillage dataset.
    
    ## Features
    - Upload plant leaf images
    - Get disease predictions with confidence scores
    - Support for 38 different plant disease classes
    - Fast inference using PyTorch
    
    ## Supported Plants
    Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato,
    Raspberry, Soybean, Squash, Strawberry, Tomato
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to PlantHealth AI API",
        "docs": "/docs",
        "version": settings.app_version,
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    classifier = get_classifier()
    
    return HealthResponse(
        status="healthy" if classifier.is_loaded else "degraded",
        model_loaded=classifier.is_loaded,
        device=str(classifier.device) if classifier.is_loaded else "N/A",
        version=settings.app_version,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else None,
        }
    )


# Entry point for direct execution
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
