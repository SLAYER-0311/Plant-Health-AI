import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import backend components
import sys
sys.path.insert(0, str(Path(__file__).parent))

from backend.app.config import get_settings
from backend.app.routers import prediction
from backend.app.services.classifier import get_classifier
from backend.app.schemas.prediction import HealthResponse

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
    logger.info("Starting PlantHealth AI on Hugging Face Spaces...")

    # Load model on startup
    classifier = get_classifier()
    if classifier.is_loaded:
        logger.info(f"Model loaded successfully on {classifier.device}")
    else:
        logger.warning("Model failed to load - running in demo mode")

    yield

    # Shutdown
    logger.info("Shutting down PlantHealth AI...")


# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="PlantHealth AI",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log all incoming requests
@app.middleware("http")
async def log_incoming_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    return response

# Include API routers with /api prefix
app.include_router(prediction.router, prefix="/api")

@app.options("/api/predict")
async def options_predict():
    return {"methods": ["OPTIONS", "POST"]}

@app.get("/api", tags=["Root"])
async def api_root():
    return {"message": "Welcome to PlantHealth AI API"}

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    classifier = get_classifier()
    return HealthResponse(
        status="healthy" if classifier.is_loaded else "degraded",
        model_loaded=classifier.is_loaded,
        device=str(classifier.device) if classifier.is_loaded else "N/A",
        version="1.0.0",
    )

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")
    @app.get("/", tags=["Frontend"])
    @app.get("/{full_path:path}", tags=["Frontend"])
    async def serve_frontend(full_path: str = ""):
        if full_path.startswith("api/"):
            return {"error": "Not found"}
        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"error": "Frontend not built"}
else:
    logger.warning("Frontend dist directory not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")