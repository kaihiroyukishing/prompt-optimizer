import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from backend.app.core.config import settings
from backend.app.core.database import init_db
from backend.app.routes import context, optimization

# Configure basic logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Prompt Optimizer API",
    description="Context-aware AI prompt optimization system with Python backend and C++ performance modules",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development (Chrome extensions need this)
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

app.include_router(optimization.router, prefix="/api/v1", tags=["optimization"])
app.include_router(context.router, prefix="/api/v1", tags=["context"])


@app.on_event("startup")
async def startup_event():
    """Initialize database and other startup tasks."""
    logger.info("Starting Prompt Optimizer API")

    # Validate configuration
    validation_results = settings.validate_configuration()

    if validation_results["warnings"]:
        for warning in validation_results["warnings"]:
            logger.warning(f"Configuration warning: {warning}")

    if not validation_results["valid"]:
        for error in validation_results["errors"]:
            logger.error(f"Configuration error: {error}")
        logger.error("Application startup failed due to configuration errors")
        # In production, you might want to exit here

    await init_db()
    logger.info("Database initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup tasks on shutdown."""
    logger.info("Shutting down Prompt Optimizer API")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Prompt Optimizer API",
        "version": "1.0.0",
        "docs": "/docs" if settings.DEBUG else "Documentation disabled in production",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "prompt-optimizer-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG, log_level="info"
    )
