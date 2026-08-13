"""
HireLens FastAPI application.

Models are loaded once during startup — roughly 10-30 seconds depending on
device — so that request handlers only run inference.

Run locally:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.logging_setup import configure_logging

from .routers import candidate, recruiter
from .services.ml_service import get_ml_service

DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://localhost:5173"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models before the first request and log a clean shutdown."""
    configure_logging()
    get_ml_service().startup()
    logger.info("HireLens API is ready to serve requests.")
    yield
    logger.info("HireLens API shutting down.")


app = FastAPI(
    title="HireLens API",
    description=(
        "AI-powered resume screening and job matching. The candidate endpoint "
        "scores a single resume; the recruiter endpoints rank bulk uploads and "
        "filter the results."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(candidate.router)
app.include_router(recruiter.router)


@app.get("/health", tags=["system"], summary="Health check")
async def health() -> dict:
    """Report liveness and whether the models have finished loading."""
    return {
        "status": "healthy",
        "models_loaded": get_ml_service().is_ready,
        "timestamp": time.time(),
    }


@app.get("/", tags=["system"], include_in_schema=False)
async def root() -> dict:
    return {"message": "HireLens API", "docs": "/docs", "version": "1.0.0"}
