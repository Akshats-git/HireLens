"""
HireLens FastAPI application entry point.

Startup sequence:
  1. Load ML models into memory (once, ~10-30s depending on GPU)
  2. Register routers
  3. Serve requests

Run locally:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import candidate, recruiter
from .services.ml_service import get_ml_service


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml = get_ml_service()
    ml.startup()
    logger.info("HireLens API is ready to serve requests.")
    yield
    logger.info("HireLens API shutting down.")


# ── App factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HireLens API",
    description=(
        "AI-powered resume screening and job matching. "
        "Candidate endpoint scores a single resume; recruiter endpoints "
        "process bulk uploads and expose filtering."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────────

_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(candidate.router)
app.include_router(recruiter.router)


# ── System endpoints ───────────────────────────────────────────────────────────

@app.get("/health", tags=["system"], summary="Health check")
async def health() -> dict:
    ml = get_ml_service()
    return {
        "status": "healthy",
        "models_loaded": ml.is_ready,
        "timestamp": time.time(),
    }


@app.get("/", tags=["system"], include_in_schema=False)
async def root() -> dict:
    return {"message": "HireLens API", "docs": "/docs", "version": "1.0.0"}
