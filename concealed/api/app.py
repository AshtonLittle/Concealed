"""FastAPI application factory and middleware configuration for Concealed API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from concealed.api.routes import router as api_router

app = FastAPI(
    title="Concealed AI Obfuscation API",
    version="0.1.0",
    description=(
        "Production-grade REST API for image obfuscation against Vision Transformers (ViT) "
        "and Multimodal Large Language Models. Supports configurable perturbation budgets, "
        "Weber's law luminance masking, chrominance damping, conforming silhouette masks, "
        "and metadata scrubbing."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration allowing frontend clients (e.g. Vite on http://127.0.0.1:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
        "X-Processing-Time-Ms",
        "X-Epsilon",
        "X-Mode",
        "X-PSNR-dB",
        "X-SSIM",
        "X-Linf-255",
        "X-Output-Format",
        "X-Feature",
        "X-Regions-Found",
        "X-Frames-Processed",
        "X-FPS",
        "X-Model",
        "X-Quality-Loss-Pct",
    ],
)

app.include_router(api_router)


@app.get("/", tags=["Root"])
async def root():
    """API root index with interactive documentation and status endpoints."""
    return {
        "service": "Concealed AI Obfuscation Engine",
        "version": "0.1.0",
        "documentation": "/docs",
        "endpoints": {
            "obfuscate_upload": "POST /api/obfuscate",
            "obfuscate_json": "POST /api/obfuscate/json",
            "parameters_catalog": "GET /api/parameters",
            "health_check": "GET /api/health",
        },
    }
