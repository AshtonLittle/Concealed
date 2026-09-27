"""
Counter-prevention module for Concealed AI.
Simulates platform ingestion degradation (Instagram, WhatsApp, Facebook)
and verifies survival metrics (PSNR, SSIM, MAE, Re-compression risk).
"""

from .verifier import CounterPreventionVerifier, IngestionSimulationResult

__all__ = ["CounterPreventionVerifier", "IngestionSimulationResult"]
