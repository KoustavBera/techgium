"""
Hardware Abstraction Layer for Chiranjeevi Health Screening.

Provides the HardwareManager singleton that owns all sensor hardware
(Camera, Radar, Thermal) and exposes them to the FastAPI application.
"""
from app.core.hardware.manager import HardwareManager

__all__ = ["HardwareManager"]
