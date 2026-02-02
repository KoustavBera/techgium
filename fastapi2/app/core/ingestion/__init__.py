"""
Data Ingestion Module

Handles multimodal data ingestion from camera, motion sensors, 
RIS streams, and auxiliary signals with time synchronization.
"""
from .camera import CameraIngestion
from .motion import MotionIngestion
from .ris import RISSimulator
from .auxiliary import AuxiliaryIngestion
from .sync import DataSynchronizer, DataPacket, ModalityType

__all__ = [
    "CameraIngestion",
    "MotionIngestion",
    "RISSimulator",
    "AuxiliaryIngestion",
    "DataSynchronizer",
    "DataPacket",
    "ModalityType",
]
