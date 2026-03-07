"""
Modal Configuration Helper
Manages environment variables, secrets, and configuration for Modal deployment
"""
import modal
import os
from typing import Optional

# Define secrets that Modal will inject
gemini_secret = modal.Secret.from_name("gemini-secrets", required=False)
huggingface_secret = modal.Secret.from_name("huggingface-secrets", required=False)

def get_modal_image(gpu_enabled: bool = False):
    """
    Create Modal image with all dependencies
    
    Args:
        gpu_enabled: Whether to include GPU libraries
    """
    image = modal.Image.debian_slim()
    
    # System dependencies
    system_packages = [
        "libgl1",
        "libglib2.0-0", 
        "libgomp1",
        "libcairo2",
        "libpango-1.0-0",
        "libpangoft2-1.0-0",
        "libgdk-pixbuf2.0-0",
        "shared-mime-info",
        "fonts-dejavu-core",
    ]
    
    # Python dependencies
    pip_packages = [
        # FastAPI & Web
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "python-multipart>=0.0.12",
        
        # Computer Vision
        "opencv-python>=4.10.0",
        "mediapipe==0.10.18",
        "scikit-image>=0.24.0",
        
        # Data Processing
        "numpy<2.0.0",
        "pandas>=2.2.3",
        "scipy>=1.14.0",
        "scikit-learn>=1.5.0",
        
        # LLM Integration
        "google-genai>=0.2.0",
        "langchain>=0.3.0",
        "langchain-google-genai>=2.0.0",
        "huggingface-hub>=0.26.0",
        "transformers>=4.45.0",
        "requests>=2.32.0",
        
        # ML Framework
        "torch>=2.5.0",
        
        # Report Generation
        "reportlab>=4.2.0",
        
        # Configuration
        "pydantic>=2.10.0",
        "pydantic-settings>=2.6.0",
        "python-dotenv>=1.0.1",
        
        # Security
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        
        # Utilities
        "aiofiles>=24.1.0",
        "pyserial>=3.5",
        "rich",
        "langchain-community>=0.3.0",
    ]
    
    image = image.apt_install(*system_packages)
    image = image.pip_install(*pip_packages)
    
    if gpu_enabled:
        # GPU-specific packages
        image = image.env({
            "CUDA_VISIBLE_DEVICES": "0",
            "TORCH_CUDA_ARCH_LIST": "8.0",
        })
    
    return image


def create_volume_config():
    """Create volume configuration for persistent storage"""
    return {
        "/app/models": modal.Volume.from_name("health-screening-models", create_if_missing=True),
        "/app/reports": modal.Volume.from_name("health-screening-reports", create_if_missing=True),
        "/app/cache": modal.Volume.from_name("health-screening-cache", create_if_missing=True),
    }


def create_function_config(
    gpu: bool = False,
    memory: int = 8192,
    timeout: int = 3600,
    keep_warm: int = 0,
    concurrency: Optional[int] = None,
):
    """
    Create function configuration for Modal
    
    Args:
        gpu: Enable GPU (A10G)
        memory: Memory in MB (default 8GB)
        timeout: Timeout in seconds
        keep_warm: Number of instances to keep warm
        concurrency: Max concurrent requests
    """
    config = {
        "image": get_modal_image(gpu_enabled=gpu),
        "volumes": create_volume_config(),
        "memory": memory,
        "timeout": timeout,
        "keep_warm": keep_warm,
    }
    
    if gpu:
        config["gpu"] = modal.gpu.A10G()
    
    if concurrency:
        config["concurrency_limit"] = concurrency
    
    # Add secrets
    if gemini_secret:
        config["secrets"] = [gemini_secret]
    if huggingface_secret:
        if "secrets" not in config:
            config["secrets"] = []
        config["secrets"].append(huggingface_secret)
    
    return config


# Configuration Presets
PRESETS = {
    "web": {
        "gpu": False,
        "memory": 8192,
        "timeout": 300,
        "keep_warm": 1,
        "concurrency": 10,
    },
    "inference": {
        "gpu": True,
        "memory": 16384,
        "timeout": 3600,
        "keep_warm": 0,
        "concurrency": 5,
    },
    "report_generation": {
        "gpu": False,
        "memory": 8192,
        "timeout": 7200,
        "keep_warm": 0,
        "concurrency": 3,
    },
    "background_job": {
        "gpu": False,
        "memory": 4096,
        "timeout": 1800,
        "keep_warm": 0,
        "concurrency": 1,
    },
}


def setup_modal_secrets():
    """
    Print commands to setup required secrets
    
    Usage: modal secret create <name> --key <KEY> --value <VALUE>
    """
    print("""
    Setup Modal Secrets:
    ===================
    
    1. Create Gemini Secret:
       modal secret create gemini-secrets
       (When prompted, enter: GOOGLE_API_KEY = your-key-here)
    
    2. Create HuggingFace Secret:
       modal secret create huggingface-secrets
       (When prompted, enter: HUGGINGFACE_TOKEN = your-token-here)
    
    3. Verify secrets created:
       modal secret list
    """)


def print_deployment_guide():
    """Print deployment instructions"""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║     Modal Deployment - Health Screening Pipeline      ║
    ╚════════════════════════════════════════════════════════╝
    
    DEPLOYMENT:
    -----------
    modal deploy modal_app.py
    
    YOUR ENDPOINT:
    https://your-username--health-screening-pipeline.modal.run
    
    COMMON COMMANDS:
    ----------------
    # View logs
    modal logs health-screening-pipeline
    
    # List volumes
    modal volume ls
    
    # Upload files
    modal volume put health-screening-models ./pose_landmarker.task /app/models/
    
    # Check costs
    modal app cost health-screening-pipeline
    
    # Stop app
    modal stop health-screening-pipeline
    """)


if __name__ == "__main__":
    setup_modal_secrets()
    print()
    print_deployment_guide()
