"""
Modal Serverless Deployment for Health Screening Pipeline
Deploys FastAPI app to Modal with GPU support for ML inference
"""
import modal
import os
from pathlib import Path

# Define the Modal app
app = modal.App("health-screening-pipeline")

# Create a custom image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "python-multipart>=0.0.12",
        "opencv-python>=4.10.0",
        "mediapipe==0.10.18",
        "scikit-image>=0.24.0",
        "numpy<2.0.0",
        "pandas>=2.2.3",
        "scipy>=1.14.0",
        "scikit-learn>=1.5.0",
        "google-genai>=0.2.0",
        "langchain>=0.3.0",
        "langchain-google-genai>=2.0.0",
        "huggingface-hub>=0.26.0",
        "transformers>=4.45.0",
        "torch>=2.5.0",
        "requests>=2.32.0",
        "reportlab>=4.2.0",
        "pydantic>=2.10.0",
        "pydantic-settings>=2.6.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-dotenv>=1.0.1",
        "aiofiles>=24.1.0",
        "pyserial>=3.5",
        "rich",
        "langchain-community>=0.3.0",
    )
    .apt_install("libgl1", "libglib2.0-0", "libgomp1", "libcairo2")
)

# Create a volume for persistent storage (models, cache, reports)
model_vol = modal.Volume.from_name("health-screening-models", create_if_missing=True)
report_vol = modal.Volume.from_name("health-screening-reports", create_if_missing=True)

# Optional: GPU configuration for ML-heavy operations
gpu_config = modal.gpu.A10G()  # or modal.gpu.A100() for faster inference


@app.function(
    image=image,
    volumes={"/app/models": model_vol, "/app/reports": report_vol},
    timeout=3600,  # 1 hour timeout for intensive processing
    memory=8192,  # 8GB RAM
    keep_warm=1,  # Keep one instance warm
)
@modal.asgi_app()
def fastapi_app():
    """
    Main FastAPI application endpoint
    """
    import sys
    sys.path.insert(0, "/app")
    
    from app.main import app as fastapi_app
    
    return fastapi_app


@app.function(
    image=image,
    volumes={"/app/models": model_vol, "/app/reports": report_vol},
    timeout=7200,  # 2 hour timeout for heavy ML inference
    gpu=gpu_config,  # Use GPU for faster inference
    memory=16384,  # 16GB RAM for ML models
    keep_warm=0,  # Spin down when idle (cost optimization)
)
def run_inference(data: dict):
    """
    GPU-accelerated inference function for risk scores, pose detection, etc.
    Call this for compute-intensive ML tasks
    """
    import sys
    sys.path.insert(0, "/app")
    
    from app.core.inference.risk_engine import RiskEngine
    
    risk_engine = RiskEngine()
    result = risk_engine.calculate_risk(data)
    
    return result


@app.function(
    image=image,
    volumes={"/app/models": model_vol, "/app/reports": report_vol},
    timeout=3600,
    memory=8192,
)
def generate_report(patient_id: str, data: dict):
    """
    Background job for PDF report generation
    """
    import sys
    sys.path.insert(0, "/app")
    
    from app.core.reports.generator import ReportGenerator
    
    generator = ReportGenerator()
    report_path = generator.generate_pdf(patient_id, data)
    
    return {"status": "completed", "report_path": report_path}


@app.function(
    image=image,
    volumes={"/app/models": model_vol},
    timeout=600,
)
def download_models():
    """
    Utility function to download and cache models
    Run once: `modal run modal_app.py::download_models`
    """
    import sys
    sys.path.insert(0, "/app")
    
    print("Downloading MediaPipe pose landmarker...")
    import mediapipe as mp
    from mediapipe.tasks import python
    
    # Models will be cached in the volume
    print("✓ Models cached successfully")


# CLI Commands
@app.local_entrypoint()
def main():
    """
    Usage:
    - Deploy: modal deploy modal_app.py
    - Deploy with custom name: modal deploy modal_app.py --name my-health-app
    - View logs: modal logs your-app-name
    - Stop: modal stop your-app-name
    """
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║  Modal Deployment Guide for Health Screening Pipeline ║
    ╚═══════════════════════════════════════════════════════╝
    
    DEPLOYMENT COMMANDS:
    
    1. Initial deployment with volume setup:
       modal deploy modal_app.py
    
    2. Access your app:
       https://your-username--health-screening-pipeline.modal.run
    
    3. View logs:
       modal logs health-screening-pipeline
    
    4. Update code:
       modal deploy modal_app.py
    
    5. Stop the app:
       modal stop health-screening-pipeline
    
    VOLUMES MANAGEMENT:
    
    - List volumes: modal volume ls
    - Upload files: modal volume put health-screening-models local/path remote/path
    - Download: modal volume get health-screening-models remote/path local/path
    
    CONFIGURATION:
    
    - GPU Types: A10G, A100, L4 (adjust in gpu_config)
    - Memory: Adjust memory parameter (default 8192 MB)
    - Timeout: Set timeout for long-running tasks
    - Keep warm: Set keep_warm=1 to reduce cold starts
    
    ENVIRONMENT VARIABLES:
    
    Add to Modal secrets (modal secret create):
    - GOOGLE_API_KEY (for Gemini)
    - HUGGINGFACE_TOKEN (for HugginFace models)
    
    modal secret create gemini-secrets \
        --key GOOGLE_API_KEY --value "your-key"
    """)
