# Modal Serverless Deployment Guide

## Overview
Modal is a serverless cloud platform perfect for deploying AI/ML applications. It handles containerization, auto-scaling, GPU support, and persistent storage automatically.

## Step 1: Install Modal

```bash
pip install modal
```

## Step 2: Authenticate with Modal

```bash
modal token new
```

This will:
- Open your browser to create an API token
- Save credentials locally (~/.modal/token_id.txt)

## Step 3: Project Structure for Modal

Your project structure is already good! Modal will:
- Mount your local code to `/app`
- Use volumes for persistent storage (models, reports)
- Auto-containerize with dependencies

## Step 4: Prepare Environment Variables

Create Modal secrets for API keys:

```bash
# Create Gemini secret
modal secret create gemini-secrets
# Enter: GOOGLE_API_KEY = your-key-here

# Create HuggingFace secret
modal secret create huggingface-secrets
# Enter: HUGGINGFACE_TOKEN = your-token-here
```

Update your FastAPI app to use Modal secrets:

```python
# In app/main.py, modify the environment loading:
import os

# For Modal environment
if "MODAL_ENVIRONMENT" in os.environ:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
else:
    from dotenv import load_dotenv
    load_dotenv()
```

## Step 5: Deploy to Modal

### Basic Deployment
```bash
cd c:\Users\Swetanjana Maity\Desktop\kblndt\techgium\fastapi2
modal deploy modal_app.py
```

Expected output:
```
✓ Created app 'health-screening-pipeline'
✓ App available at: https://your-username--health-screening-pipeline.modal.run
```

### With Custom Name
```bash
modal deploy modal_app.py --name health-app-prod
```

## Step 6: Initialize Volumes (for Models & Reports)

```bash
# Initialize volumes
modal volume create health-screening-models
modal volume create health-screening-reports

# Download models to volume (one-time setup)
modal run modal_app.py::download_models

# Upload local pose landmarker model if needed
modal volume put health-screening-models pose_landmarker.task /app/models/pose_landmarker.task
```

## Step 7: Test Your Deployment

```bash
# Check app status
modal app list

# Watch live logs
modal logs health-screening-pipeline

# Test endpoint
curl https://your-username--health-screening-pipeline.modal.run/health

# Check GPU usage (if enabled)
modal run modal_app.py --show-usage
```

## API Usage Examples

### Health Check
```bash
curl https://your-username--health-screening-pipeline.modal.run/health
```

### Submit Screening Data
```bash
curl -X POST https://your-username--health-screening-pipeline.modal.run/screen \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P123",
    "system": "respiratory",
    "data": {...}
  }'
```

### Get Results
```bash
curl https://your-username--health-screening-pipeline.modal.run/results/P123
```

## Performance Optimization

### 1. GPU Configuration (for heavy ML inference)
In `modal_app.py`, adjust the GPU type:

```python
# Faster but costlier
gpu_config = modal.gpu.A100()

# Balanced
gpu_config = modal.gpu.A10G()

# Budget-friendly
gpu_config = modal.gpu.L4()
```

### 2. Keep Warm Strategy
```python
@app.function(keep_warm=1)  # Keeps 1 instance warm
def fastapi_app():
    ...
```

### 3. Memory Allocation
```python
@app.function(
    memory=16384,  # 16GB for ML models
    timeout=3600   # 1 hour timeout
)
def run_inference(data):
    ...
```

### 4. Concurrency Control
```python
@app.function(concurrency_limit=10)  # Max concurrent requests
def fastapi_app():
    ...
```

## Volume Management

```bash
# List all volumes
modal volume ls

# Check volume size
modal volume info health-screening-models

# Upload file to volume
modal volume put health-screening-models ./pose_landmarker.task /app/models/

# Download file from volume
modal volume get health-screening-models /app/reports/report.pdf ./local/

# Delete old reports (cleanup)
modal volume delete health-screening-reports

# Mount volumes in custom function
@app.function(
    volumes={
        "/app/models": model_vol,
        "/app/reports": report_vol
    }
)
def my_function():
    ...
```

## Common Issues & Solutions

### 1. Model Download Timeout
**Problem**: MediaPipe model download times out
**Solution**: 
```bash
modal run modal_app.py::download_models
```

### 2. GPU Memory Exceeded
**Solution**:
```python
@app.function(memory=32768)  # Increase to 32GB
def run_inference(data):
    ...
```

### 3. Cold starts too slow
**Solution**:
```python
@app.function(keep_warm=1)  # Keep instance warm
def fastapi_app():
    ...
```

### 4. API Key not found
**Solution**: Ensure secrets are created and linked:
```bash
modal secret list
modal create secret <secret-name>
```

## Monitoring & Debugging

```bash
# Real-time logs
modal logs -f health-screening-pipeline

# View function input/output
modal run modal_app.py::generate_report patient_id data

# Performance metrics
modal app info health-screening-pipeline

# Cost estimation
modal app cost health-screening-pipeline
```

## Update Deployment

Simply redeploy with changes:
```bash
# After modifying code
modal deploy modal_app.py

# To stop deployment
modal stop health-screening-pipeline

# To delete deployment
modal app delete health-screening-pipeline
```

## Advanced: Background Jobs

For long-running tasks (report generation):

```python
# In your FastAPI endpoint
@app.post("/screen/async")
async def screen_async(data: ScreeningData):
    # Submit background job
    call_result = generate_report.spawn(data.patient_id, data.dict())
    
    return {
        "status": "processing",
        "job_id": str(call_result.object_id)
    }

# Check job status
@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    call = modal.functions.get_call(job_id)
    # Get result when ready
    result = call.result()
```

## Cost Optimization

1. **Use GPU only for inference functions** ✓ (already configured)
2. **Set appropriate `keep_warm` values** 
3. **Use volumes instead of re-downloading models**
4. **Set `timeout` to prevent hanging tasks**
5. **Monitor usage with `modal app cost`**

## Useful Links

- Modal Docs: https://modal.com/docs
- Modal Examples: https://github.com/modal-labs/modal-examples
- FastAPI Deployment: https://modal.com/docs/guide/fastapi

---

**Next Steps**: Run `modal deploy modal_app.py` and share your app URL!
