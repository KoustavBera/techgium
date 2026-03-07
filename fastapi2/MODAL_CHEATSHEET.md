# Modal Commands Cheat Sheet

## Installation & Authentication
```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Check version
modal --version
```

## Deployment

```bash
# Deploy app
modal deploy modal_app.py

# Deploy with custom name
modal deploy modal_app.py --name my-health-app

# Deploy specific function only
modal deploy modal_app.py::fastapi_app

# View deployment status
modal app list

# Check app URL
modal app info health-screening-pipeline
```

## Volume Management

```bash
# Create volume
modal volume create health-screening-models

# List volumes
modal volume ls

# Get volume info
modal volume info health-screening-models

# Upload file to volume
modal volume put health-screening-models ./pose_landmarker.task /app/models/pose_landmarker.task

# Download file from volume
modal volume get health-screening-models /app/reports/report.pdf ./local/report.pdf

# Delete volume (careful!)
modal volume delete health-screening-models

# List files in volume
modal volume ls health-screening-models /app/models/
```

## Secrets Management

```bash
# Create secret
modal secret create gemini-secrets
# Then paste: GOOGLE_API_KEY = your-key

# List secrets
modal secret list

# Use in code
from modal import Secret
gemini_secret = modal.Secret.from_name("gemini-secrets")

@app.function(secrets=[gemini_secret])
def my_function():
    api_key = os.environ["GOOGLE_API_KEY"]
```

## Monitoring & Logs

```bash
# Stream live logs
modal logs -f health-screening-pipeline

# View last N lines
modal logs health-screening-pipeline -n 50

# Search logs
modal logs health-screening-pipeline | grep "ERROR"

# Get function-specific logs
modal logs health-screening-pipeline::fastapi_app

# Print app info
modal app info health-screening-pipeline

# Estimate costs
modal app cost health-screening-pipeline
```

## Running Functions

```bash
# Run function directly
modal run modal_app.py::download_models

# Run with arguments
modal run modal_app.py::generate_report patient_123 '{"data":"value"}'

# Run in interactive mode with Python
python
from modal import App
app = App.lookup("health-screening-pipeline")
result = app["fastapi_app"]()
```

## Stop & Delete

```bash
# Stop (pause) deployment
modal stop health-screening-pipeline

# Delete deployment completely
modal app delete health-screening-pipeline

# Restart after stop
modal deploy modal_app.py
```

## Debugging

```bash
# Test function locally
modal run modal_app.py --local

# Dry run deployment
modal deploy modal_app.py --dry-run

# Show what would be deployed
modal deploy modal_app.py --show-step

# Verbose output
modal deploy modal_app.py -v

# Check environment
modal run modal_app.py::check_env

# List running instances
modal instance ls
```

## Environment Variables & Configuration

```bash
# Set environment
export MODAL_ENVIRONMENT=prod

# Set workspace
export MODAL_WORKSPACE=my-workspace

# View workspace info
modal workspace ls

# Switch workspace
modal config default-workspace <workspace-name>
```

## Performance Tuning

```python
# In modal_app.py configurations:

# 1. Memory adjustment
@app.function(memory=16384)  # 16GB

# 2. Timeout
@app.function(timeout=3600)  # 1 hour

# 3. Keep warm (reduce cold starts)
@app.function(keep_warm=1)  # Always 1 warm instance

# 4. Concurrency limit
@app.function(concurrency_limit=10)  # Max 10 concurrent

# 5. GPU assignment
@app.function(gpu=modal.gpu.A10G())

# 6. CPU cores
@app.function(cpu=4)

# 7. Retry logic
@app.function(retries=3, timeout=600)
def my_function():
    pass
```

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| **Secret not found** | Create it: `modal secret create name` |
| **Volume mount empty** | Upload: `modal volume put vol-name file /path/` |
| **Model download timeout** | Run: `modal run modal_app.py::download_models` |
| **GPU out of memory** | Increase: `memory=32768` or use smaller model |
| **Cold starts slow** | Use: `keep_warm=1` |
| **High costs** | Reduce: `keep_warm`, use smaller GPU, set timeouts |
| **Function timeout** | Increase: `timeout=7200` |
| **Can't authenticate** | Re-run: `modal token new` |

## Useful Environment Variables

```python
# In app code:
import os

WORKSPACE = os.environ.get("MODAL_WORKSPACE")
TOKEN_ID = os.environ.get("MODAL_TOKEN_ID")
ENVIRONMENT = os.environ.get("MODAL_ENVIRONMENT")

# API endpoints
FASTAPI_URL = os.environ.get("FASTAPI_URL")
```

## API Testing

```bash
# Health check
curl https://your-username--health-screening-pipeline.modal.run/health

# With timeout
curl --max-time 30 https://your-username--health-screening-pipeline.modal.run/health

# Debug headers
curl -v https://your-username--health-screening-pipeline.modal.run/health

# POST with data
curl -X POST https://your-username--health-screening-pipeline.modal.run/screen \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"P123","system":"respiratory"}'

# Large file upload
curl -X POST https://your-username--health-screening-pipeline.modal.run/upload \
  -F "file=@/path/to/large/file.zip"

# Check response headers
curl -i https://your-username--health-screening-pipeline.modal.run/health
```

## Useful Modal Links

- **Docs**: https://modal.com/docs
- **Examples**: https://github.com/modal-labs/modal-examples
- **Status**: https://status.modal.io
- **Pricing**: https://modal.com/pricing
- **Discord**: https://discord.gg/modal

## Sample Python Usage

```python
# Run Modal function from another script
from modal import App

app = App.lookup("health-screening-pipeline")

# Call serverless function
result = app["fastapi_app"].call()

# Or with arguments
result = app["run_inference"].call({"respiratory": {...}})

# Get status
status = app.get_status()
print(status.deployment)
```

## Pro Tips

1. **Keep models in volumes** - Download once, reuse always
2. **Use GPU only for inference** - Web functions don't need GPU
3. **Set appropriate timeouts** - Prevent zombie processes
4. **Monitor costs** - Regular cost checks: `modal app cost app-name`
5. **Use secrets for API keys** - Never hardcode credentials
6. **Test locally first** - Before deploying to Modal
7. **Enable keep_warm strategically** - Only for frequently used functions
8. **Use background jobs** - For long-running tasks (reports, exports)

---

**Need help?** `modal --help` or visit https://modal.com/docs
