"""
Integration Tests for FastAPI Backend (Phase 9)

Tests for API endpoints: screening, reports, health checks.
Uses async httpx for ASGI app testing.
"""
import pytest
import httpx
from typing import Dict, Any

from app.main import app


@pytest.fixture
async def async_client():
    """Create async test client."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


@pytest.mark.asyncio
class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    async def test_root_endpoint(self, async_client):
        """Test root endpoint returns health info."""
        response = await async_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    async def test_health_endpoint(self, async_client):
        """Test /health endpoint."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
    
    async def test_list_systems(self, async_client):
        """Test systems listing endpoint."""
        response = await async_client.get("/api/v1/systems")
        assert response.status_code == 200
        
        data = response.json()
        assert "systems" in data
        assert len(data["systems"]) > 0


@pytest.mark.asyncio
class TestScreeningEndpoint:
    """Tests for screening endpoint."""
    
    async def test_screening_single_system(self, async_client):
        """Test screening with single system."""
        request_data = {
            "patient_id": "TEST-001",
            "systems": [
                {
                    "system": "cardiovascular",
                    "biomarkers": [
                        {"name": "heart_rate", "value": 72, "unit": "bpm", "status": "normal"}
                    ]
                }
            ],
            "include_validation": True
        }
        
        response = await async_client.post("/api/v1/screening", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "screening_id" in data
        assert data["patient_id"] == "TEST-001"
        assert "overall_risk_level" in data
    
    async def test_screening_multiple_systems(self, async_client):
        """Test screening with multiple systems."""
        request_data = {
            "systems": [
                {"system": "cardiovascular", "biomarkers": [{"name": "hr", "value": 72}]},
                {"system": "cns", "biomarkers": [{"name": "gait", "value": 0.05}]}
            ]
        }
        
        response = await async_client.post("/api/v1/screening", json=request_data)
        assert response.status_code == 200
        assert len(response.json()["system_results"]) == 2
    
    async def test_screening_invalid_system(self, async_client):
        """Test screening with invalid system name."""
        request_data = {
            "systems": [
                {"system": "nonexistent_system", "biomarkers": [{"name": "x", "value": 1}]}
            ]
        }
        
        response = await async_client.post("/api/v1/screening", json=request_data)
        assert response.status_code == 400
    
    async def test_get_nonexistent_screening(self, async_client):
        """Test getting nonexistent screening returns 404."""
        response = await async_client.get("/api/v1/screening/NONEXISTENT")
        assert response.status_code == 404


@pytest.mark.asyncio
class TestReportEndpoints:
    """Tests for report generation endpoints."""
    
    async def test_report_for_nonexistent_screening(self, async_client):
        """Test report generation for nonexistent screening."""
        response = await async_client.post("/api/v1/reports/generate", json={
            "screening_id": "NONEXISTENT",
            "report_type": "patient"
        })
        assert response.status_code == 404
    
    async def test_download_nonexistent_report(self, async_client):
        """Test downloading nonexistent report returns 404."""
        response = await async_client.get("/api/v1/reports/NONEXISTENT/download")
        assert response.status_code == 404


@pytest.mark.asyncio
class TestAPIDocumentation:
    """Tests for API documentation availability."""
    
    async def test_docs_endpoint(self, async_client):
        """Test Swagger docs are available."""
        response = await async_client.get("/docs")
        assert response.status_code == 200
    
    async def test_redoc_endpoint(self, async_client):
        """Test ReDoc is available."""
        response = await async_client.get("/redoc")
        assert response.status_code == 200


@pytest.mark.asyncio
class TestIntegration:
    """Full integration tests."""
    
    async def test_full_workflow(self, async_client):
        """Test complete screening and report workflow."""
        # Create screening
        screening_resp = await async_client.post("/api/v1/screening", json={
            "patient_id": "INTEGRATION-TEST",
            "systems": [
                {"system": "cardiovascular", "biomarkers": [{"name": "hr", "value": 72}]},
                {"system": "pulmonary", "biomarkers": [{"name": "spo2", "value": 98}]}
            ],
            "include_validation": True
        })
        assert screening_resp.status_code == 200
        
        screening = screening_resp.json()
        assert len(screening["system_results"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
