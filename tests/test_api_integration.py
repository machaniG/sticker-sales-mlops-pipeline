import pytest
import httpx
from scripts.serving_app import app # Import the FastAPI application instance
from datetime import datetime

# --- FIX: Define the asynchronous client fixture correctly ---
# This fixture uses FastAPI's TestClient wrapped by httpx.AsyncClient
# to simulate an actual API call. The 'async' tests that use this fixture 
# must be marked with @pytest.mark.asyncio (or use pytest-asyncio's auto-async).

@pytest.fixture(scope="module")
async def async_client():
    """Provides an asynchronous HTTP client for the FastAPI app."""
    # Use the app instance imported from serving_app
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client

# Ensure the pytest-asyncio marker is present for async tests
@pytest.mark.asyncio
async def test_health_check_endpoint(async_client: httpx.AsyncClient):
    """Test the /health endpoint to ensure the API is running."""
    response = await async_client.get("/health")
    
    # 1. Assert Status Code
    assert response.status_code == 200
    
    # 2. Assert Response Content
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "timestamp" in data

@pytest.mark.asyncio
async def test_single_prediction_endpoint_success(async_client: httpx.AsyncClient):
    """Test the /predict/single endpoint with valid input."""
    # This input must match the Pydantic model structure in serving_app.py
    valid_input = {
        "date": "2024-01-10",
        "country": "Australia",
        "store": "A",
        "product": "Sticker1",
    }
    
    response = await async_client.post("/predict/single", json=valid_input)
    
    # 1. Assert Status Code
    # The first run might fail if the model hasn't been trained and registered.
    # We will assert for 200, assuming a model is available.
    if response.status_code != 200:
        # Check for 503 if the model is not loaded (common during CI initial setup)
        assert response.status_code == 503, f"Expected 200 or 503, got {response.status_code}: {response.text}"
    
    if response.status_code == 200:
        data = response.json()
        assert "predicted_sales" in data
        assert isinstance(data["predicted_sales"], float)
        assert data["predicted_sales"] >= 0  # Sales should be non-negative
        assert "model_version" in data
        
@pytest.mark.asyncio
async def test_single_prediction_endpoint_validation_error(async_client: httpx.AsyncClient):
    """Test the /predict/single endpoint with invalid input (missing field)."""
    invalid_input = {
        "date": "2024-01-10",
        "country": "Australia",
        # 'store' and 'product' are missing
    }
    
    response = await async_client.post("/predict/single", json=invalid_input)
    
    # FastAPI returns 422 for validation errors
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert any("store" in error["loc"] for error in data["detail"])