import httpx
import pytest
import asyncio
from typing import Dict, Any


@pytest.mark.asyncio
async def test_dtc_lookup():
    """Test DTC code lookup functionality."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Test search for P0420
        response = await client.post("/search", json={"query": "P0420", "k": 3})
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert len(data["results"]) > 0
        
        # Check if P0420 is in the results
        found_p0420 = any("P0420" in str(result.get("code", "")) for result in data["results"])
        assert found_p0420, "P0420 should be found in search results"


@pytest.mark.asyncio
async def test_search_with_vin():
    """Test search with VIN filtering."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Test search with VIN
        response = await client.post("/search", json={
            "query": "P0420",
            "vin": "2HGFC2F59JH000001",
            "k": 5
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data


@pytest.mark.asyncio
async def test_empty_query():
    """Test handling of empty queries."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post("/search", json={"query": "", "k": 5})
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_long_query():
    """Test handling of very long queries."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        long_query = "A" * 600  # Exceeds 500 character limit
        response = await client.post("/search", json={"query": long_query, "k": 5})
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_dtc_endpoint():
    """Test direct DTC endpoint."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.get("/dtc/P0420")
        assert response.status_code == 200
        
        data = response.json()
        assert "code" in data
        assert "description" in data
        assert data["code"] == "P0420"


@pytest.mark.asyncio
async def test_recalls_endpoint():
    """Test recalls endpoint."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.get("/recalls/2HGFC2F59JH000001")
        assert response.status_code == 200
        
        data = response.json()
        assert "vin" in data
        assert "recalls" in data
        assert "count" in data


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check endpoint."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data


@pytest.mark.asyncio
async def test_search_with_filter():
    """Test search with type filter."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post("/search", json={
            "query": "catalyst",
            "filter_type": "dtc",
            "k": 5
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        
        # All results should be DTC type
        for result in data["results"]:
            assert result.get("type") == "dtc"


@pytest.mark.asyncio
async def test_invalid_dtc_code():
    """Test handling of invalid DTC codes."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.get("/dtc/INVALID")
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_search_response_structure():
    """Test that search response has correct structure."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post("/search", json={"query": "P0420", "k": 3})
        assert response.status_code == 200
        
        data = response.json()
        required_fields = ["results", "query", "total_found"]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert isinstance(data["results"], list)
        assert isinstance(data["query"], str)
        assert isinstance(data["total_found"], int)


@pytest.mark.asyncio
async def test_search_result_structure():
    """Test that individual search results have correct structure."""
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post("/search", json={"query": "P0420", "k": 1})
        assert response.status_code == 200
        
        data = response.json()
        if data["results"]:
            result = data["results"][0]
            assert "score" in result
            assert "type" in result
            
            # Check type-specific fields
            if result["type"] == "dtc":
                assert "code" in result
                assert "description" in result
            elif result["type"] == "recall":
                assert "rid" in result
                assert "summary" in result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 