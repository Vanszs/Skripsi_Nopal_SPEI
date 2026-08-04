import sys
from pathlib import Path
import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data

@pytest.mark.asyncio
async def test_predict_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as ac:
        payload = {"city_id": "surabaya", "forecast_days": 5}
        res = await ac.post("/api/v1/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["city_id"] == "surabaya"
        assert len(data["predictions"]["p50"]) == 5

@pytest.mark.asyncio
async def test_spei_calculate_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as ac:
        payload = {"precipitation": [10.0, 5.0], "evapotranspiration": [2.0, 3.0], "scale": 3}
        res = await ac.post("/api/v1/spei/calculate", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert len(data["water_deficit"]) == 2

@pytest.mark.asyncio
async def test_ingest_status_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as ac:
        res = await ac.get("/api/v1/ingest/status")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "synced"

@pytest.mark.asyncio
async def test_stream_weather_sse():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as ac:
        res = await ac.get("/api/v1/stream/weather?city_id=surabaya&interval=0.1&max_steps=1")
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")
        lines = res.text.splitlines()
        assert any("weather_update" in line or "weather_sync" in line for line in lines)
