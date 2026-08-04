from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()

class PredictRequest(BaseModel):
    city_id: str = Field(..., example="surabaya")
    forecast_days: int = Field(default=30, ge=1, le=90)

class QuantilePredictions(BaseModel):
    p10: List[float]
    p50: List[float]
    p90: List[float]

class PredictResponse(BaseModel):
    city_id: str
    dates: List[str]
    predictions: QuantilePredictions
    drought_risk_level: str

@router.post("/predict", response_model=PredictResponse)
async def predict_spei(payload: PredictRequest):
    """
    TFT Model Inference Endpoint.
    """
    # ponytail: hardcoded mock array for basic endpoint check, load TFT checkpoint in production lifespan
    dates = [f"2026-08-{i+1:02d}" for i in range(payload.forecast_days)]
    p50 = [-0.5 - (i * 0.01) for i in range(payload.forecast_days)]
    p10 = [val - 0.4 for val in p50]
    p90 = [val + 0.4 for val in p50]

    return PredictResponse(
        city_id=payload.city_id,
        dates=dates,
        predictions=QuantilePredictions(p10=p10, p50=p50, p90=p90),
        drought_risk_level="Mild Drought"
    )
