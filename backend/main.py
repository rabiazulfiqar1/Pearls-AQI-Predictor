from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.predict import run_prediction_pipeline

app = FastAPI(title="Pearls AQI Predictor API", version="1.0.0")


class PredictionResponse(BaseModel):
    generated_at: str
    forecasts: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predictions", response_model=PredictionResponse)
def get_predictions() -> PredictionResponse:
    try:
        predictions = run_prediction_pipeline()
    except Exception as exc:  # pragma: no cover - defensive API layer
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if predictions.empty:
        raise HTTPException(status_code=500, detail="No predictions generated")

    return PredictionResponse(
        generated_at=str(predictions["generated_at"].iloc[0]),
        forecasts=predictions.to_dict(orient="records"),
    )
