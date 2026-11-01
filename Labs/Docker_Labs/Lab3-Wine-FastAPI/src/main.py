"""FastAPI service that serves the wine-cultivar classifier trained by train.py.

Based on API_Labs/FastAPI_Labs (Iris) and Docker_Labs/Lab2 (multi-stage build), with these changes:
  - Wine dataset (13 features, 3 cultivars) instead of Iris
  - The model is loaded once at startup (lifespan) instead of on every request
  - Named, validated input fields (Pydantic) with an example shown in /docs
  - Single and batch prediction endpoints that return class names and probabilities
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))
MAX_BATCH_SIZE = 100

# Filled in at startup by lifespan().
state = {}


class WineSample(BaseModel):
    """Chemical analysis of one wine. Field order matches the training features."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2,
        "magnesium": 100, "total_phenols": 2.65, "flavanoids": 2.76,
        "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38,
        "hue": 1.05, "od280_od315_of_diluted_wines": 3.4, "proline": 1050,
    }})

    alcohol: float = Field(ge=0)
    malic_acid: float = Field(ge=0)
    ash: float = Field(ge=0)
    alcalinity_of_ash: float = Field(ge=0)
    magnesium: float = Field(ge=0)
    total_phenols: float = Field(ge=0)
    flavanoids: float = Field(ge=0)
    nonflavanoid_phenols: float = Field(ge=0)
    proanthocyanins: float = Field(ge=0)
    color_intensity: float = Field(ge=0)
    hue: float = Field(ge=0)
    od280_od315_of_diluted_wines: float = Field(ge=0)
    proline: float = Field(ge=0)

    def to_row(self):
        return list(self.model_dump().values())


class WineBatch(BaseModel):
    instances: List[WineSample] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    predicted_class: str
    probabilities: Dict[str, float]


class BatchPrediction(BaseModel):
    predictions: List[Prediction]
    count: int
    latency_ms: float


@asynccontextmanager
async def lifespan(app):
    """Loads the model once when the server starts."""
    state["pipeline"] = joblib.load(MODEL_DIR / "wine_model.joblib")
    state["metadata"] = json.loads((MODEL_DIR / "metadata.json").read_text())
    print(f"Model loaded from {MODEL_DIR}, test accuracy = {state['metadata']['test_accuracy']}")
    yield
    state.clear()


app = FastAPI(title="Wine Classifier API", version="1.0.0", lifespan=lifespan)


def run_model(samples):
    """Returns one Prediction per sample."""
    classes = state["metadata"]["classes"]
    return [
        Prediction(
            predicted_class=classes[probs.argmax()],
            probabilities={c: round(float(p), 4) for c, p in zip(classes, probs)},
        )
        for probs in state["pipeline"].predict_proba([s.to_row() for s in samples])
    ]


@app.get("/")
def health():
    """Health check, used by the Docker HEALTHCHECK."""
    return {"status": "healthy", "model_loaded": "pipeline" in state}


@app.get("/model-info")
def model_info():
    """Model metadata, including the held-out test accuracy."""
    return state["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(sample: WineSample):
    """Predicts the cultivar of one wine."""
    try:
        return run_model([sample])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(batch: WineBatch):
    """Predicts the cultivar of up to 100 wines in one request."""
    start = time.perf_counter()
    try:
        predictions = run_model(batch.instances)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return BatchPrediction(predictions=predictions, count=len(predictions), latency_ms=latency_ms)
