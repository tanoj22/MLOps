"""Trains the wine-cultivar classifier and saves it with its metadata.

Run from the lab folder:  python src/train.py
Writes model/wine_model.joblib and model/metadata.json.
"""
import json
import time
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def train(model_dir=MODEL_DIR):
    """Fits the pipeline, saves it to model_dir and returns its metadata."""
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    metadata = {
        "model": "RandomForestClassifier",
        "dataset": "sklearn wine",
        "n_features": len(wine.feature_names),
        "feature_names": list(wine.feature_names),
        "classes": list(wine.target_names),
        "test_accuracy": round(pipeline.score(X_test, y_test), 4),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "wine_model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Model saved to {model_dir}, test accuracy = {metadata['test_accuracy']}")
    return metadata


if __name__ == "__main__":
    train()
