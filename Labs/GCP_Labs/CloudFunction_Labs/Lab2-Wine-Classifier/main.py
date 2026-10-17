"""HTTP-triggered Cloud Run Function that serves a wine-cultivar classifier.

Based on the MLOps GCP_Labs CloudFunction Lab1 (ML-Example), with these changes:
  - Wine dataset (13 features, 3 cultivars) instead of Iris
  - scikit-learn Pipeline (StandardScaler + RandomForest) instead of a separate scaler/model
  - Model is trained once per instance at cold start and kept in memory
  - GET returns model info and held-out test accuracy (a health check)
  - POST accepts a single sample or a batch, returns class names and probabilities
"""
import time

import functions_framework
from flask import jsonify
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MAX_BATCH_SIZE = 100


def train_model():
    """Trains the pipeline and returns it with its metadata."""
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    info = {
        "model": "RandomForestClassifier",
        "dataset": "sklearn wine",
        "n_features": len(wine.feature_names),
        "feature_names": list(wine.feature_names),
        "classes": list(wine.target_names),
        "test_accuracy": round(pipeline.score(X_test, y_test), 4),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"Model trained, test accuracy = {info['test_accuracy']}")
    return pipeline, info


# Runs once per instance (cold start); warm requests reuse the trained model.
PIPELINE, MODEL_INFO = train_model()


def parse_samples(body):
    """Returns (samples, error). Accepts {"features": [...]} or {"instances": [[...], ...]}."""
    if "features" in body:
        samples = [body["features"]]
    elif "instances" in body:
        samples = body["instances"]
    else:
        return None, "Request body must contain 'features' (one sample) or 'instances' (a batch)."

    if not isinstance(samples, list) or not samples:
        return None, "'instances' must be a non-empty list."
    if len(samples) > MAX_BATCH_SIZE:
        return None, f"Batch too large: at most {MAX_BATCH_SIZE} samples per request."

    n = MODEL_INFO["n_features"]
    for i, sample in enumerate(samples):
        if not isinstance(sample, list) or len(sample) != n:
            return None, f"Sample {i} must be a list of {n} numeric values."
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in sample):
            return None, f"Sample {i} contains non-numeric values."
    return samples, None


@functions_framework.http
def predict(request):
    """Entry point. GET -> model info, POST -> predictions."""
    if request.method == "GET":
        return jsonify({"status": "ok", **MODEL_INFO})

    if request.method != "POST":
        return jsonify({"error": "Use GET for model info or POST for predictions."}), 405

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    samples, error = parse_samples(body)
    if error:
        return jsonify({"error": error}), 400

    start = time.perf_counter()
    classes = MODEL_INFO["classes"]
    predictions = [
        {
            "predicted_class": classes[probs.argmax()],
            "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        }
        for probs in PIPELINE.predict_proba(samples)
    ]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return jsonify({"predictions": predictions, "count": len(predictions), "latency_ms": latency_ms})
