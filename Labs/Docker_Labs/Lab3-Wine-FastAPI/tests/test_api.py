"""API tests. Trains a fresh model into a temp folder, so no saved model is needed.

Run from the lab folder:  pytest
"""
import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SAMPLES = json.loads((Path(__file__).resolve().parent.parent / "sample_request.json").read_text())["instances"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("model")
    import train
    train.train(model_dir)

    os.environ["MODEL_DIR"] = str(model_dir)
    import main
    importlib.reload(main)  # picks up MODEL_DIR
    with TestClient(main.app) as c:  # the with block runs the lifespan (model load)
        yield c


def test_health(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "model_loaded": True}


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["n_features"] == 13
    assert info["classes"] == ["class_0", "class_1", "class_2"]
    assert info["test_accuracy"] > 0.9


def test_predict_single(client):
    r = client.post("/predict", json=SAMPLES[0])
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == "class_0"
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_batch_one_per_class(client):
    r = client.post("/predict/batch", json={"instances": SAMPLES})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert [p["predicted_class"] for p in body["predictions"]] == ["class_0", "class_1", "class_2"]


def test_missing_field_rejected(client):
    sample = dict(SAMPLES[0])
    del sample["proline"]
    assert client.post("/predict", json=sample).status_code == 422


def test_non_numeric_rejected(client):
    assert client.post("/predict", json={**SAMPLES[0], "alcohol": "high"}).status_code == 422


def test_negative_value_rejected(client):
    assert client.post("/predict", json={**SAMPLES[0], "hue": -1}).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422


def test_oversized_batch_rejected(client):
    assert client.post("/predict/batch", json={"instances": [SAMPLES[0]] * 101}).status_code == 422
