"""
Integration tests for FastAPI inference endpoints and monitoring logging.
"""

import io
import os
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.api import app
from src.evaluate import check_quality_gates


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_health_endpoint(client):
    """Verify /health returns 200 OK and model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data
    assert data["num_classes"] == 15


def test_api_model_info_endpoint(client):
    """Verify /model-info returns metadata and 15 classes."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["num_classes"] == 15
    assert len(data["class_names"]) == 15
    assert "mean" in data["normalization"]


def test_api_predict_single_image(client):
    """Verify POST /predict returns valid diagnosis on synthetic image."""
    dummy_img = Image.fromarray(np.uint8(np.random.randint(0, 255, (224, 224, 3))))
    buf = io.BytesIO()
    dummy_img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_leaf.jpg", buf, "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "top_prediction" in data
    assert "confidence" in data
    assert "crop" in data
    assert "disease" in data
    assert len(data["top_k"]) == 3
    assert data["inference_time_ms"] > 0


def test_api_predict_invalid_file(client):
    """Verify POST /predict returns 400 Bad Request on invalid file."""
    fake_txt = io.BytesIO(b"Hello world, this is not an image")
    files = {"file": ("document.txt", fake_txt, "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_api_predict_batch(client):
    """Verify POST /predict/batch handles multiple image uploads."""
    img1 = Image.fromarray(np.uint8(np.random.randint(0, 255, (224, 224, 3))))
    buf1 = io.BytesIO()
    img1.save(buf1, format="JPEG")
    buf1.seek(0)

    img2 = Image.fromarray(np.uint8(np.random.randint(0, 255, (224, 224, 3))))
    buf2 = io.BytesIO()
    img2.save(buf2, format="JPEG")
    buf2.seek(0)

    files = [
        ("files", ("leaf1.jpg", buf1, "image/jpeg")),
        ("files", ("leaf2.jpg", buf2, "image/jpeg")),
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_images"] == 2
    assert data["successful"] == 2
    assert len(data["results"]) == 2


def test_quality_gate_logic():
    """Verify quality gate pass/fail criteria under different conditions."""
    good_metrics = {
        "accuracy": 0.952,
        "macro_f1": 0.941,
        "min_class_recall": 0.820,
        "avg_latency_ms": 12.5
    }
    passed, report = check_quality_gates(good_metrics)
    assert passed is True
    assert report["checks"]["accuracy_check"]["passed"] is True

    failing_metrics = {
        "accuracy": 0.820,  # below 0.90
        "macro_f1": 0.790,
        "min_class_recall": 0.500,
        "avg_latency_ms": 15.0
    }
    passed_fail, report_fail = check_quality_gates(failing_metrics)
    assert passed_fail is False
    assert report_fail["checks"]["accuracy_check"]["passed"] is False
