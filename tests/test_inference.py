"""
Unit tests for inference pipeline, preprocessing, batch prediction, uncertainty warnings, and ONNX parity.
"""

import io
import os
import numpy as np
from PIL import Image
import pytest

from src.predict import PlantDiseasePredictor


@pytest.fixture(scope="module")
def predictor():
    checkpoint_path = "plant_disease_resnet50.pth"
    if not os.path.exists(checkpoint_path):
        pytest.skip("Baseline checkpoint not found.")
    return PlantDiseasePredictor(checkpoint_path=checkpoint_path, device="cpu")


def test_single_image_prediction(predictor):
    """Verify single image prediction structure, confidence range, and top-3 results."""
    dummy_img = Image.fromarray(np.uint8(np.random.randint(0, 255, (224, 224, 3))))
    result = predictor.predict(dummy_img, top_k=3)

    assert "top_prediction" in result
    assert "confidence" in result
    assert "crop" in result
    assert "disease" in result
    assert "top_k" in result
    assert "inference_time_ms" in result
    assert len(result["top_k"]) == 3
    assert 0.0 <= result["confidence"] <= 1.0

    # Top-1 confidence should be greater than or equal to top-2
    assert result["top_k"][0]["confidence"] >= result["top_k"][1]["confidence"]


def test_predict_from_bytes(predictor):
    """Verify prediction accepts raw image bytes."""
    dummy_img = Image.fromarray(np.uint8(np.random.randint(0, 255, (224, 224, 3))))
    buf = io.BytesIO()
    dummy_img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    result = predictor.predict(raw_bytes)
    assert result["top_prediction"] in predictor.class_names


def test_batch_prediction(predictor):
    """Verify batch inference produces results for all valid inputs."""
    img1 = Image.fromarray(np.uint8(np.random.randint(0, 255, (224, 224, 3))))
    img2 = Image.fromarray(np.uint8(np.random.randint(0, 255, (224, 224, 3))))

    results = predictor.predict_batch([img1, img2])
    assert len(results) == 2
    assert results[0]["success"] is True
    assert results[1]["success"] is True
    assert results[0]["confidence"] > 0.0


def test_invalid_image_handling(predictor):
    """Verify corrupted or invalid byte payloads raise descriptive ValueError."""
    corrupted_bytes = b"not_an_image_file_content_12345"
    with pytest.raises(ValueError) as excinfo:
        predictor.preprocess_image(corrupted_bytes)
    assert "valid image" in str(excinfo.value).lower() or "parsed" in str(excinfo.value).lower()


def test_uncertainty_threshold_flag(predictor):
    """Verify low confidence triggers uncertainty warning."""
    predictor.uncertainty_threshold = 0.999  # Guaranteed to trigger warning
    dummy_img = Image.fromarray(np.uint8(np.random.randint(0, 255, (224, 224, 3))))
    result = predictor.predict(dummy_img)

    assert result["is_uncertain"] is True
    assert result["warning"] is not None

    # Reset
    predictor.uncertainty_threshold = 0.60


def test_onnx_parity_if_available():
    """Verify numerical parity between PyTorch and ONNX Runtime if ONNX model exists."""
    onnx_path = "artifacts/model.onnx"
    checkpoint_path = "plant_disease_resnet50.pth"

    if not os.path.exists(onnx_path) or not os.path.exists(checkpoint_path):
        pytest.skip("ONNX model or checkpoint not available for parity check.")

    try:
        import onnxruntime as ort
        from src.export import benchmark_models

        benchmark_res = benchmark_models(checkpoint_path, onnx_path, num_iterations=2)
        assert benchmark_res["parity_check"]["parity_passed"] is True
        assert benchmark_res["parity_check"]["max_absolute_difference"] < 1e-4
    except ImportError:
        pytest.skip("ONNXRuntime not installed.")
