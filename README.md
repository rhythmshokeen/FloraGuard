# 🌿 FloraGuard: Production-Grade Plant Disease Classification & MLOps Platform

[![CI/CD Pipeline](https://github.com/rhythmshokeen/FloraGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/rhythmshokeen/FloraGuard/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet.svg)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)](https://streamlit.io/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.6x_Speedup-brightgreen.svg)](https://onnxruntime.ai/)

A modular, production-ready MLOps platform for automated plant leaf disease diagnosis across **Bell Pepper**, **Potato**, and **Tomato** crops. The project features reproducible PyTorch model training, experiment tracking with MLflow, automated quality gate evaluations, candidate-to-production model registry versioning, real-time REST API inference, an interactive Streamlit UI, continuous data drift monitoring, and optimized ONNX runtime deployment.

---

## 1. System Architecture & Workflow

```text
       +-----------------------------------------------------------+
       |                  PlantVillage Dataset                     |
       +-----------------------------------------------------------+
                                     |
                                     v
                       +---------------------------+
                       |    Stratified Split &     |
                       | Preprocessing Transforms  |
                       +---------------------------+
                                     |
                                     v
+------------------+   +---------------------------+   +-------------------+
|  Quality Gates   |<--|    PyTorch ResNet-50      |-->|  MLflow Tracking  |
| Accuracy >= 90%  |   | (Frozen / Fine-Tuned head)|   |  (sqlite:///)     |
| Macro F1 >= 0.88 |   +---------------------------+   +-------------------+
+------------------+                 |                           |
         |                           v                           v
         |                 Candidate Checkpoint         MLflow Registry
         +--------[Pass]--------> Promotion                      |
                                     |                           |
                                     v                           v
                       +---------------------------+   +-------------------+
                       |    Production Registry    |---|  ONNX Optimizer   |
                       | (model_manifest.json)     |   |  (1.6x Speedup)   |
                       +---------------------------+   +-------------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
       +--------------------+                 +--------------------+
       |   FastAPI Engine   |                 |   Streamlit App    |
       |  (REST Inference)  |                 | (Interactive Web)  |
       +--------------------+                 +--------------------+
                 |                                       |
                 +-------------------+-------------------+
                                     |
                                     v
                       +---------------------------+
                       |    Monitoring & Drift     |
                       | (Latency, PSI, Review Q)  |
                       +---------------------------+
```

---

## 2. Directory Structure

```text
FloraGuard/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated CI/CD GitHub Actions workflow
├── app/
│   ├── __init__.py
│   ├── api.py                     # Production FastAPI REST service
│   └── streamlit_app.py           # Interactive Streamlit Web UI
├── configs/
│   └── train.yaml                 # Centralized training & quality gate parameters
├── src/
│   ├── __init__.py
│   ├── data.py                    # Dataset resolution, transforms & stratified splits
│   ├── model.py                   # ResNet-50 architecture & checkpoint serialization
│   ├── train.py                   # Reproducible training & MLflow logging
│   ├── evaluate.py                # Metrics, confusion matrix, and quality gate logic
│   ├── predict.py                 # Predictor engine, batch inference & uncertainty
│   ├── registry.py                # Model manifest, candidate promotion & rollback
│   ├── monitoring.py              # Structured logging, PSI drift analysis, review queue
│   └── export.py                  # ONNX export, graph checker & latency benchmark
├── tests/
│   ├── __init__.py
│   ├── test_model.py              # Architecture, device & checkpoint unit tests
│   ├── test_data.py               # Transforms, labels & zero-leakage split tests
│   ├── test_inference.py          # Single/batch prediction & uncertainty tests
│   └── test_api.py                # FastAPI endpoints & quality gate integration tests
├── artifacts/
│   ├── checkpoints/               # Trained candidate model weights
│   ├── monitoring/                # Structured prediction logs & review queue
│   ├── model.onnx                 # Exported ONNX graph
│   └── model_manifest.json        # Production version manifest & candidate tracking
├── reports/
│   ├── classification_report.json # Detailed per-class precision, recall, F1
│   ├── confusion_matrix.png       # 15-class heatmap visualization
│   ├── misclassified_samples.csv  # Error analysis table with confidence scores
│   ├── quality_gate_report.json   # Quality gate pass/fail summary
│   ├── monitoring_report.json     # Drift PSI and latency metrics
│   └── onnx_benchmark.json        # PyTorch CPU vs ONNX Runtime benchmark
├── plant_disease_resnet50.pth     # Preserved baseline trained weights (95.01% accuracy)
├── model.ipynb                    # Original training notebook (preserved)
├── start.ipynb                    # Environment setup notebook (preserved)
├── new.ipynb                      # Data inspection notebook (preserved)
├── PlantVillage/                  # Dataset with 15 disease/healthy crop classes
├── requirements.txt               # Pinned project dependencies
├── Dockerfile                     # Multi-stage production container
├── docker-compose.yml             # API, Streamlit UI & MLflow multi-service stack
├── Makefile                       # Developer command interface
└── README.md                      # Comprehensive documentation
```

---

## 3. Supported Crops & Disease Classes (15 Total)

The model is trained specifically on the 15 classes from the PlantVillage dataset:

| # | Crop | Category Name | Condition Type |
|---|------|---------------|----------------|
| 1 | **Bell Pepper** | `Pepper__bell___Bacterial_spot` | Bacterial Infection ⚠️ |
| 2 | **Bell Pepper** | `Pepper__bell___healthy` | Healthy Condition ✅ |
| 3 | **Potato** | `Potato___Early_blight` | Fungal Blight ⚠️ |
| 4 | **Potato** | `Potato___Late_blight` | Fungal Blight ⚠️ |
| 5 | **Potato** | `Potato___healthy` | Healthy Condition ✅ |
| 6 | **Tomato** | `Tomato_Bacterial_spot` | Bacterial Spot ⚠️ |
| 7 | **Tomato** | `Tomato_Early_blight` | Early Blight ⚠️ |
| 8 | **Tomato** | `Tomato_Late_blight` | Late Blight ⚠️ |
| 9 | **Tomato** | `Tomato_Leaf_Mold` | Fungal Mold ⚠️ |
| 10 | **Tomato** | `Tomato_Septoria_leaf_spot` | Septoria Leaf Spot ⚠️ |
| 11 | **Tomato** | `Tomato_Spider_mites_Two_spotted_spider_mite` | Pest Infestation ⚠️ |
| 12 | **Tomato** | `Tomato__Target_Spot` | Target Spot ⚠️ |
| 13 | **Tomato** | `Tomato__Tomato_YellowLeaf__Curl_Virus` | Viral Infection ⚠️ |
| 14 | **Tomato** | `Tomato__Tomato_mosaic_virus` | Mosaic Virus ⚠️ |
| 15 | **Tomato** | `Tomato_healthy` | Healthy Condition ✅ |

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.11.9)
- PyTorch 2.0+ with CUDA or Apple Silicon (MPS) support

### Local Installation
```bash
# Clone the repository
git clone https://github.com/rhythmshokeen/FloraGuard.git
cd FloraGuard

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies using Makefile
make setup
```

---

## 5. Quickstart Commands (Makefile)

| Command | Action |
|---------|--------|
| `make test` | Run complete unit and integration test suite via `pytest` |
| `make evaluate` | Evaluate baseline checkpoint and generate confusion matrix & metrics |
| `make train-smoke` | Run rapid 1-epoch training smoke test with MLflow tracking |
| `make train` | Run full training with `configs/train.yaml` |
| `make export` | Export model to ONNX and run latency & throughput benchmarks |
| `make api` | Launch FastAPI inference server (`http://localhost:8000`) |
| `make ui` | Launch Streamlit web application (`http://localhost:8501`) |
| `make mlflow` | Launch local MLflow dashboard (`http://localhost:5000`) |
| `make docker-up` | Run full API, UI, and MLflow stack in Docker Compose |

---

## 6. Training Pipeline & MLflow Experiment Tracking

The training pipeline in [`src/train.py`](src/train.py) supports:
- **Parameter Freezing & Fine-Tuning Modes:**
  - `frozen`: Only classification head `Linear(2048, 15)` is trainable (default baseline).
  - `fine_tune_layer4`: Unfreezes top ResNet layer 4 with lower differential learning rate (`1e-4`).
  - `fine_tune_layer3_layer4`: Unfreezes layers 3 and 4.
  - `fine_tune_all`: End-to-end backpropagation.
- **Stratified Split:** 80/20 train/validation split strictly checked for zero sample leakage.
- **Hardware Acceleration:** Automatically detects Apple Silicon `mps`, NVIDIA `cuda`, or `cpu`.

### Running Experiments
```bash
# Rapid 1-epoch smoke test
python3 src/train.py --smoke-test

# Full 5-epoch training with default config
python3 src/train.py --config configs/train.yaml --mode frozen

# Fine-tuning Layer 4 with custom run name
python3 src/train.py --mode fine_tune_layer4 --epochs 5 --run-name ft_layer4_exp1
```

### Launching the MLflow UI
```bash
# Launch dashboard pointing to the SQLite tracking database
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open **`http://localhost:5000`** in your browser to view experiment runs, loss curves, confusion matrices, and parameter comparisons.

---

## 7. Automated Model Evaluation & Quality Gates

Run evaluation on any checkpoint using [`src/evaluate.py`](src/evaluate.py):

```bash
python3 src/evaluate.py --checkpoint plant_disease_resnet50.pth
```

### Generated Artifacts:
- **`reports/confusion_matrix.png`**: High-resolution heatmap of true vs. predicted classes.
- **`reports/classification_report.json`**: Precision, recall, and F1 per class.
- **`reports/misclassified_samples.csv`**: Detailed audit table of incorrect classifications with model confidences.
- **`reports/quality_gate_report.json`**: Automated quality check status.

### Quality Gate Thresholds:
| Metric | Threshold | Baseline Actual | Status |
|--------|-----------|-----------------|--------|
| **Validation Accuracy** | $\ge 90.0\%$ | **93.72%** (subset) / **95.01%** (full) | ✅ PASSED |
| **Macro F1 Score** | $\ge 0.880$ | **0.9373** | ✅ PASSED |
| **Worst-Class Recall** | $\ge 75.0\%$ | **80.00%** | ✅ PASSED |
| **Inference Latency** | $\le 250\text{ ms}$ | **3.23 ms** (GPU) / **20.5 ms** (CPU) | ✅ PASSED |

---

## 8. Model Registry & Version Management

Model versions are tracked in [`artifacts/model_manifest.json`](artifacts/model_manifest.json) and registered in MLflow:
- **Champion (Production):** Current actively deployed model (`1.0.0-baseline`).
- **Challengers (Candidates):** Newly trained models are automatically evaluated against the baseline.
- **Quality Guard:** Candidate models failing quality gates are **never** promoted.
- **Rollback Support:** Previous production weights are automatically archived into `artifacts/backups/`.

To inspect or rollback programmatically:
```python
from src.registry import get_current_production_model, rollback_production

print(get_current_production_model())
# rollback_production()  # Restores previous known-good model
```

---

## 9. Inference API (FastAPI)

Launch the REST service:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at **`http://localhost:8000/docs`**.

### Endpoints
- `GET /health`: Health status, device, model version, and uptime.
- `GET /model-info`: Supported 15 classes and image normalization constants.
- `POST /predict`: Single image upload diagnosis (multipart form).
- `POST /predict/batch`: Simultaneous multi-image diagnosis (up to 32 images).

### Example cURL Request
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@PlantVillage/PlantVillage/Tomato_Early_blight/0012b5d2-0f10-45ec-903f-9242a82bda22___RS_Erly.B 7889.JPG"
```

### Example JSON Response
```json
{
  "status": "success",
  "top_prediction": "Tomato_Early_blight",
  "confidence": 0.9842,
  "crop": "Tomato",
  "disease": "Early Blight",
  "display_name": "Tomato - Early Blight",
  "is_healthy": false,
  "is_uncertain": false,
  "warning": null,
  "top_k": [
    {
      "class_name": "Tomato_Early_blight",
      "confidence": 0.9842,
      "crop": "Tomato",
      "disease": "Early Blight",
      "is_healthy": false,
      "display_name": "Tomato - Early Blight"
    },
    {
      "class_name": "Tomato__Target_Spot",
      "confidence": 0.0112,
      "crop": "Tomato",
      "disease": "Target Spot",
      "is_healthy": false,
      "display_name": "Tomato - Target Spot"
    },
    {
      "class_name": "Tomato_Late_blight",
      "confidence": 0.0031,
      "crop": "Tomato",
      "disease": "Late Blight",
      "is_healthy": false,
      "display_name": "Tomato - Late Blight"
    }
  ],
  "inference_time_ms": 22.4,
  "model_version": "resnet50"
}
```

---

## 10. Streamlit Web Application

Launch the interactive UI:
```bash
streamlit run app/streamlit_app.py --server.port 8501
```
Visit **`http://localhost:8501`** to:
1. Upload any leaf photograph or pick directly from sample dataset categories.
2. View real-time disease classification, confidence gauges, and top 3 differential predictions.
3. Observe out-of-distribution / low-confidence warnings.
4. Track diagnosis history across the active session.

---

## 11. ONNX Model Export & Benchmarking

Export the PyTorch model to ONNX format with dynamic batch dimensions:

```bash
python3 src/export.py --checkpoint plant_disease_resnet50.pth --output artifacts/model.onnx
```

### Measured Benchmark (Intel / Apple Silicon CPU):
- **Numerical Parity:** $\text{Max Difference} = 9.54 \times 10^{-6}$ (Tolerances $\le 10^{-4}$ ✅)
- **PyTorch CPU Latency:** $20.47\text{ ms}$ ($48.9\text{ fps}$)
- **ONNX Runtime Latency:** $12.79\text{ ms}$ ($78.2\text{ fps}$)
- **Inference Speedup:** **1.60x faster** on CPU
- **File Footprint:** `90.1 MB` (`.pth`) $\rightarrow$ `0.23 MB` weights graph (`.onnx`)

---

## 12. Monitoring, Drift Detection & Review Queue

The inference layer logs structured events to [`artifacts/monitoring/predictions.log`](artifacts/monitoring/predictions.log) (without storing raw user image pixels):
- **Metrics Tracked:** Request volume, average latency, p95 latency, error rate, confidence distribution.
- **Data Drift Detection:** Computes Population Stability Index (PSI) against training class distributions.
- **Human Review Queue:** Predictions with low confidence ($< 60\%$) or errors are routed to `artifacts/monitoring/review_queue.jsonl` for agronomist auditing.

Generate an on-demand monitoring summary:
```bash
python3 -c "from src.monitoring import generate_monitoring_report; print(generate_monitoring_report())"
```

---

## 13. Docker & Container Deployment

### Build and Run with Docker
```bash
# Build standalone Docker image
docker build -t floraguard:latest .

# Run FastAPI container
docker run -p 8000:8000 floraguard:latest
```

### Multi-Service Stack (Docker Compose)
To start the complete environment (FastAPI API on 8000, Streamlit UI on 8501, and MLflow UI on 5000):
```bash
docker compose up --build
```

---

## 14. Testing

Run all 23 unit and integration tests:
```bash
pytest -v
```

Tests cover:
- Architecture construction and parameter freezing modes
- Checkpoint loading, class count (15), and metadata verification
- Preprocessing normalization and resize dimensions
- Stratified dataset loading and zero data leakage
- Single image inference, byte stream input, and batch inference
- Corrupted image error handling
- Low-confidence uncertainty triggers
- PyTorch vs ONNX numerical parity
- FastAPI `/health`, `/model-info`, `/predict`, and `/predict/batch` responses
- Quality gate pass/fail criteria

---

## 15. Known Limitations & Agricultural Disclaimer

1. **Closed-Set Assumption:** The classifier recognizes only its 15 trained classes across Bell Pepper, Potato, and Tomato. It will produce low-confidence or arbitrary predictions if presented with other plants, non-plant imagery, or unseen diseases.
2. **Laboratory vs. Field Conditions:** The PlantVillage dataset was primarily captured under controlled background lighting. In-field photos with complex soil backgrounds or severe shadows may require additional data augmentation or field-collected training samples.
3. **Advisory Tool Only:** This software does not constitute a certified agricultural diagnostic certificate.
