.PHONY: help setup test train train-smoke evaluate predict export api ui mlflow docker-build docker-up clean

PYTHON := python3
PIP := pip

help:
	@echo "Plant Disease Classification MLOps - Available Commands"
	@echo "========================================================="
	@echo "  make setup         Install project dependencies"
	@echo "  make test          Run full pytest test suite"
	@echo "  make train         Run full training pipeline with MLflow"
	@echo "  make train-smoke   Run 1-epoch lightweight training smoke test"
	@echo "  make evaluate      Run model evaluation, confusion matrix & quality gates"
	@echo "  make export        Export PyTorch model to ONNX & benchmark"
	@echo "  make api           Start FastAPI inference server (port 8000)"
	@echo "  make ui            Start Streamlit interactive web app (port 8501)"
	@echo "  make mlflow        Start local MLflow UI server (port 5000)"
	@echo "  make docker-build  Build Docker production container"
	@echo "  make docker-up     Start API, UI, and MLflow with Docker Compose"
	@echo "  make clean         Remove cache files and temporary artifacts"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install onnxscript flake8 pytest pytest-cov

test:
	pytest -v

train:
	$(PYTHON) src/train.py --config configs/train.yaml

train-smoke:
	$(PYTHON) src/train.py --smoke-test

evaluate:
	$(PYTHON) src/evaluate.py --checkpoint plant_disease_resnet50.pth

export:
	$(PYTHON) src/export.py --checkpoint plant_disease_resnet50.pth --output artifacts/model.onnx

api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run app/streamlit_app.py --server.port 8501

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:
	docker build -t plant-disease-mlops:latest .

docker-up:
	docker compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
