"""
FastAPI Production Inference API for Plant Disease Classification.
Provides health checks, model metadata, single-image, and batch predictions with monitoring.
"""

from contextlib import asynccontextmanager
import io
import os
import sys
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monitoring import PredictionLogger
from src.predict import PlantDiseasePredictor
from src.registry import get_current_production_model

# Global instances
predictor: Optional[PlantDiseasePredictor] = None
logger: Optional[PredictionLogger] = None
app_start_time: float = 0.0

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/bmp", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads model and monitoring logger on application startup.
    """
    global predictor, logger, app_start_time
    app_start_time = time.time()
    
    prod_info = get_current_production_model()
    checkpoint_path = prod_info.get("checkpoint_path", "plant_disease_resnet50.pth")
    if not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", checkpoint_path))

    print(f"[API] Initializing predictor with checkpoint: {checkpoint_path}")
    predictor = PlantDiseasePredictor(checkpoint_path=checkpoint_path)
    logger = PredictionLogger()
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="Plant Disease Classification API",
    description="Production-grade REST API for identifying plant leaf diseases across Pepper, Potato, and Tomato crops.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    uptime_seconds: float = Field(..., examples=[124.5])
    model_version: str = Field(..., examples=["resnet50"])
    device: str = Field(..., examples=["mps"])
    num_classes: int = Field(..., examples=[15])


class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., examples=["resnet50"])
    version: str = Field(..., examples=["1.0.0-baseline"])
    num_classes: int = Field(..., examples=[15])
    class_names: List[str]
    best_val_accuracy: Optional[float] = Field(None, examples=[0.9501])
    image_size: int = Field(224, examples=[224])
    normalization: Dict[str, List[float]]


class PredictionClassItem(BaseModel):
    class_name: str = Field(..., examples=["Tomato_Early_blight"])
    confidence: float = Field(..., examples=[0.9412])
    crop: str = Field(..., examples=["Tomato"])
    disease: str = Field(..., examples=["Early Blight"])
    is_healthy: bool = Field(..., examples=[False])
    display_name: str = Field(..., examples=["Tomato - Early Blight"])


class PredictionResponse(BaseModel):
    status: str = Field(..., examples=["success"])
    top_prediction: str = Field(..., examples=["Tomato_Early_blight"])
    confidence: float = Field(..., examples=[0.9412])
    crop: str = Field(..., examples=["Tomato"])
    disease: str = Field(..., examples=["Early Blight"])
    display_name: str = Field(..., examples=["Tomato - Early Blight"])
    is_healthy: bool = Field(..., examples=[False])
    is_uncertain: bool = Field(..., examples=[False])
    warning: Optional[str] = Field(None, examples=[None])
    top_k: List[PredictionClassItem]
    inference_time_ms: float = Field(..., examples=[24.5])
    model_version: str = Field(..., examples=["resnet50"])


class BatchItemResponse(BaseModel):
    filename: str
    success: bool
    prediction: Optional[PredictionResponse] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total_images: int = Field(..., examples=[2])
    successful: int = Field(..., examples=[2])
    failed: int = Field(..., examples=[0])
    results: List[BatchItemResponse]


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns system status, device, model version, and uptime."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not yet initialized.")
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - app_start_time, 2),
        model_version=predictor.model_version,
        device=str(predictor.target_device),
        num_classes=len(predictor.class_names)
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Returns active model details, class list, and normalization parameters."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    prod_info = get_current_production_model()
    return ModelInfoResponse(
        model_name=predictor.metadata.get("model_name", "resnet50"),
        version=prod_info.get("version", "1.0.0-baseline"),
        num_classes=len(predictor.class_names),
        class_names=predictor.class_names,
        best_val_accuracy=prod_info.get("val_accuracy", predictor.metadata.get("best_val_accuracy")),
        image_size=predictor.image_size,
        normalization={"mean": predictor.mean, "std": predictor.std}
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single_image(file: UploadFile = File(...)):
    """
    Submits a single leaf image for plant disease diagnosis.
    Returns crop, disease classification, confidence, top 3 predictions, and uncertainty flag.
    """
    if predictor is None or logger is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready.")

    # Validate Content-Type
    if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported media type: '{file.content_type}'. Must be JPEG, PNG, BMP, or WEBP."
        )

    # Read content with size guard
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file size ({len(contents) / (1024*1024):.1f}MB) exceeds 15MB limit."
        )

    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    # Verify image dimensions
    try:
        pil_img = Image.open(io.BytesIO(contents))
        img_width, img_height = pil_img.size
    except Exception:
        if logger:
            logger.log_prediction(
                model_version=predictor.model_version,
                predicted_class="None",
                confidence=0.0,
                latency_ms=0.0,
                image_width=0,
                image_height=0,
                status="invalid_image_error",
                error_message="Image decoding failed"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is corrupted or not a recognized image format."
        )

    try:
        result = predictor.predict(contents)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference execution error.")

    # Structured Prediction Monitoring Log
    logger.log_prediction(
        model_version=result["model_version"],
        predicted_class=result["top_prediction"],
        confidence=result["confidence"],
        latency_ms=result["inference_time_ms"],
        image_width=img_width,
        image_height=img_height,
        status="success",
        is_uncertain=result["is_uncertain"]
    )

    return PredictionResponse(
        status="success",
        top_prediction=result["top_prediction"],
        confidence=result["confidence"],
        crop=result["crop"],
        disease=result["disease"],
        display_name=result["display_name"],
        is_healthy=result["is_healthy"],
        is_uncertain=result["is_uncertain"],
        warning=result["warning"],
        top_k=[PredictionClassItem(**item) for item in result["top_k"]],
        inference_time_ms=result["inference_time_ms"],
        model_version=result["model_version"]
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """
    Submits multiple leaf images for simultaneous batch inference.
    """
    if predictor is None or logger is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready.")

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided in request.")

    if len(files) > 32:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Batch size exceeds maximum limit of 32 images.")

    batch_payloads = []
    file_metadata = []

    for file in files:
        contents = await file.read()
        filename = file.filename or "unknown"
        if len(contents) > MAX_FILE_SIZE:
            file_metadata.append({"filename": filename, "valid": False, "error": "File size exceeds 15MB"})
            batch_payloads.append(None)
        else:
            try:
                img = Image.open(io.BytesIO(contents))
                file_metadata.append({"filename": filename, "valid": True, "size": img.size, "bytes": contents})
                batch_payloads.append(contents)
            except Exception:
                file_metadata.append({"filename": filename, "valid": False, "error": "Invalid image format"})
                batch_payloads.append(None)

    valid_payloads = [p for p in batch_payloads if p is not None]
    raw_results = predictor.predict_batch(valid_payloads) if valid_payloads else []

    item_responses = []
    success_count = 0
    fail_count = 0
    valid_res_idx = 0

    for meta in file_metadata:
        if not meta["valid"]:
            fail_count += 1
            item_responses.append(BatchItemResponse(
                filename=meta["filename"],
                success=False,
                error=meta["error"]
            ))
        else:
            res = raw_results[valid_res_idx]
            valid_res_idx += 1
            if res.get("success", True):
                success_count += 1
                pred_obj = PredictionResponse(
                    status="success",
                    top_prediction=res["top_prediction"],
                    confidence=res["confidence"],
                    crop=res["crop"],
                    disease=res["disease"],
                    display_name=res["display_name"],
                    is_healthy=res["is_healthy"],
                    is_uncertain=res["is_uncertain"],
                    warning=res["warning"],
                    top_k=[PredictionClassItem(**item) for item in res["top_k"]],
                    inference_time_ms=res["inference_time_ms"],
                    model_version=res["model_version"]
                )
                item_responses.append(BatchItemResponse(
                    filename=meta["filename"],
                    success=True,
                    prediction=pred_obj
                ))
                logger.log_prediction(
                    model_version=res["model_version"],
                    predicted_class=res["top_prediction"],
                    confidence=res["confidence"],
                    latency_ms=res["inference_time_ms"],
                    image_width=meta["size"][0],
                    image_height=meta["size"][1],
                    status="success",
                    is_uncertain=res["is_uncertain"]
                )
            else:
                fail_count += 1
                item_responses.append(BatchItemResponse(
                    filename=meta["filename"],
                    success=False,
                    error=res.get("error", "Inference error")
                ))

    return BatchPredictionResponse(
        total_images=len(files),
        successful=success_count,
        failed=fail_count,
        results=item_responses
    )
