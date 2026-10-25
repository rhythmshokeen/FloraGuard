"""
Inference Pipeline: Single-Image, Batch Prediction, Preprocessing, and Uncertainty Estimation.
"""

import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.data import CLASS_NAMES, DEFAULT_MEAN, DEFAULT_STD, parse_label
from src.model import load_checkpoint, get_device


class PlantDiseasePredictor:
    """
    Production inference engine for plant disease classification.
    Supports single-image and batch inference with uncertainty flags.
    """

    MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit
    SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "BMP", "WEBP"}

    def __init__(
        self,
        checkpoint_path: str = "plant_disease_resnet50.pth",
        device: Optional[Union[str, torch.device]] = None,
        uncertainty_threshold: float = 0.60
    ):
        self.checkpoint_path = checkpoint_path
        self.target_device = get_device(str(device)) if isinstance(device, str) else (device or get_device("auto"))
        self.uncertainty_threshold = uncertainty_threshold

        # Load model and checkpoint metadata
        self.model, self.metadata = load_checkpoint(checkpoint_path, device=self.target_device)
        self.model.eval()

        self.class_names = self.metadata.get("class_names") or CLASS_NAMES
        self.image_size = self.metadata.get("image_size", 224)
        self.mean = self.metadata.get("mean", DEFAULT_MEAN)
        self.std = self.metadata.get("std", DEFAULT_STD)
        self.model_version = self.metadata.get("model_name", "resnet50")

        # Deterministic inference transform matching validation
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def preprocess_image(self, image_input: Union[str, bytes, io.BytesIO, Image.Image]) -> torch.Tensor:
        """
        Validates, cleans, and transforms raw image input into normalized PyTorch tensor [1, 3, 224, 224].
        """
        image: Image.Image

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file does not exist: {image_input}")
            file_size = os.path.getsize(image_input)
            if file_size > self.MAX_IMAGE_SIZE_BYTES:
                raise ValueError(f"Image file exceeds maximum allowed size ({file_size / (1024*1024):.1f}MB > 15MB)")
            try:
                image = Image.open(image_input)
                image.load()
            except UnidentifiedImageError:
                raise ValueError("The provided file could not be decoded as a valid image.")
            except Exception as e:
                raise ValueError(f"Failed to read image file: {str(e)}")

        elif isinstance(image_input, bytes):
            if len(image_input) > self.MAX_IMAGE_SIZE_BYTES:
                raise ValueError(f"Uploaded byte payload exceeds 15MB limit ({len(image_input)} bytes)")
            try:
                image = Image.open(io.BytesIO(image_input))
                image.load()
            except Exception:
                raise ValueError("Byte data could not be parsed as a valid image.")

        elif isinstance(image_input, io.BytesIO):
            try:
                image = Image.open(image_input)
                image.load()
            except Exception:
                raise ValueError("Byte stream could not be parsed as a valid image.")

        elif isinstance(image_input, Image.Image):
            image = image_input

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Validate format
        fmt = (image.format or "JPEG").upper()
        if fmt not in self.SUPPORTED_FORMATS and image.format is not None:
            raise ValueError(f"Unsupported image format: {fmt}. Supported formats: {self.SUPPORTED_FORMATS}")

        # Validate dimensions
        if image.width < 10 or image.height < 10:
            raise ValueError(f"Image dimensions ({image.width}x{image.height}) are too small for analysis.")

        # Convert to RGB (handles RGBA, Grayscale, CMYK, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transformation and add batch dimension
        tensor = self.transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
        return tensor

    @torch.inference_mode()
    def predict(
        self,
        image_input: Union[str, bytes, io.BytesIO, Image.Image],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Runs single image prediction with top_k results and uncertainty estimation.
        """
        start_time = time.perf_counter()

        tensor = self.preprocess_image(image_input).to(self.target_device)

        logits = self.model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)  # Shape: [num_classes]

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.class_names)))
        top_probs_list = top_probs.cpu().tolist()
        top_indices_list = top_indices.cpu().tolist()

        predictions = []
        for prob, idx in zip(top_probs_list, top_indices_list):
            cls_name = self.class_names[idx]
            parsed = parse_label(cls_name)
            predictions.append({
                "class_name": cls_name,
                "confidence": round(float(prob), 4),
                "crop": parsed["crop"],
                "disease": parsed["disease"],
                "is_healthy": parsed["is_healthy"],
                "display_name": parsed["display_name"]
            })

        top_pred = predictions[0]
        top_confidence = top_pred["confidence"]
        is_uncertain = top_confidence < self.uncertainty_threshold

        warning = None
        if is_uncertain:
            warning = (
                f"Low confidence warning ({top_confidence:.1%} < {self.uncertainty_threshold:.0%}). "
                "The image may be out-of-distribution, blurry, or contain an unrecognized condition."
            )

        return {
            "top_prediction": top_pred["class_name"],
            "confidence": top_confidence,
            "crop": top_pred["crop"],
            "disease": top_pred["disease"],
            "display_name": top_pred["display_name"],
            "is_healthy": top_pred["is_healthy"],
            "is_uncertain": is_uncertain,
            "warning": warning,
            "top_k": predictions,
            "inference_time_ms": inference_time_ms,
            "model_version": self.model_version,
            "device": str(self.target_device)
        }

    @torch.inference_mode()
    def predict_batch(
        self,
        images: List[Union[str, bytes, io.BytesIO, Image.Image]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Runs batch prediction efficiently on multiple images.
        """
        if not images:
            return []

        start_time = time.perf_counter()
        tensors = []
        valid_indices = []
        errors = {}

        for idx, img in enumerate(images):
            try:
                t = self.preprocess_image(img)
                tensors.append(t)
                valid_indices.append(idx)
            except Exception as e:
                errors[idx] = str(e)

        results: List[Dict[str, Any]] = [None] * len(images)  # type: ignore

        for err_idx, err_msg in errors.items():
            results[err_idx] = {
                "success": False,
                "error": err_msg,
                "inference_time_ms": 0.0
            }

        if tensors:
            batch_tensor = torch.cat(tensors, dim=0).to(self.target_device)
            logits = self.model(batch_tensor)
            probs = F.softmax(logits, dim=1)

            total_batch_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            per_item_time_ms = round(total_batch_time_ms / len(valid_indices), 2)

            for i, original_idx in enumerate(valid_indices):
                item_probs = probs[i]
                top_probs, top_indices = torch.topk(item_probs, k=min(top_k, len(self.class_names)))
                top_probs_list = top_probs.cpu().tolist()
                top_indices_list = top_indices.cpu().tolist()

                top_list = []
                for p, cls_idx in zip(top_probs_list, top_indices_list):
                    cls_name = self.class_names[cls_idx]
                    parsed = parse_label(cls_name)
                    top_list.append({
                        "class_name": cls_name,
                        "confidence": round(float(p), 4),
                        "crop": parsed["crop"],
                        "disease": parsed["disease"],
                        "is_healthy": parsed["is_healthy"],
                        "display_name": parsed["display_name"]
                    })

                top_item = top_list[0]
                conf = top_item["confidence"]
                is_unc = conf < self.uncertainty_threshold
                warn = (
                    f"Low confidence ({conf:.1%} < {self.uncertainty_threshold:.0%})."
                    if is_unc else None
                )

                results[original_idx] = {
                    "success": True,
                    "top_prediction": top_item["class_name"],
                    "confidence": conf,
                    "crop": top_item["crop"],
                    "disease": top_item["disease"],
                    "display_name": top_item["display_name"],
                    "is_healthy": top_item["is_healthy"],
                    "is_uncertain": is_unc,
                    "warning": warn,
                    "top_k": top_list,
                    "inference_time_ms": per_item_time_ms,
                    "model_version": self.model_version
                }

        return results
