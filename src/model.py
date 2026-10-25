"""
Model Architecture, Checkpoint Management, and Parameter Freezing.
"""

import os
import ssl
from typing import Dict, List, Optional, Tuple, Union
import certifi
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Selects the best available device according to preference.
    Supports CUDA, Apple Silicon MPS, and CPU.
    """
    pref = (device_preference or "auto").lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif pref == "cpu":
        return torch.device("cpu")
    elif pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def build_model(
    num_classes: int = 15,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    fine_tune_layers: Optional[List[str]] = None,
    init_from_checkpoint: Optional[str] = None
) -> nn.Module:
    """
    Constructs ResNet-50 matching the baseline project architecture.
    Can initialize weights from an existing local checkpoint to avoid re-downloading.
    """
    if init_from_checkpoint and os.path.exists(init_from_checkpoint):
        # Initialize locally from existing checkpoint without network downloads
        model = models.resnet50(weights=None)
        ckpt = torch.load(init_from_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Handle linear head mismatch if num_classes differs
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Load backbone only if head size changed
            backbone_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
            model.load_state_dict(backbone_dict, strict=False)
    elif pretrained:
        try:
            weights = ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        except Exception:
            # Fallback using certifi ssl context if default macOS certificates fail
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
                weights = ResNet50_Weights.DEFAULT
                model = models.resnet50(weights=weights)
            except Exception:
                # If network is completely offline, fall back to uninitialized
                model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    # Freeze or unfreeze layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    # Selective unfreezing for fine-tuning
    if fine_tune_layers:
        for layer_name in fine_tune_layers:
            if hasattr(model, layer_name):
                for param in getattr(model, layer_name).parameters():
                    param.requires_grad = True

    # Ensure final linear head is always trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def load_checkpoint(
    checkpoint_path: str,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[nn.Module, Dict]:
    """
    Safely loads trained model weights and metadata from checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    target_device = get_device(str(device)) if isinstance(device, str) else (device or torch.device("cpu"))
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Invalid checkpoint format in {checkpoint_path}. Missing 'model_state_dict'.")

    num_classes = ckpt.get("num_classes", 15)
    model = build_model(num_classes=num_classes, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(target_device)
    model.eval()

    metadata = {
        "model_name": ckpt.get("model_name", "resnet50"),
        "num_classes": num_classes,
        "class_names": ckpt.get("class_names", []),
        "best_val_accuracy": ckpt.get("best_val_accuracy", None),
        "image_size": ckpt.get("image_size", 224),
        "mean": ckpt.get("mean", [0.485, 0.456, 0.406]),
        "std": ckpt.get("std", [0.229, 0.224, 0.225]),
    }

    return model, metadata


def save_checkpoint(
    model: nn.Module,
    save_path: str,
    metadata: Dict
) -> str:
    """
    Saves model state dictionary along with all essential metadata.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    state_dict = model.state_dict()
    # Move tensors to CPU before saving for cross-platform portability
    cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}

    checkpoint = {
        "model_name": metadata.get("model_name", "resnet50"),
        "num_classes": metadata.get("num_classes", 15),
        "class_names": metadata.get("class_names", []),
        "model_state_dict": cpu_state_dict,
        "best_val_accuracy": metadata.get("best_val_accuracy", 0.0),
        "image_size": metadata.get("image_size", 224),
        "mean": metadata.get("mean", [0.485, 0.456, 0.406]),
        "std": metadata.get("std", [0.229, 0.224, 0.225]),
        "training_mode": metadata.get("training_mode", "frozen"),
        "epochs": metadata.get("epochs", 0),
        "timestamp": metadata.get("timestamp", ""),
    }

    torch.save(checkpoint, save_path)
    return save_path
