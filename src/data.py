"""
Dataset Resolution, Preprocessing Transforms, and Stratified DataLoaders.
"""

import hashlib
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# Exact 15 classes matching the baseline checkpoint and PlantVillage folder structure
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def parse_label(class_name: str) -> Dict[str, str]:
    """
    Parses a raw class folder name into clean crop and condition labels.
    Example:
        'Pepper__bell___Bacterial_spot' -> {'crop': 'Bell Pepper', 'disease': 'Bacterial spot', 'is_healthy': False}
        'Tomato_healthy' -> {'crop': 'Tomato', 'disease': 'Healthy', 'is_healthy': True}
    """
    name = class_name.strip()
    is_healthy = "healthy" in name.lower()

    if name.startswith("Pepper__bell"):
        crop = "Bell Pepper"
        disease = "Healthy" if is_healthy else name.replace("Pepper__bell___", "").replace("_", " ").title()
    elif name.startswith("Potato"):
        crop = "Potato"
        disease = "Healthy" if is_healthy else name.replace("Potato___", "").replace("_", " ").title()
    elif name.startswith("Tomato"):
        crop = "Tomato"
        clean = name.replace("Tomato__Tomato_", "").replace("Tomato__", "").replace("Tomato_", "")
        disease = "Healthy" if is_healthy else clean.replace("_", " ").title()
    else:
        parts = name.split("_")
        crop = parts[0].title()
        disease = " ".join(parts[1:]).title() if len(parts) > 1 else name

    return {
        "crop": crop,
        "disease": disease,
        "is_healthy": is_healthy,
        "display_name": f"{crop} - {disease}"
    }


def resolve_dataset_dir(primary_path: str = "PlantVillage/PlantVillage", fallback_path: Optional[str] = "PlantVillage") -> str:
    """
    Resolves the actual dataset directory containing the 15 class subdirectories.
    Handles the nested PlantVillage directory anomaly automatically.
    """
    candidates = [primary_path]
    if fallback_path:
        candidates.append(fallback_path)
    
    # Also test relative to current working directory
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            # Check if this directory directly contains the class folders
            subdirs = [d for d in os.listdir(candidate) if os.path.isdir(os.path.join(candidate, d))]
            # If nested PlantVillage exists inside candidate, check inner first
            nested = os.path.join(candidate, "PlantVillage")
            if os.path.exists(nested) and os.path.isdir(nested):
                inner_subdirs = [d for d in os.listdir(nested) if os.path.isdir(os.path.join(nested, d))]
                matched = [c for c in CLASS_NAMES if c in inner_subdirs]
                if len(matched) >= 10:
                    return os.path.abspath(nested)

            matched = [c for c in CLASS_NAMES if c in subdirs]
            if len(matched) >= 10:
                return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"Could not locate PlantVillage dataset containing expected class folders in candidates: {candidates}"
    )


def get_transforms(
    image_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns data augmentation pipeline for training and deterministic pipeline for validation/inference.
    Matches the exact baseline settings from model.ipynb.
    """
    norm_mean = mean or DEFAULT_MEAN
    norm_std = std or DEFAULT_STD

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    return train_transform, val_transform


def compute_dataset_fingerprint(data_dir: str) -> str:
    """
    Computes a deterministic MD5 fingerprint for dataset versioning in MLflow.
    Based on class directory names and file counts.
    """
    if not os.path.exists(data_dir):
        return "dataset_not_found"

    summary = []
    for cls in sorted(os.listdir(data_dir)):
        cls_dir = os.path.join(data_dir, cls)
        if os.path.isdir(cls_dir):
            file_count = len([f for f in os.listdir(cls_dir) if not f.startswith(".")])
            summary.append(f"{cls}:{file_count}")

    raw_string = ";".join(summary)
    return hashlib.md5(raw_string.encode("utf-8")).hexdigest()[:12]


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    random_state: int = 42,
    num_workers: int = 0,
    image_size: int = 224,
    subset_ratio: Optional[float] = None,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, List[str], Dict]:
    """
    Creates stratified training and validation DataLoaders.
    Strictly prevents data leakage by guaranteeing disjoint indices.
    """
    train_transform, val_transform = get_transforms(image_size=image_size)

    # Load separate ImageFolder instances with their respective transforms
    train_full = datasets.ImageFolder(data_dir, transform=train_transform)
    val_full = datasets.ImageFolder(data_dir, transform=val_transform)

    class_names = train_full.classes
    total_samples = len(train_full)

    indices = np.arange(total_samples)
    targets = np.array(train_full.targets)

    # Stratified split to preserve class distribution across train/val
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        random_state=random_state,
        stratify=targets
    )

    # Optional subset for rapid smoke tests / development
    if subset_ratio is not None and 0.0 < subset_ratio < 1.0:
        _, train_indices = train_test_split(
            train_indices,
            test_size=subset_ratio,
            random_state=random_state,
            stratify=targets[train_indices]
        )
        _, val_indices = train_test_split(
            val_indices,
            test_size=subset_ratio,
            random_state=random_state,
            stratify=targets[val_indices]
        )

    # Verify zero data leakage
    overlap = set(train_indices).intersection(set(val_indices))
    assert len(overlap) == 0, f"Critical: Data leakage detected! {len(overlap)} overlapping samples."

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    split_info = {
        "data_dir": data_dir,
        "total_samples": total_samples,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "val_split": val_split,
        "random_state": random_state,
        "num_classes": len(class_names),
        "dataset_fingerprint": compute_dataset_fingerprint(data_dir)
    }

    return train_loader, val_loader, class_names, split_info
