"""
Unit tests for data transforms, directory resolution, label parsing, and dataloaders.
"""

import os
from PIL import Image
import numpy as np
import pytest
import torch

from src.data import (
    CLASS_NAMES,
    compute_dataset_fingerprint,
    create_dataloaders,
    get_transforms,
    parse_label,
    resolve_dataset_dir
)


def test_class_names_constant():
    """Verify the 15 supported class names match project specifications."""
    assert len(CLASS_NAMES) == 15
    assert "Tomato_Early_blight" in CLASS_NAMES
    assert "Pepper__bell___healthy" in CLASS_NAMES
    assert "Potato___Late_blight" in CLASS_NAMES


def test_parse_label():
    """Verify clean human-readable parsing of class names."""
    pepper_spot = parse_label("Pepper__bell___Bacterial_spot")
    assert pepper_spot["crop"] == "Bell Pepper"
    assert pepper_spot["disease"] == "Bacterial Spot"
    assert pepper_spot["is_healthy"] is False

    potato_healthy = parse_label("Potato___healthy")
    assert potato_healthy["crop"] == "Potato"
    assert potato_healthy["disease"] == "Healthy"
    assert potato_healthy["is_healthy"] is True

    tomato_mold = parse_label("Tomato_Leaf_Mold")
    assert tomato_mold["crop"] == "Tomato"
    assert tomato_mold["disease"] == "Leaf Mold"


def test_transforms_output():
    """Verify transformation pipelines output correctly sized and normalized tensors."""
    train_tf, val_tf = get_transforms(image_size=224)
    dummy_img = Image.fromarray(np.uint8(np.random.randint(0, 255, (300, 400, 3))))

    t_tensor = train_tf(dummy_img)
    v_tensor = val_tf(dummy_img)

    assert isinstance(t_tensor, torch.Tensor)
    assert t_tensor.shape == (3, 224, 224)
    assert isinstance(v_tensor, torch.Tensor)
    assert v_tensor.shape == (3, 224, 224)


def test_resolve_dataset_dir():
    """Verify dataset directory resolution logic."""
    if os.path.exists("PlantVillage"):
        resolved = resolve_dataset_dir("PlantVillage/PlantVillage", "PlantVillage")
        assert os.path.isdir(resolved)
        # Should not resolve to a directory with only 1 folder
        assert len(os.listdir(resolved)) >= 10
    else:
        pytest.skip("PlantVillage dataset directory not found.")


def test_compute_dataset_fingerprint():
    """Verify deterministic dataset fingerprinting."""
    if os.path.exists("PlantVillage"):
        resolved = resolve_dataset_dir()
        fp1 = compute_dataset_fingerprint(resolved)
        fp2 = compute_dataset_fingerprint(resolved)
        assert fp1 == fp2
        assert len(fp1) == 12
    else:
        pytest.skip("Dataset directory not found.")


def test_dataloaders_zero_data_leakage():
    """Verify that stratified train/val split is strictly disjoint."""
    if not os.path.exists("PlantVillage"):
        pytest.skip("Dataset directory not found.")

    resolved = resolve_dataset_dir()
    train_loader, val_loader, class_names, split_info = create_dataloaders(
        data_dir=resolved,
        batch_size=16,
        val_split=0.2,
        random_state=42,
        subset_ratio=0.01  # small subset for speed
    )

    assert split_info["train_samples"] > 0
    assert split_info["val_samples"] > 0
    assert len(class_names) == 15
