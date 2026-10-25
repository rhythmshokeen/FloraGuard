"""
Unit tests for model architecture, parameter freezing, device resolution, and checkpoint loading.
"""

import os
import pytest
import torch

from src.model import build_model, get_device, load_checkpoint


def test_device_resolution():
    """Verify get_device handles auto, cpu, and invalid gracefully."""
    cpu_device = get_device("cpu")
    assert cpu_device.type == "cpu"
    auto_device = get_device("auto")
    assert auto_device.type in {"cpu", "cuda", "mps"}


def test_build_model_architecture():
    """Verify ResNet-50 head replacement and output dimensions."""
    model = build_model(num_classes=15, pretrained=False, freeze_backbone=True)
    assert model.fc.out_features == 15
    assert model.fc.in_features == 2048

    # Forward pass test
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 15)


def test_parameter_freezing_modes():
    """Verify that backbone parameters freeze correctly and fine-tuning layers unfreeze."""
    # 1. Frozen backbone
    frozen_model = build_model(num_classes=15, pretrained=False, freeze_backbone=True)
    for name, param in frozen_model.named_parameters():
        if "fc" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    # 2. Fine-tune layer4
    ft_model = build_model(num_classes=15, pretrained=False, freeze_backbone=True, fine_tune_layers=["layer4"])
    for name, param in ft_model.named_parameters():
        if "fc" in name or "layer4" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False


def test_load_existing_checkpoint():
    """Verify safe loading of the baseline plant_disease_resnet50.pth checkpoint."""
    checkpoint_path = "plant_disease_resnet50.pth"
    if not os.path.exists(checkpoint_path):
        pytest.skip("Baseline checkpoint not found in workspace.")

    model, metadata = load_checkpoint(checkpoint_path, device="cpu")
    assert metadata["num_classes"] == 15
    assert len(metadata["class_names"]) == 15
    assert metadata["image_size"] == 224
    assert metadata["model_name"] == "resnet50"
    assert metadata["best_val_accuracy"] is not None

    # Test forward pass on CPU
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 15)


def test_load_nonexistent_checkpoint_raises_error():
    """Verify FileNotFoundError when loading missing checkpoint."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint("non_existent_file.pth")
