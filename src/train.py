"""
Reproducible Training Pipeline with MLflow Experiment Tracking, Parameter Freezing, and Checkpoint Management.
"""

import argparse
import copy
from datetime import datetime
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import mlflow
import mlflow.pytorch

from src.data import CLASS_NAMES, DEFAULT_MEAN, DEFAULT_STD, create_dataloaders, resolve_dataset_dir
from src.evaluate import check_quality_gates, evaluate_model
from src.model import build_model, get_device, save_checkpoint
from src.registry import register_candidate_model


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "configs/train.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_optimizer_and_params(
    model: nn.Module,
    training_mode: str,
    head_lr: float = 0.001,
    backbone_lr: float = 0.0001,
    weight_decay: float = 0.0001
) -> optim.Optimizer:
    """
    Sets up optimizer with differential learning rates for pretrained layers and linear head.
    """
    if training_mode == "frozen":
        params = [{"params": model.fc.parameters(), "lr": head_lr}]
    else:
        # Differential learning rate
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad]
        head_params = list(model.fc.parameters())
        params = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr}
        ]

    return optim.Adam(params, weight_decay=weight_decay)


def train_model(
    config_path: str = "configs/train.yaml",
    mode_override: Optional[str] = None,
    epochs_override: Optional[int] = None,
    subset_ratio: Optional[float] = None,
    run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main training execution function.
    """
    cfg = load_config(config_path)
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    mlflow_cfg = cfg.get("mlflow", {})
    quality_cfg = cfg.get("quality_gates", {})

    seed = data_cfg.get("random_seed", 42)
    set_seed(seed)

    # Resolve execution device
    device_pref = train_cfg.get("device", "auto")
    device = get_device(device_pref)
    print(f"Executing training on device: {device}")

    # Resolve dataset path
    data_dir = resolve_dataset_dir(
        primary_path=data_cfg.get("path", "PlantVillage/PlantVillage"),
        fallback_path=data_cfg.get("fallback_path", "PlantVillage")
    )
    print(f"Resolved dataset path: {data_dir}")

    # Hyperparameters
    training_mode = mode_override or train_cfg.get("mode", "frozen")
    epochs = epochs_override or train_cfg.get("epochs", 5)
    batch_size = train_cfg.get("batch_size", 16)
    head_lr = train_cfg.get("learning_rate", 0.001)
    backbone_lr = train_cfg.get("backbone_learning_rate", 0.0001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    early_stopping_patience = train_cfg.get("early_stopping_patience", 3)
    image_size = data_cfg.get("image_size", 224)

    # Prepare DataLoaders
    train_loader, val_loader, class_names, split_info = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=data_cfg.get("val_split", 0.2),
        random_state=seed,
        num_workers=data_cfg.get("num_workers", 0),
        image_size=image_size,
        subset_ratio=subset_ratio
    )
    print(f"Loaded {split_info['train_samples']} training samples and {split_info['val_samples']} validation samples across {len(class_names)} classes.")

    # Configure Architecture and Freezing
    fine_tune_layers = None
    freeze_backbone = True
    if training_mode == "frozen":
        freeze_backbone = True
        fine_tune_layers = None
    elif training_mode == "fine_tune_layer4":
        freeze_backbone = True
        fine_tune_layers = ["layer4"]
    elif training_mode == "fine_tune_layer3_layer4":
        freeze_backbone = True
        fine_tune_layers = ["layer3", "layer4"]
    elif training_mode == "fine_tune_all":
        freeze_backbone = False
        fine_tune_layers = None

    model = build_model(
        num_classes=len(class_names),
        pretrained=model_cfg.get("pretrained", True),
        freeze_backbone=freeze_backbone,
        fine_tune_layers=fine_tune_layers,
        init_from_checkpoint=model_cfg.get("baseline_checkpoint", "plant_disease_resnet50.pth")
    )
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized: {trainable_params:,} trainable / {total_params:,} total parameters ({trainable_params/total_params:.1%})")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer_and_params(
        model=model,
        training_mode=training_mode,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay
    )

    # Setup MLflow
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = mlflow_cfg.get("experiment_name", "plant-disease-classification")
    mlflow.set_experiment(experiment_name)

    actual_run_name = run_name or f"{training_mode}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=actual_run_name) as run:
        run_id = run.info.run_id
        print(f"\nStarted MLflow Run: {actual_run_name} (Run ID: {run_id})")

        # Log parameters
        mlflow.log_params({
            "architecture": model_cfg.get("architecture", "resnet50"),
            "training_mode": training_mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "head_learning_rate": head_lr,
            "backbone_learning_rate": backbone_lr if training_mode != "frozen" else 0.0,
            "weight_decay": weight_decay,
            "optimizer": "Adam",
            "device": str(device),
            "random_seed": seed,
            "train_samples": split_info["train_samples"],
            "val_samples": split_info["val_samples"],
            "dataset_fingerprint": split_info["dataset_fingerprint"],
            "trainable_parameters": trainable_params
        })

        best_val_accuracy = 0.0
        best_epoch = 0
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0

        train_start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            epoch_train_loss = running_loss / total_train
            epoch_train_acc = correct_train / total_train

            # Validation Loop
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)

            epoch_val_loss = val_loss / total_val
            epoch_val_acc = correct_val / total_val
            epoch_duration = time.time() - epoch_start

            print(
                f"Epoch [{epoch+1}/{epochs}] ({epoch_duration:.1f}s) | "
                f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2%} | "
                f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2%}"
            )

            # Log metrics per epoch
            mlflow.log_metrics({
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_accuracy": epoch_val_acc,
                "epoch_duration_seconds": epoch_duration
            }, step=epoch + 1)

            # Checkpoint tracking
            if epoch_val_acc > best_val_accuracy:
                best_val_accuracy = epoch_val_acc
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

        total_training_duration = time.time() - train_start_time
        mlflow.log_metric("total_training_duration_seconds", total_training_duration)
        mlflow.log_metric("best_val_accuracy", best_val_accuracy)
        mlflow.log_metric("best_epoch", best_epoch)

        # Restore best model state
        model.load_state_dict(best_model_state)

        # Comprehensive Evaluation on Validation Set
        eval_output_dir = os.path.join("reports", actual_run_name)
        eval_metrics, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            class_names=class_names,
            output_dir=eval_output_dir
        )

        # Log Evaluation Metrics to MLflow
        mlflow.log_metrics({
            "eval_accuracy": eval_metrics["accuracy"],
            "eval_macro_f1": eval_metrics["macro_f1"],
            "eval_weighted_f1": eval_metrics["weighted_f1"],
            "eval_macro_precision": eval_metrics["macro_precision"],
            "eval_macro_recall": eval_metrics["macro_recall"],
            "eval_min_class_recall": eval_metrics["min_class_recall"],
            "eval_avg_latency_ms": eval_metrics["avg_latency_ms"],
            "eval_p95_latency_ms": eval_metrics["p95_latency_ms"],
            "eval_total_misclassified": eval_metrics["total_misclassified"]
        })

        # Log Evaluation Artifacts
        mlflow.log_artifact(eval_metrics["confusion_matrix_path"], artifact_path="evaluation")
        mlflow.log_artifact(eval_metrics["classification_report_path"], artifact_path="evaluation")
        mlflow.log_artifact(eval_metrics["misclassified_samples_path"], artifact_path="evaluation")

        # Save Candidate Checkpoint
        os.makedirs("artifacts/checkpoints", exist_ok=True)
        checkpoint_filename = f"model_{actual_run_name}.pth"
        checkpoint_path = os.path.join("artifacts/checkpoints", checkpoint_filename)

        metadata = {
            "model_name": "resnet50",
            "num_classes": len(class_names),
            "class_names": class_names,
            "best_val_accuracy": best_val_accuracy,
            "image_size": image_size,
            "mean": data_cfg.get("mean", DEFAULT_MEAN),
            "std": data_cfg.get("std", DEFAULT_STD),
            "training_mode": training_mode,
            "epochs": epochs,
            "timestamp": datetime.utcnow().isoformat()
        }

        save_checkpoint(model, checkpoint_path, metadata)
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
        try:
            mlflow.pytorch.log_model(model, artifact_path="model")
        except Exception as e:
            print(f"[Notice] mlflow.pytorch.log_model notice: {e}")

        # Quality Gate Check
        passed_gates, gate_report = check_quality_gates(eval_metrics, quality_cfg)
        mlflow.log_dict(gate_report, "quality_gate_report.json")

        print(f"\nFinal Best Val Accuracy: {best_val_accuracy:.2%}")
        print(f"Quality Gates Check: {'PASSED' if passed_gates else 'FAILED'}")

        # Register candidate in manifest
        candidate_version = f"2.0.0-{actual_run_name}"
        register_candidate_model(
            checkpoint_path=checkpoint_path,
            metrics=eval_metrics,
            version=candidate_version,
            run_id=run_id,
            mlflow_tracking_uri=tracking_uri
        )

        return {
            "run_id": run_id,
            "checkpoint_path": checkpoint_path,
            "best_val_accuracy": best_val_accuracy,
            "eval_metrics": eval_metrics,
            "passed_quality_gates": passed_gates,
            "gate_report": gate_report
        }


def main():
    parser = argparse.ArgumentParser(description="Train Plant Disease Classification Model with MLflow")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to YAML configuration")
    parser.add_argument("--mode", type=str, default=None, choices=["frozen", "fine_tune_layer4", "fine_tune_layer3_layer4", "fine_tune_all"], help="Training mode")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--subset-ratio", type=float, default=None, help="Fraction of dataset to use for quick test (0.01 to 1.0)")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid 1-epoch smoke test with small subset")
    parser.add_argument("--run-name", type=str, default=None, help="Custom MLflow run name")
    args = parser.parse_args()

    epochs = 1 if args.smoke_test else args.epochs
    subset = 0.02 if args.smoke_test else args.subset_ratio
    run_name = args.run_name or ("smoke_test" if args.smoke_test else None)

    train_model(
        config_path=args.config,
        mode_override=args.mode,
        epochs_override=epochs,
        subset_ratio=subset,
        run_name=run_name
    )


if __name__ == "__main__":
    main()
