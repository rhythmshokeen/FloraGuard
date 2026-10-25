"""
Model Versioning, Promotion, Rollback, and Registry Management.
Integrates with MLflow Model Registry and maintains a local model manifest.
"""

from datetime import datetime
import json
import os
import shutil
from typing import Any, Dict, List, Optional
import mlflow
from mlflow.tracking import MlflowClient

MANIFEST_PATH = "artifacts/model_manifest.json"
DEFAULT_MODEL_NAME = "plant-disease-resnet50"


def load_manifest() -> Dict[str, Any]:
    """
    Loads local model manifest containing production and staging versions.
    """
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass

    # Default initial manifest pointing to the baseline checkpoint
    return {
        "model_name": DEFAULT_MODEL_NAME,
        "production": {
            "version": "1.0.0-baseline",
            "checkpoint_path": "plant_disease_resnet50.pth",
            "promoted_at": "2026-10-25T00:00:00",
            "val_accuracy": 0.9501,
            "macro_f1": 0.9480,
            "architecture": "resnet50",
            "num_classes": 15,
            "status": "production"
        },
        "history": [],
        "candidates": []
    }


def save_manifest(manifest: Dict[str, Any]) -> None:
    """
    Saves manifest to disk atomically.
    """
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    tmp_path = f"{MANIFEST_PATH}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, MANIFEST_PATH)


def register_candidate_model(
    checkpoint_path: str,
    metrics: Dict[str, Any],
    version: str,
    run_id: Optional[str] = None,
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
) -> Dict[str, Any]:
    """
    Registers a new candidate model in both the local manifest and MLflow registry.
    """
    manifest = load_manifest()
    candidate_record = {
        "version": version,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "created_at": datetime.utcnow().isoformat(),
        "run_id": run_id,
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "min_class_recall": metrics.get("min_class_recall"),
            "avg_latency_ms": metrics.get("avg_latency_ms")
        },
        "status": "candidate"
    }

    manifest["candidates"].append(candidate_record)
    save_manifest(manifest)

    # Attempt MLflow model registry registration if run_id provided
    if run_id:
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri, DEFAULT_MODEL_NAME)
        except Exception as e:
            print(f"[Warning] MLflow registry registration skipped: {e}")

    return candidate_record


def promote_candidate_to_production(
    version: str,
    quality_gate_report: Dict[str, Any],
    backup_dir: str = "artifacts/backups"
) -> Dict[str, Any]:
    """
    Promotes a candidate version to production ONLY if quality gates passed.
    Archives previous production version into history for rollback support.
    """
    if not quality_gate_report.get("passed", False):
        raise ValueError("Cannot promote model: Candidate failed quality gate checks!")

    manifest = load_manifest()
    candidate = next((c for c in manifest["candidates"] if c["version"] == version), None)

    if not candidate:
        raise ValueError(f"Candidate version '{version}' not found in manifest candidates.")

    current_prod = manifest.get("production")
    if current_prod:
        current_prod["archived_at"] = datetime.utcnow().isoformat()
        current_prod["status"] = "archived"
        manifest["history"].append(current_prod)

        # Create safe backup copy of previous checkpoint
        os.makedirs(backup_dir, exist_ok=True)
        prev_ckpt = current_prod.get("checkpoint_path")
        if prev_ckpt and os.path.exists(prev_ckpt):
            backup_name = f"backup_{current_prod.get('version', 'prev')}_{os.path.basename(prev_ckpt)}"
            shutil.copy2(prev_ckpt, os.path.join(backup_dir, backup_name))

    # Promote candidate
    promoted_prod = {
        "version": candidate["version"],
        "checkpoint_path": candidate["checkpoint_path"],
        "promoted_at": datetime.utcnow().isoformat(),
        "val_accuracy": candidate["metrics"].get("accuracy"),
        "macro_f1": candidate["metrics"].get("macro_f1"),
        "run_id": candidate.get("run_id"),
        "status": "production"
    }

    manifest["production"] = promoted_prod
    # Remove from candidates
    manifest["candidates"] = [c for c in manifest["candidates"] if c["version"] != version]
    save_manifest(manifest)

    print(f"Successfully promoted model version {version} to production.")
    return promoted_prod


def rollback_production() -> Dict[str, Any]:
    """
    Rolls back the current production model to the immediate previous known-good model.
    """
    manifest = load_manifest()
    if not manifest.get("history"):
        raise RuntimeError("No previous model found in history to rollback to!")

    current_prod = manifest["production"]
    previous_prod = manifest["history"].pop()

    current_prod["status"] = "rolled_back"
    current_prod["archived_at"] = datetime.utcnow().isoformat()
    manifest["history"].append(current_prod)

    previous_prod["status"] = "production"
    previous_prod["restored_at"] = datetime.utcnow().isoformat()
    manifest["production"] = previous_prod

    save_manifest(manifest)
    print(f"Successfully rolled back production model to version {previous_prod.get('version')}.")
    return previous_prod


def get_current_production_model() -> Dict[str, Any]:
    """
    Returns current production model information.
    """
    manifest = load_manifest()
    return manifest.get("production", {})
