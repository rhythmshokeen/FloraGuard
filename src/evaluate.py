"""
Automated Model Evaluation, Quality Gates, Confusion Matrix, and Error Analysis.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import CLASS_NAMES, create_dataloaders, resolve_dataset_dir
from src.model import get_device, load_checkpoint


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
    output_dir: str = "reports"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluates model on dataloader, computes all metrics, saves confusion matrix and error analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []
    all_confidences = []
    latencies = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)

            start = time.perf_counter()
            outputs = model(images)
            batch_time = (time.perf_counter() - start) * 1000  # ms
            latencies.append(batch_time / images.size(0))

            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_confidences.extend(confs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_confidences = np.array(all_confidences)

    # Core Metrics
    accuracy = float((all_preds == all_targets).mean())
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
    macro_precision = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_recall = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))

    # Per-Class Metrics
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    per_class_recalls = {cls: report_dict[cls]["recall"] for cls in class_names if cls in report_dict}
    min_class_recall = min(per_class_recalls.values()) if per_class_recalls else 0.0
    worst_class = min(per_class_recalls, key=per_class_recalls.get) if per_class_recalls else "None"

    # Save Classification Report JSON
    report_path = os.path.join(output_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    # Compute & Plot Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2%}, Macro F1: {macro_f1:.4f})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    # Error / Misclassification Analysis
    misclassified_mask = all_preds != all_targets
    misclassified_df = pd.DataFrame({
        "sample_index": np.where(misclassified_mask)[0],
        "true_class": [class_names[t] for t in all_targets[misclassified_mask]],
        "predicted_class": [class_names[p] for p in all_preds[misclassified_mask]],
        "confidence": all_confidences[misclassified_mask]
    })
    misclassified_path = os.path.join(output_dir, "misclassified_samples.csv")
    misclassified_df.to_csv(misclassified_path, index=False)

    avg_latency_ms = float(np.mean(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))

    metrics_summary = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "min_class_recall": min_class_recall,
        "worst_performing_class": worst_class,
        "total_evaluated_samples": len(all_targets),
        "total_misclassified": int(misclassified_mask.sum()),
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "confusion_matrix_path": cm_path,
        "classification_report_path": report_path,
        "misclassified_samples_path": misclassified_path,
        "per_class_recalls": per_class_recalls
    }

    return metrics_summary, misclassified_df


def check_quality_gates(
    metrics: Dict[str, Any],
    gates_config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates candidate metrics against automated quality gate criteria.
    Candidate must satisfy accuracy, macro F1, class recall, and latency limits.
    """
    gates = gates_config or {
        "min_val_accuracy": 0.90,
        "min_macro_f1": 0.88,
        "min_class_recall": 0.75,
        "max_latency_ms": 250.0
    }

    checks = {
        "accuracy_check": {
            "value": metrics["accuracy"],
            "threshold": gates["min_val_accuracy"],
            "passed": bool(metrics["accuracy"] >= gates["min_val_accuracy"])
        },
        "macro_f1_check": {
            "value": metrics["macro_f1"],
            "threshold": gates["min_macro_f1"],
            "passed": bool(metrics["macro_f1"] >= gates["min_macro_f1"])
        },
        "min_class_recall_check": {
            "value": metrics["min_class_recall"],
            "threshold": gates["min_class_recall"],
            "worst_class": metrics.get("worst_performing_class", ""),
            "passed": bool(metrics["min_class_recall"] >= gates["min_class_recall"])
        },
        "latency_check": {
            "value": metrics["avg_latency_ms"],
            "threshold": gates["max_latency_ms"],
            "passed": bool(metrics["avg_latency_ms"] <= gates["max_latency_ms"])
        }
    }

    all_passed = all(check["passed"] for check in checks.values())

    report = {
        "passed": all_passed,
        "checks": checks,
        "summary": "Passed all quality gates" if all_passed else "Failed one or more quality gates"
    }

    return all_passed, report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Plant Disease Classification Model")
    parser.add_argument("--checkpoint", type=str, default="plant_disease_resnet50.pth", help="Path to model checkpoint")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to PlantVillage dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, mps, cuda, cpu)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory for reports and plots")
    parser.add_argument("--subset-ratio", type=float, default=None, help="Optional subset fraction for quick smoke check")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Loading checkpoint from: {args.checkpoint} on {device}")
    model, metadata = load_checkpoint(args.checkpoint, device=device)
    class_names = metadata.get("class_names") or CLASS_NAMES

    data_dir = args.dataset_dir or resolve_dataset_dir()
    print(f"Evaluating on dataset at: {data_dir}")

    _, val_loader, _, split_info = create_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        val_split=0.2,
        random_state=42,
        subset_ratio=args.subset_ratio
    )

    metrics, _ = evaluate_model(model, val_loader, device, class_names, output_dir=args.output_dir)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy:       {metrics['accuracy']:.2%}")
    print(f"Macro F1 Score:         {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score:      {metrics['weighted_f1']:.4f}")
    print(f"Worst Class Recall:     {metrics['min_class_recall']:.2%} ({metrics['worst_performing_class']})")
    print(f"Total Misclassified:    {metrics['total_misclassified']} / {metrics['total_evaluated_samples']}")
    print(f"Avg Latency per image:  {metrics['avg_latency_ms']:.2f} ms")
    print(f"Confusion Matrix saved: {metrics['confusion_matrix_path']}")
    print(f"Report saved:           {metrics['classification_report_path']}")

    passed, gate_report = check_quality_gates(metrics)
    print("\n" + "=" * 50)
    print(f"QUALITY GATES: {'PASSED' if passed else 'FAILED'}")
    print("=" * 50)
    for name, c in gate_report["checks"].items():
        status = "PASSED" if c["passed"] else "FAILED"
        print(f" - {name:<24}: {status} (val={c['value']:.4f}, thresh={c['threshold']})")

    # Save quality gate report
    gate_path = os.path.join(args.output_dir, "quality_gate_report.json")
    with open(gate_path, "w") as f:
        json.dump(gate_report, f, indent=2)


if __name__ == "__main__":
    main()
