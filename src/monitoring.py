"""
Production Inference Monitoring, Structured Logging, Drift Detection, and Quality Metrics.
"""

from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

MONITORING_LOG_FILE = "artifacts/monitoring/predictions.log"
MONITORING_REPORT_FILE = "reports/monitoring_report.json"
REVIEW_QUEUE_FILE = "artifacts/monitoring/review_queue.jsonl"


class PredictionLogger:
    """
    Appends structured JSON logs for each prediction request.
    Strictly avoids persisting raw image pixels for privacy and efficiency.
    """

    def __init__(self, log_file: str = MONITORING_LOG_FILE):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log_prediction(
        self,
        model_version: str,
        predicted_class: str,
        confidence: float,
        latency_ms: float,
        image_width: int,
        image_height: int,
        status: str = "success",
        is_uncertain: bool = False,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version,
            "predicted_class": predicted_class,
            "confidence": round(float(confidence), 4),
            "latency_ms": round(float(latency_ms), 2),
            "image_dimensions": [image_width, image_height],
            "status": status,
            "is_uncertain": is_uncertain,
            "error_message": error_message
        }

        # Thread-safe appending to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        # If flagged as uncertain or error, queue for human review
        if is_uncertain or status != "success":
            self._queue_for_review(record)

        return record

    def _queue_for_review(self, record: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(REVIEW_QUEUE_FILE), exist_ok=True)
        with open(REVIEW_QUEUE_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")


def calculate_psi(expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-4) -> float:
    """
    Computes Population Stability Index (PSI) between two discrete distributions.
    PSI < 0.1: No significant change / stable.
    0.1 <= PSI < 0.2: Moderate drift detected.
    PSI >= 0.2: Significant drift detected.
    """
    expected = np.array(expected, dtype=np.float64) + epsilon
    actual = np.array(actual, dtype=np.float64) + epsilon

    expected /= expected.sum()
    actual /= actual.sum()

    psi_val = np.sum((actual - expected) * np.log(actual / expected))
    return float(np.round(psi_val, 4))


def generate_monitoring_report(
    log_file: str = MONITORING_LOG_FILE,
    baseline_distribution: Optional[Dict[str, float]] = None,
    output_report_path: str = MONITORING_REPORT_FILE
) -> Dict[str, Any]:
    """
    Aggregates log events and produces latency, confidence, error, and distribution drift metrics.
    """
    if not os.path.exists(log_file):
        return {
            "status": "no_data",
            "message": "No prediction logs recorded yet.",
            "total_requests": 0
        }

    records = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return {"status": "empty_logs", "total_requests": 0}

    df = pd.DataFrame(records)

    total_requests = len(df)
    successful_requests = len(df[df["status"] == "success"])
    failed_requests = total_requests - successful_requests
    error_rate = failed_requests / total_requests if total_requests > 0 else 0.0

    success_df = df[df["status"] == "success"]

    # Latency Stats
    latencies = success_df["latency_ms"].tolist() if not success_df.empty else [0.0]
    avg_latency = float(np.mean(latencies))
    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))
    max_latency = float(np.max(latencies))

    # Confidence Stats
    confidences = success_df["confidence"].tolist() if not success_df.empty else [0.0]
    avg_confidence = float(np.mean(confidences))
    p10_confidence = float(np.percentile(confidences, 10))
    uncertain_count = int(success_df["is_uncertain"].sum()) if "is_uncertain" in success_df.columns else 0
    uncertainty_rate = uncertain_count / successful_requests if successful_requests > 0 else 0.0

    # Class Distribution
    class_counts = success_df["predicted_class"].value_counts().to_dict() if not success_df.empty else {}

    # Drift Detection (PSI)
    drift_status = "stable"
    psi_score = 0.0

    if baseline_distribution and class_counts:
        all_classes = sorted(list(set(list(baseline_distribution.keys()) + list(class_counts.keys()))))
        expected_vec = [baseline_distribution.get(c, 0.0) for c in all_classes]
        actual_vec = [class_counts.get(c, 0) for c in all_classes]

        psi_score = calculate_psi(expected_vec, actual_vec)
        if psi_score >= 0.25:
            drift_status = "significant_drift"
        elif psi_score >= 0.10:
            drift_status = "moderate_drift"

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "error_rate": round(error_rate, 4),
        "latency_stats": {
            "mean_ms": round(avg_latency, 2),
            "p50_ms": round(p50_latency, 2),
            "p95_ms": round(p95_latency, 2),
            "max_ms": round(max_latency, 2)
        },
        "confidence_stats": {
            "mean": round(avg_confidence, 4),
            "p10": round(p10_confidence, 4),
            "uncertain_requests": uncertain_count,
            "uncertainty_rate": round(uncertainty_rate, 4)
        },
        "drift_analysis": {
            "psi_score": psi_score,
            "status": drift_status,
            "interpretation": (
                "Distribution matches baseline" if drift_status == "stable"
                else "Moderate shift in incoming prediction classes" if drift_status == "moderate_drift"
                else "Significant change in incoming prediction classes. Human review recommended."
            )
        },
        "class_distribution": class_counts
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
