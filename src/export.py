"""
ONNX Export, Model Validation, Parity Verification, and Benchmarking.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.model import load_checkpoint

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def export_to_onnx(
    checkpoint_path: str = "plant_disease_resnet50.pth",
    output_onnx_path: str = "artifacts/model.onnx",
    opset_version: int = 18
) -> str:
    """
    Exports PyTorch model checkpoint to ONNX format with dynamic batch axis.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX or ONNXRuntime is not installed. Please install 'onnx' and 'onnxruntime'.")

    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
    model, metadata = load_checkpoint(checkpoint_path, device="cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    print(f"Exporting PyTorch model to ONNX: {output_onnx_path} (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    # Validate ONNX model graph structure
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated successfully by onnx.checker.")

    return output_onnx_path


def benchmark_models(
    checkpoint_path: str = "plant_disease_resnet50.pth",
    onnx_path: str = "artifacts/model.onnx",
    num_iterations: int = 50,
    batch_size: int = 1
) -> Dict[str, Any]:
    """
    Benchmarks and compares PyTorch (CPU) vs ONNX Runtime on latency, throughput, and numerical parity.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNXRuntime is required for benchmarking.")

    # Load PyTorch model
    pt_model, _ = load_checkpoint(checkpoint_path, device="cpu")
    pt_model.eval()

    # Load ONNX Runtime session
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Prepare random input
    np_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    torch_input = torch.from_numpy(np_input)

    # 1. Parity Check
    with torch.no_grad():
        pt_out = pt_model(torch_input).numpy()

    onnx_out = session.run(None, {input_name: np_input})[0]

    max_abs_diff = float(np.max(np.abs(pt_out - onnx_out)))
    parity_passed = max_abs_diff < 1e-4

    # 2. PyTorch Latency Warmup & Benchmark
    with torch.no_grad():
        for _ in range(5):
            _ = pt_model(torch_input)

        pt_latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = pt_model(torch_input)
            pt_latencies.append((time.perf_counter() - start) * 1000)

    # 3. ONNX Runtime Warmup & Benchmark
    for _ in range(5):
        _ = session.run(None, {input_name: np_input})

    onnx_latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = session.run(None, {input_name: np_input})
        onnx_latencies.append((time.perf_counter() - start) * 1000)

    # Metrics
    pt_mean_ms = float(np.mean(pt_latencies))
    pt_p95_ms = float(np.percentile(pt_latencies, 95))
    pt_throughput = round((batch_size * 1000) / pt_mean_ms, 2)

    onnx_mean_ms = float(np.mean(onnx_latencies))
    onnx_p95_ms = float(np.percentile(onnx_latencies, 95))
    onnx_throughput = round((batch_size * 1000) / onnx_mean_ms, 2)

    speedup = round(pt_mean_ms / onnx_mean_ms, 2) if onnx_mean_ms > 0 else 1.0

    pt_size_mb = round(os.path.getsize(checkpoint_path) / (1024 * 1024), 2)
    onnx_size_mb = round(os.path.getsize(onnx_path) / (1024 * 1024), 2)

    results = {
        "parity_check": {
            "max_absolute_difference": max_abs_diff,
            "parity_passed": parity_passed,
            "tolerance": 1e-4
        },
        "pytorch_cpu": {
            "mean_latency_ms": round(pt_mean_ms, 2),
            "p95_latency_ms": round(pt_p95_ms, 2),
            "throughput_fps": pt_throughput,
            "model_size_mb": pt_size_mb
        },
        "onnx_runtime_cpu": {
            "mean_latency_ms": round(onnx_mean_ms, 2),
            "p95_latency_ms": round(onnx_p95_ms, 2),
            "throughput_fps": onnx_throughput,
            "model_size_mb": onnx_size_mb
        },
        "speedup_factor": speedup,
        "iterations": num_iterations,
        "batch_size": batch_size
    }

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/onnx_benchmark.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX and benchmark")
    parser.add_argument("--checkpoint", type=str, default="plant_disease_resnet50.pth")
    parser.add_argument("--output", type=str, default="artifacts/model.onnx")
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    onnx_path = export_to_onnx(args.checkpoint, args.output)
    print("\nRunning benchmark comparison...")
    benchmarks = benchmark_models(args.checkpoint, onnx_path, num_iterations=args.iterations)

    print("\n" + "=" * 50)
    print("ONNX EXPORT & BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Numerical Parity:       {'PASSED' if benchmarks['parity_check']['parity_passed'] else 'FAILED'} (max diff: {benchmarks['parity_check']['max_absolute_difference']:.2e})")
    print(f"PyTorch CPU Latency:    {benchmarks['pytorch_cpu']['mean_latency_ms']} ms ({benchmarks['pytorch_cpu']['throughput_fps']} fps)")
    print(f"ONNX Runtime Latency:   {benchmarks['onnx_runtime_cpu']['mean_latency_ms']} ms ({benchmarks['onnx_runtime_cpu']['throughput_fps']} fps)")
    print(f"Speedup:                {benchmarks['speedup_factor']}x")
    print(f"PyTorch File Size:      {benchmarks['pytorch_cpu']['model_size_mb']} MB")
    print(f"ONNX File Size:         {benchmarks['onnx_runtime_cpu']['model_size_mb']} MB")
    print(f"Report saved to:        reports/onnx_benchmark.json")


if __name__ == "__main__":
    main()
