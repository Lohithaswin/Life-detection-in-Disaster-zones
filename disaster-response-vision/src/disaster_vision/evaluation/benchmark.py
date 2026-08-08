"""Real YOLO benchmark on a licensed dataset — no hardcoded metrics.

Dataset: Ultralytics COCO128 (128 images from MS COCO train2017)
License: Creative Commons Attribution 4.0 (CC BY 4.0)
  https://cocodataset.org/#termsofuse

Models are evaluated out-of-the-box (COCO-pretrained). No domain fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

COCO128_YAML = "coco128.yaml"
DEFAULT_MODELS = ("yolov5s.pt", "yolov8n.pt")


@dataclass
class ModelBenchmarkResult:
    model: str
    model_size_mb: float
    precision: float
    recall: float
    map50: float
    map50_95: float
    inference_ms_cpu: float
    inference_ms_gpu: float | None
    device_gpu: str | None


def _model_size_mb(model_path: Path) -> float:
    if model_path.is_file():
        return model_path.stat().st_size / (1024 * 1024)
    return 0.0


def _latency_ms(model, image_path: Path, *, device: str, runs: int = 15) -> float:
    for _ in range(3):
        model.predict(str(image_path), device=device, verbose=False)
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(str(image_path), device=device, verbose=False)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.mean(samples)


def _extract_val_metrics(metrics: Any) -> tuple[float, float, float, float]:
    box = metrics.box
    return float(box.mp), float(box.mr), float(box.map50), float(box.map)


def benchmark_model(
    model_name: str,
    *,
    data_yaml: str,
    weights_dir: Path,
    sample_image: Path | None,
) -> ModelBenchmarkResult:
    from ultralytics import YOLO

    weights_path = weights_dir / model_name
    load_name = str(weights_path) if weights_path.is_file() else model_name

    logger.info("Running val() for %s on %s", load_name, data_yaml)
    model = YOLO(load_name)
    metrics = model.val(data=data_yaml, split="val", plots=True, verbose=False)
    precision, recall, map50, map50_95 = _extract_val_metrics(metrics)

    size_mb = _model_size_mb(Path(getattr(model, "ckpt_path", weights_path)))

    if sample_image is None or not sample_image.is_file():
        raise FileNotFoundError(
            "Sample image for latency benchmark not found. Run scripts/download_sample_data.py"
        )

    cpu_ms = _latency_ms(model, sample_image, device="cpu")
    gpu_ms: float | None = None
    gpu_name: str | None = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_ms = _latency_ms(model, sample_image, device="0")

    return ModelBenchmarkResult(
        model=model_name,
        model_size_mb=round(size_mb, 2),
        precision=round(precision, 4),
        recall=round(recall, 4),
        map50=round(map50, 4),
        map50_95=round(map50_95, 4),
        inference_ms_cpu=round(cpu_ms, 2),
        inference_ms_gpu=round(gpu_ms, 2) if gpu_ms is not None else None,
        device_gpu=gpu_name,
    )


def save_results(results: list[ModelBenchmarkResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": COCO128_YAML,
        "dataset_license": "CC BY 4.0 (MS COCO via Ultralytics COCO128)",
        "methodology": "Ultralytics model.val() on COCO128; latency = mean ms/image after warmup",
        "fine_tuning": "none — COCO-pretrained weights only",
        "models": [asdict(r) for r in results],
    }
    json_path = output_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", json_path)
    return json_path


def generate_charts(json_path: Path, output_dir: Path) -> None:
    """Generate comparison charts strictly from benchmark_results.json."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload["models"])

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_cols = ["precision", "recall", "map50", "map50_95", "model_size_mb", "inference_ms_cpu"]
    plot_df = df[["model"] + metrics_cols].set_index("model")

    ax = plot_df.plot(kind="bar", figsize=(10, 6), rot=0)
    ax.set_title("Model comparison (from benchmark_results.json)")
    ax.set_ylabel("value")
    plt.tight_layout()
    bar_path = output_dir / "model_comparison_bar.png"
    plt.savefig(bar_path, dpi=200)
    plt.close()

    corr = plot_df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Metric correlation (computed from benchmark results)")
    plt.tight_layout()
    heatmap_path = output_dir / "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    if "map50" in df.columns:
        plt.figure(figsize=(6, 5))
        sns.barplot(data=df, x="model", y="map50")
        plt.title("mAP@0.5 per model (from val())")
        plt.tight_layout()
        plt.savefig(output_dir / "map50_comparison.png", dpi=200)
        plt.close()

    logger.info("Charts written to %s", output_dir)


def print_summary_table(results: list[ModelBenchmarkResult]) -> None:
    header = (
        f"{'Model':<14} {'mAP@0.5':>8} {'mAP@0.5:0.95':>12} "
        f"{'Prec':>8} {'Recall':>8} {'Size MB':>8} {'CPU ms':>8} {'GPU ms':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        gpu = f"{r.inference_ms_gpu:.1f}" if r.inference_ms_gpu is not None else "n/a"
        print(
            f"{r.model:<14} {r.map50:>8.4f} {r.map50_95:>12.4f} "
            f"{r.precision:>8.4f} {r.recall:>8.4f} {r.model_size_mb:>8.2f} "
            f"{r.inference_ms_cpu:>8.1f} {gpu:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real YOLO benchmark on COCO128.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model weight filenames to evaluate",
    )
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--data",
        default=COCO128_YAML,
        help="Ultralytics dataset YAML (default: coco128.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
    )
    parser.add_argument(
        "--sample-image",
        type=Path,
        default=Path("data/samples/coco_val_000000000139.jpg"),
    )
    parser.add_argument("--skip-charts", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results: list[ModelBenchmarkResult] = []
    for model_name in args.models:
        results.append(
            benchmark_model(
                model_name,
                data_yaml=args.data,
                weights_dir=args.weights_dir,
                sample_image=args.sample_image,
            )
        )

    json_path = save_results(results, args.output_dir)
    if not args.skip_charts:
        generate_charts(json_path, args.output_dir)

    print("\n=== Benchmark summary (paste into README) ===")
    print(f"Dataset: {COCO128_YAML} (CC BY 4.0) — pretrained only, no fine-tuning\n")
    print_summary_table(results)


if __name__ == "__main__":
    main()
