"""Download YOLO model weights into the weights/ directory.

Weights are fetched from Ultralytics on first use and cached locally.
They are gitignored — run this script after cloning.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS = ("yolov8n.pt", "yolov5s.pt")


def download_weights(models: tuple[str, ...], weights_dir: Path) -> None:
    """Download each model via Ultralytics and copy into weights_dir."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from exc

    weights_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        dest = weights_dir / model_name
        if dest.exists():
            logger.info("Already present: %s", dest)
            continue

        logger.info("Downloading %s via Ultralytics...", model_name)
        model = YOLO(model_name)

        # Ultralytics caches under ~/.config/Ultralytics or cwd; locate the file.
        cached = Path(getattr(model, "ckpt_path", model_name))
        if not cached.is_file():
            cached = Path(model_name)

        if cached.is_file() and cached.resolve() != dest.resolve():
            shutil.copy2(cached, dest)
        elif not dest.exists():
            # Trigger download by running a no-op predict on a 1x1 placeholder is overkill;
            # YOLO() already downloaded — search common cache locations.
            for candidate in (Path(model_name), Path.home() / ".config" / "Ultralytics" / model_name):
                if candidate.is_file():
                    shutil.copy2(candidate, dest)
                    break

        if dest.exists():
            logger.info("Saved: %s (%.1f MB)", dest, dest.stat().st_size / 1_048_576)
        else:
            logger.warning("Could not locate cached weight for %s; it will download on first inference.", model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YOLO model weights.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=f"Model filenames to download (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path(os.getenv("WEIGHTS_DIR", "weights")),
        help="Directory to store weights (default: weights/ or WEIGHTS_DIR env)",
    )
    args = parser.parse_args()
    download_weights(tuple(args.models), args.weights_dir)


if __name__ == "__main__":
    main()
