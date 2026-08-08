"""Download small, clearly-licensed sample images for local development.

Dataset: MS COCO 2017 validation subset (3 images)
License: Creative Commons Attribution 4.0 (CC BY 4.0)
  https://cocodataset.org/#termsofuse

These images are used only for smoke-testing detection pipelines.
Benchmark metrics in Phase 4 use a separate, reproducible evaluation script.
"""

from __future__ import annotations

import argparse
import logging
import os
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# COCO val2017 images known to contain person(s). Official CDN, CC BY 4.0.
# HTTP used — some environments report SSL hostname mismatch on the HTTPS CDN.
COCO_SAMPLES: list[tuple[str, str]] = [
    (
        "http://images.cocodataset.org/val2017/000000000139.jpg",
        "coco_val_000000000139.jpg",
    ),
    (
        "http://images.cocodataset.org/val2017/000000000285.jpg",
        "coco_val_000000000285.jpg",
    ),
    (
        "http://images.cocodataset.org/val2017/000000000632.jpg",
        "coco_val_000000000632.jpg",
    ),
]

# Sample video: Ultralytics assets repo — short demo clip for YOLO testing.
# License: AGPL-3.0 (Ultralytics). Suitable for development smoke tests only.
SAMPLE_VIDEO = (
    "https://github.com/ultralytics/assets/raw/main/social/docs/source/boat.gif",
    "sample_boat.gif",
)


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        logger.info("Already present: %s", dest.name)
        return
    logger.info("Downloading %s ...", dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 — fixed public URLs
    logger.info("Saved: %s (%.1f KB)", dest, dest.stat().st_size / 1024)


def download_samples(data_dir: Path, include_video: bool = True) -> None:
    """Download COCO sample images (and optional demo clip) into data/samples/."""
    samples_dir = data_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    attribution = samples_dir / "ATTRIBUTION.md"
    if not attribution.exists():
        attribution.write_text(
            "# Sample Data Attribution\n\n"
            "## Images\n\n"
            "- **Source:** [MS COCO 2017 validation set](https://cocodataset.org/)\n"
            "- **License:** [Creative Commons Attribution 4.0 (CC BY 4.0)]("
            "https://creativecommons.org/licenses/by/4.0/)\n"
            "- **Files:** `coco_val_*.jpg`\n\n"
            "## Demo clip (optional)\n\n"
            "- **Source:** [Ultralytics assets](https://github.com/ultralytics/assets)\n"
            "- **License:** AGPL-3.0\n"
            "- **File:** `sample_boat.gif` — short animated clip for smoke testing\n",
            encoding="utf-8",
        )

    for url, filename in COCO_SAMPLES:
        _download(url, samples_dir / filename)

    if include_video:
        url, filename = SAMPLE_VIDEO
        _download(url, samples_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download licensed sample media.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("DATA_DIR", "data")),
        help="Root data directory (default: data/ or DATA_DIR env)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip downloading the demo clip",
    )
    args = parser.parse_args()
    download_samples(args.data_dir, include_video=not args.no_video)
    logger.info("Sample data ready in %s", args.data_dir / "samples")


if __name__ == "__main__":
    main()
