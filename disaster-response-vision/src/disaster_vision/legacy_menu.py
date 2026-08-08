"""Legacy interactive menu from the original course project."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from disaster_vision.alerts.email_alert import EmailAlerter, send_life_detection_alert
from disaster_vision.config import get_settings
from disaster_vision.db.repository import DetectionRepository
from disaster_vision.detection.detector import Detector
from disaster_vision.detection.video import VideoProcessor
from disaster_vision.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _discover_media(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in extensions)


def run_legacy_menu() -> None:
    """Run the original img/vid/alert menu using the new package internals."""
    settings = get_settings()
    configure_logging(settings.log_level)

    samples_dir = settings.data_dir / "samples"
    typer.echo("=== Disaster Response Vision (legacy menu) ===")
    typer.echo("Run scripts/download_sample_data.py and scripts/download_weights.py first.")
    typer.echo("Options: img | vid | alert")
    choice = typer.prompt(">").lower().strip()

    detector = Detector(settings=settings)
    repo = DetectionRepository(settings)
    repo.create_tables()
    processor = VideoProcessor(detector=detector, settings=settings)

    images = _discover_media(samples_dir, {".jpg", ".jpeg", ".png"})
    videos = _discover_media(samples_dir, {".mp4", ".avi", ".mov", ".gif"})

    if choice == "img":
        if not images:
            typer.echo("No sample images found. Run: python scripts/download_sample_data.py")
            return
        for image_path in images:
            typer.echo(f"\nProcessing image: {image_path.name}")
            detections = detector.detect_image(image_path)
            repo.insert_media("image", image_path.name, str(image_path))
            repo.insert_detections_from_results(
                detections, str(image_path), settings.default_model
            )
            if Detector.has_life_signs(detections):
                typer.echo("Alert: person or animal detected.")
                send_life_detection_alert(detections, str(image_path), settings=settings)
            else:
                typer.echo("No person or animal detected.")

    elif choice == "vid":
        if not videos:
            typer.echo("No sample videos found. Run: python scripts/download_sample_data.py")
            return
        for video_path in videos:
            typer.echo(f"\nProcessing video: {video_path.name}")
            detections = processor.process(video_path)
            repo.insert_media("video", video_path.name, str(video_path))
            repo.insert_detections_from_results(
                detections, str(video_path), settings.default_model
            )
            if Detector.has_life_signs(detections):
                typer.echo("Alert: person or animal detected.")
                send_life_detection_alert(detections, str(video_path), settings=settings)
            else:
                typer.echo("No person or animal detected.")

    elif choice == "alert":
        EmailAlerter(settings=settings).send_test_alert()
        typer.echo("Test alert sent.")

    else:
        typer.echo("Invalid option. Choose 'img', 'vid', or 'alert'.")
