"""CLI entry point — expanded in Phase 3."""

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

app = typer.Typer(
    name="disaster-vision",
    help="Disaster Response Vision — detect people and animals in imagery.",
    no_args_is_help=True,
)


def _discover_media(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in extensions)


@app.callback()
def main_callback() -> None:
    """Initialize logging before any subcommand runs."""
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command("detect")
def detect_command(
    source: Path = typer.Argument(..., help="Image/video file or directory"),
    model: str = typer.Option("yolov8n", "--model", "-m", help="YOLO model name"),
    life_only: bool = typer.Option(False, "--life-only", help="Filter to person/animal classes"),
    no_alert: bool = typer.Option(False, "--no-alert", help="Skip email alerts"),
    no_db: bool = typer.Option(False, "--no-db", help="Skip database persistence"),
) -> None:
    """Run detection on an image, video, or directory of media files."""
    settings = get_settings()
    detector = Detector(model_name=model, settings=settings)
    repo = DetectionRepository(settings)
    if not no_db:
        repo.create_tables()

    sources: list[Path]
    if source.is_dir():
        sources = _discover_media(source, {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".gif"})
    else:
        sources = [source]

    if not sources:
        typer.echo(f"No media files found at {source}", err=True)
        raise typer.Exit(code=1)

    video_ext = {".mp4", ".avi", ".mov", ".gif"}
    processor = VideoProcessor(detector=detector, settings=settings, display=False)

    for media_path in sources:
        logging.info("Processing %s", media_path)
        if media_path.suffix.lower() in video_ext:
            detections = processor.process(media_path, life_only=life_only)
            media_type = "video"
        else:
            detections = detector.detect_image(media_path, life_only=life_only)
            media_type = "image"

        if not no_db:
            media = repo.insert_media(media_type, media_path.name, str(media_path))
            repo.insert_detections_from_results(
                detections,
                str(media_path),
                model,
                media_id=media.id,
            )

        classes = Detector.class_names(detections)
        typer.echo(f"{media_path.name}: {classes or 'no detections'}")

        if not no_alert and Detector.has_life_signs(detections):
            send_life_detection_alert(detections, str(media_path), settings=settings)


@app.command("alert-test")
def alert_test_command() -> None:
    """Send a test alert email using configured SMTP settings."""
    EmailAlerter().send_test_alert()
    typer.echo("Test alert sent.")


@app.command("legacy-menu")
def legacy_menu_command() -> None:
    """Interactive menu preserved from the original course project."""
    from disaster_vision.legacy_menu import run_legacy_menu

    run_legacy_menu()


def main() -> None:
    """Console entry point for the disaster-vision CLI."""
    app()


if __name__ == "__main__":
    main()
