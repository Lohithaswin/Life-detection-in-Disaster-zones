"""FastAPI REST API for detection, persistence, and alerts."""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile

from disaster_vision import __version__
from disaster_vision.alerts.email_alert import send_life_detection_alert_async
from disaster_vision.api.schemas import (
    BoundingBoxSchema,
    DetectionRecordSchema,
    DetectionSchema,
    DetectionsListResponse,
    DetectResponse,
    HealthResponse,
)
from disaster_vision.config import Settings, get_settings
from disaster_vision.db.repository import DetectionRepository
from disaster_vision.detection.detector import Detection, Detector
from disaster_vision.detection.video import VideoProcessor
from disaster_vision.logging_config import configure_logging

logger = logging.getLogger(__name__)


def get_repo(settings: Settings = Depends(get_settings)) -> DetectionRepository:
    return DetectionRepository(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    repo = DetectionRepository(settings)
    repo.create_tables()
    logger.info("API started (version %s)", __version__)
    yield


app = FastAPI(
    title="Disaster Response Vision API",
    description="Object detection API for disaster-zone imagery (COCO-pretrained YOLO).",
    version=__version__,
    lifespan=lifespan,
)


def _to_schemas(detections: list[Detection]) -> list[DetectionSchema]:
    return [
        DetectionSchema(
            class_name=d.class_name,
            confidence=d.confidence,
            bbox=BoundingBoxSchema(
                x1=d.bbox.x1,
                y1=d.bbox.y1,
                x2=d.bbox.x2,
                y2=d.bbox.y2,
            ),
        )
        for d in detections
    ]


async def _save_upload(upload: UploadFile, suffix: str) -> Path:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    tmp = Path(tempfile.mkdtemp(prefix="drv_upload_"))
    dest = tmp / f"upload{suffix}"
    with dest.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return dest


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check."""
    return HealthResponse(status="ok", version=__version__)


@app.get("/detections", response_model=DetectionsListResponse)
def list_detections(
    limit: int = 100,
    repo: DetectionRepository = Depends(get_repo),
) -> DetectionsListResponse:
    """List recent detection records from the database."""
    records = repo.list_detections(limit=limit)
    items = [DetectionRecordSchema.model_validate(r) for r in records]
    return DetectionsListResponse(count=len(items), items=items)


@app.post("/detect/image", response_model=DetectResponse)
async def detect_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(default="yolov8n"),
    life_only: bool = Form(default=False),
    send_alert: bool = Form(default=True),
    persist: bool = Form(default=True),
    settings: Settings = Depends(get_settings),
    repo: DetectionRepository = Depends(get_repo),
) -> DetectResponse:
    """Run object detection on an uploaded image."""
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    path = await _save_upload(file, suffix)

    detector = Detector(model_name=model, settings=settings)
    detections = detector.detect_image(path, save=False, life_only=life_only)
    life = Detector.has_life_signs(detections)

    if persist:
        media = repo.insert_media("image", file.filename or path.name, str(path))
        repo.insert_detections_from_results(
            detections, str(path), model, media_id=media.id
        )

    alert_sent = False
    if send_alert and life:

        def _run_alert() -> None:
            import asyncio

            asyncio.run(
                send_life_detection_alert_async(
                    detections,
                    str(path),
                    settings=settings,
                    image_reference=str(path),
                )
            )

        background_tasks.add_task(_run_alert)
        alert_sent = True

    return DetectResponse(
        source=file.filename or str(path),
        model=model,
        detections=_to_schemas(detections),
        life_detected=life,
        alert_sent=alert_sent,
    )


@app.post("/detect/video", response_model=DetectResponse)
async def detect_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form(default="yolov8n"),
    life_only: bool = Form(default=False),
    send_alert: bool = Form(default=True),
    persist: bool = Form(default=True),
    settings: Settings = Depends(get_settings),
    repo: DetectionRepository = Depends(get_repo),
) -> DetectResponse:
    """Run object detection on an uploaded video (no live display)."""
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    path = await _save_upload(file, suffix)

    detector = Detector(model_name=model, settings=settings)
    processor = VideoProcessor(detector=detector, settings=settings, display=False)
    detections = processor.process(path, life_only=life_only, early_exit_on_life=False)
    life = Detector.has_life_signs(detections)

    if persist:
        media = repo.insert_media("video", file.filename or path.name, str(path))
        repo.insert_detections_from_results(
            detections, str(path), model, media_id=media.id
        )

    alert_sent = False
    if send_alert and life:

        def _run_alert() -> None:
            import asyncio

            asyncio.run(
                send_life_detection_alert_async(
                    detections,
                    str(path),
                    settings=settings,
                    image_reference=str(path),
                )
            )

        background_tasks.add_task(_run_alert)
        alert_sent = True

    return DetectResponse(
        source=file.filename or str(path),
        model=model,
        detections=_to_schemas(detections),
        life_detected=life,
        alert_sent=alert_sent,
    )


def run_server() -> None:
    """Console script entry point for uvicorn."""
    import os

    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(
        "disaster_vision.api.main:app",
        host=host,
        port=port,
        reload=os.getenv("API_RELOAD", "").lower() in {"1", "true", "yes"},
    )
