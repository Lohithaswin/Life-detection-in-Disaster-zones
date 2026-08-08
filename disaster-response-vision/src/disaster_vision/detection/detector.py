"""Unified YOLOv5/YOLOv8 detection interface."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from disaster_vision.config import Settings, get_settings

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO classes relevant to disaster-zone life detection
LIFE_CLASSES: frozenset[str] = frozenset(
    {
        "person",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }
)


class ModelFamily(str, Enum):
    """Supported YOLO model families."""

    YOLOV5 = "yolov5"
    YOLOV8 = "yolov8"


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Detection:
    """Single object detection result."""

    class_name: str
    confidence: float
    bbox: BoundingBox


class Detector:
    """Unified detector over Ultralytics YOLO (v5 and v8 weights)."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        settings: Settings | None = None,
        model_path: Path | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model_name = model_name or self._settings.default_model
        self._model_path = model_path or self._settings.resolve_model_path(self._model_name)
        self._model: YOLO | None = None

    @property
    def model_family(self) -> ModelFamily:
        """Infer model family from the weights filename."""
        name = self._model_path.stem.lower()
        if "yolov5" in name or name.startswith("yolov5"):
            return ModelFamily.YOLOV5
        return ModelFamily.YOLOV8

    def _load_model(self) -> YOLO:
        if self._model is not None:
            return self._model

        from ultralytics import YOLO

        source = self._model_path if self._model_path.is_file() else self._model_name
        logger.info("Loading model: %s (family=%s)", source, self.model_family.value)
        self._model = YOLO(str(source))
        return self._model

    def detect_image(
        self,
        source: str | Path,
        *,
        save: bool = True,
        life_only: bool = False,
    ) -> list[Detection]:
        """Run detection on a single image and return structured results."""
        model = self._load_model()
        results = model.predict(
            source=str(source),
            save=save,
            conf=self._settings.confidence_threshold,
            project=str(self._settings.runs_dir),
            name="detect",
            exist_ok=True,
        )

        detections: list[Detection] = []
        boxes = results[0].boxes  # type: ignore[index, union-attr]
        if boxes is None:
            return detections

        for box in boxes:  # type: ignore[union-attr]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if life_only and class_name not in LIFE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(box.conf[0]),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        logger.info("Detected %d object(s) in %s", len(detections), source)
        return detections

    def detect_batch(
        self,
        sources: list[str | Path],
        *,
        life_only: bool = False,
    ) -> dict[str, list[Detection]]:
        """Run detection on multiple images."""
        return {str(source): self.detect_image(source, life_only=life_only) for source in sources}

    @staticmethod
    def class_names(detections: list[Detection]) -> list[str]:
        """Return class names from a list of detections."""
        return [detection.class_name for detection in detections]

    @staticmethod
    def has_life_signs(detections: list[Detection]) -> bool:
        """Return True if any detection is a person or animal."""
        return any(d.class_name in LIFE_CLASSES for d in detections)
