"""Video and real-time detection processing."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from disaster_vision.config import Settings, get_settings
from disaster_vision.detection.detector import LIFE_CLASSES, BoundingBox, Detection, Detector

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process video files or camera streams with YOLO detection."""

    def __init__(
        self,
        detector: Detector | None = None,
        *,
        settings: Settings | None = None,
        display: bool = True,
        frame_delay_ms: int = 33,
        post_detection_frames: int = 30,
    ) -> None:
        self._settings = settings or get_settings()
        self._detector = detector or Detector(settings=self._settings)
        self._display = display
        self._frame_delay_ms = frame_delay_ms
        self._post_detection_frames = post_detection_frames

    def _draw_detections(self, frame, detections: list[Detection]) -> None:
        for detection in detections:
            box = detection.bbox
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            color = (0, 255, 0) if detection.class_name == "person" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    def _predict_frame(self, frame) -> list[Detection]:
        model = self._detector._load_model()
        results = model.predict(
            source=frame,
            show=False,
            conf=self._settings.confidence_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(box.conf[0]),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return detections

    def process(
        self,
        video_source: str | Path,
        *,
        life_only: bool = False,
        early_exit_on_life: bool = True,
    ) -> list[Detection]:
        """Process a video and return all detections encountered."""
        cap = cv2.VideoCapture(str(video_source))
        if not cap.isOpened():
            logger.error("Could not open video source: %s", video_source)
            return []

        all_detections: list[Detection] = []
        life_detected = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_detections = self._predict_frame(frame)
                if life_only:
                    frame_detections = [d for d in frame_detections if d.class_name in LIFE_CLASSES]

                all_detections.extend(frame_detections)

                if self._display:
                    self._draw_detections(frame, frame_detections)
                    cv2.imshow("Disaster Response Vision", frame)
                    if cv2.waitKey(self._frame_delay_ms) & 0xFF == ord("q"):
                        break

                if early_exit_on_life and Detector.has_life_signs(frame_detections):
                    life_detected = True
                    for _ in range(self._post_detection_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_detections = self._predict_frame(frame)
                        all_detections.extend(frame_detections)
                        if self._display:
                            self._draw_detections(frame, frame_detections)
                            cv2.imshow("Disaster Response Vision", frame)
                            if cv2.waitKey(self._frame_delay_ms) & 0xFF == ord("q"):
                                break
                    break

        except KeyboardInterrupt:
            logger.info("Video processing interrupted by user.")

        finally:
            cap.release()
            if self._display:
                cv2.destroyAllWindows()

        logger.info(
            "Processed video %s — %d detection(s), life_detected=%s",
            video_source,
            len(all_detections),
            life_detected,
        )
        return all_detections
