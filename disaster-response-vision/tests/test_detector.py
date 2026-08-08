"""Tests for detection wrapper with mocked YOLO."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from disaster_vision.config import Settings
from disaster_vision.detection.detector import Detector


class _FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy: list[float]) -> None:
        self.cls = [cls_id]
        self.conf = [conf]
        xyxy_tensor = MagicMock()
        xyxy_tensor.tolist.return_value = xyxy
        self.xyxy = [xyxy_tensor]


@patch("ultralytics.YOLO")
def test_detector_detect_image_parses_results(mock_yolo: MagicMock, tmp_path: Path) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(b"fake")

    fake_model = MagicMock()
    fake_model.names = {0: "person"}
    fake_model.predict.return_value = [
        SimpleNamespace(boxes=[_FakeBox(0, 0.88, [10, 20, 30, 40])])
    ]
    mock_yolo.return_value = fake_model

    settings = Settings(weights_dir=tmp_path, runs_dir=tmp_path / "runs")
    detector = Detector(model_name="yolov8n", settings=settings)
    results = detector.detect_image(image, save=False)

    assert len(results) == 1
    assert results[0].class_name == "person"
    assert results[0].confidence == 0.88
