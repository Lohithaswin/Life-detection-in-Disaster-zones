"""Tests for SQLAlchemy repository layer."""

from disaster_vision.config import Settings
from disaster_vision.db.repository import DetectionRepository
from disaster_vision.detection.detector import BoundingBox, Detection


def test_repository_insert_and_list() -> None:
    settings = Settings(database_url="sqlite:///:memory:")
    repo = DetectionRepository(settings)
    repo.create_tables()

    media = repo.insert_media("image", "test.jpg", "/tmp/test.jpg")
    detections = [
        Detection(
            class_name="person",
            confidence=0.91,
            bbox=BoundingBox(x1=1, y1=2, x2=3, y2=4),
        )
    ]
    repo.insert_detections_from_results(
        detections, "/tmp/test.jpg", "yolov8n", media_id=media.id
    )

    rows = repo.list_detections(limit=10)
    assert len(rows) == 1
    assert rows[0].class_name == "person"
    assert rows[0].confidence == 0.91
