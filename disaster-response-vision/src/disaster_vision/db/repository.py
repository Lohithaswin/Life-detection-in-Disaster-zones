"""Single data-access layer for detection and media records."""

from __future__ import annotations

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from disaster_vision.config import Settings, get_settings
from disaster_vision.db.models import Base, DetectionRecord, MediaRecord

logger = logging.getLogger(__name__)


class DetectionRepository:
    """Persist and query detection results via SQLAlchemy."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        connect_args: dict = {}
        if self._settings.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        self._engine: Engine = create_engine(
            self._settings.database_url,
            echo=False,
            future=True,
            connect_args=connect_args,
        )
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    def create_tables(self) -> None:
        """Create database tables if they do not exist (Alembic in Phase 3)."""
        Base.metadata.create_all(self._engine)
        logger.debug("Ensured database tables exist.")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provide a transactional database session."""
        db = self._session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            logger.exception("Database transaction rolled back.")
            raise
        finally:
            db.close()

    def insert_media(self, media_type: str, file_name: str, file_path: str) -> MediaRecord:
        """Insert a media processing record."""
        with self.session() as db:
            record = MediaRecord(media_type=media_type, file_name=file_name, file_path=file_path)
            db.add(record)
            db.flush()
            logger.info("Inserted %s record: %s", media_type, file_name)
            return record

    def insert_detection(
        self,
        class_name: str,
        confidence: float,
        source_path: str,
        model_name: str,
        *,
        media_id: int | None = None,
    ) -> DetectionRecord:
        """Insert a single detection result."""
        with self.session() as db:
            record = DetectionRecord(
                media_id=media_id,
                class_name=class_name,
                confidence=confidence,
                source_path=source_path,
                model_name=model_name,
            )
            db.add(record)
            db.flush()
            logger.info(
                "Inserted detection: %s (%.2f) from %s",
                class_name,
                confidence,
                source_path,
            )
            return record

    def insert_detections_from_results(
        self,
        detections: Sequence,
        source_path: str,
        model_name: str,
        *,
        media_id: int | None = None,
    ) -> list[DetectionRecord]:
        """Bulk-insert detections from Detector output objects."""
        from disaster_vision.detection.detector import Detection

        records: list[DetectionRecord] = []
        with self.session() as db:
            for item in detections:
                if not isinstance(item, Detection):
                    continue
                record = DetectionRecord(
                    media_id=media_id,
                    class_name=item.class_name,
                    confidence=item.confidence,
                    source_path=source_path,
                    model_name=model_name,
                )
                db.add(record)
                records.append(record)
            db.flush()
        logger.info("Inserted %d detection record(s) for %s", len(records), source_path)
        return records

    def list_detections(self, limit: int = 100) -> list[DetectionRecord]:
        """Return recent detection records."""
        with self.session() as db:
            stmt = select(DetectionRecord).order_by(DetectionRecord.detected_at.desc()).limit(limit)
            return list(db.scalars(stmt).all())

    def list_media(self, media_type: str | None = None, limit: int = 100) -> list[MediaRecord]:
        """Return recent media records, optionally filtered by type."""
        with self.session() as db:
            stmt = select(MediaRecord).order_by(MediaRecord.created_at.desc()).limit(limit)
            if media_type:
                stmt = stmt.where(MediaRecord.media_type == media_type)
            return list(db.scalars(stmt).all())
