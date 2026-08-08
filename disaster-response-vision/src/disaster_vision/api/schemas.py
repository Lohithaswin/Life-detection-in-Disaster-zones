"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str


class BoundingBoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionSchema(BaseModel):
    class_name: str
    confidence: float
    bbox: BoundingBoxSchema


class DetectResponse(BaseModel):
    source: str
    model: str
    detections: list[DetectionSchema]
    life_detected: bool
    alert_sent: bool = False


class DetectionRecordSchema(BaseModel):
    id: int
    class_name: str
    confidence: float
    source_path: str
    model_name: str
    detected_at: datetime
    media_id: int | None = None

    model_config = {"from_attributes": True}


class DetectionsListResponse(BaseModel):
    count: int
    items: list[DetectionRecordSchema]


class DetectOptions(BaseModel):
    model: str = Field(default="yolov8n", description="YOLO model name")
    life_only: bool = Field(default=False, description="Return only person/animal classes")
    send_alert: bool = Field(default=True, description="Send email alert if life signs detected")
    persist: bool = Field(default=True, description="Save results to the database")
