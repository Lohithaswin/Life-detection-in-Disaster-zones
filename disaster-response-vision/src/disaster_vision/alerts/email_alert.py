"""Email alert delivery with optional async send and deduplication."""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from disaster_vision.alerts.dedup import AlertDeduplicator, DedupKey
from disaster_vision.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlertPayload:
    """Structured alert content sent via email."""

    detected_class: str
    confidence: float
    timestamp: datetime
    source_path: str | None = None
    image_reference: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    def to_message_body(self) -> str:
        """Render a plain-text email body from this payload."""
        lines = [
            "Disaster Response Vision — Detection Alert",
            "",
            f"Time (UTC): {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Class: {self.detected_class}",
            f"Confidence: {self.confidence:.2%}",
        ]
        if self.source_path:
            lines.append(f"Source: {self.source_path}")
        if self.image_reference:
            lines.append(f"Detection image: {self.image_reference}")
        if self.latitude is not None and self.longitude is not None:
            lines.append(f"GPS: {self.latitude:.6f}, {self.longitude:.6f}")
        return "\n".join(lines)

    def dedup_key(self) -> DedupKey:
        return DedupKey(detected_class=self.detected_class, source_path=self.source_path)


class EmailAlerter:
    """Send detection alerts via SMTP (sync or async)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        deduplicator: AlertDeduplicator | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._deduplicator = deduplicator or AlertDeduplicator(
            window_seconds=self._settings.alert_dedup_seconds
        )

    def _require(self, value: str, name: str) -> str:
        if not value:
            raise RuntimeError(
                f"Missing required setting: {name}. Copy .env.example to .env and configure."
            )
        return value

    def _build_message(
        self, payload: AlertPayload, *, to_email: str
    ) -> tuple[str, str, MIMEMultipart]:
        sender = self._require(
            self._settings.smtp_user or self._settings.alert_from,
            "SMTP_USER or ALERT_FROM",
        )
        from_address = self._settings.alert_from or sender

        message = MIMEMultipart()
        message["From"] = from_address
        message["To"] = to_email
        message["Subject"] = f"[Disaster Vision] {payload.detected_class} detected"
        message.attach(MIMEText(payload.to_message_body(), "plain"))
        return sender, from_address, message

    def send(
        self, payload: AlertPayload, *, to_email: str | None = None, skip_dedup: bool = False
    ) -> bool:
        """Send an alert email synchronously. Returns True if sent."""
        if not skip_dedup and not self._deduplicator.should_send(payload.dedup_key()):
            return False

        password = self._require(self._settings.smtp_password, "SMTP_PASSWORD")
        recipient = to_email or self._require(self._settings.alert_to, "ALERT_TO")
        sender, from_address, message = self._build_message(payload, to_email=recipient)

        with smtplib.SMTP(self._settings.smtp_host, self._settings.smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(from_address, recipient, message.as_string())

        logger.info("Alert email sent to %s for class=%s", recipient, payload.detected_class)
        return True

    async def send_async(
        self,
        payload: AlertPayload,
        *,
        to_email: str | None = None,
        skip_dedup: bool = False,
    ) -> bool:
        """Send an alert email asynchronously via aiosmtplib."""
        if not skip_dedup and not self._deduplicator.should_send(payload.dedup_key()):
            return False

        try:
            import aiosmtplib
        except ImportError as exc:
            raise RuntimeError("Install aiosmtplib for async email alerts.") from exc

        password = self._require(self._settings.smtp_password, "SMTP_PASSWORD")
        recipient = to_email or self._require(self._settings.alert_to, "ALERT_TO")
        sender, from_address, message = self._build_message(payload, to_email=recipient)

        await aiosmtplib.send(
            message,
            hostname=self._settings.smtp_host,
            port=self._settings.smtp_port,
            username=sender,
            password=password,
            start_tls=True,
            sender=from_address,
            recipients=[recipient],
        )
        logger.info("Async alert email sent to %s for class=%s", recipient, payload.detected_class)
        return True

    def send_test_alert(self) -> None:
        """Send a test alert using configured recipients."""
        payload = AlertPayload(
            detected_class="person",
            confidence=0.95,
            timestamp=datetime.now(timezone.utc),
            source_path="test/smoke-check",
        )
        self.send(payload, skip_dedup=True)


def send_life_detection_alert(
    detections: list,
    source_path: str | None = None,
    *,
    settings: Settings | None = None,
    image_reference: str | None = None,
) -> bool:
    """Send an alert for the highest-confidence life-sign detection."""
    from disaster_vision.detection.detector import LIFE_CLASSES, Detection

    life = [d for d in detections if isinstance(d, Detection) and d.class_name in LIFE_CLASSES]
    if not life:
        return False

    best = max(life, key=lambda d: d.confidence)
    alerter = EmailAlerter(settings=settings)
    return alerter.send(
        AlertPayload(
            detected_class=best.class_name,
            confidence=best.confidence,
            timestamp=datetime.now(timezone.utc),
            source_path=source_path,
            image_reference=image_reference,
        )
    )


async def send_life_detection_alert_async(
    detections: list,
    source_path: str | None = None,
    *,
    settings: Settings | None = None,
    image_reference: str | None = None,
) -> bool:
    """Async variant of send_life_detection_alert."""
    from disaster_vision.detection.detector import LIFE_CLASSES, Detection

    life = [d for d in detections if isinstance(d, Detection) and d.class_name in LIFE_CLASSES]
    if not life:
        return False

    best = max(life, key=lambda d: d.confidence)
    alerter = EmailAlerter(settings=settings)
    return await alerter.send_async(
        AlertPayload(
            detected_class=best.class_name,
            confidence=best.confidence,
            timestamp=datetime.now(timezone.utc),
            source_path=source_path,
            image_reference=image_reference,
        )
    )
