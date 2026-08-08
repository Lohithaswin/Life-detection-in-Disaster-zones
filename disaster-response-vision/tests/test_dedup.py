"""Tests for alert deduplication."""

from disaster_vision.alerts.dedup import AlertDeduplicator, DedupKey


def test_deduplicator_suppresses_repeated_alerts() -> None:
    dedup = AlertDeduplicator(window_seconds=300)
    key = DedupKey(detected_class="person", source_path="/tmp/a.jpg")

    assert dedup.should_send(key) is True
    assert dedup.should_send(key) is False


def test_deduplicator_clear_allows_resend() -> None:
    dedup = AlertDeduplicator(window_seconds=300)
    key = DedupKey(detected_class="dog", source_path="/tmp/b.jpg")

    assert dedup.should_send(key) is True
    dedup.clear()
    assert dedup.should_send(key) is True
