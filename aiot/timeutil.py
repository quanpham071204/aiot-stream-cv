from datetime import datetime, timezone


def iso(ts: float) -> str:
    """Epoch seconds -> ISO-8601 in local time with explicit UTC offset."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="milliseconds")
