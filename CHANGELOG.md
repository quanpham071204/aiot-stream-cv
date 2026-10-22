# Changelog

## 0.2.0 – 2026-10-22

### Added
- Evidence clips: an in-memory JPEG ring buffer and an `.mp4` clip per event
  (`clip_pre_s` before, `clip_post_s` after); events carry `evidence_clip`.
- Run metrics at `GET /metrics` and in `output/metrics.json`: throughput, inference time and
  latency percentiles, drop rate, event and clip counts.
- `source_ended` event when a video file finishes.
- Pipeline tests for health events, clips and metrics (16 tests in total).
- Data contract, status page, third-party notices, asset download script, CI workflow.

### Fixed
- A source that never delivered a frame (for example an unreachable RTSP camera) was never reported
  as offline. It is now reported once after `offline_grace_s`.

## 0.1.0 – 2026-10-22

### Added
- Stream ingest from webcam, RTSP or file with timestamps, bounded buffer, reconnect with backoff
  and per-source health.
- YOLOv8n + ByteTrack person tracking.
- Restricted-zone rule with dwell time and de-duplication.
- JSONL event store with snapshot evidence and an HTTP API.
