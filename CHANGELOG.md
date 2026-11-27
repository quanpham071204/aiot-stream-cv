# Changelog

## Unreleased – branch `quan/analyzers` (2026-11-22)

### Added
- Analyzer framework: analyzers switched on/off per source in config; failures reported as
  `status: error`, never as 0 %. See [docs/analyzers.md](docs/analyzers.md).
- `foam_coverage`: % of the tank surface covered by foam (perspective ROI, HSV presets white/brown/
  grey/shadow, water-level calibration from start-up frames or a clean-tank photo, local brightness
  filters, smoothing) in `/observations`.
- `overflow`: change detection in a check zone outside the tank wall with hysteresis, confirmation
  and cooldown; `overflow` events with snapshot and clip.
- YOLO in its own thread (`model.async: true`) and `model.enabled: false`; model warm-up before the
  stream starts (the first GPU call took 8.4 s).
- `scripts/pick_roi.py` (pick regions on a camera frame, print YAML), `scripts/foam_eval.py` and
  `scripts/foam_rois.json` (accuracy on the advisor's 29 photos: MAE 8.6 points).
- `foreign_object` events: zones take `event_type: foreign_object` (default `zone_intrusion`); `/zones`
  shows it. Example config has a tank-surface zone.
- `analyzer_ms` and `analyzer_errors` in metrics. 158 tests in `tests/` (this base had no tests; the 16 older
  pipeline tests are being restored separately).

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
