# Status

Last updated 2026-10-22, version 0.2.0.

## Implemented

| Area | State | Code |
|---|---|---|
| Webcam / RTSP / file ingest with timestamps | done | `aiot/stream.py` |
| Bounded buffer, drop-oldest when inference is slow | done | `aiot/stream.py` |
| Reconnect with exponential backoff (1 s → 30 s) | done | `aiot/stream.py` |
| Per-source health and offline / recovered / ended events | done | `aiot/pipeline.py` |
| Person detection and tracking (YOLOv8n COCO + ByteTrack) | done, pretrained, not fine-tuned | `aiot/vision.py` |
| Restricted-zone rule with dwell time and de-duplication | done | `aiot/rules.py` |
| Snapshot evidence per event | done | `aiot/events.py` |
| Clip evidence (5 s before + 5 s after) from a ring buffer | done | `aiot/evidence.py` |
| JSONL event log and HTTP API | done | `aiot/events.py`, `aiot/api.py` |
| Run metrics (fps, latency percentiles, drop rate) | done | `aiot/pipeline.py`, `run.py` |
| Automated tests | 16 passing | `tests/` |

## Reference run

60 s on `data/vtest.avi` (768x576, 10 fps, paced in real time) on a **CPU-only** machine, with one
unreachable RTSP source configured to exercise reconnect. Raw files: [`sample-run/`](sample-run/).

| Metric | cam01 (vtest.avi) |
|---|---|
| Frames received / processed / dropped | 602 / 590 / 11 (1.8 %) |
| Processing rate | 10.1 fps (equal to the source rate) |
| Inference time p50 / p95 | 46.9 / 67.9 ms |
| Frame-to-result latency p50 / p95 | 49.6 / 77.0 ms |
| `zone_intrusion` events | 23, each with snapshot and clip (23/23 clips written) |
| Unreachable source | 1 `source_offline`, 6 reconnect attempts, no repeated alerts |

How to read these numbers:

- They come from a single short run on one public video and are indicative only.
- Throughput is capped by the source frame rate, so 10 fps is not the maximum the pipeline can reach.
- The maximum latency (~1.5 s) and most dropped frames come from model warm-up on the first frame.
- The zone polygon is illustrative, so the event count is the number of people crossing it,
  not an accuracy figure.
- GPU figures (target hardware: RTX 3060) still have to be recorded with
  `python run.py --duration 300`.

## Open items

| Item | Needs |
|---|---|
| Accuracy: missed events, false alarms, ID switches | labelled video (ground truth) |
| GPU / CPU load and 2–4 concurrent cameras | runs on the target machine |
| Integration with the dashboard and workflow (n8n / Node-RED via `/events`) | owners of those components |
| Agreement on the data contract with the rest of the team | review of [data-contract.md](data-contract.md) |
| Industrial scenario (PPE or a real restricted area) | choice of pilot scenario |
| Line-crossing counter | depends on the pilot scenario |
| H.264 clips for in-browser playback | only if the dashboard plays clips inline |
| Offline packaging and simulated sensor source (MQTT / CSV) | if assigned |
