# aiot-stream-cv

Data streaming and computer vision block of an **industrial AI monitoring framework**
(Computer Engineering Project, HCMUT, semester HK261).

It reads live video sources, tracks people with YOLOv8 + ByteTrack, applies a restricted-zone rule
and publishes structured events with image and video evidence through a JSONL log and a small HTTP API.
The current build runs on public sample video in place of factory cameras (the brief's *immediate start*
phase) and is meant to be wired into the group's web dashboard and workflow tools through the API.

```
camera / RTSP / file ──► StreamSource ──► Tracker ──────────► ZoneIntrusionRule ──► EventStore ──► HTTP API :8080
                         timestamps,      YOLOv8n +            dwell time,           events.jsonl   /health  /events
                         frame buffer,    ByteTrack            de-duplication        evidence/*.jpg /metrics /observations
                         reconnect,          │                                       evidence/*.mp4 /snapshot /evidence
                         source health       └──► ClipRecorder (ring buffer → clip per event)
```

## Wastewater tank analyzers (branch `quan/analyzers`)

Foam coverage (% of the tank surface, continuous) and overflow (event with evidence) run as OpenCV
analyzers next to YOLO; set-up, configuration, accuracy and limits are in
[docs/analyzers.md](docs/analyzers.md), the JSON fields in [docs/data-contract.md](docs/data-contract.md).
Quick start: `python scripts/pick_roi.py --source 0 --clean calib/clean.jpg`, paste the printed YAML into
a copy of `config.analyzers.example.yaml`, then `python run.py --config <file> --show`.

## Features

- **Ingest**: webcam, RTSP/HTTP stream or video file behind one interface; capture timestamps; bounded
  buffer that drops the oldest frame when inference falls behind, so results stay close to real time.
- **Resilience**: automatic reconnect with exponential backoff (1 s → 30 s); per-source health
  (state, fps, dropped frames, reconnects, last error); `source_offline` / `source_recovered` /
  `source_ended` events, including sources that never connect.
- **Vision**: YOLOv8n (COCO) + ByteTrack person tracking, class filter and thresholds from config.
- **Rules**: polygon restricted zones; an event fires once per visit after a minimum dwell time, and
  border jitter does not re-fire it.
- **Evidence**: annotated snapshot for every event, plus an `.mp4` clip covering a configurable window
  before and after the event, cut from an in-memory JPEG ring buffer.
- **Observability**: `/metrics` and `output/metrics.json` with throughput, inference time and
  end-to-end latency percentiles, drop rate and event counts.
- **Fails loudly**: an inference error is reported as `status: "error"`, never as "nothing detected".

## Quick start

Requires Python 3.10+. An NVIDIA GPU is optional.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # GPU build; omit for CPU
pip install -r requirements.txt
python scripts/download_assets.py      # sample video + YOLOv8n weights
python run.py --show                   # preview window, press q to quit
```

Other options:

```bash
python run.py --duration 300           # run 5 minutes, then write output/metrics.json
python run.py --config my_site.yaml    # another configuration
python run.py --no-api                 # without the HTTP server
```

All parameters (sources, zones, thresholds, buffer, backoff, clip length, API port) live in
[`config.yaml`](config.yaml).

## Outputs

| Path | Content |
|---|---|
| `output/events.jsonl` | one JSON event per line |
| `output/evidence/<event_id>.jpg` | annotated frame at the moment of the event |
| `output/evidence/<event_id>.mp4` | clip from `clip_pre_s` before to `clip_post_s` after the event |
| `output/metrics.json` | run statistics per source, written on exit |

HTTP API (default `http://127.0.0.1:8080`, CORS enabled):

| Endpoint | Returns |
|---|---|
| `GET /health` | health of every source |
| `GET /events?limit=N` | latest events, newest first, restored across restart |
| `GET /events/history` | filtered, paged persisted history with stable snapshot |
| `GET /events/<event_id>` | event detail and current evidence lifecycle |
| `GET /events/export?format=csv` | complete CSV/JSON export for the selected filters |
| `GET /observations` | current tracks and counts per source |
| `GET /zones` | effective read-only restricted-zone configuration per source |
| `GET /metrics` | live run statistics |
| `GET /snapshot/<source_id>.jpg` | latest annotated frame |
| `GET /evidence/<file>` | evidence image or clip referenced by an event |

The event schema and every field are described in [docs/data-contract.md](docs/data-contract.md).

## Event history and evidence retention

The dashboard can query past sessions from `output/events.jsonl`. Preserve the entire `output/` folder,
including `evidence-metadata/`, to retain evidence states across restarts. New clips are converted to
browser-friendly H.264 when FFmpeg is installed; Docker includes FFmpeg. Without it, clips remain
downloadable and the UI explains playback limitations.

Evidence cleanup is disabled by default. Set positive `events.retention_days` and/or
`events.max_evidence_mb` in the config to enable periodic removal (oldest first, active clips protected).
`events.cleanup_interval_s` defaults to 3600 seconds. Event records remain in history after their images
or clips expire. See `docs/data-contract.md` for filters, pagination, export, statuses and storage limits.

## Project layout

```
aiot/
  stream.py     ingest: capture, timestamps, buffer, reconnect, health
  vision.py     YOLOv8 + ByteTrack -> Track
  rules.py      restricted-zone rule
  evidence.py   ring buffer and evidence clips
  events.py     event store (JSONL + snapshots)
  pipeline.py   per-source worker: stream -> track -> rule -> event, health events, metrics
  api.py        HTTP API
  timeutil.py   ISO-8601 timestamps
scripts/        asset download
docs/           data contract and implementation status
run.py          entry point
config.yaml     sample configuration
```

## Status and limitations

What is implemented, measured and still open is tracked in [docs/status.md](docs/status.md).
The pipeline runs end to end. Detection accuracy (missed events, false alarms, ID switches)
has not been measured because no labelled video is available.

## Licensing

This repository has no license file yet. It depends on Ultralytics YOLOv8, which is licensed under
**AGPL-3.0**; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for all third-party components and
the sample data.
