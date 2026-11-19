# Implementation status

Updated 2026-11-19. This repository contains application code and operating documentation.
Automated test suites, evaluation tools and detailed validation artifacts have been removed from
this distribution to keep it focused on running the application.

## Available features

- Webcam, RTSP/HTTP and video-file ingestion with bounded buffers, reconnect and source health.
- Pretrained YOLOv8n + ByteTrack person tracking.
- Restricted-zone dwell rules and duplicate suppression, including re-entry after long frame gaps.
- Annotated JPEG evidence and pre/post-event clips; FFmpeg H.264 with a download fallback.
- Persistent JSONL history, camera/type/time filters, pagination and CSV/JSON export.
- Evidence lifecycle metadata and optional age/size retention; event history is preserved.
- Read APIs for health, observations, zones, metrics, snapshots, history and evidence.
- Sibling frontend provides camera details, live/demo modes, history and local file validation.

## Known limits and remaining work

- Detection accuracy has not been measured against independently labelled video.
- CPU/GPU and physical-camera capacity require longer measurements on deployment hardware.
- Camera sources and zone polygons are configured in YAML; changes require a CV restart.
- History/export scans JSONL and is intended for the prototype scale.
- Authentication, permissions, device management, sensor ingestion and workflow are later phases.
- PPE and line-crossing detection are not implemented.

See [the data contract](data-contract.md) for API fields and storage behavior, and the project
README for installation and startup instructions.
