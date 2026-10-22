# Data contract

This block hands data to the rest of the framework (rules/workflow, backend, dashboard) in three
shapes: **events**, **source health** and **observations**. Field names are stable; new fields may be
added, existing ones are not renamed.

Timestamps are ISO-8601 with milliseconds and an explicit UTC offset, e.g.
`2026-10-22T17:35:12.345+07:00`.

## Event

One JSON object per line in `output/events.jsonl`, also returned by `GET /events`.

```json
{
  "event_id": "cam01-20261022T173512-a1b2c3",
  "source_id": "cam01",
  "event_type": "zone_intrusion",
  "occurred_at": "2026-10-22T17:35:12.345+07:00",
  "created_at": "2026-10-22T17:35:12.391+07:00",
  "result": {
    "zone_id": "restricted_a",
    "track_id": 7,
    "class": "person",
    "confidence": 0.81,
    "dwell_s": 1.03,
    "bbox": [412.0, 188.5, 451.2, 290.7]
  },
  "evidence_path": "evidence/cam01-20261022T173512-a1b2c3.jpg",
  "evidence_clip": "evidence/cam01-20261022T173512-a1b2c3.mp4"
}
```

| Field | Type | Meaning |
|---|---|---|
| `event_id` | string | `<source_id>-<local time>-<random>`; unique, also the evidence file name |
| `source_id` | string | source `id` from the config |
| `event_type` | string | see table below |
| `occurred_at` | string | capture time of the frame that triggered the event |
| `created_at` | string | time the event was written |
| `result` | object | type-specific payload |
| `evidence_path` | string or null | annotated snapshot, relative to the output directory |
| `evidence_clip` | string or null | video clip, relative to the output directory; the file is complete `clip_post_s` seconds after `created_at` |

### Event types

| `event_type` | When | `result` |
|---|---|---|
| `zone_intrusion` | a tracked object stays inside a zone for `min_dwell_s`; fires once per visit | `zone_id`, `track_id`, `class`, `confidence`, `dwell_s`, `bbox` (x1, y1, x2, y2 in pixels) |
| `source_offline` | a running source delivers no frame for `stale_after_s`, or a source has not delivered any frame `offline_grace_s` after start | `state`, `last_frame_age_s`, `reconnects`, `last_error` |
| `source_recovered` | an offline source delivers frames again | same as above |
| `source_ended` | a video file reached its end (`loop: false`) | same as above |

Health events carry no evidence (`null`).

## Source health

`GET /health` returns `{"time": ..., "sources": [...]}` with one entry per source:

| Field | Meaning |
|---|---|
| `source_id`, `uri` | identity of the source |
| `state` | `connecting`, `running`, `reconnecting`, `ended` or `stopped` |
| `healthy` | `running` and last frame younger than `stale_after_s` |
| `fps` | input frame rate (exponential moving average) |
| `processing_fps` | frames per second through detection and rules |
| `frames_received`, `frames_dropped`, `buffered` | ingest counters; drops happen when inference is slower than the source |
| `reconnects`, `last_error` | reconnect attempts and the most recent failure |
| `last_frame_age_s` | seconds since the last frame |

## Observation

`GET /observations` returns the latest result per source:

```json
{
  "cam01": {
    "source_id": "cam01",
    "status": "ok",
    "seq": 1532,
    "measured_at": "2026-10-22T17:35:12.345+07:00",
    "counts": {"person": 3},
    "tracks": [
      {"track_id": 7, "class": "person", "confidence": 0.81,
       "bbox": [412.0, 188.5, 451.2, 290.7], "zones": ["restricted_a"]}
    ],
    "infer_ms": 12.4,
    "latency_ms": 15.0
  }
}
```

If inference fails, the entry is `{"status": "error", "error": "...", ...}`. Consumers must treat
this as "unknown", not as "no detections".

## Metrics

`GET /metrics` and `output/metrics.json` (written on exit), one entry per source:
`uptime_s`, `frames_received`, `frames_processed`, `frames_dropped`, `drop_rate`, `input_fps`,
`processing_fps`, `infer_ms` and `latency_ms` (each with `p50`, `p95`, `max`), `reconnects`,
`inference_errors`, `events` (count per type), `clips_written`, `clips_failed`.
Percentiles cover the last 1000 processed frames.

## Clock

`measured_at` / `occurred_at` is the time the frame was grabbed by this process, because the
cameras do not provide a trusted clock. The same convention (`source_id`, `measured_at`,
`received_at`, `quality`) is intended for sensor sources so both streams can be aligned.
