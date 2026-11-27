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
| `foreign_object` | same rule on a zone configured with `event_type: foreign_object` (the tank surface): a detected object of the zone's `classes` stays on it for `min_dwell_s`; once per object visit | same as `zone_intrusion` |
| `source_offline` | a running source delivers no frame for `stale_after_s`, or a source has not delivered any frame `offline_grace_s` after start | `state`, `last_frame_age_s`, `reconnects`, `last_error` |
| `source_recovered` | an offline source delivers frames again | same as above |
| `source_ended` | a video file reached its end (`loop: false`) | same as above |
| `overflow` | the overflow check zone stays changed (`on_ratio`, `confirm_frames`); at most one per `cooldown_s` ([analyzers](analyzers.md)) | `zone_id`, `changed_ratio` (0–1, share of the zone that changed), `on_ratio` |

Example `overflow` result: `{"zone_id": "tray_front", "changed_ratio": 1.0, "on_ratio": 0.15}`, with
`evidence_path` and `evidence_clip` like other alerts.

Health events carry no evidence (`null`).

## History, detail and export (P2)

The JSONL log is append-only and remains the source of truth. Recent events are reloaded on startup;
full history queries scan the file, so they are not limited to the 500-event recent cache. Invalid or
crash-truncated lines are logged and skipped. A later append starts on a fresh line. Existing log files
without lifecycle metadata remain readable. Timestamps are compared as instants, including UTC offsets.

- `GET /events?limit=N`: backward-compatible newest-first array, `1 <= N <= 500`, default 50.
- `GET /events/history`: paged history, newest occurrence first (then creation time and event ID).
- `GET /events/{event_id}`: one event with current evidence status, or 404.
- `GET /events/export?format=csv|json`: all matching events, including events outside the current page.

History/export filters: `source_id`, `event_type`, `from`, `to`. Date boundaries are inclusive ISO-8601
instants with a timezone; invalid timestamps and reversed intervals return 400. History supports
`limit` (1–200, default 20), `offset` (>=0), and optional `snapshot` (>=0).

History response:

```json
{
  "items": [], "total": 0, "limit": 20, "offset": 0, "snapshot": 0,
  "source_ids": [], "event_types": []
}
```

The `snapshot` is the number of valid append-only log records visible at the initial query. Pass it
back on subsequent pages and export to exclude newer appended events and prevent shifting pages.
Omit it to refresh. Facets include all source IDs/event types within that snapshot, including cameras
no longer configured. A snapshot pins event membership, not changing evidence status. Do not externally
rewrite/rotate the log while browsing a snapshot. Exports ignore pagination and preserve filters/snapshot;
CSV is UTF-8 with BOM, proper quoting, and spreadsheet formula prefixes neutralized.

API events add optional `image_status`, `clip_status`, `clip_error`, and `clip_playable`:

| Status | Meaning |
|---|---|
| `none` | No evidence was requested for this kind (e.g. a source-health event) |
| `pending` | The current process is still collecting or encoding the clip |
| `ready` | The final file exists and can be downloaded |
| `failed` | Snapshot/clip encoding failed; clip details may be in `clip_error` |
| `expired` | Retention deliberately removed the file |
| `missing` | A referenced file is absent, including legacy evidence without metadata |
| `interrupted` | A previous process stopped before finalizing a pending clip |

Lifecycle metadata is stored atomically under `output/evidence-metadata/`, separately from the original
JSONL contract. Preserve this directory with the log/evidence across restarts. Clips are written to
private temporary names and published at the final path after encoding. Even if a camera stops sending
frames, its pending clip finishes with available frames after the post-event window. Failed writes are
reported rather than left permanently pending. A crash during writing becomes `interrupted` on restart.

New clips use FFmpeg H.264/yuv420p with `faststart` when FFmpeg/libx264 is available (`clip_playable: true`).
The Docker image includes FFmpeg. Without it, or if conversion fails, a downloadable MPEG-4 clip is kept
with `clip_playable: false`; the UI offers a download fallback. Older clips have unknown playability and
may require an external player. `ready` describes a final file, not guaranteed browser codec support.

`GET /evidence/{file}` supports single byte ranges (`206`, `Content-Range`, `Accept-Ranges`), invalid ranges
return 416, and `?download=1` sets an attachment disposition. Missing files return 404. Clients must show
an image/player error independently of JSON API health.

Retention is opt-in through the `events` configuration:

```yaml
events:
  retention_days: null       # positive days, or null to disable age retention
  max_evidence_mb: null      # positive MiB, or null to disable size retention
  cleanup_interval_s: 3600
```

At startup and periodically, referenced evidence exceeding age limits is removed; if over the size
budget, oldest referenced files are removed first. Active clips and their images are protected. This
is a periodic budget, not a hard disk quota: active/temp/unrelated files may exceed it. Unrelated files
and the event log are never deleted; removed evidence retains a durable `expired` status. History and
metadata continue to grow. Large deployments should move querying/indexing and long-term storage to
the planned management backend rather than rely on full JSONL scans/exports in memory.

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

Before the first inference result (including a source that never connects), the entry is
`{"source_id": "cam01", "status": "waiting"}`. This is unknown, not zero detections.
Older service versions may return `null` for this state; clients should accept both forms.
`waiting` does not mean the camera is online: use `/health` for camera connectivity.

If inference fails, the entry is `{"status": "error", "error": "...", ...}`. Consumers must treat
this as "unknown", not as "no detections".

### Analyzer measurements

When analyzers are configured ([analyzers](analyzers.md)), each observation also has `analyzers`
(one entry per analyzer) and a flat copy `foam_coverage` (the field the dashboard reads):

```json
"analyzers": {
  "foam_coverage": {"status": "ok", "foam_coverage_pct": 94.3, "raw_pct": 96.6,
                    "calibrated": true, "updated_at": "2026-11-22T12:30:03.702+07:00"},
  "overflow": {"status": "ok", "zone_id": "tray_front", "active": false, "changed_ratio": 0.0}
},
"foam_coverage": {"status": "ok", "foam_coverage_pct": 94.3, "raw_pct": 96.6,
                  "calibrated": true, "updated_at": "2026-11-22T12:30:03.702+07:00"}
```

| `status` | Meaning | Show |
|---|---|---|
| `ok` | value is valid | `foam_coverage_pct` (smoothed, 0–100, one decimal); `raw_pct` is this frame only |
| `calibrating` | `foam_coverage` is learning the water level after start-up; `frames_left` frames to go, no % yet | "calibrating" |
| `learning` | `overflow` is learning the empty check zone; no `active` field yet | "learning" |
| `error` | the analyzer failed on this frame; `error` has the message | "unknown", never 0 % |

`overflow.active` is `true` while the zone is in alarm. `foam_coverage` is absent when the analyzer is
not configured. Consumers must treat a missing value or a non-`ok` status as unknown, not as 0 %.

## Restricted zones

`GET /zones` returns the effective read-only rule configuration, including defaults applied at startup:

```json
{
  "time": "2026-11-18T20:00:00.000+07:00",
  "sources": [
    {
      "source_id": "cam01",
      "zones": [
        {
          "zone_id": "restricted_a",
          "event_type": "zone_intrusion",
          "polygon": [[380, 215], [560, 215], [580, 330], [360, 330]],
          "min_dwell_s": 1.0,
          "lost_timeout_s": 2.0,
          "classes": ["person"]
        }
      ]
    }
  ]
}
```

Every configured source is included, even offline sources and sources with `zones: []`.
`event_type` is the event the zone raises: `zone_intrusion` (default) or `foreign_object`.
Polygon coordinates are pixels in the original frame. `classes: null` means all tracked classes.
`min_dwell_s` is the time inside a zone before an intrusion event; `lost_timeout_s` is the grace
period outside/unseen before a visit is considered ended. Merely being in a zone is not evidence
that the dwell threshold has been reached. Current occupancy comes from `observations.tracks[].zones`;
count people once per source even if multiple zones overlap. Only use fresh successful observations
from healthy sources for current occupancy. Track IDs are local to the source/tracker session and
may change after lost tracking or process restart.

This endpoint does not edit configuration. Change the YAML configuration and restart the CV service
to apply changes. It contains no camera connection credentials.

## Metrics

`GET /metrics` and `output/metrics.json` (written on exit), one entry per source:
`uptime_s`, `frames_received`, `frames_processed`, `frames_dropped`, `drop_rate`, `input_fps`,
`processing_fps`, `infer_ms` and `latency_ms` (each with `p50`, `p95`, `max`), `analyzer_ms`
(per analyzer, same percentiles), `analyzer_errors` (count per analyzer), `reconnects`,
`inference_errors`, `events` (count per type), `clips_written`, `clips_failed`.
Percentiles cover the last 1000 processed frames.

## Clock

`measured_at` / `occurred_at` is the time the frame was grabbed by this process, because the
cameras do not provide a trusted clock. The same convention (`source_id`, `measured_at`,
`received_at`, `quality`) is intended for sensor sources so both streams can be aligned.
