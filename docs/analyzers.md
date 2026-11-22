# Analyzers: foam coverage and overflow (wastewater tank, phase 1)

Branch `quan/analyzers`. Two OpenCV analyzers run on every processed frame, next to YOLO
(object detection/tracking for foreign objects). They need no model and take 1–5 ms per frame.

| Analyzer | Output | Alert |
|---|---|---|
| `foam_coverage` | % of the tank surface covered by foam, every frame (`/observations`) | none in phase 1 |
| `overflow` | changed ratio of a check zone outside the tank wall | one `overflow` event per spill, with snapshot + clip |

Foreign objects use the existing YOLO path (`zone_intrusion` events, `tracks` in observations); the
team's own model plugs in through `model.weights`. With `model.async: true` YOLO runs in its own
thread, so a slow model never slows down the two analyzers.

The JSON fields are listed in [data-contract.md](data-contract.md) (sections *Observation*, *Event
types* and *Metrics*).

## Set-up for a camera (demo tank)

1. Fix the camera: the regions below are pixel coordinates, so the camera must not move. Plug the laptop
   in, use a high-performance power plan and turn sleep off (in one test YOLO on the CPU went from 34 to
   87 ms per frame while the battery was at 10 %, and the laptop slept during another run).
2. Pick the regions on a live frame and get the YAML:

   ```
   python scripts/pick_roi.py --source 0 --clean calib/clean.jpg
   ```

   Click the 4 corners of the tank surface (inside the wall), press N, click the corners of the
   overflow check zone (a dark drip tray outside the wall), press Enter. With `--clean` the grabbed
   frame is also saved as the clean-tank photo and the YAML uses it for calibration. Start this on a
   calm, foam-free tank. Without a window: `--image frame.jpg --points "x,y x,y x,y x,y" --overflow "..."`.
3. Paste the YAML under the source in the config (see `config.analyzers.example.yaml`) and run
   `python run.py --config <file> --show`.

## Configuration

```yaml
analyzers:
  foam_coverage:
    enabled: true
    roi: [[120, 80], [520, 80], [560, 400], [80, 400]]  # tank surface, any corner order, convex
    colors: [white, brown, grey, shadow]  # HSV presets; grey/shadow also match concrete -> tight ROI
    calibrate: first_frames     # or a path to a clean-tank photo from this camera; remove = off
    calibrate_frames: 10
    darker_than_local: 10       # drop water cells darker than their surroundings
    brighter_than_local: 8      # add dull foam veins brighter than their surroundings
    kernel_px: 5                # speckle filter; 3 for a far camera with thin foam lines
    smooth_alpha: 0.3           # exponential smoothing of the reported %
  overflow:
    enabled: true
    zone_id: tray_front
    roi: [[80, 410], [560, 410], [560, 470], [80, 470]]  # polygon, 3+ points
    diff_thresh: 30             # grey-level change that counts as "changed"
    on_ratio: 0.15              # alarm when >= 15 % of the zone changed ...
    off_ratio: 0.05             # ... for confirm_frames frames; clears below 5 %
    confirm_frames: 3
    cooldown_s: 30              # at most one event per 30 s
```

Unknown analyzer names or option names stop the service at start-up with a clear error.

## How it works

**Foam coverage.** The tank surface is warped to a top-down rectangle (railings and sky are cut away).
A pixel is foam if its colour matches a preset (`white`, `brown`, `grey`, `shadow`) or, after
calibration, if it is at least `brighter_than_water` (40) grey levels brighter than the water. The
water level is learnt at start-up (`first_frames`) or from a clean-tank photo: the median brightness of
the pixels that do not look like white foam. Two local filters compare each pixel with its
neighbourhood: brown/grey water between foam is dropped, dull grey-brown foam veins are added. The
result is cleaned with open/close and smoothed. While calibrating, the status is `calibrating` and no
% is reported.

**Overflow.** The empty check zone is learnt from the first frames. A pixel counts as changed when its
grey level differs from that reference by more than `diff_thresh`. The alarm needs `confirm_frames`
frames in a row above `on_ratio`, clears below `off_ratio` (hysteresis), and fires at most once per
`cooldown_s`; a spill that starts during the cooldown is reported when the cooldown ends. The reference
follows slow lighting drift only on unchanged pixels, so a slowly creeping spill is not absorbed.

## Accuracy (what was measured)

- 29 reference photos from the advisor, one tank ROI each (`scripts/foam_rois.json`):
  mean absolute error **8.6 points** against a visual estimate (Gemini, T056); two visual estimates
  of the same photos differ by 6.9 points. Run: `python scripts/foam_eval.py <dirs> --colors white
  brown grey shadow --rois scripts/foam_rois.json --darker-than-local 10 --brighter-than-local 8`.
- Pexels drone videos (moving camera, not a fair test): tank without foam 2 %, aeration tank 29 %
  (by eye ~30 %).
- Synthetic demo video: one `overflow` event per spill, evidence image + clip.
- 154 tests (`pytest -q`), including three rounds of cross-tests by other AIs.

Not verified yet: a real webcam on the demo tank; the foreign-object model.

## Known limits

- The camera must be fixed. A moved camera means wrong regions (on a drone video the ROI slid onto
  concrete and read 73 %).
- Calibrate on a calm tank. Dull or brown foam present at calibration is taken as water.
- After calibration, a uniform lighting jump (sun comes out) reads as foam: calibrate again
  (restart). Without calibration, water brighter than V=200 cannot be told from foam.
- Hard cases on the photos: golden bubble foam (`sh-bot-nau-vang-09`), grey-brown foam the colour of
  the water (`SH-bot-den-xam-05`), sun glare (`SH-bot-nau-ben`). These need a segmentation model trained
  on hand-labelled masks.
- Overflow treats a slow, uniform brightening of the whole zone as lighting drift.
