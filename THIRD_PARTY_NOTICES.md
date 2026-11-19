# Third-party notices

This project uses the following components. They are not redistributed in this repository unless
stated; they are installed with `pip` or downloaded by `scripts/download_assets.py`.

## Software

| Component | Use | License |
|---|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8, ByteTrack tracker) | detection and tracking | AGPL-3.0 |
| [OpenCV](https://github.com/opencv/opencv) (`opencv-python`) | capture, drawing, video encoding | Apache-2.0 |
| [NumPy](https://numpy.org) | arrays | BSD-3-Clause |
| [PyYAML](https://github.com/yaml/pyyaml) | configuration | MIT |
| [lap](https://github.com/gatagat/lap) | assignment solver used by ByteTrack | BSD-2-Clause |
| [PyTorch](https://pytorch.org) | inference backend (pulled in by Ultralytics) | BSD-3-Clause |

ByteTrack was published by Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box" (ECCV 2022); this project uses the Ultralytics implementation.

**AGPL-3.0 note:** Ultralytics is licensed under AGPL-3.0 (or a commercial Ultralytics license).
Distributing this software, or offering it as a network service, must respect those terms.

## Model weights

| File | Source | License |
|---|---|---|
| `yolov8n.pt` | Ultralytics release assets, pretrained on MS COCO | AGPL-3.0 (Ultralytics); COCO annotations are CC BY 4.0 |

## Sample data

| File | Source | Notes |
|---|---|---|
| `data/vtest.avi` | [OpenCV sample data](https://github.com/opencv/opencv/tree/4.x/samples/data) | pedestrian scene, 768x576, 10 fps; distributed with OpenCV (Apache-2.0); no separate license is stated for the clip; used for testing only |
