"""Train the floating-waste detector (what was run on 27/11/2026, RTX 3060 laptop 6 GB).

Dataset: Mendeley j26w4m645z v2, "Floating Waste and Aquatic Vegetation Image Dataset from Turbid Rivers",
CC BY 4.0, 1,577 images, YOLO labels, classes floating_waste / river_vegetation.
  zip: https://data.mendeley.com/public-files/datasets/j26w4m645z/files/fe81eec3-65c8-48c3-809b-9a9a3a359c82/file_downloaded
  1,758,122,784 bytes, sha256 e43166dde54f82fa9dc00adc5f3943ea7a9083f4beda1490a774fcea6dc7d8b3
Unzip, rename the folder "Floating Waste and River Vegetation Dataset" to fw, pass its parent as --data-root.

  python models/rac-noi/train.py --data-root D:/datasets            # step 1: n, 640, 30 epochs (~15 min)
  python models/rac-noi/train.py --data-root D:/datasets --step 2   # step 2: continue at 1280 (~45 min)
  python models/rac-noi/train.py --data-root D:/datasets --step 3 --data my_tank.yaml   # tank frames (to do)
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True, help="folder that contains fw/ (train, val, test)")
ap.add_argument("--step", type=int, default=1, choices=[1, 2, 3])
ap.add_argument("--data", help="data yaml for step 3 (frames from the demo tank, same class names)")
a = ap.parse_args()

yaml_path = Path(a.data_root) / "fw.yaml"
yaml_path.write_text(f"path: {Path(a.data_root, 'fw').as_posix()}\ntrain: train/images\nval: val/images\n"
                     "test: test/images\nnames: ['floating_waste', 'river_vegetation']\n", encoding="utf-8")
out = dict(project=str(HERE / "runs"), exist_ok=True, workers=4, device=0, plots=True)

if a.step == 1:   # val mAP50 0.417 (at 640) -> rac-noi_n640.pt
    YOLO("yolov8n.pt").train(data=str(yaml_path), epochs=30, imgsz=640, batch=16, name="n640", **out)
elif a.step == 2:  # val mAP50 0.307 (at 1280) -> rac-noi_n1280.pt
    YOLO(str(HERE / "rac-noi_n640.pt")).train(data=str(yaml_path), epochs=20, time=0.75, imgsz=1280, batch=8,
                                              lr0=0.002, warmup_epochs=1, name="n1280ft", **out)
else:              # recommended next step: fine-tune on 100-200 labelled frames from the demo tank
    if not a.data:
        ap.error("--step 3 needs --data <tank yaml>")
    YOLO(str(HERE / "rac-noi_n640.pt")).train(data=a.data, epochs=50, imgsz=640, batch=16, lr0=0.002,
                                              name="tank", **out)
