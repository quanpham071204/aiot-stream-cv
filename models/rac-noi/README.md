# Model rác nổi (thử nghiệm, 27/11/2026)

Bản thử của Quân để chạy luồng `foreign_object` trước khi có model của Đạt. Đây là mốc so sánh, chưa phải model cho demo.

## File

| File | Là gì |
|---|---|
| `rac-noi_n640.pt` | YOLOv8n, train 30 epoch ở ảnh 640. Val mAP50 0,417 (đo ở 640). **Đang dùng mặc định** |
| `rac-noi_n1280.pt` | Train tiếp từ bản trên ở ảnh 1280, 15 epoch. Val mAP50 0,307 (đo ở 1280, không so thẳng với số trên) |
| `train.py` | Lệnh train cả 2 bước + bước 3 (train trên khung hình hộp bể, chưa làm). Trong file có link và sha256 của bộ ảnh |
| `config.river.example.yaml` | Cấu hình chạy pipeline trên các clip sông trong `data/` |
| `results_*.csv/png` | Đường loss/mAP theo epoch |
| `so-sanh.jpg`, `events_montage.jpg` | Ảnh so 2 model; 10 sự kiện pipeline tạo ra |

Lớp: `floating_waste`, `river_vegetation`. Trong cấu hình vùng mặt bể, để cả hai lớp vào `classes` thì vật nổi nào cũng báo.

## Bộ ảnh

Mendeley j26w4m645z v2, CC BY 4.0, 1.577 ảnh sông đục, chụp từ bờ. Đã quét người/thuyền bằng YOLO COCO: 28 ảnh bị gắn cờ, xem tay đều là bóng người chụp hoặc bóng nhà phản chiếu, nên không lọc.

## Kết quả trên clip (`data/river_*.mp4`)

Đếm khung bao từng khung hình, ngưỡng 0,35:

| Clip | n640 | n1280 |
|---|---|---|
| `river_lake` hồ rác vụn, nhìn chéo xuống (giống bể nhất) | 151 hộp / 133 trên 329 khung | 107 / 95 |
| `river_murky` nước đục, có 1 vật nhỏ ở mép phải | 40 / 40 trên 600 khung | 0 |
| `river_canal` kênh nhìn từ bờ | 387 / 184 trên 250 khung | 484 / 230 |

Pipeline thật (vùng mặt nước, `min_dwell_s: 1`): n640 ra 10 sự kiện, n1280 ra 7. Clip hồ ra 0 sự kiện với cả hai, vì vật bắt được lúc có lúc không, không giữ đủ 1 giây.

YOLO gốc COCO (`yolov8n.pt`) trên cùng clip: chỉ 1/3 clip Mixkit ra "bottle" (độ tin cậy trung vị 0,26), clip khác gọi rác là "apple".

## Việc tiếp theo (gợi ý cho Đạt)

1. Quay hộp bể demo từ góc camera thật, thả chai, cốc, túi, xốp.
2. Gán nhãn 100–200 khung với cùng tên lớp (Roboflow hoặc CVAT, xuất YOLO).
3. `python models/rac-noi/train.py --data-root <...> --step 3 --data <tank.yaml>`.
4. Chạy `config.river.example.yaml` (đổi `uri` sang video hộp bể) để xem sự kiện.

## Nguồn clip

- Mixkit 25556, 25549, 25541: Mixkit Stock Video Free License.
- Pexels 9736659 (`river_lake`), 10394114 (`river_murky`): Pexels License.
- Pixabay 41849 (`river_canal`, cắt 10 s đầu), 305743 (`river_bottle`, cắt 15 s): Pixabay Content License.

`*.pt` và `data/*.mp4` không vào git. Các file này đang nằm trong worktree của phiên Claude (`.claude/worktrees/ello-866c1a/...`). Dọn worktree là mất, nên muốn chia sẻ thì gửi tay hoặc qua Drive.
