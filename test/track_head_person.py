"""对视频中的人(person)和人头(head)进行多目标跟踪。

使用自训练的 YOLO 检测模型 + ByteTrack 跟踪器,输出:
  - 带跟踪框和 track ID 的可视化视频
  - 每帧每个目标的跟踪结果 CSV(frame, track_id, class, conf, x1, y1, x2, y2)

用法:
  python test/track_head_person.py \
    --source data/1.10/clipleft/clipped_testdata2.mp4 \
    --model test/model/trackheadpeople.pt \
    --output test/output/clipped_testdata2
"""

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

# 类别颜色 (BGR):person 绿色,head 橙色
CLASS_COLORS = {
    "person": (80, 200, 60),
    "head": (0, 140, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(description="跟踪视频中的人和人头")
    parser.add_argument("--source", type=str,
                        default="data/1.10/clipleft/clipped_testdata2.mp4",
                        help="输入视频路径")
    parser.add_argument("--model", type=str,
                        default="test/model/trackheadpeople.pt",
                        help="YOLO 模型权重路径")
    parser.add_argument("--output", type=str,
                        default="test/output/clipped_testdata2",
                        help="输出目录")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                        choices=["bytetrack.yaml", "botsort.yaml"],
                        help="跟踪器配置")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="检测置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="推理分辨率")
    parser.add_argument("--device", type=str, default="0",
                        help="推理设备,如 0 或 cpu")
    parser.add_argument("--classes", type=str, default="head,person",
                        help="要跟踪的类别,逗号分隔,如 head 或 head,person")
    return parser.parse_args()


def draw_track(frame, box, track_id, cls_name, conf):
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS.get(cls_name, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{cls_name} #{track_id} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = y1 - 6 if y1 - th - 8 >= 0 else y2 + th + 6
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 3), color, -1)
    cv2.putText(frame, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"找不到输入视频: {source}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    name_to_id = {v: k for k, v in model.names.items()}
    class_ids = [name_to_id[c.strip()] for c in args.classes.split(",")
                 if c.strip() in name_to_id]

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    video_path = out_dir / f"{source.stem}_tracked.mp4"
    csv_path = out_dir / f"{source.stem}_tracks.csv"
    writer = cv2.VideoWriter(str(video_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    results = model.track(
        source=str(source),
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        classes=class_ids,
        persist=True,
        stream=True,
        verbose=False,
    )

    track_ids_seen = {name: set() for name in model.names.values()}

    with open(csv_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["frame", "track_id", "class", "conf",
                             "x1", "y1", "x2", "y2"])

        for frame_idx, result in enumerate(results):
            frame = result.orig_img
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                clss = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                for box, tid, cid, conf in zip(xyxy, ids, clss, confs):
                    cls_name = model.names[cid]
                    track_ids_seen[cls_name].add(tid)
                    draw_track(frame, box, tid, cls_name, conf)
                    csv_writer.writerow(
                        [frame_idx, tid, cls_name, f"{conf:.4f}",
                         f"{box[0]:.1f}", f"{box[1]:.1f}",
                         f"{box[2]:.1f}", f"{box[3]:.1f}"])

            cv2.putText(frame, f"frame {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
            writer.write(frame)

            if frame_idx % 100 == 0:
                print(f"进度: {frame_idx}/{total}")

    writer.release()

    print(f"\n完成,共处理 {total} 帧")
    for cls_name, ids in track_ids_seen.items():
        if ids:
            print(f"  {cls_name}: {len(ids)} 个跟踪目标")
    print(f"输出视频: {video_path}")
    print(f"跟踪数据: {csv_path}")


if __name__ == "__main__":
    main()
