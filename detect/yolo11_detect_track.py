"""
YOLOv11 视频检测 + 跟踪（ID）脚本（独立可用）
仅支持：输入视频 -> 输出带检测框+ID的视频

用法:
1) 仅检测:
python detect/yolo11_detect_track.py --source data/clipped_00000000028000000.mp4 --weights model/yolo11sbest.pt  --img-size 1280 --conf 0.18 --iou 0.6 --device 0 --half

2) 检测 + 跟踪（推荐）:
python detect/yolo11_detect_track.py --source data/clipped_00000000028000000.mp4 --weights model/yolo11sbest.pt --track --tracker bytetrack.yaml --img-size 1280 --conf 0.18 --iou 0.6 --device 0 --half

3) 用 BoT-SORT（更稳但略慢）:
python detect/yolo11_detect_track.py --source data/clip_video/clipped_00000000028000000.mp4 --weights model/yolo11sbest.pt --track --tracker botsort.yaml   --img-size 2560 --conf 0.1 --iou 0.6 --device 0 --half

4) 高分辨率小目标（教室远处人更稳）:
python detect/yolo11_detect_track.py --source data/clip_video/clipped_00000000028000000.mp4 --weights model/yolo11sbest.pt --track --img-size 1280 --conf 0.18 --iou 0.6 --device 0 --half
"""

import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


def draw_boxes(frame, boxes, names=None, show_id=True):
    """绘制 bbox + conf + class (+ track_id)"""
    img = frame.copy()

    # boxes.id: 跟踪ID（开启 track + persist 后才有）
    ids = None
    if show_id and hasattr(boxes, "id") and boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()

    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0])
        cls = int(b.cls[0])

        cls_name = names[cls] if names else str(cls)
        if ids is not None:
            label = f"ID{ids[i]} {cls_name}:{conf:.2f}"
        else:
            label = f"{cls_name}:{conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return img


def detect_video(
    source: str,
    weights: str,
    output_dir: str,
    conf: float,
    iou: float,
    device: str,
    img_size: int | None,
    half: bool,
    display: bool,
    track: bool,
    tracker: str,
    classes: list[int] | None,
):
    if not os.path.isfile(source):
        raise FileNotFoundError(f"视频不存在: {source}")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"权重不存在: {weights}")

    model = YOLO(weights)
    names = getattr(model, "names", None)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(source).stem
    out_path = os.path.join(output_dir, f"detected_{video_name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))

    frame_id = 0

    # 跟踪模式下：persist=True 很关键，用于保持跨帧的 tracker 状态
    persist = True if track else False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            if track:
                results = model.track(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    device=device if device else None,
                    imgsz=img_size if img_size else None,
                    half=half,
                    verbose=False,
                    persist=True,
                    tracker=tracker,
                    classes=classes,
                )
            else:
                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    device=device if device else None,
                    imgsz=img_size if img_size else None,
                    half=half,
                    verbose=False,
                    save=False,
                    classes=classes,
                )

            r = results[0]
            boxes = r.boxes

            annotated = draw_boxes(frame, boxes, names=names, show_id=track)

            info = f"Frame {frame_id}/{total_frames if total_frames > 0 else '?'} | Det {len(boxes)}"
            if track and hasattr(boxes, "id") and boxes.id is not None:
                info += f" | Tracks {len(boxes.id)}"
            cv2.putText(
                annotated,
                info,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            out.write(annotated)

            if display:
                cv2.imshow("YOLOv11 Detect/Track", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_id % 30 == 0:
                if total_frames > 0:
                    prog = 100.0 * frame_id / total_frames
                    print(f"进度: {frame_id}/{total_frames} ({prog:.1f}%)")
                else:
                    print(f"已处理帧数: {frame_id}")

    finally:
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

    print(f"完成! 输出视频: {out_path}")


def parse_classes(s: str | None):
    """
    解析 --classes 参数:
    - None: 不过滤
    - "0" / "0,1,2": 过滤指定类别
    """
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser("YOLOv11 视频检测 + 跟踪（独立版）")
    parser.add_argument("--source", type=str, required=True, help="输入视频路径")
    parser.add_argument("--weights", type=str, required=True, help="权重路径")
    parser.add_argument("--output", type=str, default="detect/track", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.18, help="置信度阈值（教室小目标建议 0.12~0.25）")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IOU 阈值（建议 0.5~0.7）")
    parser.add_argument("--device", type=str, default="0", help="cpu / cuda / 0 / 1 ...（空=自动）")
    parser.add_argument("--img-size", type=int, default=None, help="推理尺寸，如 640/1280")
    parser.add_argument("--half", action="store_true", help="FP16（仅GPU）")
    parser.add_argument("--display", action="store_true", help="实时显示，按 q 退出")

    # 跟踪相关
    parser.add_argument("--track", action="store_true", help="开启多目标跟踪（输出带ID）")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="bytetrack.yaml 或 botsort.yaml")
    parser.add_argument("--classes", type=str, default=None, help="只检测指定类别，如 '0' 或 '0,1'（person 常为 0）")

    args = parser.parse_args()
    classes = parse_classes(args.classes)

    detect_video(
        source=args.source,
        weights=args.weights,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        img_size=args.img_size,
        half=args.half,
        display=args.display,
        track=args.track,
        tracker=args.tracker,
        classes=classes,
    )


if __name__ == "__main__":
    main()