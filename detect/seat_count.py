# seat_count.py
import argparse
import json
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


def load_seats(seats_json: str):
    with open(seats_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    seats = data["seats"]
    # normalize ints
    for s in seats:
        s["x1"] = int(s["x1"]); s["y1"] = int(s["y1"])
        s["x2"] = int(s["x2"]); s["y2"] = int(s["y2"])
        s["id"] = int(s["id"])
    return seats


def point_in_rect(px, py, r):
    return (r["x1"] <= px <= r["x2"]) and (r["y1"] <= py <= r["y2"])


def seat_center(r):
    return ((r["x1"] + r["x2"]) * 0.5, (r["y1"] + r["y2"]) * 0.5)


def rect_iou(a, b):
    # a,b: (x1,y1,x2,y2)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def assign_boxes_to_seats(
    boxes, seats,
    alpha=0.75,              # 坐姿点位置：0.7~0.85 都可试
    bottom_margin=10,        # 底部点上移像素（避免触地噪声）
    iou_th=0.05              # IoU 保险阈值：0.03~0.08
):
    seat_hit = {s["id"]: False for s in seats}

    # 预取 seat bbox & center
    seat_info = []
    for s in seats:
        sb = (float(s["x1"]), float(s["y1"]), float(s["x2"]), float(s["y2"]))
        sx, sy = seat_center(s)
        seat_info.append((s["id"], sb, sx, sy))

    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx = 0.5 * (x1 + x2)
        h = (y2 - y1)

        # 两个 anchor：坐姿点 + 底部点
        p1 = (cx, y1 + alpha * h)
        p2 = (cx, y2 - bottom_margin)

        pb = (float(x1), float(y1), float(x2), float(y2))

        candidates = []
        for sid, sb, sx, sy in seat_info:
            hit = False
            if point_in_rect(p1[0], p1[1], {"x1": sb[0], "y1": sb[1], "x2": sb[2], "y2": sb[3]}):
                hit = True
            elif point_in_rect(p2[0], p2[1], {"x1": sb[0], "y1": sb[1], "x2": sb[2], "y2": sb[3]}):
                hit = True
            else:
                if rect_iou(pb, sb) > iou_th:
                    hit = True

            if hit:
                dist2 = (cx - sx) ** 2 + ((y1 + 0.75*h) - sy) ** 2
                candidates.append((dist2, sid))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            seat_hit[candidates[0][1]] = True

    return seat_hit


class SeatStateMachine:
    def __init__(self, seat_ids, inc=0.15, dec=0.03, on_th=0.6, off_th=0.3):
        self.score = {sid: 0.0 for sid in seat_ids}
        self.occ = {sid: False for sid in seat_ids}
        self.inc = float(inc)
        self.dec = float(dec)
        self.on_th = float(on_th)
        self.off_th = float(off_th)

    def update(self, seat_hit: dict):
        for sid, hit in seat_hit.items():
            if hit:
                self.score[sid] = min(1.0, self.score[sid] + self.inc)
            else:
                self.score[sid] = max(0.0, self.score[sid] - self.dec)

            # hysteresis
            if self.occ[sid]:
                if self.score[sid] < self.off_th:
                    self.occ[sid] = False
            else:
                if self.score[sid] > self.on_th:
                    self.occ[sid] = True

        return self.occ, self.score


def draw_seats(frame, seats, occ, score):
    img = frame.copy()
    for s in seats:
        sid = s["id"]
        is_occ = occ[sid]
        color = (0, 255, 0) if is_occ else (0, 0, 255)
        cv2.rectangle(img, (s["x1"], s["y1"]), (s["x2"], s["y2"]), color, 2)
        txt = f"S{sid} {score[sid]:.2f}"
        cv2.putText(img, txt, (s["x1"], max(0, s["y1"] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def main():
    ap = argparse.ArgumentParser("Seat-based Counting (Rect ROI) for Exam Room")
    ap.add_argument("--source", type=str, required=True, help="输入视频")
    ap.add_argument("--weights", type=str, required=True, help="YOLOv11 权重")
    ap.add_argument("--seats", type=str, required=True, help="seats.json（用 seat_label_tool.py 生成）")
    ap.add_argument("--output", type=str, default="detect/output", help="输出目录")
    ap.add_argument("--conf", type=float, default=0.15, help="检测 conf（建议 0.12~0.2）")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS iou（建议 0.5~0.7）")
    ap.add_argument("--img-size", type=int, default=1280, help="推理尺寸（远处人建议>=1280）")
    ap.add_argument("--device", type=str, default="0", help="0/cpu/cuda...")
    ap.add_argument("--half", action="store_true", help="FP16（仅GPU）")
    ap.add_argument("--classes", type=str, default="0", help="只检测的类别id，person一般是0")

    # seat 状态机参数
    ap.add_argument("--inc", type=float, default=0.15)
    ap.add_argument("--dec", type=float, default=0.03)
    ap.add_argument("--on", type=float, default=0.6)
    ap.add_argument("--off", type=float, default=0.3)

    # 归属规则参数
    ap.add_argument("--bottom-margin", type=int, default=5, help="底部点上移像素，避免触地噪声")

    # 可视化
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.source):
        raise FileNotFoundError(args.source)
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(args.weights)
    if not os.path.isfile(args.seats):
        raise FileNotFoundError(args.seats)

    seats = load_seats(args.seats)
    seat_ids = [s["id"] for s in seats]
    sm = SeatStateMachine(seat_ids, inc=args.inc, dec=args.dec, on_th=args.on, off_th=args.off)

    classes = [int(x) for x in args.classes.split(",")] if args.classes else None

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"seatcount_{Path(args.source).stem}.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                device=args.device if args.device else None,
                imgsz=args.img_size if args.img_size else None,
                half=args.half,
                verbose=False,
                classes=classes,
            )
            r = results[0]
            boxes = r.boxes

            seat_hit = assign_boxes_to_seats(
                boxes,
                seats,
                alpha=0.75,
                bottom_margin=args.bottom_margin,
                iou_th=0.05
            )
            occ, score = sm.update(seat_hit)

            # count
            occupied_cnt = sum(1 for sid in seat_ids if occ[sid])

            vis = draw_seats(frame, seats, occ, score)
            info = f"SeatCount={occupied_cnt} | Frame {frame_id}/{total if total>0 else '?'} | Det {len(boxes)}"
            cv2.putText(vis, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            out.write(vis)

            if args.display:
                cv2.imshow("Seat-based Counting", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_id % 30 == 0:
                if total > 0:
                    print(f"进度: {frame_id}/{total} ({100*frame_id/total:.1f}%)  seat={occupied_cnt} det={len(boxes)}")
                else:
                    print(f"已处理帧数: {frame_id}  seat={occupied_cnt} det={len(boxes)}")

    finally:
        cap.release()
        out.release()
        if args.display:
            cv2.destroyAllWindows()

    print(f"完成! 输出: {out_path}")


if __name__ == "__main__":
    main()