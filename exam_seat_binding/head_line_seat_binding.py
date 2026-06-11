"""
Head-line based exam seat binding.

Pipeline:
1. Use the first N seconds of video to build the fixed desk layout with
   exam_seat_binding/desk_layout_detector.py.
2. Track only head detections from model/yolo26m/best2.pt.
3. Bind each tracked head to the nearest desk column line, then to the nearest
   seat depth in that column.

Example:
python exam_seat_binding/head_line_seat_binding.py \
  --source data/1.10/clipleft/merged_output.mp4 \
  --weights model/yolo26m/best2.pt \
  --desk-reference-seconds 30 \
  --output exam_seat_binding/output/head_line_binding
"""

import argparse
import csv
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def _load_module_from_file(name: str, filename: str):
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    from exam_seat_binding import desk_layout_detector as desk_layout_mod
    from exam_seat_binding.person_desk_binding_v3 import (
        HeadLineSeatAssigner,
        StableSeatManager,
    )
except Exception:
    desk_layout_mod = _load_module_from_file(
        "desk_layout_detector", "desk_layout_detector.py"
    )
    binding_mod = _load_module_from_file(
        "person_desk_binding_v3", "person_desk_binding_v3.py"
    )
    HeadLineSeatAssigner = binding_mod.HeadLineSeatAssigner
    StableSeatManager = binding_mod.StableSeatManager


def parse_classes(raw: str | None):
    if raw is None or str(raw).strip() == "":
        return None
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def class_name(names, cls_id: int) -> str:
    cls_id = int(cls_id)
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def box_area(xyxy):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def is_head_detection(det, names, head_classes: set[int], head_area_max: float) -> bool:
    cls_id = int(det["cls"])
    if head_classes:
        return cls_id in head_classes
    cname = class_name(names, cls_id).lower()
    compact = cname.replace("_", "").replace("-", "").replace(" ", "")
    if "head" in compact or "face" in compact:
        return True
    if any(word in compact for word in ("person", "human", "body", "visible")):
        return False
    return box_area(det["xyxy"]) <= float(head_area_max)


def head_anchor_from_box(xyxy, mode="center", y_offset=0.0):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    cx = (x1 + x2) / 2.0
    if mode == "top":
        y = y1
    elif mode == "bottom":
        y = y2
    else:
        y = (y1 + y2) / 2.0
    return np.asarray([cx, y + float(y_offset)], dtype=np.float32)


class DeskCornerZoneBuilder:
    """
    Build seat zones by connecting detected desk box corners directly.

    For adjacent desks in the same column:
        current top-left  -> current bottom-right
        next bottom-right -> next top-left

    This keeps the near-camera/front row tied to the real detected desk box
    instead of extending it outside the desk.
    """

    def __init__(self, last_extend_ratio: float = 1.2):
        self.last_extend_ratio = float(last_extend_ratio)

    def build(self, desks: list) -> list:
        columns: dict[int, list] = {}
        for desk in desks:
            columns.setdefault(int(desk["column_index"]), []).append(desk)

        zones = []
        for col_id in sorted(columns):
            col_desks = sorted(columns[col_id], key=lambda d: int(d["row_index"]))
            for idx, desk in enumerate(col_desks):
                x1, y1, x2, y2 = [float(v) for v in desk["xyxy"]]
                if idx < len(col_desks) - 1:
                    next_desk = col_desks[idx + 1]
                    nx1, ny1, nx2, ny2 = [float(v) for v in next_desk["xyxy"]]
                    poly = np.asarray(
                        [
                            [x1, y1],
                            [x2, y2],
                            [nx2, ny2],
                            [nx1, ny1],
                        ],
                        dtype=np.float32,
                    )
                elif idx > 0:
                    prev = col_desks[idx - 1]
                    px1, py1, px2, py2 = [float(v) for v in prev["xyxy"]]
                    dx_left = x1 - px1
                    dy_left = y1 - py1
                    dx_right = x2 - px2
                    dy_right = y2 - py2
                    poly = np.asarray(
                        [
                            [x1, y1],
                            [x2, y2],
                            [x2 + dx_right * self.last_extend_ratio, y2 + dy_right * self.last_extend_ratio],
                            [x1 + dx_left * self.last_extend_ratio, y1 + dy_left * self.last_extend_ratio],
                        ],
                        dtype=np.float32,
                    )
                else:
                    poly = np.asarray(
                        [
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2],
                        ],
                        dtype=np.float32,
                    )

                zones.append(
                    {
                        "desk_id": desk["desk_id"],
                        "desk_no": int(desk["desk_no"]),
                        "column_index": int(desk["column_index"]),
                        "row_index": int(desk["row_index"]),
                        "desk_xyxy": desk["xyxy"],
                        "desk_center": desk["center"],
                        "polygon": poly,
                        "zone_center": [
                            float(np.mean(poly[:, 0])),
                            float(np.mean(poly[:, 1])),
                        ],
                    }
                )
        return zones


def video_fps(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return fps if fps and fps > 1e-6 else 25.0


def draw_label(img, text, x, y, color, scale=0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(max(0, round(x)))
    y = int(max(th + baseline + 4, round(y)))
    cv2.rectangle(
        img,
        (x, y - th - baseline - 5),
        (x + tw + 6, y),
        color,
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x + 3, y - baseline - 3),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_column_lines(img, column_lines, height):
    out = img
    colors = [
        (255, 80, 80),
        (80, 210, 80),
        (80, 120, 255),
        (255, 200, 80),
        (220, 80, 255),
    ]
    for item in column_lines or []:
        col = int(item.get("column_index", 0))
        color = colors[col % len(colors)]
        start = item.get("segment_start")
        end = item.get("segment_end")
        if start and end:
            p1 = tuple(int(round(v)) for v in start)
            p2 = tuple(int(round(v)) for v in end)
            cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
            draw_label(out, f"C{col + 1}", p2[0] + 6, p2[1], color, scale=0.45)
            continue

        k, b = [float(v) for v in item["line_kb"]]
        ys = [0.0, float(height - 1)]
        pts = []
        for y in ys:
            if abs(k) < 1e-6:
                x = float(item.get("x_min", 0.0))
            else:
                x = (-y - b) / k
            pts.append((int(round(x)), int(round(y))))
        cv2.line(out, pts[0], pts[1], color, 2, cv2.LINE_AA)
    return out


def draw_layout(img, desks, zones, column_lines):
    out = img.copy()
    for zone in zones:
        poly = np.asarray(zone["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], True, (80, 190, 255), 1, cv2.LINE_AA)
    out = draw_column_lines(out, column_lines, out.shape[0])

    for desk in sorted(desks, key=lambda d: int(d["desk_no"])):
        x1, y1, x2, y2 = [int(round(v)) for v in desk["xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (80, 255, 80), 1, cv2.LINE_AA)
        draw_label(out, desk["desk_id"], x1, y1, (80, 255, 80), scale=0.42)
    return out


def build_layout(args):
    reference_source = args.desk_reference_source or args.source
    detector = desk_layout_mod.DeskLayoutDetector(
        weights_path=args.desk_weights,
        conf_threshold=args.desk_conf,
        iou_threshold=args.desk_iou,
        device=args.device,
        img_size=args.img_size,
        half=args.half,
        mode=args.desk_mode,
        num_cols=args.desk_num_cols,
        required_per_col=args.desk_required_per_col,
        init_sample_count=args.desk_init_sample_count,
        pending_confirm_hits=args.desk_pending_confirm_hits,
    )

    max_frames = 0
    if args.desk_reference_seconds > 0:
        max_frames = max(1, int(round(args.desk_reference_seconds * video_fps(reference_source))))
    elif args.reference_max_frames > 0:
        max_frames = int(args.reference_max_frames)

    reference = detector.select_best_layout_from_video(
        reference_source,
        max_frames=max_frames,
        sample_step=args.reference_sample_step,
        save_dir=args.output,
        log=True,
    )
    desks = reference["layout"]["desks"]
    zones = DeskCornerZoneBuilder(
        last_extend_ratio=args.last_extend,
    ).build(desks)
    column_lines = reference["layout"].get("column_lines", [])
    return reference, desks, zones, column_lines


def write_layout_preview(output_dir, video_name, reference, desks, zones, column_lines):
    preview = draw_layout(reference["frame"], desks, zones, column_lines)
    path = os.path.join(output_dir, f"head_line_layout_{video_name}.jpg")
    cv2.imwrite(path, preview)
    return path


def summarize_tracks(track_seat_votes: dict[int, Counter], seat_track_votes: dict[int, Counter]):
    tracks = []
    for tid in sorted(track_seat_votes):
        votes = track_seat_votes[tid]
        if not votes:
            continue
        seat_no, frames = votes.most_common(1)[0]
        tracks.append(
            {
                "track_id_raw": int(tid),
                "seat_no": int(seat_no),
                "vote_frames": int(frames),
                "observed_bound_frames": int(sum(votes.values())),
                "seat_votes": {str(k): int(v) for k, v in sorted(votes.items())},
            }
        )

    seats = {}
    for seat_no in sorted(seat_track_votes):
        votes = seat_track_votes[seat_no]
        if not votes:
            continue
        tid, frames = votes.most_common(1)[0]
        seats[str(seat_no)] = {
            "track_id_raw": int(tid),
            "vote_frames": int(frames),
            "track_votes": {str(k): int(v) for k, v in sorted(votes.items())},
        }
    return tracks, seats


def run(args):
    if not os.path.isfile(args.source):
        raise FileNotFoundError(f"Video not found: {args.source}")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not os.path.isfile(args.desk_weights):
        raise FileNotFoundError(f"Desk weights not found: {args.desk_weights}")

    os.makedirs(args.output, exist_ok=True)
    video_name = Path(args.source).stem
    output_video = os.path.join(args.output, f"head_line_bound_{video_name}.mp4")
    output_csv = os.path.join(args.output, f"head_line_bound_{video_name}.csv")
    output_json = os.path.join(args.output, f"head_line_bound_{video_name}.json")

    print("Building desk layout from the reference segment...")
    reference, desks, zones, column_lines = build_layout(args)
    layout_preview = write_layout_preview(
        args.output, video_name, reference, desks, zones, column_lines
    )
    print(f"Desk layout frame: {reference['frame_idx']}")
    print(f"Desks: {len(desks)}, column lines: {len(column_lines)}")
    print(f"Layout preview: {layout_preview}")

    assigner = HeadLineSeatAssigner(zones=zones, column_lines=column_lines)

    model = YOLO(args.weights)
    names = getattr(model, "names", None)
    print(f"Loaded head/person model: {args.weights}")
    print(f"Model classes: {names}")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bind_start_frame = (
        max(0, int(round(args.bind_start_seconds * fps)))
        if args.bind_start_seconds is not None
        else max(0, int(round(args.desk_reference_seconds * fps)))
    )

    seat_manager = StableSeatManager(
        fps=fps,
        switch_seconds=args.switch_seconds,
        miss_hold_seconds=args.miss_hold_seconds,
        initial_bind_seconds=args.initial_bind_seconds,
        reservation_seconds=args.reservation_seconds,
        reacquire_seconds=args.reacquire_seconds,
    )

    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )

    head_classes = set(parse_classes(args.head_classes) or [])
    model_classes = parse_classes(args.classes)
    track_seat_votes: dict[int, Counter] = {}
    seat_track_votes: dict[int, Counter] = {}
    seen_tracks: set[int] = set()

    csv_file = open(output_csv, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "frame_idx",
            "timestamp",
            "track_id_raw",
            "cls",
            "class_name",
            "conf",
            "head_x1",
            "head_y1",
            "head_x2",
            "head_y2",
            "head_cx",
            "head_cy",
            "anchor_x",
            "anchor_y",
            "seat_no_current",
            "seat_no_display",
            "desk_id_current",
            "col_idx",
            "row_idx",
            "head_line_distance",
            "head_depth_gap",
            "cost",
        ],
    )
    csv_writer.writeheader()

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            track_kwargs = {
                "source": frame,
                "conf": args.conf,
                "iou": args.iou,
                "persist": True,
                "tracker": args.tracker,
                "verbose": False,
                "half": args.half,
            }
            if args.device:
                track_kwargs["device"] = args.device
            if args.img_size is not None:
                track_kwargs["imgsz"] = args.img_size
            if model_classes is not None:
                track_kwargs["classes"] = model_classes

            results = model.track(**track_kwargs)
            boxes = results[0].boxes
            ids = boxes.id.int().cpu().tolist() if getattr(boxes, "id", None) is not None else None

            annotated = draw_layout(frame, desks, zones, column_lines) if args.draw_layout else frame.copy()
            head_people = []
            head_anchor_pts = {}
            head_dets = {}

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                tid = int(ids[idx]) if ids is not None else -1
                det = {
                    "track_id": tid,
                    "xyxy": [x1, y1, x2, y2],
                    "conf": conf,
                    "cls": cls_id,
                    "class_name": class_name(names, cls_id),
                }
                if tid < 0 or not is_head_detection(det, names, head_classes, args.head_area_max):
                    continue

                anchor = head_anchor_from_box(
                    det["xyxy"],
                    mode=args.head_anchor,
                    y_offset=args.head_y_offset,
                )
                seen_tracks.add(tid)
                head_people.append(
                    {
                        "track_id": tid,
                        "xyxy": det["xyxy"],
                        "conf": conf,
                        "cls": cls_id,
                        "class_name": det["class_name"],
                    }
                )
                head_anchor_pts[tid] = anchor
                head_dets[tid] = det

            if frame_idx >= bind_start_frame:
                assignments = assigner.assign_batch(
                    people=head_people,
                    head_anchor_pts=head_anchor_pts,
                    occupied_seat_nos=set(),
                )
            else:
                assignments = {}

            for person in head_people:
                tid = int(person["track_id"])
                det = head_dets[tid]
                anchor = head_anchor_pts[tid]
                assignment = assignments.get(tid)
                current_seat = int(assignment["desk_no"]) if assignment else None
                display_seat = (
                    seat_manager.update(tid, frame_idx, current_seat)
                    if frame_idx >= bind_start_frame
                    else None
                )

                if display_seat is not None:
                    track_seat_votes.setdefault(tid, Counter())[int(display_seat)] += 1
                    seat_track_votes.setdefault(int(display_seat), Counter())[tid] += 1

                x1, y1, x2, y2 = det["xyxy"]
                color = (0, 255, 0) if display_seat is not None else (0, 220, 255)
                if frame_idx < bind_start_frame:
                    color = (180, 180, 180)
                    label = f"H{tid} layout"
                elif display_seat is not None:
                    label = f"H{tid}->S{int(display_seat):02d}"
                elif assignment:
                    label = f"H{tid}->S{int(current_seat):02d} pending"
                else:
                    color = (0, 0, 255)
                    label = f"H{tid}->unbound"

                cv2.rectangle(
                    annotated,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    annotated,
                    (int(round(anchor[0])), int(round(anchor[1]))),
                    4,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
                draw_label(annotated, label, x1, max(0, y1 - 8), color)

                csv_writer.writerow(
                    {
                        "frame_idx": frame_idx,
                        "timestamp": round(frame_idx / fps, 3),
                        "track_id_raw": tid,
                        "cls": int(det["cls"]),
                        "class_name": det["class_name"],
                        "conf": round(float(det["conf"]), 5),
                        "head_x1": round(x1, 2),
                        "head_y1": round(y1, 2),
                        "head_x2": round(x2, 2),
                        "head_y2": round(y2, 2),
                        "head_cx": round((x1 + x2) / 2.0, 2),
                        "head_cy": round((y1 + y2) / 2.0, 2),
                        "anchor_x": round(float(anchor[0]), 2),
                        "anchor_y": round(float(anchor[1]), 2),
                        "seat_no_current": current_seat,
                        "seat_no_display": int(display_seat) if display_seat is not None else None,
                        "desk_id_current": assignment["desk_id"] if assignment else None,
                        "col_idx": assignment["col_idx"] if assignment else None,
                        "row_idx": assignment["row_idx"] if assignment else None,
                        "head_line_distance": assignment.get("head_line_distance") if assignment else None,
                        "head_depth_gap": assignment.get("head_depth_gap") if assignment else None,
                        "cost": assignment.get("cost") if assignment else None,
                    }
                )

            cv2.putText(
                annotated,
                f"frame {frame_idx}/{total_frames} bind_start={bind_start_frame}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(annotated)
            if args.display:
                cv2.imshow("head-line seat binding", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_process_frames > 0 and frame_idx >= args.max_process_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        if args.display:
            cv2.destroyAllWindows()

    tracks, seat_bindings = summarize_tracks(track_seat_votes, seat_track_votes)
    payload = {
        "source": args.source,
        "weights": args.weights,
        "desk_weights": args.desk_weights,
        "fps": float(fps),
        "total_frames": int(total_frames),
        "processed_frames": int(frame_idx),
        "desk_reference_seconds": float(args.desk_reference_seconds),
        "bind_start_frame": int(bind_start_frame),
        "bind_start_seconds": round(bind_start_frame / fps, 3),
        "desk_reference_frame": int(reference["frame_idx"]),
        "desk_count": int(len(desks)),
        "column_line_count": int(len(column_lines)),
        "head_classes": sorted(int(v) for v in head_classes),
        "head_anchor": args.head_anchor,
        "head_y_offset": float(args.head_y_offset),
        "layout_preview": layout_preview,
        "output_video": output_video if writer is not None else None,
        "output_csv": output_csv,
        "desks": desks,
        "zones": [
            {
                "desk_id": z["desk_id"],
                "desk_no": int(z["desk_no"]),
                "column_index": int(z["column_index"]),
                "row_index": int(z["row_index"]),
                "desk_xyxy": z["desk_xyxy"],
                "zone_center": z["zone_center"],
                "polygon": np.asarray(z["polygon"]).astype(float).tolist(),
            }
            for z in zones
        ],
        "column_lines": column_lines,
        "tracks": tracks,
        "seat_track_bindings": seat_bindings,
        "seen_track_count": int(len(seen_tracks)),
    }
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print(f"Done. Video: {output_video if writer is not None else '(disabled)'}")
    print(f"CSV: {output_csv}")
    print(f"JSON: {output_json}")
    return payload


def build_arg_parser():
    p = argparse.ArgumentParser("Head-line person-desk binding")
    p.add_argument("--source", required=True, help="Input video path")
    p.add_argument(
        "--desk-reference-source",
        default=None,
        help="Optional video used only for desk layout; defaults to --source",
    )
    p.add_argument("--weights", default="model/yolo26m/best2.pt", help="Head/person YOLO weights")
    p.add_argument(
        "--desk-weights",
        default="exam_seat_binding/weight/yolo11desk.pt",
        help="Desk YOLO weights",
    )
    p.add_argument("--output", default="exam_seat_binding/output/head_line_binding")

    p.add_argument("--conf", type=float, default=0.18)
    p.add_argument("--iou", type=float, default=0.60)
    p.add_argument("--desk-conf", type=float, default=0.70)
    p.add_argument("--desk-iou", type=float, default=0.45)
    p.add_argument("--device", default="")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--half", action="store_true")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--classes", default=None, help="Optional YOLO class IDs to track, comma-separated")
    p.add_argument("--head-classes", default=None, help="Head class IDs, comma-separated; auto by class name if empty")
    p.add_argument("--head-area-max", type=float, default=6500.0, help="Fallback head area limit when class names are unclear")
    p.add_argument("--head-anchor", choices=["center", "top", "bottom"], default="center")
    p.add_argument("--head-y-offset", type=float, default=0.0, help="Pixel offset added to the selected head anchor y")

    p.add_argument("--desk-mode", default="scheme1", choices=["normal", "scheme1", "scheme2", "auto"])
    p.add_argument("--desk-num-cols", type=int, default=5)
    p.add_argument("--desk-required-per-col", type=int, default=6)
    p.add_argument("--desk-init-sample-count", type=int, default=5)
    p.add_argument("--desk-pending-confirm-hits", type=int, default=1)
    p.add_argument("--desk-reference-seconds", type=float, default=30.0)
    p.add_argument("--reference-max-frames", type=int, default=0)
    p.add_argument("--reference-sample-step", type=int, default=5)
    p.add_argument(
        "--bind-start-seconds",
        type=float,
        default=None,
        help="When to start binding heads; defaults to --desk-reference-seconds",
    )
    p.add_argument(
        "--first-extend",
        type=float,
        default=0.0,
        help="Compatibility only; front row now connects detected desk corners directly",
    )
    p.add_argument(
        "--last-extend",
        type=float,
        default=1.2,
        help="Only used for the farthest row, which has no next desk to connect",
    )

    p.add_argument("--initial-bind-seconds", type=float, default=0.6)
    p.add_argument("--switch-seconds", type=float, default=3.0)
    p.add_argument("--miss-hold-seconds", type=float, default=5.0)
    p.add_argument("--reservation-seconds", type=float, default=120.0)
    p.add_argument("--reacquire-seconds", type=float, default=0.25)

    p.add_argument("--draw-layout", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--max-process-frames", type=int, default=0, help="Debug only; 0 means full video")
    p.add_argument("--display", action="store_true")
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
