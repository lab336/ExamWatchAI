"""
Head-to-desk detection based exam seat binding.

Pipeline:
1. Use the first N seconds of video to build the fixed desk layout with
   exam_seat_binding/desk_layout_detector.py.
2. Track only head detections from model/yolo26m/best2.pt.
3. Bind each tracked head to the detected desk box at its lower-left side.

Example:
python exam_seat_binding/head_line_seat_binding.py \
  --source data/1.10/clipleft/merged_output.mp4 \
  --weights model/yolo26m/best2.pt \
  --desk-reference-seconds 30 \
  --output exam_seat_binding/output/head_line_binding2
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module_from_file(name: str, filename: str):
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    from exam_seat_binding import desk_layout_detector as desk_layout_mod
except Exception:
    desk_layout_mod = _load_module_from_file(
        "desk_layout_detector", "desk_layout_detector.py"
    )


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


def is_body_detection(det, names, body_classes: set[int], head_classes: set[int], head_area_max: float) -> bool:
    cls_id = int(det["cls"])
    if body_classes:
        return cls_id in body_classes
    if head_classes and cls_id in head_classes:
        return False
    cname = class_name(names, cls_id).lower()
    compact = cname.replace("_", "").replace("-", "").replace(" ", "")
    if any(word in compact for word in ("person", "human", "body", "visible")):
        return True
    if "head" in compact or "face" in compact:
        return False
    return box_area(det["xyxy"]) > float(head_area_max)


def xyxy_center(xyxy):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def point_in_xyxy(point_xy, xyxy, margin=0.0):
    x, y = [float(v) for v in point_xy]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def point_xyxy_distance(point_xy, xyxy):
    x, y = [float(v) for v in point_xy]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    dx = max(float(x1) - x, 0.0, x - float(x2))
    dy = max(float(y1) - y, 0.0, y - float(y2))
    return float(np.hypot(dx, dy))


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
    Build seat zones by connecting adjacent desk-box corners in each column.

    Zone construction (per column, desks sorted near-to-far):
      - non-last row: current top-left -> current bottom-right ->
        next bottom-right -> next top-left
      - last (farthest) row: extend the same diagonal corner region upward
      - single-desk column: use the desk bounding box itself as the zone
    """

    def __init__(self, last_extend_ratio: float = 1.2):
        self.last_extend_ratio = float(last_extend_ratio)

    def build(self, desks: list) -> list:
        columns: dict[int, list] = {}
        for desk in desks:
            columns.setdefault(int(desk["column_index"]), []).append(desk)

        zones = []
        for col_id in sorted(columns):
            # Sort near-to-far: largest y first (row_index ascending)
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
                    display_poly = poly.copy()
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
                    display_poly = poly.copy()
                else:
                    # Single-desk column: use desk bounding box
                    poly = np.asarray(
                        [
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2],
                        ],
                        dtype=np.float32,
                    )
                    display_poly = poly.copy()

                zones.append(
                    {
                        "desk_id": desk["desk_id"],
                        "desk_no": int(desk["desk_no"]),
                        "column_index": int(desk["column_index"]),
                        "row_index": int(desk["row_index"]),
                        "desk_xyxy": desk["xyxy"],
                        "desk_center": desk["center"],
                        "polygon": poly,
                        "display_polygon": display_poly,
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


def normalize_depth_map(depth_map, low_percentile: float = 5.0, high_percentile: float = 95.0):
    if depth_map is None:
        return None
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        return None
    lo = float(np.percentile(valid, float(low_percentile)))
    hi = float(np.percentile(valid, float(high_percentile)))
    if hi <= lo + 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    norm = (depth - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def median_depth_in_xyxy(depth_map, xyxy, sample_ratio: float = 0.35):
    if depth_map is None:
        return None
    depth = np.asarray(depth_map, dtype=np.float32)
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    if x2 <= x1 or y2 <= y1:
        return None
    ratio = max(0.05, min(1.0, float(sample_ratio)))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(2.0, (x2 - x1) * ratio)
    bh = max(2.0, (y2 - y1) * ratio)
    sx1 = int(max(0, min(w - 1, round(cx - bw / 2.0))))
    sx2 = int(max(0, min(w, round(cx + bw / 2.0))))
    sy1 = int(max(0, min(h - 1, round(cy - bh / 2.0))))
    sy2 = int(max(0, min(h, round(cy + bh / 2.0))))
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    patch = depth[sy1:sy2, sx1:sx2]
    values = patch[np.isfinite(patch)]
    if values.size == 0:
        return None
    return float(np.median(values))


class VideoDepthPrior:
    def __init__(
        self,
        weights_path: str,
        encoder: str = "vitb",
        device: str = "",
        input_size: int = 518,
        max_res: int = 960,
        fp32: bool = False,
    ):
        weights_path = str(weights_path)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Depth weights not found: {weights_path}")

        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Depth prior requires torch to be installed") from exc

        depth_root = Path(__file__).resolve().parent / "Video-Depth-Anything"
        if str(depth_root) not in sys.path:
            sys.path.insert(0, str(depth_root))
        from video_depth_anything.video_depth_stream import VideoDepthAnything

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        if encoder not in model_configs:
            raise ValueError(f"Unsupported depth encoder: {encoder}")

        self.torch = torch
        raw_device = str(device or "").strip()
        if raw_device.isdigit():
            raw_device = f"cuda:{raw_device}"
        if not raw_device:
            raw_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_device = raw_device
        self.infer_device = "cuda" if raw_device.startswith("cuda") else raw_device
        self.input_size = int(input_size)
        self.max_res = int(max_res)
        self.fp32 = bool(fp32)
        self.model = VideoDepthAnything(**model_configs[encoder])
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        self.model = self.model.to(self.model_device).eval()

    def infer(self, frame_bgr):
        original_h, original_w = frame_bgr.shape[:2]
        frame = frame_bgr
        if self.max_res > 0 and max(original_h, original_w) > self.max_res:
            scale = float(self.max_res) / float(max(original_h, original_w))
            resized_w = max(1, int(round(original_w * scale)))
            resized_h = max(1, int(round(original_h * scale)))
            frame = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth = self.model.infer_video_depth_one(
            frame_rgb,
            input_size=self.input_size,
            device=self.infer_device,
            fp32=self.fp32,
        )
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape[:2] != (original_h, original_w):
            depth = cv2.resize(depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        return depth


def build_student_area_polygon(desks: list, padding: float = 80.0):
    if not desks:
        return None

    points = []
    for desk in desks:
        x1, y1, x2, y2 = [float(v) for v in desk["xyxy"]]
        points.extend([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
    padding = float(padding)
    if padding <= 0:
        return hull.astype(np.float32)

    center = np.mean(hull, axis=0)
    expanded = []
    for point in hull:
        vec = point - center
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            expanded.append(point)
        else:
            expanded.append(point + vec / norm * padding)
    return np.asarray(expanded, dtype=np.float32)


def build_first_column_head_area_polygon(
    desks: list,
    base_polygon,
    pad_x: float = 160.0,
    pad_y: float = 60.0,
):
    """Expand only the leftmost desk column for tolerant head binding."""
    if base_polygon is None or not desks:
        return base_polygon
    pad_x = float(pad_x)
    pad_y = float(pad_y)
    if pad_x <= 0 and pad_y <= 0:
        return np.asarray(base_polygon, dtype=np.float32)

    by_col: dict[int, list] = {}
    for desk in desks:
        by_col.setdefault(int(desk["column_index"]), []).append(desk)
    if not by_col:
        return np.asarray(base_polygon, dtype=np.float32)

    left_col = min(
        by_col,
        key=lambda col: float(
            np.median(
                [
                    (float(d["xyxy"][0]) + float(d["xyxy"][2])) / 2.0
                    for d in by_col[col]
                ]
            )
        ),
    )

    points = [list(map(float, p)) for p in np.asarray(base_polygon, dtype=np.float32).reshape(-1, 2)]
    for desk in by_col[left_col]:
        x1, y1, x2, y2 = [float(v) for v in desk["xyxy"]]
        points.extend(
            [
                [x1 - pad_x, y1 - pad_y],
                [x1 - pad_x, y2 + pad_y],
                [x2 + pad_x * 0.20, y1 - pad_y],
                [x2 + pad_x * 0.20, y2 + pad_y],
            ]
        )

    return cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2)


def expand_polygon_by_padding(polygon, padding: float):
    if polygon is None or len(polygon) < 3:
        return polygon
    padding = float(padding)
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if padding <= 0:
        return poly
    center = np.mean(poly, axis=0)
    expanded = []
    for point in poly:
        vec = point - center
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            expanded.append(point)
        else:
            expanded.append(point + vec / norm * padding)
    return np.asarray(expanded, dtype=np.float32)


def point_in_polygon(point_xy, polygon) -> bool:
    if polygon is None or len(polygon) < 3:
        return True
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(point_xy[0]), float(point_xy[1])), False) >= 0


def point_polygon_signed_distance(point_xy, polygon) -> float:
    if polygon is None or len(polygon) < 3:
        return 0.0
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return float(
        cv2.pointPolygonTest(
            poly,
            (float(point_xy[0]), float(point_xy[1])),
            True,
        )
    )


def xyxy_polygon_area_ratio(xyxy, polygon) -> float:
    if polygon is None or len(polygon) < 3:
        return 1.0
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    box_area_value = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if box_area_value <= 1e-6:
        return 0.0
    box_poly = np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    try:
        inter_area, _ = cv2.intersectConvexConvex(box_poly, poly)
    except cv2.error:
        return 0.0
    return max(0.0, float(inter_area)) / box_area_value


def l2(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
        )
    )


def point_line_distance_kb(point_xy, line_kb):
    x = float(point_xy[0])
    y_up = -float(point_xy[1])
    k, b = [float(v) for v in line_kb]
    return abs(k * x - y_up + b) / max(1e-6, float(np.sqrt(k * k + 1.0)))


def three_point_angle_deg(prev_point, mid_point, next_point):
    a = np.asarray(prev_point, dtype=np.float32) - np.asarray(mid_point, dtype=np.float32)
    b = np.asarray(next_point, dtype=np.float32) - np.asarray(mid_point, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-6:
        return 180.0
    value = float(np.dot(a, b) / denom)
    value = max(-1.0, min(1.0, value))
    return float(np.degrees(np.arccos(value)))


def line_fit_rms(points) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) <= 2:
        return 0.0
    center = np.mean(pts, axis=0)
    shifted = pts - center
    _, _, vh = np.linalg.svd(shifted, full_matrices=False)
    direction = vh[0]
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    distances = shifted @ normal
    return float(np.sqrt(np.mean(distances * distances)))


class HeadLineSeatAssigner:
    """
    Assign tracked heads to seats by column line and row depth.

    Each head first chooses the nearest fitted desk-column line, then projects
    to that column's depth axis and selects the nearest seat in the column.
    """

    def __init__(self, zones: list, column_lines: list | None):
        self.zones_by_col: dict[int, list] = {}
        self.column_lines = column_lines or []
        self.column_line_by_col = {
            int(item["column_index"]): item for item in self.column_lines
        }
        self.col_center_x: dict[int, float] = {}
        self.col_origin: dict[int, np.ndarray] = {}
        self.col_axis_unit: dict[int, np.ndarray] = {}
        self.col_zone_depths: dict[int, dict[int, float]] = {}
        self.col_line_limit: dict[int, float] = {}
        self.col_depth_limit: dict[int, float] = {}
        self.col_required_count: dict[int, int] = {}
        self.lateral_axis = None
        self.col_lateral_center: dict[int, float] = {}
        self.col_lateral_min: dict[int, float] = {}
        self.col_lateral_max: dict[int, float] = {}

        for zone in zones:
            col = int(zone["column_index"])
            self.zones_by_col.setdefault(col, []).append(zone)

        for col, col_zones in self.zones_by_col.items():
            col_zones.sort(key=lambda z: int(z["row_index"]))
            self.col_center_x[col] = float(
                np.mean([float(z["desk_center"][0]) for z in col_zones])
            )
            line_item = self.column_line_by_col.get(col)
            if line_item is not None:
                avg_step = float(line_item.get("avg_step", 0.0))
                avg_w = float(line_item.get("avg_box_width", 0.0))
                avg_h = float(line_item.get("avg_box_height", 0.0))
            else:
                avg_step = 0.0
                avg_w = 0.0
                avg_h = 0.0

            origin = np.asarray(col_zones[0]["desk_center"], dtype=np.float32)
            far_point = np.asarray(col_zones[-1]["desk_center"], dtype=np.float32)
            axis = far_point - origin
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-6:
                axis = np.asarray([0.0, -1.0], dtype=np.float32)
                axis_norm = 1.0
            self.col_origin[col] = origin
            self.col_axis_unit[col] = axis / axis_norm

            zone_depths = {}
            for zone in col_zones:
                seat_no = int(zone["desk_no"])
                zone_depths[seat_no] = self._project_depth(
                    np.asarray(zone["desk_center"], dtype=np.float32),
                    col,
                )
            self.col_zone_depths[col] = zone_depths

            ordered_depths = [zone_depths[int(zone["desk_no"])] for zone in col_zones]
            if len(ordered_depths) >= 2:
                depth_steps = [
                    abs(b - a) for a, b in zip(ordered_depths[:-1], ordered_depths[1:])
                ]
                typical_step = float(np.median(depth_steps))
            else:
                typical_step = avg_step
            self.col_line_limit[col] = max(130.0, avg_w * 1.45, avg_step * 0.55)
            self.col_depth_limit[col] = max(120.0, typical_step * 0.95, avg_h * 1.35)
            self.col_required_count[col] = len(col_zones)

        if self.col_axis_unit:
            mean_axis = np.mean(
                np.asarray(list(self.col_axis_unit.values()), dtype=np.float32),
                axis=0,
            )
            norm = float(np.linalg.norm(mean_axis))
            if norm <= 1e-6:
                mean_axis = np.asarray([0.0, -1.0], dtype=np.float32)
            else:
                mean_axis = mean_axis / norm
            self.lateral_axis = np.asarray([-mean_axis[1], mean_axis[0]], dtype=np.float32)
            for col, col_zones in self.zones_by_col.items():
                values = [
                    float(np.dot(np.asarray(zone["desk_center"], dtype=np.float32), self.lateral_axis))
                    for zone in col_zones
                ]
                self.col_lateral_center[col] = float(np.median(values))
                self.col_lateral_min[col] = float(min(values))
                self.col_lateral_max[col] = float(max(values))

    def _project_depth(self, point_xy, col: int) -> float:
        point = np.asarray(point_xy, dtype=np.float32)
        return float(np.dot(point - self.col_origin[col], self.col_axis_unit[col]))

    def _axis_distance(self, point_xy, col: int) -> float:
        point = np.asarray(point_xy, dtype=np.float32)
        vec = point - self.col_origin[col]
        axis = self.col_axis_unit[col]
        proj = self.col_origin[col] + axis * float(np.dot(vec, axis))
        return float(np.linalg.norm(point - proj))

    def _lateral_value(self, point_xy) -> float:
        point = np.asarray(point_xy, dtype=np.float32)
        if self.lateral_axis is None:
            return float(point[0])
        return float(np.dot(point, self.lateral_axis))

    def _pick_column_by_lateral_order(self, point_xy):
        if not self.col_lateral_center:
            return self._pick_column(point_xy)
        value = self._lateral_value(point_xy)
        col = min(self.col_lateral_center, key=lambda c: abs(value - self.col_lateral_center[c]))
        return col, abs(value - self.col_lateral_center[col])

    def _pick_column(self, point_xy, overflow_weight: float = 1.2):
        if not self.zones_by_col:
            return None, float("inf")
        if self.col_axis_unit:
            lat = self._lateral_value(point_xy)

            def _penalized_dist(c):
                base = self._axis_distance(point_xy, c)
                lat_min = self.col_lateral_min.get(c, lat)
                lat_max = self.col_lateral_max.get(c, lat)
                overflow = max(0.0, lat_min - lat, lat - lat_max)
                return base + overflow * float(overflow_weight)

            col = min(self.zones_by_col, key=_penalized_dist)
            return col, self._axis_distance(point_xy, col)
        col = min(
            self.zones_by_col,
            key=lambda c: abs(float(point_xy[0]) - self.col_center_x[c]),
        )
        return col, abs(float(point_xy[0]) - self.col_center_x[col])

    def _seat_match_info(
        self,
        zone: dict,
        head_anchor,
        body_xyxy=None,
        head_padding: float = 35.0,
        first_col_head_padding: float = 35.0,
        max_head_outside: float = 35.0,
        body_padding: float = 20.0,
        body_min_overlap: float = 0.025,
    ):
        col = int(zone["column_index"])
        left_col = min(self.zones_by_col) if self.zones_by_col else col
        padding = float(first_col_head_padding) if col == left_col else float(head_padding)
        head_poly = expand_polygon_by_padding(zone["polygon"], padding)
        body_poly = expand_polygon_by_padding(zone["polygon"], float(body_padding))
        head_signed = point_polygon_signed_distance(head_anchor, head_poly)
        head_ok = head_signed >= -float(max_head_outside)

        body_overlap = None
        body_ok = True
        if body_xyxy is not None:
            body_overlap = xyxy_polygon_area_ratio(body_xyxy, body_poly)
            body_ok = body_overlap >= float(body_min_overlap)

        ok = bool(head_ok and body_ok)
        return {
            "ok": ok,
            "head_signed_distance": float(head_signed),
            "body_overlap": None if body_overlap is None else float(body_overlap),
        }

    def seat_match_ok(
        self,
        zone: dict,
        head_anchor,
        body_xyxy=None,
        head_padding: float = 35.0,
        first_col_head_padding: float = 35.0,
        max_head_outside: float = 35.0,
        body_padding: float = 20.0,
        body_min_overlap: float = 0.025,
    ) -> tuple[bool, dict]:
        info = self._seat_match_info(
            zone,
            head_anchor,
            body_xyxy=body_xyxy,
            head_padding=head_padding,
            first_col_head_padding=first_col_head_padding,
            max_head_outside=max_head_outside,
            body_padding=body_padding,
            body_min_overlap=body_min_overlap,
        )
        return bool(info["ok"]), info

    def assign_zone_hit_batch(
        self,
        people: list,
        head_anchor_pts: dict[int, np.ndarray],
        occupied_seat_nos: set[int],
        body_xyxy_by_tid: dict[int, list] | None = None,
        seat_head_padding: float = 0.0,
        first_col_seat_head_padding: float = 0.0,
        seat_head_max_outside: float = 0.0,
        seat_head_min_overlap: float = 0.10,
        seat_body_padding: float = 0.0,
        seat_body_min_overlap: float = 0.0,
        seat_body_bind_min_overlap: float = 0.08,
    ):
        body_xyxy_by_tid = body_xyxy_by_tid or {}
        occupied = set(int(v) for v in occupied_seat_nos)
        candidates = []
        left_col = min(self.zones_by_col) if self.zones_by_col else 0

        for person in people:
            tid = int(person["track_id"])
            head_anchor = head_anchor_pts.get(tid)
            if head_anchor is None:
                continue

            for col, col_zones in self.zones_by_col.items():
                depth_value = self._project_depth(head_anchor, int(col))
                line_dist = self._axis_distance(head_anchor, int(col))
                for zone in col_zones:
                    seat_no = int(zone["desk_no"])
                    if seat_no in occupied:
                        continue

                    padding = (
                        float(first_col_seat_head_padding)
                        if int(col) == int(left_col)
                        else float(seat_head_padding)
                    )
                    head_poly = expand_polygon_by_padding(zone["polygon"], padding)
                    head_signed = point_polygon_signed_distance(head_anchor, head_poly)
                    head_overlap = xyxy_polygon_area_ratio(person["xyxy"], head_poly)
                    head_ok = (
                        head_signed >= -float(seat_head_max_outside)
                        or head_overlap >= float(seat_head_min_overlap)
                    )
                    body_overlap = None
                    body_hit = False
                    body_xyxy = body_xyxy_by_tid.get(tid)
                    if body_xyxy is not None:
                        body_poly = expand_polygon_by_padding(zone["polygon"], float(seat_body_padding))
                        body_overlap = xyxy_polygon_area_ratio(body_xyxy, body_poly)
                        body_hit = body_overlap >= float(seat_body_bind_min_overlap)
                    if not (head_ok or body_hit):
                        continue

                    depth_gap = abs(self.col_zone_depths[int(col)][seat_no] - depth_value)
                    center_dist = l2(head_anchor, zone["zone_center"])
                    inside_bonus = max(0.0, float(head_signed))
                    body_bonus = 0.0 if body_overlap is None else float(body_overlap) * 260.0
                    match_info = {
                        "head_signed_distance": float(head_signed),
                        "head_overlap": float(head_overlap),
                        "body_overlap": None if body_overlap is None else float(body_overlap),
                        "body_hit": bool(body_hit),
                    }
                    score = (
                        center_dist
                        + depth_gap * 0.25
                        + line_dist * 0.15
                        - inside_bonus * 0.35
                        - body_bonus
                    )
                    candidates.append(
                        (
                            float(score),
                            -inside_bonus,
                            float(center_dist),
                            tid,
                            zone,
                            float(line_dist),
                            float(depth_gap),
                            match_info,
                        )
                    )

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        used_people = set()
        used_seats = set(occupied)
        assignments = {}
        for score, _inside_rank, _center_dist, tid, zone, line_dist, depth_gap, match_info in candidates:
            seat_no = int(zone["desk_no"])
            if tid in used_people or seat_no in used_seats:
                continue
            assignments[tid] = self._assignment_from_zone(
                tid,
                zone,
                head_anchor_pts[tid],
                line_dist,
                depth_gap,
                score,
                "zone_hit",
                match_info=match_info,
            )
            used_people.add(tid)
            used_seats.add(seat_no)
        return assignments

    @staticmethod
    def _solve_cost_matrix(cost_matrix: np.ndarray):
        if cost_matrix.size == 0:
            return []
        try:
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return list(zip(row_ind.tolist(), col_ind.tolist()))
        except Exception:
            pairs = []
            used_rows = set()
            used_cols = set()
            flat = [
                (float(cost_matrix[r, c]), int(r), int(c))
                for r in range(cost_matrix.shape[0])
                for c in range(cost_matrix.shape[1])
            ]
            for _cost, row, col in sorted(flat):
                if row in used_rows or col in used_cols:
                    continue
                used_rows.add(row)
                used_cols.add(col)
                pairs.append((row, col))
            return pairs

    def assign_left_down_desk_batch(
        self,
        people: list,
        head_anchor_pts: dict[int, np.ndarray],
        occupied_seat_nos: set[int] | None = None,
        max_right_offset: float = 120.0,
        max_up_offset: float = 80.0,
        max_box_distance: float = 0.0,
        column_order_min_count: int = 3,
        column_order_weight: float = 0.85,
    ):
        """
        Bind each head to the desk detection that lies down-left of it.

        Image coordinates grow right/down. A valid desk should not be far to
        the right of the head, and should not sit clearly above it. Among valid
        desks, the score prefers the nearest box with a lower-left relation.
        """
        occupied = set(int(v) for v in (occupied_seat_nos or set()))
        all_zones = [
            zone
            for col in sorted(self.zones_by_col)
            for zone in self.zones_by_col[col]
        ]
        if not people or not all_zones:
            return {}

        desk_widths = []
        desk_heights = []
        for zone in all_zones:
            x1, y1, x2, y2 = [float(v) for v in zone["desk_xyxy"]]
            desk_widths.append(max(1.0, x2 - x1))
            desk_heights.append(max(1.0, y2 - y1))
        median_w = float(np.median(desk_widths)) if desk_widths else 100.0
        median_h = float(np.median(desk_heights)) if desk_heights else 70.0
        auto_box_distance = max(180.0, float(np.hypot(median_w, median_h)) * 2.8)
        box_distance_limit = (
            float(max_box_distance)
            if float(max_box_distance) > 0
            else auto_box_distance
        )

        candidate_rows = []
        for person in people:
            tid = int(person["track_id"])
            anchor = head_anchor_pts.get(tid)
            if anchor is None:
                continue
            hx, hy = float(anchor[0]), float(anchor[1])

            for zone in all_zones:
                seat_no = int(zone["desk_no"])
                if seat_no in occupied:
                    continue

                x1, y1, x2, y2 = [float(v) for v in zone["desk_xyxy"]]
                cx, cy = [float(v) for v in zone["desk_center"]]

                right_violation = max(0.0, cx - hx - float(max_right_offset))
                up_violation = max(0.0, hy - cy - float(max_up_offset))
                if right_violation > median_w * 0.80 or up_violation > median_h * 1.20:
                    continue

                box_dist = point_xyxy_distance(anchor, zone["desk_xyxy"])
                if box_dist > box_distance_limit:
                    continue

                desk_left_gap = max(0.0, hx - x2)
                desk_right_gap = max(0.0, x1 - hx)
                desk_down_gap = max(0.0, y1 - hy)
                desk_up_gap = max(0.0, hy - y2)
                center_right_gap = max(0.0, cx - hx)
                center_up_gap = max(0.0, hy - cy)

                lower_left_bonus = 0.0
                if cx <= hx and cy >= hy:
                    lower_left_bonus = min(80.0, median_w * 0.35 + median_h * 0.35)
                if x1 <= hx <= x2 and y1 >= hy:
                    lower_left_bonus += 30.0

                score = (
                    box_dist * 0.90
                    + desk_left_gap * 0.45
                    + desk_down_gap * 0.60
                    + desk_right_gap * 2.80
                    + desk_up_gap * 3.20
                    + center_right_gap * 1.30
                    + center_up_gap * 1.65
                    - lower_left_bonus
                )
                candidate_rows.append(
                    {
                        "score": float(score),
                        "box_dist": float(box_dist),
                        "desk_down_gap": float(desk_down_gap),
                        "center_right_gap": float(center_right_gap),
                        "tid": tid,
                        "zone": zone,
                        "col": int(zone["column_index"]),
                        "info": {
                            "left_down_box_distance": float(box_dist),
                            "left_down_desk_left_gap": float(desk_left_gap),
                            "left_down_desk_down_gap": float(desk_down_gap),
                            "left_down_desk_right_gap": float(desk_right_gap),
                            "left_down_desk_up_gap": float(desk_up_gap),
                        },
                    }
                )

        candidate_rows.sort(
            key=lambda item: (
                item["score"],
                item["box_dist"],
                item["desk_down_gap"],
                item["center_right_gap"],
                item["tid"],
            )
        )
        used_people = set()
        used_seats = set(occupied)
        assignments = {}

        by_col: dict[int, list] = {}
        for item in candidate_rows:
            by_col.setdefault(int(item["col"]), []).append(item)

        for col in sorted(by_col):
            col_items = by_col[col]
            tids = sorted({int(item["tid"]) for item in col_items})
            if len(tids) < int(column_order_min_count):
                continue

            candidate_seat_nos = {int(item["zone"]["desk_no"]) for item in col_items}
            col_zones = [
                zone for zone in self.zones_by_col.get(int(col), [])
                if int(zone["desk_no"]) in candidate_seat_nos
                and int(zone["desk_no"]) not in used_seats
            ]
            if not col_zones:
                continue
            col_zones.sort(
                key=lambda zone: self.col_zone_depths[int(col)][int(zone["desk_no"])]
            )
            tid_depth = {
                tid: self._project_depth(head_anchor_pts[tid], int(col))
                for tid in tids
                if tid in head_anchor_pts
            }
            ordered_tids = sorted(tid_depth, key=lambda tid: tid_depth[tid])
            if not ordered_tids:
                continue

            people_rank_den = max(1, len(ordered_tids) - 1)
            zone_rank_den = max(1, len(col_zones) - 1)
            tid_rank = {
                tid: idx / people_rank_den
                for idx, tid in enumerate(ordered_tids)
            }
            zone_rank = {
                int(zone["desk_no"]): idx / zone_rank_den
                for idx, zone in enumerate(col_zones)
            }
            if len(col_zones) >= 2:
                zone_depths = [
                    self.col_zone_depths[int(col)][int(zone["desk_no"])]
                    for zone in col_zones
                ]
                typical_step = float(np.median([abs(b - a) for a, b in zip(zone_depths[:-1], zone_depths[1:])]))
            else:
                typical_step = self.col_depth_limit.get(int(col), median_h)

            item_by_pair = {
                (int(item["tid"]), int(item["zone"]["desk_no"])): item
                for item in col_items
            }
            cost_matrix = np.full((len(ordered_tids), len(col_zones)), 1e6, dtype=np.float32)
            for row, tid in enumerate(ordered_tids):
                for col_idx, zone in enumerate(col_zones):
                    seat_no = int(zone["desk_no"])
                    item = item_by_pair.get((int(tid), seat_no))
                    if item is None:
                        continue
                    rank_gap = abs(float(tid_rank[int(tid)]) - float(zone_rank[seat_no]))
                    cost_matrix[row, col_idx] = float(item["score"]) + rank_gap * typical_step * float(column_order_weight)

            for row, col_idx in self._solve_cost_matrix(cost_matrix):
                if row >= len(ordered_tids) or col_idx >= len(col_zones):
                    continue
                if float(cost_matrix[row, col_idx]) >= 1e6:
                    continue
                tid = int(ordered_tids[row])
                zone = col_zones[col_idx]
                seat_no = int(zone["desk_no"])
                if tid in used_people or seat_no in used_seats:
                    continue
                item = item_by_pair.get((tid, seat_no))
                if item is None:
                    continue
                rank_gap = abs(float(tid_rank[tid]) - float(zone_rank[seat_no]))
                assignment = self._assignment_from_zone(
                    tid,
                    zone,
                    head_anchor_pts[tid],
                    item["box_dist"],
                    item["desk_down_gap"],
                    float(cost_matrix[row, col_idx]),
                    "head_left_down_desk_column_order",
                )
                for key, value in item["info"].items():
                    assignment[key] = round(float(value), 4)
                assignment["left_down_column_rank_gap"] = round(rank_gap, 4)
                assignments[tid] = assignment
                used_people.add(tid)
                used_seats.add(seat_no)

        for item in candidate_rows:
            score = item["score"]
            box_dist = item["box_dist"]
            desk_down_gap = item["desk_down_gap"]
            tid = int(item["tid"])
            zone = item["zone"]
            info = item["info"]
            seat_no = int(zone["desk_no"])
            if tid in used_people or seat_no in used_seats:
                continue
            assignment = self._assignment_from_zone(
                tid,
                zone,
                head_anchor_pts[tid],
                box_dist,
                desk_down_gap,
                score,
                "head_left_down_desk_detection",
            )
            for key, value in info.items():
                assignment[key] = round(float(value), 4)
            assignments[tid] = assignment
            used_people.add(tid)
            used_seats.add(seat_no)
        return assignments

    def assign_column_order_batch(
        self,
        people: list,
        head_anchor_pts: dict[int, np.ndarray],
        occupied_seat_nos: set[int],
        body_xyxy_by_tid: dict[int, list] | None = None,
        line_scale: float = 2.8,
        depth_scale: float = 2.2,
        min_line: float = 260.0,
        min_depth: float = 230.0,
        order_weight: float = 0.65,
        zone_hit_bonus: float = 85.0,
        seat_head_padding: float = 0.0,
        first_col_seat_head_padding: float = 0.0,
        seat_head_max_outside: float = 0.0,
        seat_head_min_overlap: float = 0.10,
        seat_body_padding: float = 0.0,
        seat_body_min_overlap: float = 0.0,
    ):
        body_xyxy_by_tid = body_xyxy_by_tid or {}
        occupied = set(int(v) for v in occupied_seat_nos)
        assignments = {}
        used_people = set()
        used_seats = set(occupied)
        left_col = min(self.zones_by_col) if self.zones_by_col else 0

        people_items = []
        for person in people:
            tid = int(person["track_id"])
            head_anchor = head_anchor_pts.get(tid)
            if head_anchor is None:
                continue
            col, lateral_gap = self._pick_column_by_lateral_order(head_anchor)
            if col is None:
                continue
            line_dist = self._axis_distance(head_anchor, int(col))
            line_limit = max(
                float(min_line),
                self.col_line_limit.get(int(col), 180.0) * float(line_scale),
            )
            if line_dist > line_limit:
                continue
            people_items.append(
                {
                    "person": person,
                    "tid": tid,
                    "col": int(col),
                    "anchor": head_anchor,
                    "depth": self._project_depth(head_anchor, int(col)),
                    "line_dist": float(line_dist),
                    "lateral_gap": float(lateral_gap),
                    "line_limit": float(line_limit),
                }
            )

        for col in sorted(self.zones_by_col):
            col_people = [item for item in people_items if int(item["col"]) == int(col)]
            col_people = [item for item in col_people if int(item["tid"]) not in used_people]
            col_zones = [
                zone for zone in sorted(
                    self.zones_by_col[col],
                    key=lambda z: self.col_zone_depths[int(col)][int(z["desk_no"])],
                )
                if int(zone["desk_no"]) not in used_seats
            ]
            if not col_people or not col_zones:
                continue

            col_people.sort(key=lambda item: item["depth"])
            zone_depths = [
                self.col_zone_depths[int(col)][int(zone["desk_no"])]
                for zone in col_zones
            ]
            depth_limit = max(
                float(min_depth),
                self.col_depth_limit.get(int(col), 150.0) * float(depth_scale),
            )
            typical_step = self.col_depth_limit.get(int(col), 150.0)
            cost_matrix = np.full(
                (len(col_people), len(col_zones)),
                1e6,
                dtype=np.float32,
            )
            info_matrix = {}
            people_den = max(1, len(col_people) - 1)
            zone_den = max(1, len(col_zones) - 1)
            padding = (
                float(first_col_seat_head_padding)
                if int(col) == int(left_col)
                else float(seat_head_padding)
            )

            for row, item in enumerate(col_people):
                tid = int(item["tid"])
                head_anchor = item["anchor"]
                head_xyxy = item["person"]["xyxy"]
                body_xyxy = body_xyxy_by_tid.get(tid)
                for col_idx, zone in enumerate(col_zones):
                    seat_no = int(zone["desk_no"])
                    zone_depth = zone_depths[col_idx]
                    depth_gap = abs(float(zone_depth) - float(item["depth"]))
                    if depth_gap > depth_limit:
                        continue

                    head_poly = expand_polygon_by_padding(zone["polygon"], padding)
                    head_signed = point_polygon_signed_distance(head_anchor, head_poly)
                    head_overlap = xyxy_polygon_area_ratio(head_xyxy, head_poly)
                    head_hit = (
                        head_signed >= -float(seat_head_max_outside)
                        or head_overlap >= float(seat_head_min_overlap)
                    )

                    body_overlap = None
                    body_ok = True
                    if body_xyxy is not None:
                        body_poly = expand_polygon_by_padding(zone["polygon"], float(seat_body_padding))
                        body_overlap = xyxy_polygon_area_ratio(body_xyxy, body_poly)
                        body_ok = body_overlap >= float(seat_body_min_overlap)
                    if not body_ok:
                        continue

                    center_dist = l2(head_anchor, zone["zone_center"])
                    rank_gap = abs((row / people_den) - (col_idx / zone_den))
                    score = (
                        float(item["line_dist"]) * 1.20
                        + depth_gap * 0.95
                        + center_dist * 0.18
                        + rank_gap * typical_step * float(order_weight)
                    )
                    if head_hit:
                        score -= float(zone_hit_bonus)
                    score -= max(0.0, float(head_signed)) * 0.20
                    cost_matrix[row, col_idx] = float(score)
                    info_matrix[(row, col_idx)] = {
                        "line_dist": float(item["line_dist"]),
                        "depth_gap": float(depth_gap),
                        "score": float(score),
                        "head_signed_distance": float(head_signed),
                        "head_overlap": float(head_overlap),
                        "body_overlap": None if body_overlap is None else float(body_overlap),
                        "rank_gap": float(rank_gap),
                        "head_hit": bool(head_hit),
                    }

            max_accept_cost = max(
                float(min_line) * 1.3,
                depth_limit * 1.8 + typical_step * float(order_weight),
            )
            for row, col_idx in self._solve_cost_matrix(cost_matrix):
                if row >= len(col_people) or col_idx >= len(col_zones):
                    continue
                info = info_matrix.get((row, col_idx))
                if info is None or float(cost_matrix[row, col_idx]) > max_accept_cost:
                    continue
                item = col_people[row]
                tid = int(item["tid"])
                zone = col_zones[col_idx]
                seat_no = int(zone["desk_no"])
                if tid in used_people or seat_no in used_seats:
                    continue
                assignment = self._assignment_from_zone(
                    tid,
                    zone,
                    item["anchor"],
                    info["line_dist"],
                    info["depth_gap"],
                    info["score"],
                    "column_order",
                    match_info=info,
                )
                assignment["column_order_rank_gap"] = round(float(info["rank_gap"]), 4)
                assignment["column_order_head_hit"] = bool(info["head_hit"])
                assignments[tid] = assignment
                used_people.add(tid)
                used_seats.add(seat_no)

        return assignments

    def assign_head_line_fit_batch(
        self,
        people: list,
        head_anchor_pts: dict[int, np.ndarray],
        occupied_seat_nos: set[int],
        min_angle: float = 150.0,
        max_angle: float = 180.0,
        max_candidates: int = 11,
        line_scale: float = 2.4,
        first_col_line_scale: float = 1.6,
        min_line: float = 220.0,
        depth_scale: float = 1.35,
        min_depth: float = 150.0,
        max_rms: float = 55.0,
        min_count: int = 3,
        max_seat_distance: float = 0.0,
        head_depth_by_tid: dict[int, float] | None = None,
        seat_depth_by_no: dict[int, float] | None = None,
        depth_max_diff: float = 0.0,
        depth_weight: float = 0.0,
        body_xyxy_by_tid: dict[int, list] | None = None,
        body_seat_min_overlap: float = 0.02,
        body_seat_weight: float = 120.0,
    ):
        import itertools

        def zone_for_projected_depth(col_zones, col_depths, depth_value, tolerance):
            if not col_zones:
                return None, float("inf"), None
            if depth_value < col_depths[0]:
                gap = abs(float(col_depths[0]) - float(depth_value))
                return (col_zones[0], gap, "before_first") if gap <= tolerance else (None, gap, None)
            if depth_value > col_depths[-1]:
                gap = abs(float(depth_value) - float(col_depths[-1]))
                return (col_zones[-1], gap, "after_last") if gap <= tolerance else (None, gap, None)
            for idx in range(len(col_depths) - 1):
                start = float(col_depths[idx])
                end = float(col_depths[idx + 1])
                lo = min(start, end)
                hi = max(start, end)
                if lo <= float(depth_value) <= hi:
                    center_gap = min(abs(float(depth_value) - start), abs(float(depth_value) - end))
                    return col_zones[idx], center_gap, "between_seats"
            gaps = [abs(float(depth_value) - float(v)) for v in col_depths]
            best_idx = int(np.argmin(gaps))
            return (
                col_zones[best_idx],
                float(gaps[best_idx]),
                "nearest_center",
            ) if gaps[best_idx] <= tolerance else (None, float(gaps[best_idx]), None)

        people_by_tid = {int(person["track_id"]): person for person in people}
        head_depth_by_tid = head_depth_by_tid or {}
        seat_depth_by_no = seat_depth_by_no or {}
        body_xyxy_by_tid = body_xyxy_by_tid or {}
        used_people = set()
        used_seats = set(int(v) for v in occupied_seat_nos)
        assignments = {}
        left_col = min(self.zones_by_col) if self.zones_by_col else 0
        min_angle = float(min_angle)
        max_angle = min(180.0, float(max_angle))
        if max_angle < min_angle:
            min_angle, max_angle = max_angle, min_angle

        for col in sorted(self.zones_by_col):
            all_zones = sorted(
                self.zones_by_col[col],
                key=lambda z: self.col_zone_depths[col][int(z["desk_no"])],
            )
            if not all_zones:
                continue

            col_line_scale = float(line_scale)
            if int(col) == int(left_col):
                col_line_scale *= float(first_col_line_scale)
            line_limit = max(
                float(min_line),
                self.col_line_limit.get(col, 180.0) * col_line_scale,
            )
            depth_limit = max(
                float(min_depth),
                self.col_depth_limit.get(col, 150.0) * float(depth_scale),
            )

            candidates = []
            zone_depth_values = [self.col_zone_depths[col][int(z["desk_no"])] for z in all_zones]
            min_zone_depth = min(zone_depth_values) - depth_limit
            max_zone_depth = max(zone_depth_values) + depth_limit
            seat_close_limit = (
                float(max_seat_distance)
                if float(max_seat_distance) > 0
                else max(90.0, self.col_depth_limit.get(col, 150.0) * 0.90)
            )
            for tid, person in people_by_tid.items():
                if tid in used_people:
                    continue
                anchor = head_anchor_pts.get(tid)
                if anchor is None:
                    continue
                lateral_col, lateral_gap = self._pick_column_by_lateral_order(anchor)
                if int(lateral_col) != int(col):
                    continue
                axis_dist = self._axis_distance(anchor, col)
                if axis_dist > line_limit:
                    continue
                depth_value = self._project_depth(anchor, col)
                if depth_value < min_zone_depth or depth_value > max_zone_depth:
                    continue
                projected_zone, projection_gap, projection_region = zone_for_projected_depth(
                    all_zones,
                    zone_depth_values,
                    depth_value,
                    depth_limit,
                )
                if projected_zone is None:
                    continue
                seat_no = int(projected_zone["desk_no"])
                if seat_no in used_seats:
                    continue
                seat_box_distance = point_xyxy_distance(anchor, projected_zone["desk_xyxy"])
                if seat_box_distance > seat_close_limit:
                    continue
                body_overlap = None
                body_xyxy = body_xyxy_by_tid.get(int(tid))
                if body_xyxy is not None:
                    body_poly = projected_zone.get("display_polygon") or projected_zone.get("polygon")
                    body_overlap = xyxy_polygon_area_ratio(body_xyxy, body_poly)
                    if body_overlap < float(body_seat_min_overlap):
                        continue
                depth_prior_head = head_depth_by_tid.get(int(tid))
                depth_prior_seat = seat_depth_by_no.get(int(seat_no))
                depth_prior_diff = None
                if depth_prior_head is not None and depth_prior_seat is not None:
                    if np.isfinite(depth_prior_head) and np.isfinite(depth_prior_seat):
                        depth_prior_diff = abs(float(depth_prior_head) - float(depth_prior_seat))
                        if float(depth_max_diff) > 0 and depth_prior_diff > float(depth_max_diff):
                            continue
                nearest_bonus = -40.0 if lateral_col == col else 0.0
                candidate_score = (
                    axis_dist
                    + float(projection_gap) * 0.45
                    + float(lateral_gap) * 0.35
                    + float(seat_box_distance) * 0.85
                    + (float(depth_prior_diff) * float(depth_weight) if depth_prior_diff is not None else 0.0)
                    - (float(body_overlap) * float(body_seat_weight) if body_overlap is not None else 0.0)
                    + nearest_bonus
                )
                candidates.append(
                    {
                        "track_id": tid,
                        "person": person,
                        "anchor": anchor,
                        "axis_dist": float(axis_dist),
                        "depth": float(depth_value),
                        "nearest_depth_gap": float(projection_gap),
                        "nearest_col": lateral_col,
                        "projected_zone": projected_zone,
                        "projection_region": projection_region,
                        "seat_box_distance": float(seat_box_distance),
                        "depth_prior_head": None if depth_prior_head is None else float(depth_prior_head),
                        "depth_prior_seat": None if depth_prior_seat is None else float(depth_prior_seat),
                        "depth_prior_diff": depth_prior_diff,
                        "body_seat_overlap": body_overlap,
                        "score": float(candidate_score),
                    }
                )

            if len(candidates) < int(min_count):
                continue
            candidates.sort(key=lambda item: (item["score"], item["track_id"]))
            candidates = candidates[: max(int(min_count), int(max_candidates))]

            best = None
            max_group_size = min(len(candidates), len(all_zones), int(max_candidates))
            min_group_size = min(int(min_count), max_group_size)
            for group_size in range(max_group_size, min_group_size - 1, -1):
                for group in itertools.combinations(candidates, group_size):
                    ordered = sorted(group, key=lambda item: item["depth"])
                    angles = [
                        three_point_angle_deg(a["anchor"], b["anchor"], c["anchor"])
                        for a, b, c in zip(ordered[:-2], ordered[1:-1], ordered[2:])
                    ]
                    min_group_angle = min(angles) if angles else 180.0
                    max_group_angle = max(angles) if angles else 180.0
                    if min_group_angle < min_angle or max_group_angle > max_angle:
                        continue
                    rms = line_fit_rms([item["anchor"] for item in ordered])
                    if rms > float(max_rms):
                        continue

                    score = rms * 3.0 + max(0.0, 180.0 - min_group_angle) * 8.0
                    pair_rows = []
                    seen_seats = set()
                    failed = False
                    for item in ordered:
                        zone = item["projected_zone"]
                        seat_no = int(zone["desk_no"])
                        if seat_no in seen_seats or seat_no in used_seats:
                            failed = True
                            break
                        depth_gap = abs(self.col_zone_depths[col][seat_no] - item["depth"])
                        if depth_gap > depth_limit:
                            failed = True
                            break
                        score += (
                            depth_gap * 0.75
                            + item["axis_dist"] * 0.85
                            + item["seat_box_distance"] * 1.10
                            + (
                                float(item["depth_prior_diff"]) * float(depth_weight)
                                if item.get("depth_prior_diff") is not None
                                else 0.0
                            )
                            - (
                                float(item["body_seat_overlap"]) * float(body_seat_weight)
                                if item.get("body_seat_overlap") is not None
                                else 0.0
                            )
                        )
                        if item["nearest_col"] == col:
                            score -= 35.0
                        seen_seats.add(seat_no)
                        pair_rows.append((item, zone, depth_gap))
                    if failed:
                        continue
                    best_key = (-len(pair_rows), score)
                    if best is None or best_key < best[0]:
                        best = (best_key, score, min_group_angle, max_group_angle, rms, pair_rows)
                if best is not None:
                    break

            if best is None:
                continue

            _, score, min_group_angle, max_group_angle, rms, pair_rows = best
            for item, zone, depth_gap in pair_rows:
                tid = int(item["track_id"])
                seat_no = int(zone["desk_no"])
                if tid in used_people or seat_no in used_seats:
                    continue
                assignment = self._assignment_from_zone(
                    tid,
                    zone,
                    item["anchor"],
                    item["axis_dist"],
                    depth_gap,
                    score,
                    "head_line_perp_projection",
                )
                assignment["head_column_min_angle"] = round(float(min_group_angle), 3)
                assignment["head_column_max_angle"] = round(float(max_group_angle), 3)
                assignment["head_column_rms"] = round(float(rms), 4)
                assignment["head_projected_depth"] = round(float(item["depth"]), 4)
                assignment["head_projection_region"] = item["projection_region"]
                assignment["head_seat_box_distance"] = round(float(item["seat_box_distance"]), 4)
                assignment["depth_prior_head"] = (
                    None
                    if item.get("depth_prior_head") is None
                    else round(float(item["depth_prior_head"]), 6)
                )
                assignment["depth_prior_seat"] = (
                    None
                    if item.get("depth_prior_seat") is None
                    else round(float(item["depth_prior_seat"]), 6)
                )
                assignment["depth_prior_diff"] = (
                    None
                    if item.get("depth_prior_diff") is None
                    else round(float(item["depth_prior_diff"]), 6)
                )
                assignment["body_seat_overlap"] = (
                    None
                    if item.get("body_seat_overlap") is None
                    else round(float(item["body_seat_overlap"]), 6)
                )
                assignments[tid] = assignment
                used_people.add(tid)
                used_seats.add(seat_no)

        return assignments

    def _assignment_from_zone(
        self,
        tid: int,
        zone: dict,
        head_anchor,
        line_dist: float,
        depth_gap: float,
        score: float,
        method: str,
        match_info: dict | None = None,
    ):
        assignment = {
            "desk_id": zone["desk_id"],
            "desk_no": int(zone["desk_no"]),
            "col_idx": int(zone["column_index"]),
            "row_idx": int(zone["row_index"]),
            "cost": round(float(score), 4),
            "head_line_binding": True,
            "binding_method": method,
            "head_depth_gap": round(float(depth_gap), 4),
            "head_line_distance": round(float(line_dist), 4),
            "anchor_x": round(float(head_anchor[0]), 2),
            "anchor_y": round(float(head_anchor[1]), 2),
            "zone_cx": round(float(zone["zone_center"][0]), 2),
            "zone_cy": round(float(zone["zone_center"][1]), 2),
        }
        if match_info is not None:
            assignment["seat_match_head_signed_distance"] = round(
                float(match_info.get("head_signed_distance", 0.0)),
                4,
            )
            body_overlap = match_info.get("body_overlap")
            assignment["seat_match_body_overlap"] = (
                None if body_overlap is None else round(float(body_overlap), 6)
            )
            head_overlap = match_info.get("head_overlap")
            if head_overlap is not None:
                assignment["seat_match_head_overlap"] = round(float(head_overlap), 6)
            if "body_hit" in match_info:
                assignment["seat_match_body_hit"] = bool(match_info.get("body_hit"))
        return assignment

    def assignment_for_seat(
        self,
        track_id: int,
        seat_no: int,
        head_anchor,
        method: str = "bound_sticky",
    ):
        for col, col_zones in self.zones_by_col.items():
            for zone in col_zones:
                if int(zone["desk_no"]) != int(seat_no):
                    continue
                line_item = self.column_line_by_col.get(int(col))
                if line_item is not None:
                    line_dist = point_line_distance_kb(head_anchor, line_item["line_kb"])
                else:
                    line_dist = abs(float(head_anchor[0]) - self.col_center_x[int(col)])
                depth_value = self._project_depth(head_anchor, int(col))
                depth_gap = abs(self.col_zone_depths[int(col)][int(seat_no)] - depth_value)
                score = depth_gap + line_dist * 1.8 + l2(head_anchor, zone["zone_center"]) * 0.12
                return self._assignment_from_zone(
                    int(track_id),
                    zone,
                    head_anchor,
                    line_dist,
                    depth_gap,
                    score,
                    method,
                )
        return None

    def is_anchor_near_seat(
        self,
        seat_no: int,
        head_anchor,
        line_scale: float,
        depth_scale: float,
        center_scale: float,
    ) -> tuple[bool, dict | None]:
        assignment = self.assignment_for_seat(-1, seat_no, head_anchor, method="bound_sticky")
        if assignment is None:
            return False, None
        col = int(assignment["col_idx"])
        zone = None
        for candidate in self.zones_by_col.get(col, []):
            if int(candidate["desk_no"]) == int(seat_no):
                zone = candidate
                break
        if zone is None:
            return False, assignment
        line_limit = self.col_line_limit.get(col, 180.0) * float(line_scale)
        depth_limit = self.col_depth_limit.get(col, 150.0) * float(depth_scale)
        center_limit = max(line_limit, depth_limit) * float(center_scale)
        center_dist = l2(head_anchor, zone["zone_center"])
        ok = (
            float(assignment["head_line_distance"]) <= line_limit
            and float(assignment["head_depth_gap"]) <= depth_limit
            and center_dist <= center_limit
        )
        assignment["bound_sticky_center_distance"] = round(float(center_dist), 4)
        assignment["bound_sticky_line_limit"] = round(float(line_limit), 4)
        assignment["bound_sticky_depth_limit"] = round(float(depth_limit), 4)
        return ok, assignment

    def is_anchor_closest_to_seat(
        self,
        seat_no: int,
        head_anchor,
        max_rank_gap: int = 0,
    ) -> tuple[bool, dict | None]:
        target_assignment = self.assignment_for_seat(
            -1,
            int(seat_no),
            head_anchor,
            method="bound_sticky_closest",
        )
        if target_assignment is None:
            return False, None

        ranked = []
        for col, col_zones in self.zones_by_col.items():
            for zone in col_zones:
                candidate_seat = int(zone["desk_no"])
                assignment = self.assignment_for_seat(
                    -1,
                    candidate_seat,
                    head_anchor,
                    method="closest_check",
                )
                if assignment is None:
                    continue
                center_dist = l2(head_anchor, zone["desk_center"])
                box_dist = point_xyxy_distance(head_anchor, zone["desk_xyxy"])
                score = (
                    float(assignment["head_line_distance"]) * 1.25
                    + float(assignment["head_depth_gap"]) * 1.0
                    + center_dist * 0.22
                    + box_dist * 0.90
                )
                ranked.append((float(score), candidate_seat, center_dist, box_dist))

        if not ranked:
            return False, target_assignment
        ranked.sort(key=lambda item: (item[0], item[1]))
        target_rank = next(
            (idx for idx, item in enumerate(ranked) if int(item[1]) == int(seat_no)),
            None,
        )
        closest_score, closest_seat, _, _ = ranked[0]
        target_score = float(target_assignment["cost"])
        for score, candidate_seat, center_dist, box_dist in ranked:
            if int(candidate_seat) == int(seat_no):
                target_score = float(score)
                target_assignment["bound_closest_rank"] = int(target_rank or 0)
                target_assignment["bound_closest_best_seat"] = int(closest_seat)
                target_assignment["bound_closest_best_score"] = round(float(closest_score), 4)
                target_assignment["bound_closest_score"] = round(float(score), 4)
                target_assignment["bound_closest_center_distance"] = round(float(center_dist), 4)
                target_assignment["bound_closest_box_distance"] = round(float(box_dist), 4)
                break

        ok = target_rank is not None and int(target_rank) <= int(max_rank_gap)
        return ok, target_assignment

    def is_anchor_far_from_seat(
        self,
        seat_no: int,
        head_anchor,
        line_scale: float,
        depth_scale: float,
        center_scale: float,
    ) -> tuple[bool, dict | None]:
        assignment = self.assignment_for_seat(-1, seat_no, head_anchor, method="bound_locked")
        if assignment is None:
            return True, None
        col = int(assignment["col_idx"])
        zone = None
        for candidate in self.zones_by_col.get(col, []):
            if int(candidate["desk_no"]) == int(seat_no):
                zone = candidate
                break
        if zone is None:
            return True, assignment
        line_limit = self.col_line_limit.get(col, 180.0) * float(line_scale)
        depth_limit = self.col_depth_limit.get(col, 150.0) * float(depth_scale)
        center_limit = max(line_limit, depth_limit) * float(center_scale)
        center_dist = l2(head_anchor, zone["zone_center"])
        far = (
            float(assignment["head_line_distance"]) > line_limit
            or float(assignment["head_depth_gap"]) > depth_limit
            or center_dist > center_limit
        )
        assignment["large_move_center_distance"] = round(float(center_dist), 4)
        assignment["large_move_line_limit"] = round(float(line_limit), 4)
        assignment["large_move_depth_limit"] = round(float(depth_limit), 4)
        return far, assignment

    def assign_projection_batch(
        self,
        people: list,
        head_anchor_pts: dict[int, np.ndarray],
        occupied_seat_nos: set[int],
        line_scale: float,
        depth_scale: float,
        min_line: float,
        min_depth: float,
        body_xyxy_by_tid: dict[int, list] | None = None,
        seat_head_padding: float = 35.0,
        first_col_seat_head_padding: float = 35.0,
        seat_head_max_outside: float = 35.0,
        seat_body_padding: float = 20.0,
        seat_body_min_overlap: float = 0.025,
    ):
        body_xyxy_by_tid = body_xyxy_by_tid or {}
        candidates = []
        for person in people:
            tid = int(person["track_id"])
            head_anchor = head_anchor_pts.get(tid)
            if head_anchor is None:
                continue
            col, line_dist = self._pick_column(head_anchor)
            if col is None:
                continue

            line_limit = max(float(min_line), self.col_line_limit.get(col, 180.0) * float(line_scale))
            if line_dist > line_limit:
                continue

            depth_value = self._project_depth(head_anchor, col)
            best = None
            for zone in self.zones_by_col.get(col, []):
                seat_no = int(zone["desk_no"])
                if seat_no in occupied_seat_nos:
                    continue
                depth_gap = abs(self.col_zone_depths[col][seat_no] - depth_value)
                depth_limit = max(float(min_depth), self.col_depth_limit.get(col, 150.0) * float(depth_scale))
                if depth_gap > depth_limit:
                    continue
                match_ok, match_info = self.seat_match_ok(
                    zone,
                    head_anchor,
                    body_xyxy=body_xyxy_by_tid.get(tid),
                    head_padding=seat_head_padding,
                    first_col_head_padding=first_col_seat_head_padding,
                    max_head_outside=seat_head_max_outside,
                    body_padding=seat_body_padding,
                    body_min_overlap=seat_body_min_overlap,
                )
                if not match_ok:
                    continue
                score = depth_gap + line_dist * 2.0 + l2(head_anchor, zone["zone_center"]) * 0.08
                item = (score, depth_gap, line_dist, tid, zone, match_info)
                if best is None or item < best:
                    best = item
            if best is not None:
                candidates.append(best)

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        used_people = set()
        used_seats = set(occupied_seat_nos)
        assignments = {}
        for score, depth_gap, line_dist, tid, zone, match_info in candidates:
            seat_no = int(zone["desk_no"])
            if tid in used_people or seat_no in used_seats:
                continue
            assignments[tid] = self._assignment_from_zone(
                tid,
                zone,
                head_anchor_pts[tid],
                line_dist,
                depth_gap,
                score,
                "projection_fallback",
                match_info=match_info,
            )
            used_people.add(tid)
            used_seats.add(seat_no)
        return assignments

def tune_head_line_assigner(assigner, line_scale: float, depth_scale: float, min_line: float, min_depth: float):
    for col in list(assigner.col_line_limit):
        assigner.col_line_limit[col] = max(
            float(min_line),
            float(assigner.col_line_limit[col]) * float(line_scale),
        )
    for col in list(assigner.col_depth_limit):
        assigner.col_depth_limit[col] = max(
            float(min_depth),
            float(assigner.col_depth_limit[col]) * float(depth_scale),
        )
    return assigner


class TrackMotionFilter:
    """
    Classify raw tracker IDs as stationary or moving from recent head anchors.

    Binding is allowed only after a track has stayed relatively still for a
    short window. This keeps walking teachers/proctors from being absorbed into
    nearby seats while still allowing seated students to bind after tracker
    jitter settles.
    """

    def __init__(
        self,
        fps: float,
        window_seconds: float = 2.0,
        moving_distance: float = 120.0,
        moving_speed: float = 45.0,
        stationary_distance: float = 45.0,
        stationary_seconds: float = 1.2,
        bind_cooldown_seconds: float = 8.0,
    ):
        self.fps = max(1.0, float(fps))
        self.window_frames = max(2, int(round(float(window_seconds) * self.fps)))
        self.moving_distance = float(moving_distance)
        self.moving_speed = float(moving_speed)
        self.stationary_distance = float(stationary_distance)
        self.stationary_frames = max(1, int(round(float(stationary_seconds) * self.fps)))
        self.bind_cooldown_frames = max(0, int(round(float(bind_cooldown_seconds) * self.fps)))
        self.history: dict[int, deque] = {}
        self.states: dict[int, dict] = {}

    def update(self, track_id: int, frame_idx: int, anchor) -> dict:
        tid = int(track_id)
        point = np.asarray(anchor, dtype=np.float32)
        hist = self.history.setdefault(tid, deque())
        hist.append((int(frame_idx), point))
        min_frame = int(frame_idx) - self.window_frames
        while len(hist) > 1 and int(hist[0][0]) < min_frame:
            hist.popleft()

        if len(hist) >= 2:
            elapsed_frames = max(1, int(hist[-1][0]) - int(hist[0][0]))
            net_dist = l2(hist[0][1], hist[-1][1])
            speed = net_dist / max(1e-6, elapsed_frames / self.fps)
        else:
            net_dist = 0.0
            speed = 0.0

        prev = self.states.get(
            tid,
            {
                "stationary_count": 0,
                "moving_count": 0,
                "motion_state": "unknown",
                "last_moving_frame": None,
            },
        )
        moving = net_dist >= self.moving_distance or speed >= self.moving_speed
        stationary = net_dist <= self.stationary_distance and len(hist) >= 2

        if moving:
            moving_count = int(prev.get("moving_count", 0)) + 1
            stationary_count = 0
            motion_state = "moving"
            last_moving_frame = int(frame_idx)
        elif stationary:
            stationary_count = int(prev.get("stationary_count", 0)) + 1
            moving_count = 0
            motion_state = "stationary" if stationary_count >= self.stationary_frames else "settling"
            last_moving_frame = prev.get("last_moving_frame")
        else:
            stationary_count = 0
            moving_count = 0
            motion_state = "settling"
            last_moving_frame = prev.get("last_moving_frame")

        in_cooldown = (
            last_moving_frame is not None
            and int(frame_idx) - int(last_moving_frame) <= self.bind_cooldown_frames
        )
        can_bind = bool(motion_state == "stationary" and not in_cooldown)

        state = {
            "motion_state": motion_state,
            "is_moving": bool(moving),
            "can_bind": can_bind,
            "motion_bind_cooldown": bool(in_cooldown),
            "stationary_count": int(stationary_count),
            "moving_count": int(moving_count),
            "last_moving_frame": last_moving_frame,
            "motion_net_distance": float(net_dist),
            "motion_speed": float(speed),
        }
        self.states[tid] = state
        return state

    def get(self, track_id: int) -> dict:
        return self.states.get(
            int(track_id),
            {
                "motion_state": "unknown",
                "is_moving": False,
                "can_bind": False,
                "motion_bind_cooldown": False,
                "stationary_count": 0,
                "moving_count": 0,
                "last_moving_frame": None,
                "motion_net_distance": 0.0,
                "motion_speed": 0.0,
            },
        )


class EvidenceSeatManager:
    """
    Seat state machine with evidence-based bind, switch, and release.

    A track must gather enough sliding-window votes for one seat before binding.
    A confirmed track keeps its seat through short misses and minor drift.
    It is released only after sustained large-movement evidence, or after the
    whole track has been missing longer than the hold window.
    """

    def __init__(
        self,
        fps: float,
        initial_bind_seconds: float = 3.0,
        switch_seconds: float = 6.0,
        release_seconds: float = 8.0,
        miss_hold_seconds: float = 12.0,
        reacquire_seconds: float = 1.0,
        large_move_release_seconds: float | None = None,
        evidence_vote_window_seconds: float = 2.5,
        initial_vote_ratio: float = 0.65,
        switch_vote_ratio: float = 0.80,
    ):
        self.fps = max(1.0, float(fps))
        self.initial_confirm_frames = max(1, int(round(float(initial_bind_seconds) * self.fps)))
        self.switch_confirm_frames = max(1, int(round(float(switch_seconds) * self.fps)))
        self.release_confirm_frames = max(1, int(round(float(release_seconds) * self.fps)))
        self.miss_hold_frames = max(self.release_confirm_frames, int(round(float(miss_hold_seconds) * self.fps)))
        self.reacquire_confirm_frames = max(1, int(round(float(reacquire_seconds) * self.fps)))
        self.evidence_vote_window_frames = max(
            self.initial_confirm_frames,
            int(round(float(evidence_vote_window_seconds) * self.fps)),
        )
        self.switch_vote_window_frames = max(
            self.switch_confirm_frames,
            self.evidence_vote_window_frames,
        )
        self.initial_vote_ratio = max(0.0, min(1.0, float(initial_vote_ratio)))
        self.switch_vote_ratio = max(0.0, min(1.0, float(switch_vote_ratio)))
        large_move_seconds = (
            float(release_seconds)
            if large_move_release_seconds is None
            else float(large_move_release_seconds)
        )
        self.large_move_release_frames = max(
            self.release_confirm_frames,
            int(round(large_move_seconds * self.fps)),
        )
        self.states: dict[int, dict] = {}
        self.seat_owner: dict[int, int] = {}
        self.ever_bound_seats: set[int] = set()

    def _state(self, track_id: int):
        if track_id not in self.states:
            self.states[track_id] = {
                "bound_seat_no": None,
                "last_seen_frame": -1,
                "last_bound_frame": -1,
                "pending_seat_no": None,
                "pending_count": 0,
                "release_count": 0,
                "large_move_count": 0,
                "soft_unmatched_count": 0,
                "evidence_history": deque(),
                "evidence_vote_ratio": 0.0,
            }
        return self.states[track_id]

    def _clear_pending(self, track_id: int):
        st = self._state(track_id)
        st["pending_seat_no"] = None
        st["pending_count"] = 0

    def _push_vote(
        self,
        track_id: int,
        frame_idx: int,
        seat_no: int,
        window_frames: int | None = None,
    ) -> tuple[int, int, float]:
        st = self._state(track_id)
        hist = st["evidence_history"]
        hist.append((int(frame_idx), int(seat_no)))
        self._prune_votes(track_id, frame_idx, window_frames=window_frames)
        votes = Counter(int(item[1]) for item in hist)
        best_seat, best_count = votes.most_common(1)[0]
        valid_count = max(1, len(hist))
        ratio = float(best_count) / float(valid_count)
        st["pending_seat_no"] = int(best_seat)
        st["pending_count"] = int(best_count)
        st["evidence_vote_ratio"] = ratio
        return int(best_seat), int(best_count), ratio

    def _prune_votes(self, track_id: int, frame_idx: int, window_frames: int | None = None):
        st = self._state(track_id)
        hist = st["evidence_history"]
        vote_window = self.evidence_vote_window_frames if window_frames is None else int(window_frames)
        min_frame = int(frame_idx) - vote_window
        while hist and int(hist[0][0]) < min_frame:
            hist.popleft()

    def _clear_votes(self, track_id: int):
        st = self._state(track_id)
        st["evidence_history"].clear()
        st["evidence_vote_ratio"] = 0.0

    def _is_track_active(self, track_id: int, frame_idx: int) -> bool:
        st = self.states.get(int(track_id))
        return st is not None and int(frame_idx) - int(st["last_seen_frame"]) <= self.miss_hold_frames

    def _release_track(self, track_id: int):
        tid = int(track_id)
        st = self.states.get(tid)
        if st is None:
            return
        seat_no = st.get("bound_seat_no")
        if seat_no is not None and self.seat_owner.get(int(seat_no)) == tid:
            del self.seat_owner[int(seat_no)]
        st["bound_seat_no"] = None
        st["last_bound_frame"] = -1
        st["release_count"] = 0
        st["large_move_count"] = 0
        st["soft_unmatched_count"] = 0
        self._clear_pending(tid)
        self._clear_votes(tid)

    def cleanup(self, frame_idx: int):
        for tid, st in list(self.states.items()):
            if st.get("bound_seat_no") is None:
                continue
            if int(frame_idx) - int(st["last_seen_frame"]) > self.miss_hold_frames:
                self._release_track(tid)

    def _seat_is_available(self, seat_no: int, track_id: int, frame_idx: int) -> bool:
        seat_no = int(seat_no)
        tid = int(track_id)
        owner = self.seat_owner.get(seat_no)
        if owner is None or owner == tid:
            return True
        if not self._is_track_active(owner, frame_idx):
            self._release_track(owner)
            return True
        return False

    def _bind(self, track_id: int, seat_no: int, frame_idx: int):
        tid = int(track_id)
        seat_no = int(seat_no)
        st = self._state(tid)
        old_seat = st.get("bound_seat_no")
        if old_seat is not None and self.seat_owner.get(int(old_seat)) == tid:
            del self.seat_owner[int(old_seat)]

        owner = self.seat_owner.get(seat_no)
        if owner is not None and owner != tid:
            self._release_track(owner)

        st["bound_seat_no"] = seat_no
        st["last_bound_frame"] = int(frame_idx)
        st["release_count"] = 0
        st["large_move_count"] = 0
        st["soft_unmatched_count"] = 0
        self.seat_owner[seat_no] = tid
        self.ever_bound_seats.add(seat_no)
        self._clear_pending(tid)
        self._clear_votes(tid)

    def update(
        self,
        track_id: int,
        frame_idx: int,
        proposed_seat_no: int | None,
        release_evidence: str | None = None,
        allow_occupied_takeover: bool = False,
    ):
        tid = int(track_id)
        self.cleanup(frame_idx)
        st = self._state(tid)
        st["last_seen_frame"] = int(frame_idx)
        current_seat = st.get("bound_seat_no")

        if current_seat is None:
            if proposed_seat_no is None:
                self._prune_votes(
                    tid,
                    frame_idx,
                    window_frames=self.evidence_vote_window_frames,
                )
                self._clear_pending(tid)
                return None
            proposed = int(proposed_seat_no)
            best_seat, count, vote_ratio = self._push_vote(
                tid,
                frame_idx,
                proposed,
                window_frames=self.evidence_vote_window_frames,
            )
            if not self._seat_is_available(proposed, tid, frame_idx):
                if not allow_occupied_takeover:
                    self._clear_votes(tid)
                    self._clear_pending(tid)
                    return None
                required = self.reacquire_confirm_frames
                if count >= required and best_seat == proposed:
                    self._bind(tid, proposed, frame_idx)
                return proposed

            if not self._seat_is_available(best_seat, tid, frame_idx):
                self._clear_pending(tid)
                return None
            required = (
                self.reacquire_confirm_frames
                if best_seat in self.ever_bound_seats
                else self.initial_confirm_frames
            )
            required_ratio = (
                min(self.initial_vote_ratio, 0.55)
                if best_seat in self.ever_bound_seats
                else self.initial_vote_ratio
            )
            if count >= required and vote_ratio >= required_ratio:
                self._bind(tid, best_seat, frame_idx)
                return best_seat
            return None

        if proposed_seat_no is None:
            self._prune_votes(
                tid,
                frame_idx,
                window_frames=self.switch_vote_window_frames,
            )
            self._clear_pending(tid)
            if release_evidence == "large_move":
                st["large_move_count"] += 1
                st["release_count"] = st["large_move_count"]
            else:
                st["soft_unmatched_count"] += 1
                st["release_count"] = 0
                st["large_move_count"] = 0
            if st["large_move_count"] >= self.large_move_release_frames:
                self._release_track(tid)
                return None
            return int(current_seat)

        proposed = int(proposed_seat_no)
        if proposed == int(current_seat):
            st["release_count"] = 0
            st["large_move_count"] = 0
            st["soft_unmatched_count"] = 0
            st["last_bound_frame"] = int(frame_idx)
            self._clear_pending(tid)
            self._clear_votes(tid)
            return int(current_seat)

        st["release_count"] = 0
        st["large_move_count"] = 0
        st["soft_unmatched_count"] = 0
        best_seat, count, vote_ratio = self._push_vote(
            tid,
            frame_idx,
            proposed,
            window_frames=self.switch_vote_window_frames,
        )
        if best_seat == int(current_seat):
            self._clear_pending(tid)
            self._clear_votes(tid)
            return int(current_seat)
        if not self._seat_is_available(best_seat, tid, frame_idx):
            self._clear_pending(tid)
            return int(current_seat)
        if count >= self.switch_confirm_frames and vote_ratio >= self.switch_vote_ratio:
            self._bind(tid, best_seat, frame_idx)
            return best_seat
        return int(current_seat)

    def get_display_seat(self, track_id: int, frame_idx: int):
        self.cleanup(frame_idx)
        st = self.states.get(int(track_id))
        if st is None:
            return None
        return st.get("bound_seat_no")

    def debug_state(self, track_id: int):
        st = self.states.get(int(track_id))
        return dict(st) if st is not None else {}


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


def draw_desk_center_lines(img, desks, num_cols=None):
    out = img
    colors = [
        (255, 80, 80),
        (80, 210, 80),
        (80, 120, 255),
        (255, 200, 80),
        (220, 80, 255),
    ]
    by_col: dict[int, list] = {}
    for desk in desks or []:
        by_col.setdefault(int(desk["column_index"]), []).append(desk)

    col_ids = sorted(by_col)
    if num_cols is not None:
        col_ids = list(range(int(num_cols)))

    for col in col_ids:
        col_desks = sorted(
            by_col.get(int(col), []),
            key=lambda d: int(d["row_index"]),
        )
        if not col_desks:
            continue
        color = colors[int(col) % len(colors)]
        centers = [
            tuple(int(round(v)) for v in desk["center"])
            for desk in col_desks
        ]
        for p1, p2 in zip(centers[:-1], centers[1:]):
            cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
        for point in centers:
            cv2.circle(out, point, 3, color, -1, cv2.LINE_AA)
        end = centers[-1]
        draw_label(out, f"C{int(col) + 1}", end[0] + 6, end[1], color, scale=0.45)
    return out


def draw_layout(
    img,
    desks,
    zones,
    column_lines,
    student_area_polygon=None,
    draw_student_area=False,
    draw_seat_zones=False,
):
    out = img.copy()
    if draw_seat_zones:
        for zone in zones:
            display_poly = zone.get("display_polygon")
            if display_poly is None:
                continue
            poly = np.asarray(display_poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [poly], True, (80, 190, 255), 1, cv2.LINE_AA)
    out = draw_desk_center_lines(out, desks)

    if draw_student_area and student_area_polygon is not None:
        area_poly = np.asarray(student_area_polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [area_poly], True, (255, 255, 255), 2, cv2.LINE_AA)

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


def write_layout_preview(
    output_dir,
    video_name,
    reference,
    desks,
    zones,
    column_lines,
    student_area_polygon=None,
    draw_student_area=False,
    draw_seat_zones=False,
):
    preview = draw_layout(
        reference["frame"],
        desks,
        zones,
        column_lines,
        student_area_polygon=student_area_polygon,
        draw_student_area=draw_student_area,
        draw_seat_zones=draw_seat_zones,
    )
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
                "track_id": int(tid),
                "student_id": int(seat_no),
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
            "track_id": int(seat_no),
            "student_id": int(seat_no),
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
    head_student_area_polygon = build_student_area_polygon(
        desks,
        padding=args.student_area_padding,
    )
    head_student_area_polygon = build_first_column_head_area_polygon(
        desks,
        head_student_area_polygon,
        pad_x=args.first_column_head_pad_x,
        pad_y=args.first_column_head_pad_y,
    )
    layout_preview = write_layout_preview(
        args.output,
        video_name,
        reference,
        desks,
        zones,
        column_lines,
        student_area_polygon=head_student_area_polygon,
        draw_student_area=args.draw_student_area,
        draw_seat_zones=args.draw_seat_zones,
    )
    print(f"Desk layout frame: {reference['frame_idx']}")
    print(f"Desks: {len(desks)}, column lines: {len(column_lines)}")
    print(f"Layout preview: {layout_preview}")

    assigner = HeadLineSeatAssigner(zones=zones, column_lines=column_lines)
    tune_head_line_assigner(
        assigner,
        line_scale=args.line_margin_scale,
        depth_scale=args.depth_margin_scale,
        min_line=args.min_line_margin,
        min_depth=args.min_depth_margin,
    )

    model = YOLO(args.weights)
    names = getattr(model, "names", None)
    print(f"Loaded head/person model: {args.weights}")
    print(f"Model classes: {names}")

    depth_prior = None
    latest_depth_frame = -1
    if args.use_depth_prior:
        print("Ignoring --use-depth-prior: this script now binds by detection only.")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detect_duration_frames = (
        max(1, int(round(float(args.detect_duration_seconds) * fps)))
        if args.detect_duration_seconds and args.detect_duration_seconds > 0
        else 0
    )
    process_frame_limit = 0
    if args.max_process_frames > 0 and detect_duration_frames > 0:
        process_frame_limit = min(int(args.max_process_frames), int(detect_duration_frames))
    elif args.max_process_frames > 0:
        process_frame_limit = int(args.max_process_frames)
    elif detect_duration_frames > 0:
        process_frame_limit = int(detect_duration_frames)
    bind_start_frame = (
        max(0, int(round(args.bind_start_seconds * fps)))
        if args.bind_start_seconds is not None
        else max(0, int(round(args.desk_reference_seconds * fps)))
    )
    if process_frame_limit > 0:
        print(
            "Detection duration limit: "
            f"{process_frame_limit} frames ({process_frame_limit / fps:.3f}s)"
        )

    seat_manager = EvidenceSeatManager(
        fps=fps,
        initial_bind_seconds=args.initial_bind_seconds,
        switch_seconds=args.switch_seconds,
        release_seconds=args.release_seconds,
        miss_hold_seconds=args.miss_hold_seconds,
        reacquire_seconds=args.reacquire_seconds,
        large_move_release_seconds=args.large_move_release_seconds,
        evidence_vote_window_seconds=args.evidence_vote_window_seconds,
        initial_vote_ratio=args.initial_vote_ratio,
        switch_vote_ratio=args.switch_vote_ratio,
    )
    motion_filter = TrackMotionFilter(
        fps=fps,
        window_seconds=args.motion_window_seconds,
        moving_distance=args.motion_moving_distance,
        moving_speed=args.motion_moving_speed,
        stationary_distance=args.motion_stationary_distance,
        stationary_seconds=args.motion_stationary_seconds,
        bind_cooldown_seconds=args.motion_bind_cooldown_seconds,
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
    body_classes = set(parse_classes(args.body_classes) or [])
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
            "track_id",
            "student_id",
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
            "role",
            "inside_student_area",
            "pending_seat_no",
            "pending_count",
            "evidence_vote_ratio",
            "release_count",
            "large_move_count",
            "soft_unmatched_count",
            "release_evidence",
            "allow_occupied_takeover",
            "motion_state",
            "motion_can_bind",
            "motion_bind_cooldown",
            "motion_stationary_count",
            "motion_moving_count",
            "motion_net_distance",
            "motion_speed",
            "binding_method",
            "head_line_distance",
            "head_depth_gap",
            "depth_prior_head",
            "depth_prior_seat",
            "depth_prior_diff",
            "depth_prior_frame",
            "body_seat_overlap",
            "cost",
        ],
    )
    csv_writer.writeheader()

    frame_idx = 0
    try:
        while True:
            if process_frame_limit > 0 and frame_idx >= process_frame_limit:
                break
            ret, frame = cap.read()
            if not ret:
                break

            depth_norm = None

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

            annotated = (
                draw_layout(
                    frame,
                    desks,
                    zones,
                    column_lines,
                    student_area_polygon=head_student_area_polygon,
                    draw_student_area=args.draw_student_area,
                    draw_seat_zones=args.draw_seat_zones,
                )
                if args.draw_layout
                else frame.copy()
            )
            head_people = []
            head_anchor_pts = {}
            head_depth_by_tid = {}
            head_dets = {}
            body_dets = []
            inside_student_area_by_tid = {}
            body_xyxy_by_tid = {}

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
                if tid < 0:
                    continue

                if is_body_detection(det, names, body_classes, head_classes, args.head_area_max):
                    det["body_anchor"] = xyxy_center(det["xyxy"])
                    body_dets.append(det)

                if is_head_detection(det, names, head_classes, args.head_area_max):
                    anchor = head_anchor_from_box(
                        det["xyxy"],
                        mode=args.head_anchor,
                        y_offset=args.head_y_offset,
                    )
                    inside_student_area = point_in_polygon(anchor, head_student_area_polygon)
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
                    inside_student_area_by_tid[tid] = inside_student_area

            for body in body_dets:
                bx1, by1, bx2, by2 = [float(v) for v in body["xyxy"]]
                bw = max(1.0, bx2 - bx1)
                bh = max(1.0, by2 - by1)
                margin = max(16.0, bw * 0.18)
                body_center = xyxy_center(body["xyxy"])
                for tid, anchor in head_anchor_pts.items():
                    same_track = int(tid) == int(body["track_id"])
                    anchor_inside = point_in_xyxy(anchor, body["xyxy"], margin=margin)
                    head_body_gap = l2(anchor, body_center)
                    near_body = (
                        abs(float(anchor[0]) - float(body_center[0])) <= bw * 0.75
                        and head_body_gap <= max(bw, bh) * 0.95
                    )
                    if not (same_track or anchor_inside or near_body):
                        continue
                    prev_body = body_xyxy_by_tid.get(int(tid))
                    if (
                        prev_body is None
                        or same_track
                        or box_area(body["xyxy"]) < box_area(prev_body)
                    ):
                        body_xyxy_by_tid[int(tid)] = body["xyxy"]
            for tid in head_dets:
                inside_student_area_by_tid[int(tid)] = True
            visible_head_track_ids = set(int(tid) for tid in head_dets)
            motion_by_tid = {
                int(tid): motion_filter.update(int(tid), frame_idx, anchor)
                for tid, anchor in head_anchor_pts.items()
            }
            bindable_motion_track_ids = {
                tid for tid, info in motion_by_tid.items()
                if bool(info.get("can_bind", False))
            }

            seat_manager.cleanup(frame_idx)

            locked_assignments = {}
            locked_seats = set()
            bound_track_ids = set()
            large_move_track_ids = set()
            if frame_idx >= bind_start_frame:
                for person in head_people:
                    tid = int(person["track_id"])
                    bound_seat = seat_manager.get_display_seat(tid, frame_idx)
                    if bound_seat is None:
                        continue
                    anchor = head_anchor_pts.get(tid)
                    if anchor is None:
                        continue
                    far_from_bound, locked_assignment = assigner.is_anchor_far_from_seat(
                        int(bound_seat),
                        anchor,
                        line_scale=args.large_move_line_scale,
                        depth_scale=args.large_move_depth_scale,
                        center_scale=args.large_move_center_scale,
                    )
                    if not far_from_bound and locked_assignment is not None:
                        bound_track_ids.add(tid)
                        locked_seats.add(int(bound_seat))
                        locked_assignment["binding_method"] = "bound_sticky_until_far"
                        locked_assignments[tid] = locked_assignment
                    else:
                        large_move_track_ids.add(tid)

            if frame_idx >= bind_start_frame:
                bindable_head_people = [
                    person for person in head_people
                    if int(person["track_id"]) not in bound_track_ids
                ]
                assignments = assigner.assign_left_down_desk_batch(
                    people=bindable_head_people,
                    head_anchor_pts=head_anchor_pts,
                    occupied_seat_nos=set(locked_seats),
                    max_right_offset=args.left_down_max_right_offset,
                    max_up_offset=args.left_down_max_up_offset,
                    max_box_distance=args.left_down_max_box_distance,
                    column_order_min_count=args.left_down_column_order_min_count,
                    column_order_weight=args.left_down_column_order_weight,
                )
            else:
                assignments = {}
            assignments.update(locked_assignments)

            for person in head_people:
                tid = int(person["track_id"])
                det = head_dets[tid]
                anchor = head_anchor_pts[tid]
                inside_student_area = inside_student_area_by_tid.get(tid, True)
                assignment = assignments.get(tid)
                bound_seat = seat_manager.get_display_seat(tid, frame_idx)
                allow_occupied_takeover = False
                if (
                    args.occupied_handoff
                    and
                    bound_seat is None
                    and assignment is None
                    and tid in bindable_motion_track_ids
                ):
                    handoff_assignment = None
                    handoff_score = None
                    for occupied_seat, owner_tid in seat_manager.seat_owner.items():
                        if int(owner_tid) == tid or int(owner_tid) in visible_head_track_ids:
                            continue
                        owner_state = seat_manager.states.get(int(owner_tid), {})
                        last_seen = int(owner_state.get("last_seen_frame", -1))
                        if (
                            last_seen >= 0
                            and int(frame_idx) - last_seen < int(round(args.occupied_handoff_missing_seconds * fps))
                        ):
                            continue
                        near_occupied, occupied_assignment = assigner.is_anchor_near_seat(
                            int(occupied_seat),
                            anchor,
                            line_scale=args.handoff_line_scale,
                            depth_scale=args.handoff_depth_scale,
                            center_scale=args.handoff_center_scale,
                        )
                        if not near_occupied or occupied_assignment is None:
                            continue
                        body_xyxy = body_xyxy_by_tid.get(int(tid))
                        if body_xyxy is not None:
                            target_zone = next(
                                (
                                    z for z in zones
                                    if int(z["desk_no"]) == int(occupied_seat)
                                ),
                                None,
                            )
                            if target_zone is not None:
                                body_poly = target_zone.get("display_polygon") or target_zone.get("polygon")
                                body_overlap = xyxy_polygon_area_ratio(body_xyxy, body_poly)
                                if body_overlap < float(args.occupied_handoff_body_min_overlap):
                                    continue
                                occupied_assignment["body_seat_overlap"] = round(float(body_overlap), 6)
                        score = float(occupied_assignment.get("cost", 0.0))
                        if handoff_assignment is None or score < float(handoff_score):
                            handoff_assignment = occupied_assignment
                            handoff_score = score
                    if handoff_assignment is not None:
                        handoff_assignment["binding_method"] = "occupied_handoff"
                        assignment = handoff_assignment
                        allow_occupied_takeover = True
                current_seat = int(assignment["desk_no"]) if assignment else None
                release_evidence = (
                    "large_move" if tid in large_move_track_ids and current_seat is None else None
                )
                display_seat = (
                    seat_manager.update(
                        tid,
                        frame_idx,
                        current_seat,
                        release_evidence=release_evidence,
                        allow_occupied_takeover=allow_occupied_takeover,
                    )
                    if frame_idx >= bind_start_frame
                    else None
                )
                seat_debug = seat_manager.debug_state(tid)
                motion_info = motion_by_tid.get(tid, motion_filter.get(tid))

                if display_seat is not None:
                    track_seat_votes.setdefault(tid, Counter())[int(display_seat)] += 1
                    seat_track_votes.setdefault(int(display_seat), Counter())[tid] += 1

                stable_track_id = int(display_seat) if display_seat is not None else None
                candidate_track_id = int(current_seat) if current_seat is not None else None
                x1, y1, x2, y2 = det["xyxy"]
                color = (0, 255, 0) if display_seat is not None else (0, 220, 255)
                if frame_idx < bind_start_frame:
                    color = (180, 180, 180)
                    label = "layout"
                elif display_seat is not None:
                    label = f"ID{stable_track_id:02d}"
                elif assignment:
                    label = f"ID{candidate_track_id:02d} pending"
                elif motion_info.get("motion_state") == "moving":
                    color = (80, 180, 255)
                    label = f"raw{tid} moving"
                elif motion_info.get("motion_bind_cooldown"):
                    color = (80, 180, 255)
                    label = f"raw{tid} cooldown"
                elif not inside_student_area:
                    color = (255, 160, 80)
                    label = f"raw{tid} outside"
                else:
                    color = (0, 0, 255)
                    label = f"raw{tid} unbound"

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
                        "track_id": stable_track_id,
                        "student_id": stable_track_id,
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
                        "role": "student" if display_seat is not None else "candidate",
                        "inside_student_area": bool(inside_student_area),
                        "pending_seat_no": seat_debug.get("pending_seat_no"),
                        "pending_count": seat_debug.get("pending_count"),
                        "evidence_vote_ratio": round(float(seat_debug.get("evidence_vote_ratio", 0.0)), 4),
                        "release_count": seat_debug.get("release_count"),
                        "large_move_count": seat_debug.get("large_move_count"),
                        "soft_unmatched_count": seat_debug.get("soft_unmatched_count"),
                        "release_evidence": release_evidence,
                        "allow_occupied_takeover": allow_occupied_takeover,
                        "motion_state": motion_info.get("motion_state"),
                        "motion_can_bind": bool(motion_info.get("can_bind", False)),
                        "motion_bind_cooldown": bool(motion_info.get("motion_bind_cooldown", False)),
                        "motion_stationary_count": motion_info.get("stationary_count"),
                        "motion_moving_count": motion_info.get("moving_count"),
                        "motion_net_distance": round(float(motion_info.get("motion_net_distance", 0.0)), 4),
                        "motion_speed": round(float(motion_info.get("motion_speed", 0.0)), 4),
                        "binding_method": assignment.get("binding_method") if assignment else None,
                        "head_line_distance": assignment.get("head_line_distance") if assignment else None,
                        "head_depth_gap": assignment.get("head_depth_gap") if assignment else None,
                        "depth_prior_head": (
                            assignment.get("depth_prior_head")
                            if assignment else head_depth_by_tid.get(int(tid))
                        ),
                        "depth_prior_seat": assignment.get("depth_prior_seat") if assignment else None,
                        "depth_prior_diff": assignment.get("depth_prior_diff") if assignment else None,
                        "depth_prior_frame": latest_depth_frame if depth_norm is not None else None,
                        "body_seat_overlap": assignment.get("body_seat_overlap") if assignment else None,
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
        "detect_duration_seconds": (
            None
            if not args.detect_duration_seconds or args.detect_duration_seconds <= 0
            else float(args.detect_duration_seconds)
        ),
        "process_frame_limit": int(process_frame_limit),
        "desk_reference_seconds": float(args.desk_reference_seconds),
        "bind_start_frame": int(bind_start_frame),
        "bind_start_seconds": round(bind_start_frame / fps, 3),
        "desk_reference_frame": int(reference["frame_idx"]),
        "desk_count": int(len(desks)),
        "column_line_count": int(len(column_lines)),
        "head_classes": sorted(int(v) for v in head_classes),
        "head_anchor": args.head_anchor,
        "head_y_offset": float(args.head_y_offset),
        "student_area_padding": float(args.student_area_padding),
        "first_column_head_pad_x": float(args.first_column_head_pad_x),
        "first_column_head_pad_y": float(args.first_column_head_pad_y),
        "draw_seat_zones": bool(args.draw_seat_zones),
        "seat_head_padding": float(args.seat_head_padding),
        "first_col_seat_head_padding": float(args.first_col_seat_head_padding),
        "seat_head_max_outside": float(args.seat_head_max_outside),
        "seat_head_min_overlap": float(args.seat_head_min_overlap),
        "seat_body_padding": float(args.seat_body_padding),
        "seat_body_min_overlap": float(args.seat_body_min_overlap),
        "seat_body_bind_min_overlap": float(args.seat_body_bind_min_overlap),
        "student_area_polygon": (
            np.asarray(head_student_area_polygon).astype(float).tolist()
            if head_student_area_polygon is not None else None
        ),
        "head_student_area_polygon": (
            np.asarray(head_student_area_polygon).astype(float).tolist()
            if head_student_area_polygon is not None else None
        ),
        "line_margin_scale": float(args.line_margin_scale),
        "depth_margin_scale": float(args.depth_margin_scale),
        "min_line_margin": float(args.min_line_margin),
        "min_depth_margin": float(args.min_depth_margin),
        "assignment_method": "head_left_down_desk_detection",
        "left_down_max_right_offset": float(args.left_down_max_right_offset),
        "left_down_max_up_offset": float(args.left_down_max_up_offset),
        "left_down_max_box_distance": float(args.left_down_max_box_distance),
        "left_down_column_order_min_count": int(args.left_down_column_order_min_count),
        "left_down_column_order_weight": float(args.left_down_column_order_weight),
        "column_order_line_scale": float(args.column_order_line_scale),
        "column_order_depth_scale": float(args.column_order_depth_scale),
        "column_order_min_line": float(args.column_order_min_line),
        "column_order_min_depth": float(args.column_order_min_depth),
        "column_order_rank_weight": float(args.column_order_rank_weight),
        "column_order_zone_hit_bonus": float(args.column_order_zone_hit_bonus),
        "head_fit_min_angle": float(args.head_fit_min_angle),
        "head_fit_max_angle": float(args.head_fit_max_angle),
        "head_fit_max_candidates": int(args.head_fit_max_candidates),
        "head_fit_line_scale": float(args.head_fit_line_scale),
        "head_fit_first_col_line_scale": float(args.head_fit_first_col_line_scale),
        "head_fit_min_line": float(args.head_fit_min_line),
        "head_fit_depth_scale": float(args.head_fit_depth_scale),
        "head_fit_min_depth": float(args.head_fit_min_depth),
        "head_fit_max_rms": float(args.head_fit_max_rms),
        "head_fit_min_count": int(args.head_fit_min_count),
        "head_fit_max_seat_distance": float(args.head_fit_max_seat_distance),
        "use_depth_prior": False,
        "depth_prior_ignored": bool(args.use_depth_prior),
        "depth_weights": args.depth_weights,
        "depth_encoder": args.depth_encoder,
        "depth_stride": int(args.depth_stride),
        "depth_input_size": int(args.depth_input_size),
        "depth_max_res": int(args.depth_max_res),
        "depth_low_percentile": float(args.depth_low_percentile),
        "depth_high_percentile": float(args.depth_high_percentile),
        "depth_head_sample_ratio": float(args.depth_head_sample_ratio),
        "depth_seat_sample_ratio": float(args.depth_seat_sample_ratio),
        "depth_max_diff": float(args.depth_max_diff),
        "depth_weight": float(args.depth_weight),
        "body_seat_min_overlap": float(args.body_seat_min_overlap),
        "body_seat_weight": float(args.body_seat_weight),
        "occupied_handoff": bool(args.occupied_handoff),
        "occupied_handoff_missing_seconds": float(args.occupied_handoff_missing_seconds),
        "occupied_handoff_body_min_overlap": float(args.occupied_handoff_body_min_overlap),
        "projection_fallback": bool(args.projection_fallback),
        "projection_line_scale": float(args.projection_line_scale),
        "projection_depth_scale": float(args.projection_depth_scale),
        "projection_min_line": float(args.projection_min_line),
        "projection_min_depth": float(args.projection_min_depth),
        "sticky_line_scale": float(args.sticky_line_scale),
        "sticky_depth_scale": float(args.sticky_depth_scale),
        "sticky_center_scale": float(args.sticky_center_scale),
        "sticky_closest_rank_gap": int(args.sticky_closest_rank_gap),
        "handoff_line_scale": float(args.handoff_line_scale),
        "handoff_depth_scale": float(args.handoff_depth_scale),
        "handoff_center_scale": float(args.handoff_center_scale),
        "motion_window_seconds": float(args.motion_window_seconds),
        "motion_moving_distance": float(args.motion_moving_distance),
        "motion_moving_speed": float(args.motion_moving_speed),
        "motion_stationary_distance": float(args.motion_stationary_distance),
        "motion_stationary_seconds": float(args.motion_stationary_seconds),
        "motion_bind_cooldown_seconds": float(args.motion_bind_cooldown_seconds),
        "large_move_line_scale": float(args.large_move_line_scale),
        "large_move_depth_scale": float(args.large_move_depth_scale),
        "large_move_center_scale": float(args.large_move_center_scale),
        "initial_bind_seconds": float(args.initial_bind_seconds),
        "switch_seconds": float(args.switch_seconds),
        "release_seconds": float(args.release_seconds),
        "large_move_release_seconds": float(args.large_move_release_seconds),
        "miss_hold_seconds": float(args.miss_hold_seconds),
        "evidence_vote_window_seconds": float(args.evidence_vote_window_seconds),
        "initial_vote_ratio": float(args.initial_vote_ratio),
        "switch_vote_ratio": float(args.switch_vote_ratio),
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
                "display_polygon": (
                    np.asarray(z["display_polygon"]).astype(float).tolist()
                    if z.get("display_polygon") is not None else None
                ),
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

    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--iou", type=float, default=0.60)
    p.add_argument("--desk-conf", type=float, default=0.70)
    p.add_argument("--desk-iou", type=float, default=0.45)
    p.add_argument("--device", default="")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--half", action="store_true")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--classes", default=None, help="Optional YOLO class IDs to track, comma-separated")
    p.add_argument("--head-classes", default=None, help="Head class IDs, comma-separated; auto by class name if empty")
    p.add_argument("--body-classes", default=None, help="Person/body class IDs, comma-separated; auto by class name if empty")
    p.add_argument("--head-area-max", type=float, default=6500.0, help="Fallback head area limit when class names are unclear")
    p.add_argument("--head-anchor", choices=["center", "top", "bottom"], default="center")
    p.add_argument("--head-y-offset", type=float, default=0.0, help="Pixel offset added to the selected head anchor y")
    p.add_argument(
        "--left-down-max-right-offset",
        type=float,
        default=120.0,
        help="Desk center may be this many pixels to the right of the head and still count as lower-left",
    )
    p.add_argument(
        "--left-down-max-up-offset",
        type=float,
        default=80.0,
        help="Desk center may be this many pixels above the head and still count as lower-left",
    )
    p.add_argument(
        "--left-down-max-box-distance",
        type=float,
        default=0.0,
        help="Maximum head-anchor to desk-box distance; 0 uses an automatic desk-size based limit",
    )
    p.add_argument(
        "--left-down-column-order-min-count",
        type=int,
        default=3,
        help="Use same-column head order correction when at least this many heads have candidates in a desk column",
    )
    p.add_argument(
        "--left-down-column-order-weight",
        type=float,
        default=0.85,
        help="Cost weight for preserving front-back order within a desk column",
    )

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
        "--last-extend",
        type=float,
        default=1.2,
        help="Only used for the farthest row, which has no next desk to connect",
    )
    p.add_argument(
        "--student-area-padding",
        type=float,
        default=90.0,
        help="Pixels added around the desk-layout hull for head binding",
    )
    p.add_argument(
        "--first-column-head-pad-x",
        type=float,
        default=180.0,
        help="Extra horizontal expansion for the leftmost column head-binding area",
    )
    p.add_argument(
        "--first-column-head-pad-y",
        type=float,
        default=70.0,
        help="Extra vertical expansion for the leftmost column head-binding area",
    )
    p.add_argument(
        "--seat-head-padding",
        type=float,
        default=0.0,
        help="Pixels used to expand each individual seat zone when matching a head to its proposed seat",
    )
    p.add_argument(
        "--first-col-seat-head-padding",
        type=float,
        default=0.0,
        help="Seat-zone head padding for the leftmost column",
    )
    p.add_argument(
        "--seat-head-max-outside",
        type=float,
        default=0.0,
        help="Maximum pixels a head anchor may be outside its expanded seat zone",
    )
    p.add_argument(
        "--seat-head-min-overlap",
        type=float,
        default=1.01,
        help="Minimum head-box overlap ratio with a seat zone when the head anchor is outside",
    )
    p.add_argument(
        "--seat-body-padding",
        type=float,
        default=0.0,
        help="Pixels used to expand each seat zone when matching person/body boxes",
    )
    p.add_argument(
        "--seat-body-min-overlap",
        type=float,
        default=0.0,
        help="Minimum body-box overlap with the proposed seat zone when a body box is available",
    )
    p.add_argument(
        "--seat-body-bind-min-overlap",
        type=float,
        default=0.08,
        help="Minimum body-box overlap ratio that can bind a head track to a seat zone",
    )
    p.add_argument(
        "--line-margin-scale",
        type=float,
        default=1.55,
        help="Scale for column-line distance tolerance, allowing normal edge movement",
    )
    p.add_argument(
        "--depth-margin-scale",
        type=float,
        default=1.35,
        help="Scale for row-depth tolerance",
    )
    p.add_argument("--min-line-margin", type=float, default=190.0)
    p.add_argument("--min-depth-margin", type=float, default=150.0)
    p.add_argument(
        "--column-order-line-scale",
        type=float,
        default=2.8,
        help="Column membership tolerance for column-order assignment",
    )
    p.add_argument(
        "--column-order-depth-scale",
        type=float,
        default=2.2,
        help="Row-depth tolerance for column-order assignment",
    )
    p.add_argument("--column-order-min-line", type=float, default=260.0)
    p.add_argument("--column-order-min-depth", type=float, default=230.0)
    p.add_argument(
        "--column-order-rank-weight",
        type=float,
        default=0.65,
        help="Cost weight for preserving front-to-back order within each column",
    )
    p.add_argument(
        "--column-order-zone-hit-bonus",
        type=float,
        default=85.0,
        help="Cost reduction when a head anchor/box hits the front-back seat zone",
    )
    p.add_argument(
        "--head-fit-min-angle",
        type=float,
        default=150.0,
        help="Minimum three-point angle for a connected 6-head column fit",
    )
    p.add_argument(
        "--head-fit-max-angle",
        type=float,
        default=180.0,
        help="Maximum three-point angle for a connected 6-head column fit",
    )
    p.add_argument(
        "--head-fit-max-candidates",
        type=int,
        default=12,
        help="Maximum candidate heads evaluated for each desk column fit",
    )
    p.add_argument("--head-fit-line-scale", type=float, default=2.6)
    p.add_argument("--head-fit-first-col-line-scale", type=float, default=1.8)
    p.add_argument("--head-fit-min-line", type=float, default=230.0)
    p.add_argument("--head-fit-depth-scale", type=float, default=1.35)
    p.add_argument("--head-fit-min-depth", type=float, default=150.0)
    p.add_argument("--head-fit-max-rms", type=float, default=55.0)
    p.add_argument(
        "--head-fit-min-count",
        type=int,
        default=3,
        help="Require this many unbound heads in a desk column before fitting/binding",
    )
    p.add_argument(
        "--head-fit-max-seat-distance",
        type=float,
        default=0.0,
        help="Maximum distance from head center to its projected seat box; 0 uses an automatic row-step based limit",
    )
    p.add_argument(
        "--use-depth-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated; ignored because binding now uses detections only",
    )
    p.add_argument(
        "--depth-weights",
        default="exam_seat_binding/Video-Depth-Anything/model/video_depth_anything_vitb.pth",
        help="Video-Depth-Anything checkpoint path",
    )
    p.add_argument("--depth-encoder", choices=["vits", "vitb", "vitl"], default="vitb")
    p.add_argument("--depth-device", default="", help="Depth device; empty follows --device/auto")
    p.add_argument("--depth-input-size", type=int, default=518)
    p.add_argument("--depth-max-res", type=int, default=960, help="Resize largest frame side before depth inference; 0 disables")
    p.add_argument("--depth-stride", type=int, default=5, help="Run depth inference every N frames and reuse the latest map")
    p.add_argument("--depth-fp32", action="store_true", help="Run depth inference in fp32 instead of autocast")
    p.add_argument("--depth-low-percentile", type=float, default=5.0)
    p.add_argument("--depth-high-percentile", type=float, default=95.0)
    p.add_argument("--depth-head-sample-ratio", type=float, default=0.35)
    p.add_argument("--depth-seat-sample-ratio", type=float, default=0.35)
    p.add_argument(
        "--depth-max-diff",
        type=float,
        default=0.08,
        help="Reject head-seat candidates whose normalized monocular-depth gap is larger; <=0 disables rejection",
    )
    p.add_argument(
        "--depth-weight",
        type=float,
        default=180.0,
        help="Cost penalty weight for normalized monocular-depth gap",
    )
    p.add_argument(
        "--body-seat-min-overlap",
        type=float,
        default=0.02,
        help="When a person_visible box exists, require this overlap ratio with the projected seat area",
    )
    p.add_argument(
        "--body-seat-weight",
        type=float,
        default=120.0,
        help="Cost bonus weight for person_visible overlap with the projected seat area",
    )
    p.add_argument(
        "--occupied-handoff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow a new track ID to take over an occupied seat after the old owner has been missing",
    )
    p.add_argument(
        "--occupied-handoff-missing-seconds",
        type=float,
        default=2.0,
        help="Old seat owner must be missing this long before occupied handoff is considered",
    )
    p.add_argument(
        "--occupied-handoff-body-min-overlap",
        type=float,
        default=0.04,
        help="If a body box exists, require this body-seat overlap for occupied handoff",
    )
    p.add_argument(
        "--projection-line-scale",
        type=float,
        default=2.3,
        help="Relaxed column-line tolerance for unbound/unstable projection fallback",
    )
    p.add_argument(
        "--projection-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable relaxed one-head projection fallback after strict 6-head column fitting",
    )
    p.add_argument(
        "--projection-depth-scale",
        type=float,
        default=2.0,
        help="Relaxed row-depth tolerance for projection fallback",
    )
    p.add_argument("--projection-min-line", type=float, default=260.0)
    p.add_argument("--projection-min-depth", type=float, default=230.0)
    p.add_argument(
        "--sticky-line-scale",
        type=float,
        default=3.5,
        help="Large tolerance for keeping an already-bound student on the same seat",
    )
    p.add_argument(
        "--sticky-depth-scale",
        type=float,
        default=3.2,
        help="Large depth tolerance for keeping an already-bound student on the same seat",
    )
    p.add_argument("--sticky-center-scale", type=float, default=1.9)
    p.add_argument(
        "--sticky-closest-rank-gap",
        type=int,
        default=0,
        help="Keep a confirmed binding when its seat is within this nearest-seat rank for the current head",
    )
    p.add_argument(
        "--handoff-line-scale",
        type=float,
        default=3.2,
        help="Tolerance for reconnecting a new track ID to an occupied seat whose owner is temporarily missing",
    )
    p.add_argument(
        "--handoff-depth-scale",
        type=float,
        default=2.8,
        help="Depth tolerance for occupied-seat track handoff",
    )
    p.add_argument("--handoff-center-scale", type=float, default=1.8)
    p.add_argument(
        "--motion-window-seconds",
        type=float,
        default=2.0,
        help="Recent time window used to classify a head track as moving or stationary",
    )
    p.add_argument(
        "--motion-moving-distance",
        type=float,
        default=180.0,
        help="Net pixel displacement in the motion window that blocks new seat binding",
    )
    p.add_argument(
        "--motion-moving-speed",
        type=float,
        default=90.0,
        help="Pixel-per-second speed in the motion window that blocks new seat binding",
    )
    p.add_argument(
        "--motion-stationary-distance",
        type=float,
        default=80.0,
        help="Maximum net pixel displacement considered stationary for new seat binding",
    )
    p.add_argument(
        "--motion-stationary-seconds",
        type=float,
        default=1.2,
        help="Seconds a new/unbound track must remain stationary before it may bind to a seat",
    )
    p.add_argument(
        "--motion-bind-cooldown-seconds",
        type=float,
        default=2.0,
        help="Seconds after a moving track is detected before it may bind to a seat",
    )
    p.add_argument("--large-move-line-scale", type=float, default=6.0)
    p.add_argument("--large-move-depth-scale", type=float, default=5.0)
    p.add_argument("--large-move-center-scale", type=float, default=3.0)

    p.add_argument("--initial-bind-seconds", type=float, default=1.0)
    p.add_argument("--switch-seconds", type=float, default=10.0)
    p.add_argument("--release-seconds", type=float, default=8.0)
    p.add_argument(
        "--large-move-release-seconds",
        type=float,
        default=30.0,
        help="Seconds of continuous large movement required before releasing a confirmed binding",
    )
    p.add_argument("--miss-hold-seconds", type=float, default=60.0)
    p.add_argument("--reacquire-seconds", type=float, default=1.0)
    p.add_argument(
        "--evidence-vote-window-seconds",
        type=float,
        default=2.5,
        help="Sliding window used to vote candidate seats before binding or switching",
    )
    p.add_argument(
        "--initial-vote-ratio",
        type=float,
        default=0.65,
        help="Minimum dominant-seat vote ratio required for initial binding",
    )
    p.add_argument(
        "--switch-vote-ratio",
        type=float,
        default=0.80,
        help="Minimum dominant-seat vote ratio required before switching a confirmed seat",
    )

    p.add_argument("--draw-layout", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--draw-student-area", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--draw-seat-zones",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw adjacent-desk binding polygons in layout/video overlays",
    )
    p.add_argument("--no-video", action="store_true")
    p.add_argument(
        "--detect-duration-seconds",
        type=float,
        default=0.0,
        help="Only process this many seconds from the input video; 0 means full video",
    )
    p.add_argument("--max-process-frames", type=int, default=0, help="Debug only; 0 means full video")
    p.add_argument("--display", action="store_true")
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
