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
import sys
from collections import Counter
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
                    display_poly = None
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


def point_in_polygon(point_xy, polygon) -> bool:
    if polygon is None or len(polygon) < 3:
        return True
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(point_xy[0]), float(point_xy[1])), False) >= 0


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

        for zone in zones:
            col = int(zone["column_index"])
            self.zones_by_col.setdefault(col, []).append(zone)

        for col, col_zones in self.zones_by_col.items():
            col_zones.sort(key=lambda z: int(z["row_index"]))
            self.col_center_x[col] = float(
                np.mean([float(z["zone_center"][0]) for z in col_zones])
            )
            line_item = self.column_line_by_col.get(col)
            if line_item is not None:
                origin = np.asarray(
                    line_item.get("segment_start", col_zones[0]["zone_center"]),
                    dtype=np.float32,
                )
                far_point = np.asarray(
                    line_item.get("segment_end", col_zones[-1]["zone_center"]),
                    dtype=np.float32,
                )
                avg_step = float(line_item.get("avg_step", 0.0))
                avg_w = float(line_item.get("avg_box_width", 0.0))
                avg_h = float(line_item.get("avg_box_height", 0.0))
            else:
                origin = np.asarray(col_zones[0]["zone_center"], dtype=np.float32)
                far_point = np.asarray(col_zones[-1]["zone_center"], dtype=np.float32)
                avg_step = 0.0
                avg_w = 0.0
                avg_h = 0.0

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
                    np.asarray(zone["zone_center"], dtype=np.float32),
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

    def _project_depth(self, point_xy, col: int) -> float:
        point = np.asarray(point_xy, dtype=np.float32)
        return float(np.dot(point - self.col_origin[col], self.col_axis_unit[col]))

    def _pick_column(self, point_xy):
        if not self.zones_by_col:
            return None, float("inf")
        if self.column_lines:
            best_item = min(
                self.column_lines,
                key=lambda item: point_line_distance_kb(point_xy, item["line_kb"]),
            )
            col = int(best_item["column_index"])
            return col, point_line_distance_kb(point_xy, best_item["line_kb"])
        col = min(
            self.zones_by_col,
            key=lambda c: abs(float(point_xy[0]) - self.col_center_x[c]),
        )
        return col, abs(float(point_xy[0]) - self.col_center_x[col])

    def assign_batch(
        self,
        people: list,
        head_anchor_pts: dict[int, np.ndarray],
        occupied_seat_nos: set[int],
    ):
        candidates = []
        for person in people:
            tid = int(person["track_id"])
            head_anchor = head_anchor_pts.get(tid)
            if head_anchor is None:
                continue
            col, line_dist = self._pick_column(head_anchor)
            if col is None:
                continue
            if line_dist > self.col_line_limit.get(col, 180.0):
                continue

            depth_value = self._project_depth(head_anchor, col)
            for zone in self.zones_by_col.get(col, []):
                seat_no = int(zone["desk_no"])
                if seat_no in occupied_seat_nos:
                    continue
                depth_gap = abs(self.col_zone_depths[col][seat_no] - depth_value)
                if depth_gap > self.col_depth_limit.get(col, 150.0):
                    continue
                center_dist = l2(head_anchor, zone["zone_center"])
                score = depth_gap + line_dist * 1.8 + center_dist * 0.12
                candidates.append((score, depth_gap, line_dist, tid, zone))

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        used_people = set()
        used_seats = set(occupied_seat_nos)
        assignments = {}
        for score, depth_gap, line_dist, tid, zone in candidates:
            seat_no = int(zone["desk_no"])
            if tid in used_people or seat_no in used_seats:
                continue
            head_anchor = head_anchor_pts[tid]
            assignments[tid] = {
                "desk_id": zone["desk_id"],
                "desk_no": seat_no,
                "col_idx": zone["column_index"],
                "row_idx": zone["row_index"],
                "cost": round(float(score), 4),
                "head_line_binding": True,
                "binding_method": "strict_line_depth",
                "head_depth_gap": round(float(depth_gap), 4),
                "head_line_distance": round(float(line_dist), 4),
                "anchor_x": round(float(head_anchor[0]), 2),
                "anchor_y": round(float(head_anchor[1]), 2),
                "zone_cx": round(float(zone["zone_center"][0]), 2),
                "zone_cy": round(float(zone["zone_center"][1]), 2),
            }
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
    ):
        return {
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

    def assign_projection_batch(
        self,
        people: list,
        head_anchor_pts: dict[int, np.ndarray],
        occupied_seat_nos: set[int],
        line_scale: float,
        depth_scale: float,
        min_line: float,
        min_depth: float,
    ):
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
                score = depth_gap + line_dist * 2.0 + l2(head_anchor, zone["zone_center"]) * 0.08
                item = (score, depth_gap, line_dist, tid, zone)
                if best is None or item < best:
                    best = item
            if best is not None:
                candidates.append(best)

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        used_people = set()
        used_seats = set(occupied_seat_nos)
        assignments = {}
        for score, depth_gap, line_dist, tid, zone in candidates:
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
            )
            used_people.add(tid)
            used_seats.add(seat_no)
        return assignments

    def assign_column_sort_batch(
        self,
        people: list,
        head_anchor_pts: dict[int, np.ndarray],
        occupied_seat_nos: set[int],
        line_scale: float,
        min_line: float,
        depth_scale: float,
    ):
        people_by_col: dict[int, list] = {}
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
            people_by_col.setdefault(col, []).append(
                {
                    "track_id": tid,
                    "anchor": head_anchor,
                    "line_dist": float(line_dist),
                    "depth": float(depth_value),
                }
            )

        assignments = {}
        used_people = set()
        used_seats = set(occupied_seat_nos)
        for col in sorted(self.zones_by_col):
            zones = sorted(
                self.zones_by_col[col],
                key=lambda z: self.col_zone_depths[col][int(z["desk_no"])],
            )
            col_people = people_by_col.get(col, [])
            if not zones or not col_people:
                continue

            if len(col_people) >= len(zones):
                selected = sorted(col_people, key=lambda item: item["line_dist"])[: len(zones)]
                selected = sorted(selected, key=lambda item: item["depth"])
                for item, zone in zip(selected, zones):
                    tid = int(item["track_id"])
                    seat_no = int(zone["desk_no"])
                    if tid in used_people or seat_no in used_seats:
                        continue
                    depth_gap = abs(self.col_zone_depths[col][seat_no] - item["depth"])
                    score = depth_gap + item["line_dist"] * 0.8
                    assignments[tid] = self._assignment_from_zone(
                        tid,
                        zone,
                        item["anchor"],
                        item["line_dist"],
                        depth_gap,
                        score,
                        "column_sort_rank",
                    )
                    used_people.add(tid)
                    used_seats.add(seat_no)
                continue

            row_candidates = []
            for item in col_people:
                tid = int(item["track_id"])
                if tid in used_people:
                    continue
                for zone in zones:
                    seat_no = int(zone["desk_no"])
                    if seat_no in used_seats:
                        continue
                    depth_gap = abs(self.col_zone_depths[col][seat_no] - item["depth"])
                    depth_limit = self.col_depth_limit.get(col, 150.0) * float(depth_scale)
                    if depth_gap > depth_limit:
                        continue
                    score = depth_gap + item["line_dist"] * 0.8
                    row_candidates.append((score, depth_gap, item["line_dist"], tid, item, zone))

            row_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            for score, depth_gap, line_dist, tid, item, zone in row_candidates:
                seat_no = int(zone["desk_no"])
                if tid in used_people or seat_no in used_seats:
                    continue
                assignments[tid] = self._assignment_from_zone(
                    tid,
                    zone,
                    item["anchor"],
                    line_dist,
                    depth_gap,
                    score,
                    "column_sort_depth_nearest",
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


class TeacherRoleManager:
    """Lock teacher tracks by sustained evidence outside the desk-layout area."""

    def __init__(
        self,
        fps: float,
        max_teachers: int = 2,
        confirm_seconds: float = 2.0,
        window_seconds: float = 4.0,
        outside_ratio: float = 0.65,
    ):
        self.fps = max(1.0, float(fps))
        self.max_teachers = max(0, int(max_teachers))
        self.confirm_frames = max(1, int(round(float(confirm_seconds) * self.fps)))
        self.window_frames = max(self.confirm_frames, int(round(float(window_seconds) * self.fps)))
        self.outside_ratio = float(outside_ratio)
        self.states: dict[int, dict] = {}
        self.teacher_tracks: set[int] = set()

    def _state(self, track_id: int):
        if track_id not in self.states:
            self.states[track_id] = {
                "outside_history": [],
                "first_seen_frame": -1,
                "last_seen_frame": -1,
                "outside_frames": 0,
            }
        return self.states[track_id]

    def update(self, track_id: int, frame_idx: int, inside_student_area: bool, has_student_binding: bool):
        tid = int(track_id)
        st = self._state(tid)
        if st["first_seen_frame"] < 0:
            st["first_seen_frame"] = int(frame_idx)
        st["last_seen_frame"] = int(frame_idx)

        outside = not bool(inside_student_area)
        st["outside_history"].append(outside)
        if len(st["outside_history"]) > self.window_frames:
            st["outside_history"] = st["outside_history"][-self.window_frames:]
        if outside:
            st["outside_frames"] += 1

        if tid in self.teacher_tracks:
            return True
        if has_student_binding or self.max_teachers <= 0:
            return False
        if len(self.teacher_tracks) >= self.max_teachers:
            return False

        history = st["outside_history"]
        outside_count = sum(1 for value in history if value)
        if outside_count < self.confirm_frames:
            return False
        if outside_count / max(1, len(history)) < self.outside_ratio:
            return False

        self.teacher_tracks.add(tid)
        return True

    def is_teacher(self, track_id: int) -> bool:
        return int(track_id) in self.teacher_tracks

    def summary(self):
        return {
            str(tid): {
                "outside_frames": int(self.states.get(tid, {}).get("outside_frames", 0)),
                "first_seen_frame": int(self.states.get(tid, {}).get("first_seen_frame", -1)),
                "last_seen_frame": int(self.states.get(tid, {}).get("last_seen_frame", -1)),
            }
            for tid in sorted(self.teacher_tracks)
        }


class EvidenceSeatManager:
    """
    Seat state machine with evidence-based bind, switch, and release.

    A track must keep proposing the same seat for a while before binding.
    A confirmed track is released only after sustained missing/unbound evidence.
    """

    def __init__(
        self,
        fps: float,
        initial_bind_seconds: float = 3.0,
        switch_seconds: float = 6.0,
        release_seconds: float = 8.0,
        miss_hold_seconds: float = 12.0,
        reacquire_seconds: float = 1.0,
    ):
        self.fps = max(1.0, float(fps))
        self.initial_confirm_frames = max(1, int(round(float(initial_bind_seconds) * self.fps)))
        self.switch_confirm_frames = max(1, int(round(float(switch_seconds) * self.fps)))
        self.release_confirm_frames = max(1, int(round(float(release_seconds) * self.fps)))
        self.miss_hold_frames = max(self.release_confirm_frames, int(round(float(miss_hold_seconds) * self.fps)))
        self.reacquire_confirm_frames = max(1, int(round(float(reacquire_seconds) * self.fps)))
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
            }
        return self.states[track_id]

    def _clear_pending(self, track_id: int):
        st = self._state(track_id)
        st["pending_seat_no"] = None
        st["pending_count"] = 0

    def _push_pending(self, track_id: int, seat_no: int) -> int:
        st = self._state(track_id)
        if st["pending_seat_no"] == int(seat_no):
            st["pending_count"] += 1
        else:
            st["pending_seat_no"] = int(seat_no)
            st["pending_count"] = 1
        return int(st["pending_count"])

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
        self._clear_pending(tid)

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
        self.seat_owner[seat_no] = tid
        self.ever_bound_seats.add(seat_no)
        self._clear_pending(tid)

    def update(self, track_id: int, frame_idx: int, proposed_seat_no: int | None):
        tid = int(track_id)
        self.cleanup(frame_idx)
        st = self._state(tid)
        st["last_seen_frame"] = int(frame_idx)
        current_seat = st.get("bound_seat_no")

        if current_seat is None:
            if proposed_seat_no is None:
                self._clear_pending(tid)
                return None
            proposed = int(proposed_seat_no)
            if not self._seat_is_available(proposed, tid, frame_idx):
                self._clear_pending(tid)
                return None
            count = self._push_pending(tid, proposed)
            required = (
                self.reacquire_confirm_frames
                if proposed in self.ever_bound_seats
                else self.initial_confirm_frames
            )
            if count >= required:
                self._bind(tid, proposed, frame_idx)
                return proposed
            return None

        if proposed_seat_no is None:
            st["release_count"] += 1
            self._clear_pending(tid)
            if st["release_count"] >= self.release_confirm_frames:
                self._release_track(tid)
                return None
            return int(current_seat)

        proposed = int(proposed_seat_no)
        if proposed == int(current_seat):
            st["release_count"] = 0
            st["last_bound_frame"] = int(frame_idx)
            self._clear_pending(tid)
            return int(current_seat)

        st["release_count"] = 0
        if not self._seat_is_available(proposed, tid, frame_idx):
            self._clear_pending(tid)
            return int(current_seat)
        count = self._push_pending(tid, proposed)
        if count >= self.switch_confirm_frames:
            self._bind(tid, proposed, frame_idx)
            return proposed
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


def draw_layout(img, desks, zones, column_lines, student_area_polygon=None, draw_student_area=False):
    out = img.copy()
    for zone in zones:
        display_poly = zone.get("display_polygon")
        if display_poly is None:
            continue
        poly = np.asarray(display_poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], True, (80, 190, 255), 1, cv2.LINE_AA)
    out = draw_column_lines(out, column_lines, out.shape[0])

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
):
    preview = draw_layout(
        reference["frame"],
        desks,
        zones,
        column_lines,
        student_area_polygon=student_area_polygon,
        draw_student_area=draw_student_area,
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
    student_area_polygon = build_student_area_polygon(
        desks,
        padding=args.student_area_padding,
    )
    teacher_area_polygon = build_student_area_polygon(
        desks,
        padding=args.teacher_area_padding,
    )
    layout_preview = write_layout_preview(
        args.output,
        video_name,
        reference,
        desks,
        zones,
        column_lines,
        student_area_polygon=student_area_polygon,
        draw_student_area=args.draw_student_area,
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

    seat_manager = EvidenceSeatManager(
        fps=fps,
        initial_bind_seconds=args.initial_bind_seconds,
        switch_seconds=args.switch_seconds,
        release_seconds=args.release_seconds,
        miss_hold_seconds=args.miss_hold_seconds,
        reacquire_seconds=args.reacquire_seconds,
    )
    teacher_manager = TeacherRoleManager(
        fps=fps,
        max_teachers=args.max_teachers,
        confirm_seconds=args.teacher_confirm_seconds,
        window_seconds=args.teacher_window_seconds,
        outside_ratio=args.teacher_outside_ratio,
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
            "role",
            "inside_student_area",
            "inside_teacher_area",
            "pending_seat_no",
            "pending_count",
            "release_count",
            "binding_method",
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

            annotated = (
                draw_layout(
                    frame,
                    desks,
                    zones,
                    column_lines,
                    student_area_polygon=student_area_polygon,
                    draw_student_area=args.draw_student_area,
                )
                if args.draw_layout
                else frame.copy()
            )
            head_people = []
            head_anchor_pts = {}
            head_dets = {}
            inside_student_area_by_tid = {}

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
                inside_student_area = point_in_polygon(anchor, student_area_polygon)
                inside_teacher_area = point_in_polygon(anchor, teacher_area_polygon)
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
                det["inside_teacher_area"] = inside_teacher_area

            seat_manager.cleanup(frame_idx)
            teacher_by_tid = {}
            for person in head_people:
                tid = int(person["track_id"])
                has_student_binding = seat_manager.get_display_seat(tid, frame_idx) is not None
                teacher_by_tid[tid] = teacher_manager.update(
                    tid,
                    frame_idx,
                    head_dets.get(tid, {}).get("inside_teacher_area", True),
                    has_student_binding=has_student_binding,
                )

            if frame_idx >= bind_start_frame:
                bindable_head_people = [
                    person for person in head_people
                    if inside_student_area_by_tid.get(int(person["track_id"]), True)
                    and not teacher_by_tid.get(int(person["track_id"]), False)
                ]
                if args.assignment_mode == "column_sort":
                    assignments = assigner.assign_column_sort_batch(
                        people=bindable_head_people,
                        head_anchor_pts=head_anchor_pts,
                        occupied_seat_nos=set(),
                        line_scale=args.column_sort_line_scale,
                        min_line=args.column_sort_min_line,
                        depth_scale=args.column_sort_depth_scale,
                    )
                else:
                    assignments = assigner.assign_batch(
                        people=bindable_head_people,
                        head_anchor_pts=head_anchor_pts,
                        occupied_seat_nos=set(),
                    )
                    fallback_people = [
                        person for person in bindable_head_people
                        if int(person["track_id"]) not in assignments
                    ]
                    if fallback_people:
                        occupied_seats = {int(item["desk_no"]) for item in assignments.values()}
                        assignments.update(
                            assigner.assign_projection_batch(
                                people=fallback_people,
                                head_anchor_pts=head_anchor_pts,
                                occupied_seat_nos=occupied_seats,
                                line_scale=args.projection_line_scale,
                                depth_scale=args.projection_depth_scale,
                                min_line=args.projection_min_line,
                                min_depth=args.projection_min_depth,
                            )
                        )
            else:
                assignments = {}

            for person in head_people:
                tid = int(person["track_id"])
                det = head_dets[tid]
                anchor = head_anchor_pts[tid]
                inside_student_area = inside_student_area_by_tid.get(tid, True)
                inside_teacher_area = bool(det.get("inside_teacher_area", True))
                is_teacher = teacher_by_tid.get(tid, False)
                assignment = assignments.get(tid)
                bound_seat = seat_manager.get_display_seat(tid, frame_idx)
                if bound_seat is not None and not is_teacher:
                    near_bound, sticky_assignment = assigner.is_anchor_near_seat(
                        int(bound_seat),
                        anchor,
                        line_scale=args.sticky_line_scale,
                        depth_scale=args.sticky_depth_scale,
                        center_scale=args.sticky_center_scale,
                    )
                    if near_bound:
                        assignment = sticky_assignment
                current_seat = int(assignment["desk_no"]) if assignment else None
                if is_teacher:
                    display_seat = None
                else:
                    display_seat = (
                        seat_manager.update(tid, frame_idx, current_seat)
                        if frame_idx >= bind_start_frame
                        else None
                    )
                seat_debug = seat_manager.debug_state(tid)

                if display_seat is not None:
                    track_seat_votes.setdefault(tid, Counter())[int(display_seat)] += 1
                    seat_track_votes.setdefault(int(display_seat), Counter())[tid] += 1

                x1, y1, x2, y2 = det["xyxy"]
                color = (0, 255, 0) if display_seat is not None else (0, 220, 255)
                if frame_idx < bind_start_frame:
                    color = (180, 180, 180)
                    label = f"H{tid} teacher" if is_teacher else f"H{tid} layout"
                elif is_teacher:
                    color = (255, 160, 80)
                    label = f"H{tid} teacher"
                elif display_seat is not None:
                    label = f"H{tid}->S{int(display_seat):02d}"
                elif assignment:
                    label = f"H{tid}->S{int(current_seat):02d} pending"
                elif not inside_student_area:
                    color = (255, 160, 80)
                    label = f"H{tid} outside"
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
                        "role": "teacher" if is_teacher else ("student" if display_seat is not None else "candidate"),
                        "inside_student_area": bool(inside_student_area),
                        "inside_teacher_area": bool(inside_teacher_area),
                        "pending_seat_no": seat_debug.get("pending_seat_no"),
                        "pending_count": seat_debug.get("pending_count"),
                        "release_count": seat_debug.get("release_count"),
                        "binding_method": assignment.get("binding_method") if assignment else None,
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
        "student_area_padding": float(args.student_area_padding),
        "student_area_polygon": (
            np.asarray(student_area_polygon).astype(float).tolist()
            if student_area_polygon is not None else None
        ),
        "teacher_area_padding": float(args.teacher_area_padding),
        "teacher_area_polygon": (
            np.asarray(teacher_area_polygon).astype(float).tolist()
            if teacher_area_polygon is not None else None
        ),
        "line_margin_scale": float(args.line_margin_scale),
        "depth_margin_scale": float(args.depth_margin_scale),
        "min_line_margin": float(args.min_line_margin),
        "min_depth_margin": float(args.min_depth_margin),
        "assignment_mode": args.assignment_mode,
        "column_sort_line_scale": float(args.column_sort_line_scale),
        "column_sort_min_line": float(args.column_sort_min_line),
        "column_sort_depth_scale": float(args.column_sort_depth_scale),
        "projection_line_scale": float(args.projection_line_scale),
        "projection_depth_scale": float(args.projection_depth_scale),
        "projection_min_line": float(args.projection_min_line),
        "projection_min_depth": float(args.projection_min_depth),
        "sticky_line_scale": float(args.sticky_line_scale),
        "sticky_depth_scale": float(args.sticky_depth_scale),
        "sticky_center_scale": float(args.sticky_center_scale),
        "max_teachers": int(args.max_teachers),
        "teacher_confirm_seconds": float(args.teacher_confirm_seconds),
        "teacher_window_seconds": float(args.teacher_window_seconds),
        "teacher_outside_ratio": float(args.teacher_outside_ratio),
        "initial_bind_seconds": float(args.initial_bind_seconds),
        "switch_seconds": float(args.switch_seconds),
        "release_seconds": float(args.release_seconds),
        "miss_hold_seconds": float(args.miss_hold_seconds),
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
        "teacher_tracks": teacher_manager.summary(),
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
    p.add_argument(
        "--student-area-padding",
        type=float,
        default=90.0,
        help="Pixels added around the desk-layout hull; heads outside this area are treated as non-students",
    )
    p.add_argument(
        "--teacher-area-padding",
        type=float,
        default=10.0,
        help="Small padding around the desk-layout hull used to collect teacher-outside evidence",
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
        "--assignment-mode",
        choices=["column_sort", "line_depth"],
        default="column_sort",
        help="column_sort maps heads by 5x6 column order; line_depth uses per-head nearest line/depth",
    )
    p.add_argument(
        "--column-sort-line-scale",
        type=float,
        default=3.0,
        help="Column membership tolerance for column_sort mode",
    )
    p.add_argument("--column-sort-min-line", type=float, default=320.0)
    p.add_argument(
        "--column-sort-depth-scale",
        type=float,
        default=2.2,
        help="Depth tolerance when fewer than 6 heads are detected in a column",
    )
    p.add_argument(
        "--projection-line-scale",
        type=float,
        default=2.3,
        help="Relaxed column-line tolerance for unbound/unstable projection fallback",
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
        default=2.8,
        help="Large tolerance for keeping an already-bound student on the same seat",
    )
    p.add_argument(
        "--sticky-depth-scale",
        type=float,
        default=2.4,
        help="Large depth tolerance for keeping an already-bound student on the same seat",
    )
    p.add_argument("--sticky-center-scale", type=float, default=1.65)

    p.add_argument("--max-teachers", type=int, default=2)
    p.add_argument("--teacher-confirm-seconds", type=float, default=2.0)
    p.add_argument("--teacher-window-seconds", type=float, default=4.0)
    p.add_argument("--teacher-outside-ratio", type=float, default=0.65)

    p.add_argument("--initial-bind-seconds", type=float, default=1.0)
    p.add_argument("--switch-seconds", type=float, default=6.0)
    p.add_argument("--release-seconds", type=float, default=8.0)
    p.add_argument("--miss-hold-seconds", type=float, default=12.0)
    p.add_argument("--reservation-seconds", type=float, default=120.0, help="Compatibility only; ignored by evidence manager")
    p.add_argument("--reacquire-seconds", type=float, default=1.0)

    p.add_argument("--draw-layout", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--draw-student-area", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--max-process-frames", type=int, default=0, help="Debug only; 0 means full video")
    p.add_argument("--display", action="store_true")
    return p


def main():
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
