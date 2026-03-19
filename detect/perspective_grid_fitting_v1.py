import argparse
import os
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


# =========================================================
# Geometry / helpers
# =========================================================

def bbox_anchor(box, alpha=0.72):
    x1, y1, x2, y2 = box
    h = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = y1 + alpha * h
    return cx, cy, h


def draw_box(img, box, color=(0, 255, 0), label=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            img, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
        )


def bilinear_point(P00, P04, P50, P54, u, v):
    """
    u: row ratio in [0,1]
    v: col ratio in [0,1]
    """
    P00 = np.asarray(P00, dtype=np.float32)
    P04 = np.asarray(P04, dtype=np.float32)
    P50 = np.asarray(P50, dtype=np.float32)
    P54 = np.asarray(P54, dtype=np.float32)

    p = (
        (1 - u) * (1 - v) * P00
        + (1 - u) * v * P04
        + u * (1 - v) * P50
        + u * v * P54
    )
    return p


# =========================================================
# ByteTrack person tracker
# =========================================================

class PersonTracker:
    def __init__(self, weights: str, device="0", half=False, tracker="bytetrack.yaml"):
        self.weights = weights
        self.device = device
        self.half = half
        self.tracker_name = tracker
        self.model = YOLO(weights)

    def track(self, frame, conf=0.15, iou=0.6, imgsz=1280, classes=None):
        results = self.model.track(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device if self.device else None,
            half=self.half,
            verbose=False,
            classes=classes,
            persist=True,
            tracker=self.tracker_name,
        )
        return results[0].boxes

    def reset(self):
        self.model = YOLO(self.weights)


def tracked_boxes_to_detections(boxes):
    dets = []
    ids = None
    if hasattr(boxes, "id") and boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()

    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = float(b.conf[0]) if b.conf is not None else 1.0
        cls = int(b.cls[0]) if b.cls is not None else 0
        tid = ids[i] if ids is not None else -1
        dets.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "conf": conf,
            "cls": cls,
            "track_id": int(tid),
        })
    return dets


# =========================================================
# Track memory
# =========================================================

class TrackMemory:
    def __init__(self, maxlen=200):
        self.history = {}

    def update(self, detections, frame_id):
        seen = set()
        for det in detections:
            tid = det.get("track_id", -1)
            if tid < 0:
                continue
            seen.add(tid)

            cx, cy, h = bbox_anchor(det["bbox"])
            if tid not in self.history:
                self.history[tid] = {
                    "anchors": deque(maxlen=200),
                    "boxes": deque(maxlen=200),
                    "first_frame": frame_id,
                    "last_frame": frame_id,
                    "seen_count": 0,
                    "miss_count": 0,
                }

            st = self.history[tid]
            st["anchors"].append((cx, cy, h))
            st["boxes"].append(det["bbox"])
            st["last_frame"] = frame_id
            st["seen_count"] += 1
            st["miss_count"] = 0

        for tid, st in self.history.items():
            if tid not in seen:
                st["miss_count"] += 1

    def get_stable_anchors(self, min_seen=10, max_miss=20):
        pts = []
        for tid, st in self.history.items():
            if st["seen_count"] >= min_seen and st["miss_count"] <= max_miss:
                pts.extend(list(st["anchors"]))
        return pts

    def get_stable_student_like_anchors(self, min_seen=10, max_miss=20, max_motion=120):
        """
        升级的候选点筛选：
        - 检查轨迹运动范围
        - 排除运动过大的轨迹（可能是教师或漂移）
        - 用轨迹的中位位置作为稳定样本
        """
        pts = []
        for tid, st in self.history.items():
            if st["seen_count"] < min_seen:
                continue
            if st["miss_count"] > max_miss:
                continue

            anchors = np.array(list(st["anchors"]), dtype=np.float32)
            if len(anchors) < 2:
                continue

            xs = anchors[:, 0]
            ys = anchors[:, 1]

            # 计算轨迹的运动范围
            motion = np.sqrt((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2)

            # 运动过大，像老师或漂移轨迹，排除
            if motion > max_motion:
                continue

            # 用该轨迹的中位 anchor 作为学生候选点
            cx = float(np.median(xs))
            cy = float(np.median(ys))
            h = float(np.median(anchors[:, 2]))
            pts.append((cx, cy, h))

        return pts


# =========================================================
# Perspective grid fitting (Scheme A)
# =========================================================

class PerspectiveGridFitterA:
    """
    方案A：四角点 + 双线性插值网格
    """
    def __init__(self, num_rows=6, num_cols=5):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.corners = None   # P00, P04, P50, P54
        self.grid_points = None
        self.row_expected_h = None

    def _init_corners_from_points(self, pts):
        """
        pts: list[(cx, cy, h)]
        改进的角点初始化：
        - 用分位数代替极值，减少异常点影响
        - 更鲁棒地找出座位区域边界
        """
        arr = np.asarray([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)
        ys = arr[:, 1]

        # 按 y 排序，分前层和后层
        idx = np.argsort(ys)
        arr_sorted = arr[idx]

        n = len(arr_sorted)
        top_group = arr_sorted[:max(6, n // 4)]
        bottom_group = arr_sorted[-max(6, n // 4):]

        # 用分位数代替极值，减少异常点影响
        top_xs = np.sort(top_group[:, 0])
        bottom_xs = np.sort(bottom_group[:, 0])

        top_y = float(np.median(top_group[:, 1]))
        bottom_y = float(np.median(bottom_group[:, 1]))

        P00 = np.array([np.percentile(top_xs, 20), top_y], dtype=np.float32)
        P04 = np.array([np.percentile(top_xs, 80), top_y], dtype=np.float32)
        P50 = np.array([np.percentile(bottom_xs, 20), bottom_y], dtype=np.float32)
        P54 = np.array([np.percentile(bottom_xs, 80), bottom_y], dtype=np.float32)

        return [P00, P04, P50, P54]

    def _build_grid(self, corners):
        P00, P04, P50, P54 = corners
        grid = []
        for r in range(self.num_rows):
            u = 0.0 if self.num_rows == 1 else r / (self.num_rows - 1)
            for c in range(self.num_cols):
                v = 0.0 if self.num_cols == 1 else c / (self.num_cols - 1)
                p = bilinear_point(P00, P04, P50, P54, u, v)
                grid.append({
                    "slot_id": r * self.num_cols + c,
                    "row_id": r,
                    "col_id": c,
                    "center_x": float(p[0]),
                    "center_y": float(p[1]),
                })
        return grid

    def _match_points_to_grid(self, pts, grid_points, dist_norm=120.0):
        """
        pts: observed anchors [(cx,cy,h), ...]
        grid_points: slot list
        返回:
          matched_pairs: [(pt_idx, slot_idx), ...]
          mean_cost
        """
        if len(pts) == 0 or len(grid_points) == 0:
            return [], 1e9

        obs = np.asarray([[p[0], p[1]] for p in pts], dtype=np.float32)
        grd = np.asarray([[g["center_x"], g["center_y"]] for g in grid_points], dtype=np.float32)

        cost = np.zeros((len(obs), len(grd)), dtype=np.float32)
        for i in range(len(obs)):
            for j in range(len(grd)):
                dx = obs[i, 0] - grd[j, 0]
                dy = obs[i, 1] - grd[j, 1]
                cost[i, j] = np.sqrt(dx * dx + dy * dy) / dist_norm

        row_ind, col_ind = linear_sum_assignment(cost)

        pairs = []
        costs = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < 2.0:   # 粗门限，过大就不接受
                pairs.append((i, j))
                costs.append(cost[i, j])

        mean_cost = float(np.mean(costs)) if len(costs) > 0 else 1e9
        return pairs, mean_cost

    def _refine_corners(self, pts, corners, steps=40, step_scale=8.0):
        """
        简单局部搜索优化四角点，不依赖额外优化库
        """
        best_corners = [c.copy() for c in corners]
        best_grid = self._build_grid(best_corners)
        _, best_cost = self._match_points_to_grid(pts, best_grid)

        directions = [
            np.array([1, 0], dtype=np.float32),
            np.array([-1, 0], dtype=np.float32),
            np.array([0, 1], dtype=np.float32),
            np.array([0, -1], dtype=np.float32),
            np.array([1, 1], dtype=np.float32),
            np.array([1, -1], dtype=np.float32),
            np.array([-1, 1], dtype=np.float32),
            np.array([-1, -1], dtype=np.float32),
        ]

        scale = step_scale
        for _ in range(steps):
            improved = False

            for corner_id in range(4):
                for d in directions:
                    trial = [c.copy() for c in best_corners]
                    trial[corner_id] = trial[corner_id] + scale * d

                    grid = self._build_grid(trial)
                    _, cost = self._match_points_to_grid(pts, grid)

                    if cost < best_cost:
                        best_cost = cost
                        best_corners = trial
                        improved = True

            if not improved:
                scale *= 0.6
                if scale < 0.8:
                    break

        return best_corners, best_cost

    def fit(self, pts, num_iters=5, keep_ratio=0.75):
        """
        pts: list[(cx, cy, h)]
        鲁棒迭代拟合版本：
        - 每轮迭代后剔除误差较大的点
        - 最后四角点会更多地被"真正的学生排布"支撑
        """
        if len(pts) < 6:
            raise ValueError("可用于拟合透视网格的点太少")

        pts = list(pts)
        corners = self._init_corners_from_points(pts)

        current_pts = pts[:]
        best_cost = 1e9
        best_corners = corners

        for iter_id in range(num_iters):
            # 1. 优化角点
            refined_corners, cost = self._refine_corners(current_pts, corners)
            grid = self._build_grid(refined_corners)

            # 2. 匹配当前点到 grid
            pairs, _ = self._match_points_to_grid(current_pts, grid)

            if len(pairs) < 4:
                break

            # 3. 计算每个匹配点的误差
            errors = []
            for pt_idx, slot_idx in pairs:
                px, py, _ = current_pts[pt_idx]
                gx = grid[slot_idx]["center_x"]
                gy = grid[slot_idx]["center_y"]
                err = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                errors.append((err, pt_idx))

            errors.sort(key=lambda x: x[0])

            # 4. 只保留误差较小的点，剔除异常点
            keep_n = max(6, int(len(errors) * keep_ratio))
            keep_ids = set(pt_idx for _, pt_idx in errors[:keep_n])
            current_pts = [p for idx, p in enumerate(current_pts) if idx in keep_ids]

            corners = refined_corners
            if cost < best_cost:
                best_cost = cost
                best_corners = refined_corners

            if len(current_pts) < 6:
                break

        self.corners = best_corners
        self.grid_points = self._build_grid(best_corners)

        # 重新估计每行的 expected_h
        row_h = defaultdict(list)
        matched_pairs, _ = self._match_points_to_grid(pts, self.grid_points)
        for pt_idx, slot_idx in matched_pairs:
            r = self.grid_points[slot_idx]["row_id"]
            row_h[r].append(pts[pt_idx][2])

        self.row_expected_h = {}
        for r in range(self.num_rows):
            vals = row_h.get(r, [])
            self.row_expected_h[r] = float(np.median(vals)) if len(vals) > 0 else 80.0

        return self.grid_points, best_cost

    def get_slots(self):
        slots = []
        for g in self.grid_points:
            r = g["row_id"]
            slots.append({
                "slot_id": g["slot_id"],
                "row_id": r,
                "col_id": g["col_id"],
                "center_x": g["center_x"],
                "center_y": g["center_y"],
                "expected_h": self.row_expected_h.get(r, 80.0),
            })
        return slots


# =========================================================
# Slot assignment using fitted perspective grid
# =========================================================

class PerspectiveGridAssigner:
    def __init__(self, seat_slots):
        self.seat_slots = seat_slots
        self.row_slots = defaultdict(list)
        for s in seat_slots:
            self.row_slots[s["row_id"]].append(s)
        for row_id in self.row_slots:
            self.row_slots[row_id] = sorted(self.row_slots[row_id], key=lambda x: x["col_id"])

    def assign(self, detections):
        slot_hits = {s["slot_id"]: False for s in self.seat_slots}
        slot_best_det = {s["slot_id"]: None for s in self.seat_slots}
        slot_best_track_id = {s["slot_id"]: None for s in self.seat_slots}
        unmatched = []

        if len(detections) == 0:
            return slot_hits, slot_best_det, slot_best_track_id, unmatched

        # 全局最小代价匹配
        obs = []
        for det in detections:
            cx, cy, h = bbox_anchor(det["bbox"])
            obs.append((cx, cy, h))

        cost = np.zeros((len(obs), len(self.seat_slots)), dtype=np.float32)
        for i, (cx, cy, h) in enumerate(obs):
            for j, slot in enumerate(self.seat_slots):
                dx = abs(cx - slot["center_x"]) / max(30.0, slot["expected_h"])
                dy = abs(cy - slot["center_y"]) / max(20.0, slot["expected_h"])
                dh = abs(h - slot["expected_h"]) / max(20.0, slot["expected_h"])
                cost[i, j] = 0.60 * dx + 0.25 * dy + 0.15 * dh

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_det_ids = set()
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] > 3.0:
                continue
            det = detections[i]
            slot = self.seat_slots[j]
            sid = slot["slot_id"]

            slot_hits[sid] = True
            slot_best_det[sid] = det
            tid = det.get("track_id", -1)
            slot_best_track_id[sid] = tid if tid >= 0 else None
            assigned_det_ids.add(i)

        for i, det in enumerate(detections):
            if i not in assigned_det_ids:
                unmatched.append(det)

        return slot_hits, slot_best_det, slot_best_track_id, unmatched


# =========================================================
# Slot memory / teacher analyzer
# =========================================================

class SlotMemory:
    def __init__(self, seat_slots):
        self.state = {}
        for s in seat_slots:
            sid = s["slot_id"]
            row_id = s["row_id"]

            if row_id <= 1:
                band = "near"
            elif row_id <= 3:
                band = "mid"
            else:
                band = "rear"

            self.state[sid] = {
                "band": band,
                "occupied_score": 0.0,
                "hit_streak": 0,
                "miss_streak": 0,
                "is_occupied": False,
                "last_box": None,
                "linked_track_id": None,
            }

        self.params = {
            "near": {"inc": 0.18, "dec": 0.05, "on_th": 0.55, "off_th": 0.25, "k_on": 2, "k_off": 12},
            "mid":  {"inc": 0.16, "dec": 0.03, "on_th": 0.55, "off_th": 0.25, "k_on": 2, "k_off": 20},
            "rear": {"inc": 0.14, "dec": 0.015, "on_th": 0.50, "off_th": 0.20, "k_on": 2, "k_off": 40},
        }

    def update(self, slot_hits, slot_best_det, slot_best_track_id):
        for sid, st in self.state.items():
            p = self.params[st["band"]]
            hit = slot_hits[sid]

            if hit:
                st["hit_streak"] += 1
                st["miss_streak"] = 0
                st["occupied_score"] = min(1.0, st["occupied_score"] + p["inc"])
                if slot_best_det[sid] is not None:
                    st["last_box"] = slot_best_det[sid]["bbox"]
                if slot_best_track_id[sid] is not None:
                    st["linked_track_id"] = slot_best_track_id[sid]
            else:
                st["hit_streak"] = 0
                st["miss_streak"] += 1
                st["occupied_score"] = max(0.0, st["occupied_score"] - p["dec"])

            if not st["is_occupied"]:
                if st["hit_streak"] >= p["k_on"] or st["occupied_score"] > p["on_th"]:
                    st["is_occupied"] = True
            else:
                if st["miss_streak"] >= p["k_off"] and st["occupied_score"] < p["off_th"]:
                    st["is_occupied"] = False
                    st["linked_track_id"] = None

        return self.state


class TeacherAnalyzer:
    def __init__(self):
        self.track_unmatched = defaultdict(int)
        self.track_matched = defaultdict(int)

    def update(self, detections, matched_track_ids):
        for det in detections:
            tid = det.get("track_id", -1)
            if tid < 0:
                continue
            if tid in matched_track_ids:
                self.track_matched[tid] += 1
            else:
                self.track_unmatched[tid] += 1

    def get_teacher_ids(self, track_memory, min_seen=15, min_unmatched=10):
        out = set()
        for tid, st in track_memory.history.items():
            seen = st["seen_count"]
            unmatched = self.track_unmatched.get(tid, 0)
            matched = self.track_matched.get(tid, 0)
            if seen >= min_seen and unmatched > matched and unmatched >= min_unmatched:
                out.add(tid)
        return out


# =========================================================
# Visualization
# =========================================================

def draw_tracks(img, detections, teacher_ids):
    vis = img.copy()
    for det in detections:
        tid = det.get("track_id", -1)
        if tid in teacher_ids:
            color = (255, 0, 255)
            label = f"teacher?:ID{tid}"
        else:
            color = (255, 255, 0)
            label = f"ID{tid}" if tid >= 0 else "det"
        draw_box(vis, det["bbox"], color=color, label=label, thickness=2)
    return vis


def draw_grid(img, seat_slots, slot_state, corners=None):
    vis = img.copy()

    if corners is not None:
        P00, P04, P50, P54 = corners
        for p, name in zip([P00, P04, P50, P54], ["P00", "P04", "P50", "P54"]):
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0, 165, 255), -1)
            cv2.putText(vis, name, (int(p[0]) + 6, int(p[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # 画外框
        pts = np.array([P00, P04, P54, P50], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 165, 255), thickness=2)

    for s in seat_slots:
        sid = s["slot_id"]
        st = slot_state[sid]
        color = (0, 255, 0) if st["is_occupied"] else (0, 0, 255)

        cx = int(s["center_x"])
        cy = int(s["center_y"])
        r = max(5, int(0.10 * s["expected_h"]))
        cv2.circle(vis, (cx, cy), r, color, -1)

        txt = f"S{sid}:{st['occupied_score']:.2f}"
        cv2.putText(vis, txt, (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        if st["is_occupied"] and st["last_box"] is not None:
            label = "student"
            if st["linked_track_id"] is not None:
                label += f":ID{st['linked_track_id']}"
            draw_box(vis, st["last_box"], color=color, label=label, thickness=2)

    return vis


def draw_grid_lines(img, seat_slots, num_rows, num_cols, color=(0, 165, 255)):
    """
    网格线可视化：将座位点按行列连接成网格线
    便于调试：哪一行弯了、哪一列偏了
    """
    vis = img.copy()

    slot_map = {(s["row_id"], s["col_id"]): s for s in seat_slots}

    # 画行线
    for r in range(num_rows):
        pts = []
        for c in range(num_cols):
            s = slot_map.get((r, c), None)
            if s is not None:
                pts.append((int(s["center_x"]), int(s["center_y"])))
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(vis, pts[i], pts[i + 1], color, 2)

    # 画列线
    for c in range(num_cols):
        pts = []
        for r in range(num_rows):
            s = slot_map.get((r, c), None)
            if s is not None:
                pts.append((int(s["center_x"]), int(s["center_y"])))
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(vis, pts[i], pts[i + 1], color, 2)

    return vis


# =========================================================
# Main pipeline
# =========================================================

class PerspectiveGridPipelineA:
    def __init__(
        self,
        weights,
        device="0",
        half=False,
        tracker="bytetrack.yaml",
        classes="0",
        num_rows=6,
        num_cols=5,
        calib_frames=120,
    ):
        self.weights = weights
        self.device = device
        self.half = half
        self.tracker_name = tracker
        self.classes = [int(x) for x in classes.split(",")] if classes else None

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.calib_frames = calib_frames

        self.tracker_model = PersonTracker(
            weights=weights,
            device=device,
            half=half,
            tracker=tracker
        )

        self.track_memory = TrackMemory()
        self.grid_fitter = PerspectiveGridFitterA(num_rows=num_rows, num_cols=num_cols)

        self.seat_slots = None
        self.assigner = None
        self.slot_memory = None
        self.teacher_analyzer = TeacherAnalyzer()

    def _reset_tracker(self):
        self.tracker_model = PersonTracker(
            weights=self.weights,
            device=self.device,
            half=self.half,
            tracker=self.tracker_name
        )

    def calibrate(self, cap, conf=0.15, iou=0.6, imgsz=1280):
        frame_id = 0
        while frame_id < self.calib_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            boxes = self.tracker_model.track(
                frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=self.classes
            )
            dets = tracked_boxes_to_detections(boxes)
            self.track_memory.update(dets, frame_id)

        # 使用升级的候选点筛选
        pts = self.track_memory.get_stable_student_like_anchors(min_seen=10, max_miss=20, max_motion=120)
        if len(pts) < 6:
            # 如果候选点不足，回退到基础筛选
            pts = self.track_memory.get_stable_anchors(min_seen=8, max_miss=20)
        self.grid_fitter.fit(pts, num_iters=5, keep_ratio=0.75)
        self.seat_slots = self.grid_fitter.get_slots()
        self.assigner = PerspectiveGridAssigner(self.seat_slots)
        self.slot_memory = SlotMemory(self.seat_slots)

        print(f"[Calibration] stable anchors: {len(pts)}")
        print(f"[Calibration] inferred slots: {len(self.seat_slots)}")
        return frame_id

    def run(self, source, output_dir, conf=0.15, iou=0.6, imgsz=1280, display=False):
        if not os.path.isfile(source):
            raise FileNotFoundError(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-6:
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"perspective_grid_A_{Path(source).stem}.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))

        # calibration
        used_calib = self.calibrate(cap, conf=conf, iou=iou, imgsz=imgsz)

        # rerun
        cap.release()
        cap = cv2.VideoCapture(source)
        self._reset_tracker()

        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1

                boxes = self.tracker_model.track(
                    frame,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    classes=self.classes
                )
                dets = tracked_boxes_to_detections(boxes)
                self.track_memory.update(dets, frame_id)

                slot_hits, slot_best_det, slot_best_track_id, unmatched = self.assigner.assign(dets)
                slot_state = self.slot_memory.update(slot_hits, slot_best_det, slot_best_track_id)

                matched_track_ids = set(
                    tid for tid in slot_best_track_id.values()
                    if tid is not None and tid >= 0
                )
                self.teacher_analyzer.update(dets, matched_track_ids)
                teacher_ids = self.teacher_analyzer.get_teacher_ids(self.track_memory)

                occupied_count = sum(1 for _, st in slot_state.items() if st["is_occupied"])

                vis = frame.copy()
                vis = draw_tracks(vis, dets, teacher_ids)
                vis = draw_grid(vis, self.seat_slots, slot_state, corners=self.grid_fitter.corners)
                # 添加网格线可视化
                vis = draw_grid_lines(vis, self.seat_slots, self.num_rows, self.num_cols)

                info = (
                    f"Occupied={occupied_count} | "
                    f"Frame {frame_id}/{total if total>0 else '?'} | "
                    f"Det={len(dets)} | Teacher={len(teacher_ids)} | Calib={used_calib}"
                )
                cv2.putText(vis, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

                out.write(vis)

                if display:
                    cv2.imshow("Perspective Grid Fitting A", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if frame_id % 30 == 0:
                    print(f"进度: {frame_id}/{total if total>0 else '?'} | occupied={occupied_count} | teacher={len(teacher_ids)}")

        finally:
            cap.release()
            out.release()
            if display:
                cv2.destroyAllWindows()

        print(f"完成! 输出视频: {out_path}")


# =========================================================
# CLI
# =========================================================

def main():
    ap = argparse.ArgumentParser("Perspective Grid Fitting V1 (Scheme A)")
    ap.add_argument("--source", type=str, required=True, help="输入视频")
    ap.add_argument("--weights", type=str, required=True, help="YOLO11 权重")
    ap.add_argument("--output", type=str, default="detect/output", help="输出目录")

    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml")
    ap.add_argument("--classes", type=str, default="0")

    ap.add_argument("--num-rows", type=int, default=6)
    ap.add_argument("--num-cols", type=int, default=5)
    ap.add_argument("--calib-frames", type=int, default=120)

    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=1280)

    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    pipeline = PerspectiveGridPipelineA(
        weights=args.weights,
        device=args.device,
        half=args.half,
        tracker=args.tracker,
        classes=args.classes,
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        calib_frames=args.calib_frames,
    )

    pipeline.run(
        source=args.source,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        display=args.display
    )


if __name__ == "__main__":
    main()