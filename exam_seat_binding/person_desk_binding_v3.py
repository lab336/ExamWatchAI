"""
人桌绑定 V3 —— 基于相邻桌子梯形区的人框重叠绑定

核心思路:
  分好每列后, 第 i 个座位的绑定区域 = 第 i 张桌子与第 i+1 张桌子之间的
  梯形区域。对每个学生框, 计算其与每个梯形座位区的交并比(IoU), 取 IoU
  最大的候选并做一对一匹配。

  视频假设:
  1. 前半段为空教室, 用于建立固定桌位布局
  2. 后半段为满座场景, 30 个学生 + 少量老师
  3. 对外输出时, 学生的统一 track_id 直接使用座位号 1..30

流水线:
  桌子检测 → 按列排序得 6×5 网格 → 相邻桌子间建梯形区域
  → 人框与梯形区算 IoU → 贪心一对一分配 → 输出座位号

用法:
python exam_seat_binding/person_desk_binding_v3.py --source data/ideotest/merged_output.mp4
python exam_seat_binding/person_desk_binding_v3.py --source data/ideotest/merged_output.mp4 --display
"""

import argparse
import csv
import importlib.util
import json
import os
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ─── 加载桌子检测模块 ────────────────────────────────────────

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


# ─── 常量 ────────────────────────────────────────────────────

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")


# ─── 几何工具函数 ─────────────────────────────────────────────

def xyxy_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def foot_point(box):
    """人框底部中心点（脚部位置），更接近桌子位置"""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


def head_point(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, y1], dtype=np.float32)


def l2(a, b):
    return float(np.linalg.norm(
        np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    ))


def box_polygon(box):
    """将 xyxy 人框转为顺时针四边形."""
    x1, y1, x2, y2 = box
    return np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ], dtype=np.float32)


def convex_polygon_iou(poly_a, poly_b):
    """计算两个凸多边形的 IoU 与交集面积."""
    poly_a = np.asarray(poly_a, dtype=np.float32)
    poly_b = np.asarray(poly_b, dtype=np.float32)
    area_a = float(abs(cv2.contourArea(poly_a)))
    area_b = float(abs(cv2.contourArea(poly_b)))
    if area_a <= 1e-6 or area_b <= 1e-6:
        return 0.0, 0.0

    try:
        inter_area, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    except cv2.error:
        return 0.0, 0.0

    inter_area = float(inter_area)
    if inter_area <= 1e-6:
        return 0.0, 0.0

    union_area = area_a + area_b - inter_area
    if union_area <= 1e-6:
        return 0.0, 0.0

    return inter_area / union_area, inter_area


def point_line_distance_kb(point_xy, line_kb):
    """点到直线 y = kx + b 的距离."""
    x = float(point_xy[0])
    y = float(point_xy[1])
    k, b = line_kb
    return abs(float(k) * x - y + float(b)) / max(
        1e-6, float(np.sqrt(float(k) * float(k) + 1.0))
    )


def seat_id_from_track(track_id: int) -> str:
    """生成与跟踪ID一致的座位ID表示."""
    return f"T{int(track_id)}"


def build_classroom_polygon_from_desks(desks: list, padding: float = 0.0):
    """根据 5x6 桌子整体外轮廓构建包围多边形(凸包)."""
    if not desks:
        return None

    pts = []
    for d in desks:
        x1, y1, x2, y2 = d["xyxy"]
        pts.append([x1, y1])
        pts.append([x2, y1])
        pts.append([x2, y2])
        pts.append([x1, y2])

    pts_np = np.asarray(pts, dtype=np.float32)
    hull = cv2.convexHull(pts_np).reshape(-1, 2)

    # 扩大padding以包含更多边缘学生
    if padding > 0:
        cx = float(np.mean(hull[:, 0]))
        cy = float(np.mean(hull[:, 1]))
        padded = []
        for px, py in hull:
            vx = float(px) - cx
            vy = float(py) - cy
            norm = max(1e-6, float(np.sqrt(vx * vx + vy * vy)))
            # 增加padding系数，确保边缘学生也在包围区内
            padded.append([px + padding * 1.5 * vx / norm, py + padding * 1.5 * vy / norm])
        hull = np.asarray(padded, dtype=np.float32)

    return hull


def point_in_polygon(point_xy, polygon: np.ndarray | None) -> bool:
    """判断点是否在多边形内或边上."""
    if polygon is None or len(polygon) < 3:
        return False
    poly = polygon.reshape(-1, 1, 2).astype(np.float32)
    return cv2.pointPolygonTest(poly, (float(point_xy[0]), float(point_xy[1])), False) >= 0


# ─── 相邻桌子区间构建 ────────────────────────────────────────

class InterDeskZoneBuilder:
    """
    在同一列相邻桌子之间构建梯形绑定区域。

    原理 (以某一列为例, 桌子从下往上排列: d0, d1, d2, ..., d5):
      - 座位 i 的绑定区域 = 从 d_i 到 d_{i+1} 的四边形
      - 四边形由 d_i 的左上角/右下角 连线到 d_{i+1} 的左上角/右下角 构成
      - 最后一个座位的区域向远端(图像上方)延伸
      - 第一个座位的区域向近端(图像下方)略做扩展

    该四边形自然适应透视畸变:
      近处桌子大, 远处桌子小, 四边形即为梯形。
    """

    def __init__(self, first_extend_ratio: float = 0.3,
                 last_extend_ratio: float = 0.8):
        """
        Args:
            first_extend_ratio: 第一个座位(最近相机)向下延伸比例
            last_extend_ratio: 最后一个座位(最远相机)向上延伸比例
        """
        self.first_extend_ratio = first_extend_ratio
        self.last_extend_ratio = last_extend_ratio

    def build(self, desks: list) -> list:
        """
        从已排好列和行的桌子列表, 构建所有座位绑定区域。

        Args:
            desks: 桌子列表 (含 desk_id, desk_no, column_index, row_index, xyxy, center ...)

        Returns:
            zones: 绑定区域列表, 每项含 polygon (4点四边形), zone_center 等
        """
        # 按列分组
        columns: dict[int, list] = {}
        for d in desks:
            col = d["column_index"]
            columns.setdefault(col, []).append(d)

        # 每列按 row_index 排序 (row 0 = 图像底部, 最近相机)
        for col in columns:
            columns[col].sort(key=lambda d: d["row_index"])

        zones = []
        for col_id in sorted(columns.keys()):
            col_desks = columns[col_id]
            n = len(col_desks)

            for i in range(n):
                curr = col_desks[i]
                x1_c, y1_c, x2_c, y2_c = curr["xyxy"]

                if i < n - 1:
                    # ── 中间座位: 当前桌子 → 下一桌子的四边形 ──
                    nxt = col_desks[i + 1]
                    x1_n, y1_n, x2_n, y2_n = nxt["xyxy"]

                    # 扩展区域以确保覆盖，向左右扩展
                    desk_w_c = x2_c - x1_c
                    desk_w_n = x2_n - x1_n
                    lateral_expand_c = desk_w_c * 0.15  # 左右扩展15%
                    lateral_expand_n = desk_w_n * 0.15
                    
                    # 四边形顶点 (顺时针):
                    #   A = 当前桌 TL    B = 当前桌 BR
                    #   D = 下一桌 BR    C = 下一桌 TL
                    poly = np.array([
                        [x1_c - lateral_expand_c, y1_c],   # A: 当前桌左上（左扩）
                        [x2_c + lateral_expand_c, y2_c],   # B: 当前桌右下（右扩）
                        [x2_n + lateral_expand_n, y2_n],   # D: 下一桌右下（右扩）
                        [x1_n - lateral_expand_n, y1_n],   # C: 下一桌左上（左扩）
                    ], dtype=np.float32)

                    # 第一个座位: 向下(近相机方向)扩展
                    if i == 0:
                        gap_h = abs(y2_c - y1_n)  # 两桌间距
                        ext = max(gap_h * self.first_extend_ratio,
                                  (y2_c - y1_c) * 0.5)  # 增加扩展
                        poly[0][1] += ext   # A.y 向下
                        poly[1][1] += ext   # B.y 向下

                else:
                    # ── 最后一个座位: 向远端延伸 ──
                    if i > 0:
                        prev = col_desks[i - 1]
                        gap_h = abs(prev["xyxy"][1] - y1_c)
                    else:
                        gap_h = y2_c - y1_c

                    ext = max(gap_h * self.last_extend_ratio,
                              (y2_c - y1_c) * 1.0)  # 增加扩展

                    # 透视缩放: 远处桌子窄些，但也要扩展
                    desk_w = x2_c - x1_c
                    cx = (x1_c + x2_c) / 2.0
                    lateral_expand = desk_w * 0.15
                    shrink = 1.0  # 不缩小，保持宽度
                    half_w = desk_w * shrink / 2.0

                    poly = np.array([
                        [x1_c - lateral_expand, y1_c],      # A: 桌左上（左扩）
                        [x2_c + lateral_expand, y2_c],      # B: 桌右下（右扩）
                        [cx + half_w, y1_c - ext],          # D: 延伸右上
                        [cx - half_w, y1_c - ext],          # C: 延伸左上
                    ], dtype=np.float32)

                zone_cx = float(np.mean(poly[:, 0]))
                zone_cy = float(np.mean(poly[:, 1]))

                zones.append({
                    "desk_id": curr["desk_id"],
                    "desk_no": curr["desk_no"],
                    "column_index": curr["column_index"],
                    "row_index": curr["row_index"],
                    "desk_xyxy": curr["xyxy"],
                    "desk_center": curr["center"],
                    "polygon": poly,
                    "zone_center": [zone_cx, zone_cy],
                })

        return zones


# ─── 区间绑定器 ──────────────────────────────────────────────

class ZoneBinder:
    """
    基于梯形区重叠的人-座位绑定器。

    规则:
    1. 计算人框与梯形座位区的 IoU
    2. 只保留有交集的候选
    3. 按 IoU 从高到低做贪心一对一匹配
    """

    def __init__(self, zones: list):
        """
        Args:
            zones: 座位区域列表
        """
        self.zones = zones
        self.zone_lookup = {z["desk_id"]: z for z in zones}

    def assign_batch(self, people: list):
        """
        批量绑定（按 IoU 最大原则）。

        Args:
            people: 列表, 每项含 track_id, xyxy

        Returns:
            assignments: {track_id: assignment_dict}
            anchor_points: {track_id: ndarray (ax, ay)}
        """
        if not people or not self.zones:
            return {}, {}

        # 提取锚点与人框多边形
        anchor_pts = {}
        person_polys = {}
        for person in people:
            tid = person["track_id"]
            anchor_pts[tid] = foot_point(person["xyxy"])
            person_polys[tid] = box_polygon(person["xyxy"])

        candidates = []  # (-iou, -inter_area, center_dist, track_id, zone_idx)
        for person in people:
            tid = person["track_id"]
            anchor = anchor_pts[tid]
            person_poly = person_polys[tid]

            for z_idx, zone in enumerate(self.zones):
                iou, inter_area = convex_polygon_iou(person_poly, zone["polygon"])
                if inter_area <= 1.0:
                    continue
                d_center = l2(anchor, zone["zone_center"])
                candidates.append((-iou, -inter_area, d_center, tid, z_idx))

        # 贪心分配: IoU 高的优先, 面积大的次之, 更靠近区中心再次之
        candidates.sort()
        used_zones: set[int] = set()
        used_people: set[int] = set()
        assignments = {}

        for neg_iou, neg_inter_area, d_center, tid, z_idx in candidates:
            if tid in used_people or z_idx in used_zones:
                continue

            zone = self.zones[z_idx]
            anchor = anchor_pts[tid]
            iou = -neg_iou
            inter_area = -neg_inter_area

            assignments[tid] = {
                "desk_id": zone["desk_id"],
                "desk_no": zone["desk_no"],
                "col_idx": zone["column_index"],
                "row_idx": zone["row_index"],
                "cost": round(d_center, 4),
                "in_zone": True,
                "overlap_iou": round(iou, 6),
                "overlap_area": round(inter_area, 2),
                "anchor_x": round(float(anchor[0]), 2),
                "anchor_y": round(float(anchor[1]), 2),
                "zone_cx": round(zone["zone_center"][0], 2),
                "zone_cy": round(zone["zone_center"][1], 2),
            }
            used_zones.add(z_idx)
            used_people.add(tid)

        return assignments, anchor_pts


class StableSeatManager:
    """
    轻量级稳定绑定器。

    目标:
    1. 一旦某个 track 绑定座位, 不因单帧抖动轻易换座
    2. 检测短时丢失或单帧未落入梯形区时, 保留旧座位
    3. 一个座位同一时刻只允许一个活跃 track 占用
    """

    def __init__(self, fps: float,
                 switch_seconds: float = 4.5,
                 miss_hold_seconds: float = 5.0,
                 initial_bind_seconds: float = 1.0):
        self.fps = max(1.0, fps)
        self.switch_confirm_frames = max(1, int(round(switch_seconds * self.fps)))
        self.miss_hold_frames = max(1, int(round(miss_hold_seconds * self.fps)))
        self.initial_confirm_frames = max(1, int(round(initial_bind_seconds * self.fps)))
        self.states: dict[int, dict] = {}
        self.seat_owner: dict[int, int] = {}
        self.ever_bound_seats: set[int] = set()

    def _get_state(self, track_id: int):
        if track_id not in self.states:
            self.states[track_id] = {
                "bound_seat_no": None,
                "last_seen_frame": -1,
                "last_bound_frame": -1,
                "pending_switch_seat_no": None,
                "pending_switch_count": 0,
            }
        return self.states[track_id]

    def _clear_pending(self, track_id: int):
        st = self._get_state(track_id)
        st["pending_switch_seat_no"] = None
        st["pending_switch_count"] = 0

    def _push_pending(self, track_id: int, seat_no: int) -> int:
        st = self._get_state(track_id)
        if st["pending_switch_seat_no"] == seat_no:
            st["pending_switch_count"] += 1
        else:
            st["pending_switch_seat_no"] = seat_no
            st["pending_switch_count"] = 1
        return st["pending_switch_count"]

    def _is_track_active(self, track_id: int, frame_idx: int) -> bool:
        st = self.states.get(track_id)
        if st is None:
            return False
        return frame_idx - st["last_seen_frame"] <= self.miss_hold_frames

    def _release_track(self, track_id: int):
        st = self.states.get(track_id)
        if st is None:
            return
        seat_no = st.get("bound_seat_no")
        if seat_no is not None and self.seat_owner.get(seat_no) == track_id:
            del self.seat_owner[seat_no]
        st["bound_seat_no"] = None
        st["last_bound_frame"] = -1
        self._clear_pending(track_id)

    def _cleanup_expired(self, frame_idx: int):
        expired = []
        for seat_no, owner_tid in list(self.seat_owner.items()):
            if not self._is_track_active(owner_tid, frame_idx):
                expired.append((seat_no, owner_tid))
        for seat_no, owner_tid in expired:
            if self.seat_owner.get(seat_no) == owner_tid:
                del self.seat_owner[seat_no]
            owner_st = self.states.get(owner_tid)
            if owner_st is not None and owner_st.get("bound_seat_no") == seat_no:
                owner_st["bound_seat_no"] = None
                owner_st["last_bound_frame"] = -1
                owner_st["pending_switch_seat_no"] = None
                owner_st["pending_switch_count"] = 0

    def _seat_is_available(self, seat_no: int, track_id: int, frame_idx: int) -> bool:
        owner_tid = self.seat_owner.get(seat_no)
        if owner_tid is None or owner_tid == track_id:
            return True
        if not self._is_track_active(owner_tid, frame_idx):
            self._release_track(owner_tid)
            return True
        return False

    def _bind(self, track_id: int, seat_no: int, frame_idx: int):
        st = self._get_state(track_id)
        old_seat_no = st.get("bound_seat_no")
        if old_seat_no is not None and self.seat_owner.get(old_seat_no) == track_id:
            del self.seat_owner[old_seat_no]

        owner_tid = self.seat_owner.get(seat_no)
        if owner_tid is not None and owner_tid != track_id:
            self._release_track(owner_tid)

        st["bound_seat_no"] = seat_no
        st["last_bound_frame"] = frame_idx
        self.seat_owner[seat_no] = track_id
        self.ever_bound_seats.add(seat_no)
        self._clear_pending(track_id)

    def has_active_binding(self, track_id: int, frame_idx: int) -> bool:
        return self.get_display_seat(track_id, frame_idx) is not None

    def force_release(self, track_id: int):
        self._release_track(track_id)

    def update(self, track_id: int, frame_idx: int, proposed_seat_no: int | None):
        """
        输入当前帧 raw IoU 绑座结果, 返回稳定后的座位号.
        """
        self._cleanup_expired(frame_idx)
        st = self._get_state(track_id)
        st["last_seen_frame"] = frame_idx
        current_seat_no = st.get("bound_seat_no")

        if current_seat_no is None:
            if proposed_seat_no is None:
                self._clear_pending(track_id)
                return None
            if not self._seat_is_available(proposed_seat_no, track_id, frame_idx):
                self._clear_pending(track_id)
                return None
            pending_count = self._push_pending(track_id, proposed_seat_no)
            required_frames = (
                self.switch_confirm_frames
                if proposed_seat_no in self.ever_bound_seats
                else self.initial_confirm_frames
            )
            if pending_count >= required_frames:
                self._bind(track_id, proposed_seat_no, frame_idx)
                return proposed_seat_no
            return None

        if proposed_seat_no is None:
            return current_seat_no

        if proposed_seat_no == current_seat_no:
            st["last_bound_frame"] = frame_idx
            self._clear_pending(track_id)
            return current_seat_no

        if not self._seat_is_available(proposed_seat_no, track_id, frame_idx):
            self._clear_pending(track_id)
            return current_seat_no

        pending_count = self._push_pending(track_id, proposed_seat_no)
        if pending_count >= self.switch_confirm_frames:
            self._bind(track_id, proposed_seat_no, frame_idx)
            return proposed_seat_no

        return current_seat_no

    def get_display_seat(self, track_id: int, frame_idx: int):
        self._cleanup_expired(frame_idx)
        st = self.states.get(track_id)
        if st is None:
            return None
        seat_no = st.get("bound_seat_no")
        if seat_no is None:
            return None
        if frame_idx - st["last_seen_frame"] > self.miss_hold_frames:
            self._release_track(track_id)
            return None
        return seat_no

    def active_seat_count(self, frame_idx: int) -> int:
        self._cleanup_expired(frame_idx)
        return len(self.seat_owner)

    def active_seat_numbers(self, frame_idx: int) -> set[int]:
        self._cleanup_expired(frame_idx)
        return {int(seat_no) for seat_no in self.seat_owner.keys()}

    def get_bound_seat(self, track_id: int, frame_idx: int):
        return self.get_display_seat(track_id, frame_idx)


class ColumnFallbackAssigner:
    """
    同列空位兜底分配器。
    """

    def __init__(self, zones: list, column_lines: list | None):
        self.zones = zones
        self.column_lines = column_lines or []
        self.zones_by_col: dict[int, list] = {}
        self.col_center_x: dict[int, float] = {}
        self.col_distance_limit: dict[int, float] = {}

        for zone in zones:
            col = int(zone["column_index"])
            self.zones_by_col.setdefault(col, []).append(zone)

        for col, col_zones in self.zones_by_col.items():
            col_zones.sort(key=lambda z: z["row_index"])
            xs = [float(z["desk_center"][0]) for z in col_zones]
            self.col_center_x[col] = float(np.mean(xs))

            line_item = next(
                (item for item in self.column_lines if int(item["column_index"]) == col),
                None,
            )
            if line_item is not None:
                self.col_distance_limit[col] = max(
                    180.0,
                    float(line_item.get("avg_step", 0.0)) * 0.9,
                    float(line_item.get("avg_box_width", 0.0)) * 2.0,
                )
            else:
                self.col_distance_limit[col] = 260.0

    def _pick_column(self, anchor_xy):
        if self.column_lines:
            best_item = min(
                self.column_lines,
                key=lambda item: point_line_distance_kb(anchor_xy, item["line_kb"]),
            )
            return int(best_item["column_index"])
        return min(
            self.zones_by_col.keys(),
            key=lambda col: abs(float(anchor_xy[0]) - self.col_center_x[col]),
        )

    def assign_point(self, anchor_xy, occupied_seat_nos: set[int]):
        if not self.zones_by_col:
            return None

        col = self._pick_column(anchor_xy)
        candidates = [
            zone for zone in self.zones_by_col.get(col, [])
            if int(zone["desk_no"]) not in occupied_seat_nos
        ]
        if not candidates:
            return None

        best_zone = min(candidates, key=lambda z: l2(anchor_xy, z["desk_center"]))
        distance = l2(anchor_xy, best_zone["desk_center"])
        if distance > self.col_distance_limit.get(col, 260.0):
            return None

        return {
            "desk_id": best_zone["desk_id"],
            "desk_no": best_zone["desk_no"],
            "col_idx": best_zone["column_index"],
            "row_idx": best_zone["row_index"],
            "cost": round(distance, 4),
            "in_zone": False,
            "column_fallback": True,
            "anchor_x": round(float(anchor_xy[0]), 2),
            "anchor_y": round(float(anchor_xy[1]), 2),
            "zone_cx": round(best_zone["zone_center"][0], 2),
            "zone_cy": round(best_zone["zone_center"][1], 2),
        }


class MovementTeacherDetector:
    """
    在视频后半段, 用移动轨迹识别布局内老师。

    规则:
    1. 只在视频后半段启用
    2. 布局内 track 在观察窗口内移动距离明显较大, 则锁定为 teacher
    3. 一旦锁定为 teacher, 后续不再参与座位绑定
    """

    def __init__(self, fps: float,
                 phase_start_frame: int,
                 observe_seconds: float = 1.5,
                 move_window_seconds: float = 2.5,
                 move_distance_thresh: float = 260.0,
                 net_displacement_thresh: float = 120.0):
        self.fps = max(1.0, fps)
        self.phase_start_frame = max(0, int(phase_start_frame))
        self.observe_frames = max(1, int(round(observe_seconds * self.fps)))
        self.move_window_frames = max(2, int(round(move_window_seconds * self.fps)))
        self.move_distance_thresh = float(move_distance_thresh)
        self.net_displacement_thresh = float(net_displacement_thresh)
        self.states: dict[int, dict] = {}

    def _get_state(self, track_id: int):
        if track_id not in self.states:
            self.states[track_id] = {
                "anchors": deque(maxlen=self.move_window_frames),
                "inside_frames": 0,
                "teacher_locked": False,
                "last_seen_frame": -1,
                "away_from_bound_frames": 0,
            }
        return self.states[track_id]

    def update(self, track_id: int, frame_idx: int, anchor,
               inside_classroom: bool, bound_seat_overlap: bool):
        st = self._get_state(track_id)
        st["last_seen_frame"] = frame_idx
        if inside_classroom:
            st["inside_frames"] += 1
            st["anchors"].append(np.asarray(anchor, dtype=np.float32))
            if bound_seat_overlap:
                st["away_from_bound_frames"] = 0
            else:
                st["away_from_bound_frames"] += 1
        else:
            st["anchors"].clear()
            st["inside_frames"] = 0
            st["away_from_bound_frames"] = 0

    def _path_length(self, track_id: int) -> float:
        st = self.states.get(track_id)
        if st is None:
            return 0.0
        pts = list(st["anchors"])
        if len(pts) < 2:
            return 0.0
        return float(sum(l2(a, b) for a, b in zip(pts[:-1], pts[1:])))

    def _net_displacement(self, track_id: int) -> float:
        st = self.states.get(track_id)
        if st is None:
            return 0.0
        pts = list(st["anchors"])
        if len(pts) < 2:
            return 0.0
        return l2(pts[0], pts[-1])

    def is_teacher_locked(self, track_id: int) -> bool:
        st = self.states.get(track_id)
        if st is None:
            return False
        return bool(st.get("teacher_locked", False))

    def should_delay_new_binding(self, track_id: int, frame_idx: int,
                                 inside_classroom: bool, has_binding: bool) -> bool:
        if frame_idx < self.phase_start_frame or has_binding or not inside_classroom:
            return False
        st = self.states.get(track_id)
        if st is None:
            return True
        if st.get("teacher_locked", False):
            return False
        return st["inside_frames"] < self.observe_frames

    def is_teacher_like(self, track_id: int, frame_idx: int,
                        inside_classroom: bool,
                        has_binding: bool) -> bool:
        st = self._get_state(track_id)
        if st.get("teacher_locked", False):
            return True
        if frame_idx < self.phase_start_frame or not inside_classroom:
            return False
        if has_binding and st["away_from_bound_frames"] < self.observe_frames:
            return False
        if st["inside_frames"] < self.observe_frames:
            return False

        move_distance = self._path_length(track_id)
        net_displacement = self._net_displacement(track_id)
        if (move_distance >= self.move_distance_thresh
                and net_displacement >= self.net_displacement_thresh):
            st["teacher_locked"] = True
            return True
        return False


class GridIntervalAssigner:
    """
    包围区内强制分配器。

    逻辑:
    1. 用各行桌子中心 y 构造“相邻两行之间”的边界带
    2. 用各列桌子中心 x 构造列中心
    3. 人点在包围区内时, 先定行带, 再定最近列, 映射到 (col,row) 座位
    """

    def __init__(self, desks: list):
        self.desk_by_rc = {}
        row_map = {}
        col_map = {}

        for d in desks:
            r = int(d["row_index"])
            c = int(d["column_index"])
            self.desk_by_rc[(c, r)] = d
            row_map.setdefault(r, []).append(float(d["center"][1]))
            col_map.setdefault(c, []).append(float(d["center"][0]))

        # 按图像坐标排序: y 大在下(近), x 小在左
        self.rows_desc = sorted(row_map.keys(), key=lambda r: np.median(row_map[r]), reverse=True)
        self.row_center_y = {r: float(np.median(row_map[r])) for r in self.rows_desc}
        self.cols_asc = sorted(col_map.keys(), key=lambda c: np.median(col_map[c]))
        self.col_center_x = {c: float(np.median(col_map[c])) for c in self.cols_asc}

        self.row_intervals = self._build_row_intervals()

    def _build_row_intervals(self):
        intervals = {}
        n = len(self.rows_desc)
        if n == 0:
            return intervals

        ys = [self.row_center_y[r] for r in self.rows_desc]

        for i, r in enumerate(self.rows_desc):
            # 上边界(更小的 y)
            if i < n - 1:
                upper = 0.5 * (ys[i] + ys[i + 1])
            else:
                gap = abs(ys[i - 1] - ys[i]) if i > 0 else 80.0
                upper = ys[i] - 0.5 * gap

            # 下边界(更大的 y)
            if i > 0:
                lower = 0.5 * (ys[i - 1] + ys[i])
            else:
                gap = abs(ys[i] - ys[i + 1]) if i < n - 1 else 80.0
                lower = ys[i] + 0.5 * gap

            intervals[r] = (float(upper), float(lower))

        return intervals

    def _pick_row(self, ay: float):
        # 优先按行带区间命中
        for r in self.rows_desc:
            upper, lower = self.row_intervals[r]
            if upper <= ay <= lower:
                return r

        # 若在外侧, 回退到最近行中心
        return min(self.rows_desc, key=lambda r: abs(ay - self.row_center_y[r]))

    def _pick_col(self, ax: float):
        return min(self.cols_asc, key=lambda c: abs(ax - self.col_center_x[c]))

    def assign_point(self, anchor_xy):
        """给定点坐标, 返回 fallback 座位分配或 None."""
        if not self.rows_desc or not self.cols_asc:
            return None

        ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
        row = self._pick_row(ay)
        col = self._pick_col(ax)
        desk = self.desk_by_rc.get((col, row))
        if desk is None:
            return None

        return {
            "desk_id": desk["desk_id"],
            "desk_no": desk["desk_no"],
            "col_idx": col,
            "row_idx": row,
            "cost": round(abs(ay - self.row_center_y[row]) + abs(ax - self.col_center_x[col]), 4),
            "in_zone": False,
            "fallback_grid": True,
            "anchor_x": round(ax, 2),
            "anchor_y": round(ay, 2),
            "zone_cx": round(float(desk["center"][0]), 2),
            "zone_cy": round(float(desk["center"][1]), 2),
        }


# ─── 时间确认器 ──────────────────────────────────────────────

class TemporalConfirmer:
    """
    时间窗口绑定确认。

    规则:
    - 维护每个 track_id 最近 window 内的帧级座位分配
    - 同一座位在窗口内占比 >= confirm_ratio → 确认绑定
    - 确认后短时丢失仍保持 (hold)
    - 已确认后需连续 re_confirm_frames 帧指向新座位才切换 (防跳变)
    - 场景切换检测：当突然出现大量学生时，启用快速确认模式
    """

    def __init__(
        self,
        fps: float,
        confirm_seconds: float = 10.0,
        confirm_ratio: float = 0.65,
        hold_seconds: float = 5.0,
        max_students: int = 30,
        max_teachers: int = 2,
        re_confirm_seconds: float = 5.0,
        lock_seconds: float = 20.0,
        release_miss_seconds: float = 12.0,
        pending_switch_seconds: float = 1.2,
        pending_hold_seconds: float = 2.0,
        scene_switch_threshold: int = 10,  # 新增学生数阈值
        quick_confirm_seconds: float = 1.5,  # 快速确认时间：1.5秒
        quick_confirm_ratio: float = 0.65,   # 快速确认比例：65%
    ):
        self.fps = max(1.0, fps)
        self.confirm_frames = int(confirm_seconds * self.fps)
        self.confirm_ratio = confirm_ratio
        self.hold_frames = int(hold_seconds * self.fps)
        self.re_confirm_frames = int(re_confirm_seconds * self.fps)
        self.lock_frames = int(lock_seconds * self.fps)
        self.release_miss_frames = int(release_miss_seconds * self.fps)
        self.pending_switch_frames = max(1, int(pending_switch_seconds * self.fps))
        self.pending_hold_frames = max(1, int(pending_hold_seconds * self.fps))
        self.max_students = max_students
        self.max_teachers = max_teachers
        
        # 场景切换检测
        self.scene_switch_threshold = scene_switch_threshold
        self.quick_confirm_frames = int(quick_confirm_seconds * self.fps)
        self.quick_confirm_ratio = quick_confirm_ratio
        self.scene_switch_frame = -1
        self.last_student_count = 0
        self.quick_confirm_mode = False
        self.quick_confirm_window = int(15.0 * self.fps)  # 快速确认窗口15秒

        self.states: dict = {}
        self._desk_owner: dict = {}

    def _get_state(self, track_id):
        if track_id not in self.states:
            self.states[track_id] = {
                "history": deque(maxlen=self.confirm_frames),
                "confirmed_desk_id": None,
                "last_seen_frame": -1,
                "total_move": 0.0,
                "last_center": None,
                "consecutive_new_desk": 0,
                "consecutive_new_desk_id": None,
                "confirmed_at_frame": -1,
                # 未确认阶段用于显示去抖的状态
                "pending_display_desk_id": None,
                "pending_candidate_desk_id": None,
                "pending_candidate_count": 0,
                "pending_last_seen_frame": -1,
            }
        return self.states[track_id]

    def _update_pending_display(self, track_id, frame_idx):
        """未确认座位显示去抖: 切换需持续若干帧一致，短时丢失保留显示。"""
        st = self.states[track_id]
        vote = self.get_current_vote(track_id)

        if vote is None:
            return

        if st["pending_display_desk_id"] is None:
            st["pending_display_desk_id"] = vote
            st["pending_last_seen_frame"] = frame_idx
            st["pending_candidate_desk_id"] = None
            st["pending_candidate_count"] = 0
            return

        if vote == st["pending_display_desk_id"]:
            st["pending_last_seen_frame"] = frame_idx
            st["pending_candidate_desk_id"] = None
            st["pending_candidate_count"] = 0
            return

        # 与当前显示不一致, 需要持续若干帧才切换
        if vote == st["pending_candidate_desk_id"]:
            st["pending_candidate_count"] += 1
        else:
            st["pending_candidate_desk_id"] = vote
            st["pending_candidate_count"] = 1

        if st["pending_candidate_count"] >= self.pending_switch_frames:
            st["pending_display_desk_id"] = vote
            st["pending_last_seen_frame"] = frame_idx
            st["pending_candidate_desk_id"] = None
            st["pending_candidate_count"] = 0

    def update(self, track_id, frame_idx, desk_id, box):
        st = self._get_state(track_id)
        st["history"].append(desk_id)
        st["last_seen_frame"] = frame_idx

        center = xyxy_center(box)
        if st["last_center"] is not None:
            st["total_move"] += l2(center, st["last_center"])
        st["last_center"] = center

        # 已确认: 防跳变逻辑
        if st["confirmed_desk_id"] is not None:
            # 锁定期内不允许换座，避免轻易丢失绑定
            if frame_idx - st["confirmed_at_frame"] <= self.lock_frames:
                return

            if desk_id is not None and desk_id != st["confirmed_desk_id"]:
                if desk_id == st["consecutive_new_desk_id"]:
                    st["consecutive_new_desk"] += 1
                else:
                    st["consecutive_new_desk"] = 1
                    st["consecutive_new_desk_id"] = desk_id
                if st["consecutive_new_desk"] >= self.re_confirm_frames:
                    self._try_confirm(track_id, desk_id,
                                      st["consecutive_new_desk"], frame_idx)
                    st["consecutive_new_desk"] = 0
                    st["consecutive_new_desk_id"] = None
            else:
                st["consecutive_new_desk"] = 0
                st["consecutive_new_desk_id"] = None
            return

        # 未确认: 常规确认或快速确认
        history = st["history"]
        
        # 快速确认模式：场景切换后快速绑定稳定学生
        if self.quick_confirm_mode and len(history) >= self.quick_confirm_frames:
            votes = [d for d in history if d is not None]
            if votes:
                best_desk, count = Counter(votes).most_common(1)[0]
                if count / len(history) >= self.quick_confirm_ratio:
                    self._try_confirm(track_id, best_desk, count, frame_idx)
                    return
        
        # 常规确认模式
        if len(history) >= min(self.confirm_frames, int(self.fps * 3)):
            votes = [d for d in history if d is not None]
            if votes:
                best_desk, count = Counter(votes).most_common(1)[0]
                if count / len(history) >= self.confirm_ratio:
                    self._try_confirm(track_id, best_desk, count, frame_idx)

        # 更新未确认显示座位(去抖)
        self._update_pending_display(track_id, frame_idx)

    def _try_confirm(self, track_id: int, desk_id: str, vote_count: int, frame_idx: int):
        st = self.states[track_id]
        if st["confirmed_desk_id"] == desk_id:
            return

        current_owner = self._desk_owner.get(desk_id)
        if current_owner is not None and current_owner != track_id:
            owner_st = self.states.get(current_owner)
            if owner_st is not None:
                owner_votes = sum(1 for d in owner_st["history"] if d == desk_id)
                if vote_count <= owner_votes:
                    return
                owner_st["confirmed_desk_id"] = None
            del self._desk_owner[desk_id]
        elif current_owner == track_id:
            return

        if len(self._desk_owner) >= self.max_students:
            weakest_desk = None
            weakest_votes = vote_count
            for d_id, t_id in list(self._desk_owner.items()):
                t_st = self.states.get(t_id)
                if t_st is None:
                    weakest_desk = d_id
                    weakest_votes = -1
                    break
                t_votes = sum(1 for d in t_st["history"] if d == d_id)
                if t_votes < weakest_votes:
                    weakest_votes = t_votes
                    weakest_desk = d_id
            if weakest_desk is None:
                return
            evicted_tid = self._desk_owner[weakest_desk]
            if evicted_tid in self.states:
                self.states[evicted_tid]["confirmed_desk_id"] = None
            del self._desk_owner[weakest_desk]

        old_desk = st["confirmed_desk_id"]
        if old_desk is not None and self._desk_owner.get(old_desk) == track_id:
            del self._desk_owner[old_desk]

        st["confirmed_desk_id"] = desk_id
        st["confirmed_at_frame"] = frame_idx
        self._desk_owner[desk_id] = track_id

    def get_confirmed(self, track_id, current_frame):
        if track_id not in self.states:
            return None
        st = self.states[track_id]
        if st["confirmed_desk_id"] is None:
            return None
        # 使用更长的释放阈值，避免绑定轻易丢失
        if current_frame - st["last_seen_frame"] > self.release_miss_frames:
            desk_id = st["confirmed_desk_id"]
            if self._desk_owner.get(desk_id) == track_id:
                del self._desk_owner[desk_id]
            st["confirmed_desk_id"] = None
            return None
        return st["confirmed_desk_id"]

    def get_current_vote(self, track_id):
        if track_id not in self.states:
            return None
        history = self.states[track_id]["history"]
        votes = [d for d in history if d is not None]
        if not votes:
            return None
        return Counter(votes).most_common(1)[0][0]

    def get_display_desk_id(self, track_id, current_frame):
        confirmed = self.get_confirmed(track_id, current_frame)
        if confirmed is not None:
            return confirmed

        if track_id not in self.states:
            return None

        st = self.states[track_id]
        pending = st.get("pending_display_desk_id")
        if pending is None:
            return None

        # 短时无票时保留显示，过期后隐藏
        if current_frame - st.get("pending_last_seen_frame", -1) <= self.pending_hold_frames:
            return pending
        return None

    def is_confirmed(self, track_id, current_frame):
        return self.get_confirmed(track_id, current_frame) is not None

    def _get_teacher_set(self, current_frame) -> set:
        candidates = []
        for tid, st in self.states.items():
            if self.is_confirmed(tid, current_frame):
                continue
            if len(st["history"]) >= self.confirm_frames:
                candidates.append((st["total_move"], tid))
        candidates.sort(reverse=True)
        return {tid for _, tid in candidates[:self.max_teachers]}

    def role(self, track_id, current_frame, min_observe_frames=None):
        if min_observe_frames is None:
            min_observe_frames = self.confirm_frames
        if self.is_confirmed(track_id, current_frame):
            return "student"
        if track_id not in self.states:
            return "unknown"
        st = self.states[track_id]
        if len(st["history"]) >= min_observe_frames:
            if track_id in self._get_teacher_set(current_frame):
                return "teacher"
        return "unknown"

    def summary(self, track_id, current_frame):
        if track_id not in self.states:
            return {
                "track_id": track_id,
                "confirmed_desk_id": None,
                "role": "unknown",
                "total_move": 0.0,
            }
        st = self.states[track_id]
        return {
            "track_id": track_id,
            "confirmed_desk_id": st["confirmed_desk_id"],
            "current_vote": self.get_current_vote(track_id),
            "role": self.role(track_id, current_frame),
            "total_move": round(st["total_move"], 3),
            "observed_frames": len(st["history"]),
        }
    
    def detect_scene_switch(self, current_student_count: int, frame_idx: int):
        """
        检测场景切换：当学生数突然大幅增加时（如视频拼接）
        
        Args:
            current_student_count: 当前帧的学生数量
            frame_idx: 当前帧索引
        """
        # 学生数突然增加超过阈值，判定为场景切换
        if current_student_count - self.last_student_count >= self.scene_switch_threshold:
            self.scene_switch_frame = frame_idx
            self.quick_confirm_mode = True
            print(f"\n[场景切换检测] 帧{frame_idx}: 学生数从{self.last_student_count}增至{current_student_count}")
            print(f"  启用快速确认模式: {self.quick_confirm_frames}帧({self.quick_confirm_frames/self.fps:.1f}秒) @ {self.quick_confirm_ratio*100:.0f}%")
            print(f"  快速确认窗口: {self.quick_confirm_window/self.fps:.1f}秒")
        
        # 场景切换后一段时间，退出快速确认模式
        if self.quick_confirm_mode and frame_idx - self.scene_switch_frame > self.quick_confirm_window:
            self.quick_confirm_mode = False
            print(f"\n[场景切换] 帧{frame_idx}: 退出快速确认模式，恢复常规确认")
        
        self.last_student_count = current_student_count


# ─── 可视化 ──────────────────────────────────────────────────

def draw_text(img, text, org, color, scale=0.55, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def color_for_desk(desk_no):
    palette = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 200, 80), (220, 80, 255), (80, 220, 255),
    ]
    return palette[(desk_no - 1) % len(palette)]


def draw_column_lines(img, column_lines, img_h):
    vis = img.copy()
    colors = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 200, 80), (220, 80, 255),
    ]
    for col_line in column_lines:
        col_idx = col_line["column_index"]
        color = colors[col_idx % len(colors)]
        start = tuple(map(int, col_line["segment_start"]))
        end = tuple(map(int, col_line["segment_end"]))
        cv2.line(vis, start, end, color, 2)
        cv2.putText(vis, f"Col{col_idx}", (start[0] + 5, start[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return vis


# ─── 主管线 V3 ───────────────────────────────────────────────

class PersonDeskBindingPipelineV3:
    def __init__(
        self,
        source: str,
        person_weights: str,
        desk_weights: str,
        output_dir: str,
        person_conf: float,
        person_iou: float,
        desk_conf: float,
        desk_iou: float,
        device: str,
        img_size: int | None,
        half: bool,
        tracker: str,
        person_classes: list[int] | None,
        desk_mode: str,
        desk_num_cols: int,
        desk_required_per_col: int,
        reference_max_frames: int,
        reference_sample_step: int,
        confirm_seconds: float,
        confirm_ratio: float,
        hold_seconds: float,
        re_confirm_seconds: float,
        lock_seconds: float,
        release_miss_seconds: float,
        pending_switch_seconds: float,
        pending_hold_seconds: float,
        first_extend_ratio: float,
        last_extend_ratio: float,
        outputs: set[str],
        display: bool,
        simple_vis: bool,
        max_students: int = 30,
        max_teachers: int = 2,
        layout_callback=None,
        frame_callback=None,
        finish_callback=None,
    ):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"视频不存在: {source}")
        if not os.path.isfile(person_weights):
            raise FileNotFoundError(f"人物模型不存在: {person_weights}")
        if not os.path.isfile(desk_weights):
            raise FileNotFoundError(f"桌子模型不存在: {desk_weights}")

        self.source = source
        self.output_dir = output_dir
        self.person_conf = person_conf
        self.person_iou = person_iou
        self.device = device
        self.img_size = img_size
        self.half = half
        self.tracker = tracker
        self.person_classes = person_classes
        self.reference_max_frames = reference_max_frames
        self.reference_sample_step = max(1, reference_sample_step)
        self.confirm_seconds = confirm_seconds
        self.confirm_ratio = confirm_ratio
        self.hold_seconds = hold_seconds
        self.re_confirm_seconds = re_confirm_seconds
        self.lock_seconds = lock_seconds
        self.release_miss_seconds = release_miss_seconds
        self.pending_switch_seconds = pending_switch_seconds
        self.pending_hold_seconds = pending_hold_seconds
        self.first_extend_ratio = first_extend_ratio
        self.last_extend_ratio = last_extend_ratio
        self.outputs = set(outputs)
        self.display = display
        self.simple_vis = simple_vis
        self.max_students = max_students
        self.max_teachers = max_teachers
        self.layout_callback = layout_callback
        self.frame_callback = frame_callback
        self.finish_callback = finish_callback

        self.person_model = YOLO(person_weights)
        self.person_names = getattr(self.person_model, "names", None)
        self.desk_detector = desk_layout_mod.DeskLayoutDetector(
            weights_path=desk_weights,
            conf_threshold=desk_conf,
            iou_threshold=desk_iou,
            device=device,
            img_size=img_size,
            half=half,
            mode=desk_mode,
            num_cols=desk_num_cols,
            required_per_col=desk_required_per_col,
        )

    def _emit_layout(self, payload: dict):
        if callable(self.layout_callback):
            self.layout_callback(payload)

    def _emit_frame(self, payload: dict):
        if callable(self.frame_callback):
            self.frame_callback(payload)

    def _emit_finish(self, payload: dict):
        if callable(self.finish_callback):
            self.finish_callback(payload)

    # ── 参考帧桌子检测 ───────────────────────────────────────

    def _select_reference_desks(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.source}")
        best = None
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx >= self.reference_max_frames:
                    break
                if frame_idx % self.reference_sample_step == 0:
                    result = self.desk_detector.detect_desks(
                        frame, tag=f"ref:{frame_idx}", annotate=False, log=False,
                    )
                    desks = result["layout"]["desks"]
                    score = sum(d["conf"] for d in desks)
                    if desks and (
                        best is None
                        or len(desks) > best["desk_count"]
                        or (len(desks) == best["desk_count"]
                            and score > best["score"])
                    ):
                        best = {
                            "frame_idx": frame_idx,
                            "frame": frame.copy(),
                            "layout": result["layout"],
                            "desk_count": len(desks),
                            "score": score,
                        }
                frame_idx += 1
        finally:
            cap.release()

        if best is None:
            raise RuntimeError("桌子检测失败，未找到可用桌子布局。")
        return best

    # ── 人物提取 ─────────────────────────────────────────────

    def _extract_people(self, boxes, frame_idx):
        people = []
        ids = None
        if hasattr(boxes, "id") and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
        for idx, box in enumerate(boxes):
            xyxy = list(map(float, box.xyxy[0].tolist()))
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            track_id = ids[idx] if ids is not None else frame_idx * 10000 + idx
            people.append({
                "track_id": int(track_id),
                "xyxy": xyxy,
                "conf": conf,
                "cls": cls,
            })
        return people

    # ── 绘制 ─────────────────────────────────────────────────

    def _draw_zones(self, frame, zones, classroom_polygon=None):
        """绘制梯形绑定区域 + 桌子框 + 区域中心."""
        annotated = frame.copy()
        overlay = annotated.copy()

        for zone in zones:
            poly_int = zone["polygon"].astype(np.int32)
            color = color_for_desk(zone["desk_no"])

            if not self.simple_vis:
                # 半透明填充区域
                cv2.fillPoly(overlay, [poly_int], color)
                # 区域边框
                cv2.polylines(annotated, [poly_int], True, color, 2)

            # 桌子实框
            dx1, dy1, dx2, dy2 = map(int, zone["desk_xyxy"])
            cv2.rectangle(annotated, (dx1, dy1), (dx2, dy2), color, 1 if self.simple_vis else 2)

            # 标签
            if not self.simple_vis:
                draw_text(annotated, zone["desk_id"],
                          (dx1, max(20, dy1 - 6)), color, scale=0.55)

            # 区域中心标记
            if not self.simple_vis:
                zcx = int(zone["zone_center"][0])
                zcy = int(zone["zone_center"][1])
                cv2.drawMarker(annotated, (zcx, zcy), color,
                               cv2.MARKER_CROSS, 10, 1)

        # 混合叠加
        if not self.simple_vis:
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        if classroom_polygon is not None and len(classroom_polygon) >= 3:
            cp = classroom_polygon.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [cp], True, (255, 255, 255), 2)
            tl = tuple(np.min(classroom_polygon, axis=0).astype(np.int32))
            draw_text(annotated, "SEAT_AREA", (int(tl[0]), max(20, int(tl[1]) - 8)),
                      (255, 255, 255), scale=0.55)

        return annotated

    def _draw_person(self, frame, person, anchor, role, display_desk_id,
                     display_physical_desk_id,
                     display_seat_id,
                     is_confirmed, assignment, zone_lookup):
        x1, y1, x2, y2 = map(int, person["xyxy"])

        if is_confirmed:
            color = (0, 255, 0)
        elif display_desk_id is not None:
            color = (0, 255, 255)
        elif role == "teacher":
            color = (0, 0, 255)
        else:
            color = (180, 180, 180)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 锚点 (人框中心)
        ax, ay = int(anchor[0]), int(anchor[1])
        if not self.simple_vis:
            cv2.circle(frame, (ax, ay), 5, (0, 255, 255), -1)

        # 到区域中心连线
        if assignment is not None and not self.simple_vis:
            zcx = int(assignment["zone_cx"])
            zcy = int(assignment["zone_cy"])
            cv2.line(frame, (ax, ay), (zcx, zcy), (255, 0, 255), 1)

        # 到物理桌子中心连线
        if (display_physical_desk_id and display_physical_desk_id in zone_lookup
                and not self.simple_vis):
            dc = zone_lookup[display_physical_desk_id]["desk_center"]
            dcx, dcy = int(dc[0]), int(dc[1])
            cv2.line(frame, (ax, ay), (dcx, dcy), color, 2)

        tid = person["track_id"]
        mark = "✓" if is_confirmed else "?"
        # 学生优先显示统一后的座位号 1..30
        sid = str(display_seat_id) if display_seat_id is not None else (
            display_physical_desk_id if display_physical_desk_id else f"#{tid}"
        )
        label1 = f"{sid} {role}"
        if is_confirmed:
            label2 = f"{mark}" if not self.simple_vis else f"{sid}"
        else:
            label2 = "PENDING" if display_desk_id is not None else "UNBOUND"
        draw_text(frame, label1, (x1, max(20, y1 - 20)), color, scale=0.50)
        if not self.simple_vis:
            draw_text(frame, label2, (x1, max(36, y1 - 2)), color, scale=0.50)

    # ── 主运行 ───────────────────────────────────────────────

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        video_name = Path(self.source).stem

        output_video_path = os.path.join(
            self.output_dir, f"bound_v3_{video_name}.mp4")
        output_csv_path = os.path.join(
            self.output_dir, f"binding_v3_{video_name}.csv")
        output_json_path = os.path.join(
            self.output_dir, f"binding_v3_{video_name}.json")
        output_zone_map_path = os.path.join(
            self.output_dir, f"zone_map_v3_{video_name}.jpg")

        # ── Step 1: 桌子检测 → 6×5 网格 ─────────────────────
        print("=" * 60)
        print("Step 1: 建立桌子布局...")
        reference = self._select_reference_desks()
        layout = reference["layout"]
        desks = layout["desks"]
        column_lines = layout.get("column_lines", [])

        print(f"  桌子参考帧: {reference['frame_idx']}")
        print(f"  检测到桌子: {len(desks)} 个")
        print(f"  列直线: {len(column_lines)} 条")
        print(f"  布局方案: {layout.get('chosen_mode', 'unknown')}")

        # ── Step 2: 构建相邻桌子区间 ─────────────────────────
        print("=" * 60)
        print("Step 2: 构建相邻桌子绑定区间 (梯形区域)...")
        zone_builder = InterDeskZoneBuilder(
            first_extend_ratio=self.first_extend_ratio,
            last_extend_ratio=self.last_extend_ratio,
        )
        zones = zone_builder.build(desks)
        zone_lookup = {z["desk_id"]: z for z in zones}
        # 增大padding确保边缘学生在包围区内
        classroom_polygon = build_classroom_polygon_from_desks(desks, padding=100.0)

        print(f"  绑定区域数: {len(zones)}")
        for z in zones:
            print(f"    {z['desk_id']}: col={z['column_index']} "
                  f"row={z['row_index']} "
                  f"center=({z['zone_center'][0]:.0f},"
                  f"{z['zone_center'][1]:.0f})")

        # 保存区间可视化
        if "zone_map" in self.outputs:
            zone_frame = self._draw_zones(reference["frame"], zones, classroom_polygon)
            if column_lines:
                zone_frame = draw_column_lines(
                    zone_frame, column_lines, reference["frame"].shape[0])
            cv2.imwrite(output_zone_map_path, zone_frame)
            print(f"  区间可视化: {output_zone_map_path}")

        self._emit_layout({
            "source": self.source,
            "reference_frame_idx": reference["frame_idx"],
            "frame_width": int(reference["frame"].shape[1]),
            "frame_height": int(reference["frame"].shape[0]),
            "desk_count": len(desks),
            "zone_count": len(zones),
            "chosen_mode": layout.get("chosen_mode", "unknown"),
            "desks": desks,
            "zones": [
                {
                    "desk_id": z["desk_id"],
                    "desk_no": int(z["desk_no"]),
                    "column_index": int(z["column_index"]),
                    "row_index": int(z["row_index"]),
                    "desk_xyxy": [float(v) for v in z["desk_xyxy"]],
                    "polygon": z["polygon"].tolist(),
                    "zone_center": [float(v) for v in z["zone_center"]],
                }
                for z in zones
            ],
            "column_lines": column_lines,
            "classroom_polygon": (
                classroom_polygon.tolist() if classroom_polygon is not None else None
            ),
        })

        # ── Step 3: 构建区间绑定器 ───────────────────────────
        print("=" * 60)
        print("Step 3: 构建梯形重叠绑定器...")
        binder = ZoneBinder(zones=zones)
        column_fallback = ColumnFallbackAssigner(zones=zones, column_lines=column_lines)
        seat_no_to_desk = {int(z["desk_no"]): z["desk_id"] for z in zones}
        seat_no_to_zone = {int(z["desk_no"]): z for z in zones}
        print("  规则1: 只对教室包围区内的人做座位绑定")
        print("  规则2: 人框与梯形座位区的 IoU 越大, 优先级越高")
        print("  规则3: 按 IoU 做一对一贪心匹配")
        print("  输出ID: 学生对外统一使用座位号 1..30")

        # ── 打开视频 ─────────────────────────────────────────
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-6:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── Step 4: 逐帧绑定 ────────────────────────────────
        print("=" * 60)
        print("Step 4: 逐帧处理 (梯形 IoU 绑定)...")
        stable_manager = StableSeatManager(
            fps=fps,
            switch_seconds=4.5,
            miss_hold_seconds=8.0,
            initial_bind_seconds=2.0,
        )
        teacher_phase_start_frame = (
            total_frames // 2 if total_frames and total_frames > 0 else 0
        )
        teacher_detector = MovementTeacherDetector(
            fps=fps,
            phase_start_frame=teacher_phase_start_frame,
            observe_seconds=1.5,
            move_window_seconds=2.5,
            move_distance_thresh=260.0,
            net_displacement_thresh=120.0,
        )
        print(f"  初次入座确认: {stable_manager.initial_confirm_frames} 帧 "
              f"({stable_manager.initial_confirm_frames / fps:.2f}s)")
        print(f"  换座确认: {stable_manager.switch_confirm_frames} 帧 "
              f"({stable_manager.switch_confirm_frames / fps:.2f}s)")
        print(f"  丢检保持: {stable_manager.miss_hold_frames} 帧 "
              f"({stable_manager.miss_hold_frames / fps:.2f}s)")
        print(f"  老师判定起点: 帧 {teacher_phase_start_frame} "
              f"({teacher_phase_start_frame / fps:.2f}s)")
        print(f"  老师观察窗口: {teacher_detector.observe_frames} 帧 / "
              f"{teacher_detector.move_window_frames} 帧")
        print(f"  老师移动阈值: 路径 {teacher_detector.move_distance_thresh:.0f}px, "
              f"位移 {teacher_detector.net_displacement_thresh:.0f}px")
        print(f"  视频: {width}x{height} @ {fps:.1f}fps, "
              f"总帧: {total_frames}")

        # ── 输出准备 ─────────────────────────────────────────
        video_writer = None
        if "binding_video" in self.outputs:
            video_writer = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps), (width, height),
            )

        csv_file = None
        csv_writer = None
        if "csv" in self.outputs:
            csv_file = open(output_csv_path, "w", newline="",
                            encoding="utf-8-sig")
            csv_writer = csv.DictWriter(csv_file, fieldnames=[
                "frame_idx", "track_id", "track_id_raw", "role",
                "is_bound", "is_confirmed",
                "person_conf",
                "seat_no_current", "seat_no_display",
                "desk_id_current", "desk_id_confirmed", "desk_id_display",
                "bind_track_id",
                "physical_desk_id_current", "physical_desk_id_display",
                "student_id_aligned",
                "seat_id_same_as_track",
                "inside_classroom",
                "in_zone",
                "overlap_iou", "overlap_area",
                "fallback_grid",
                "anchor_x", "anchor_y",
                "zone_cx", "zone_cy",
                "x1", "y1", "x2", "y2", "total_move",
            ])
            csv_writer.writeheader()

        # ── 逐帧处理 ─────────────────────────────────────────
        frame_idx = 0
        track_region_role = {}
        track_physical_desk = {}
        track_seat_votes: dict[int, Counter] = {}
        seat_track_votes: dict[int, Counter] = {
            int(z["desk_no"]): Counter() for z in zones
        }
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                track_kwargs = {
                    "source": frame,
                    "conf": self.person_conf,
                    "iou": self.person_iou,
                    "half": self.half,
                    "verbose": False,
                    "persist": True,
                    "tracker": self.tracker,
                    "classes": self.person_classes,
                }
                if self.device:
                    track_kwargs["device"] = self.device
                if self.img_size is not None:
                    track_kwargs["imgsz"] = self.img_size

                results = self.person_model.track(**track_kwargs)
                boxes = results[0].boxes
                people = self._extract_people(boxes, frame_idx)
                anchor_pts = {
                    p["track_id"]: foot_point(p["xyxy"])
                    for p in people
                }
                teacher_like_tracks = set()
                delayed_tracks = set()
                for person in people:
                    tid = person["track_id"]
                    anchor = anchor_pts[tid]
                    inside_classroom = point_in_polygon(anchor, classroom_polygon)
                    has_binding = stable_manager.has_active_binding(tid, frame_idx)
                    bound_seat_no = stable_manager.get_bound_seat(tid, frame_idx)
                    bound_seat_overlap = False
                    if bound_seat_no is not None:
                        bound_zone = seat_no_to_zone.get(int(bound_seat_no))
                        if bound_zone is not None:
                            _, keep_area = convex_polygon_iou(
                                box_polygon(person["xyxy"]),
                                bound_zone["polygon"],
                            )
                            bound_seat_overlap = (
                                point_in_polygon(anchor, bound_zone["polygon"])
                                or keep_area > 1.0
                            )
                    teacher_detector.update(
                        tid, frame_idx, anchor, inside_classroom, bound_seat_overlap
                    )
                    if teacher_detector.is_teacher_like(
                        tid, frame_idx, inside_classroom, has_binding
                    ):
                        if has_binding:
                            stable_manager.force_release(tid)
                        teacher_like_tracks.add(tid)
                    elif teacher_detector.should_delay_new_binding(
                        tid, frame_idx, inside_classroom, has_binding
                    ):
                        delayed_tracks.add(tid)

                student_people = [
                    p for p in people
                    if (point_in_polygon(anchor_pts[p["track_id"]], classroom_polygon)
                        and p["track_id"] not in teacher_like_tracks
                        and p["track_id"] not in delayed_tracks)
                ]
                assignments, student_anchor_pts = binder.assign_batch(student_people)
                anchor_pts.update(student_anchor_pts)
                occupied_seat_nos = (
                    stable_manager.active_seat_numbers(frame_idx)
                    | {int(a["desk_no"]) for a in assignments.values()}
                )
                for person in student_people:
                    tid = person["track_id"]
                    if tid in assignments:
                        continue
                    if stable_manager.has_active_binding(tid, frame_idx):
                        continue
                    fb = column_fallback.assign_point(anchor_pts[tid], occupied_seat_nos)
                    if fb is None:
                        continue
                    assignments[tid] = fb
                    occupied_seat_nos.add(int(fb["desk_no"]))

                frame_rows = []

                # 绘制
                annotated = None
                if (
                    video_writer is not None
                    or self.display
                    or callable(self.frame_callback)
                ):
                    annotated = self._draw_zones(frame, zones, classroom_polygon)
                    if column_lines and not self.simple_vis:
                        annotated = draw_column_lines(
                            annotated, column_lines, height)

                student_count = 0
                teacher_count = 0
                unknown_count = 0

                for person in people:
                    tid = person["track_id"]
                    anchor = anchor_pts.get(
                        tid, foot_point(person["xyxy"]))
                    assignment = assignments.get(tid)
                    inside_classroom = point_in_polygon(anchor, classroom_polygon)
                    is_teacher_like = tid in teacher_like_tracks

                    current_seat_no = int(assignment["desk_no"]) if assignment else None
                    current_desk_id = assignment["desk_id"] if assignment else None
                    bound_seat_no = stable_manager.get_display_seat(tid, frame_idx)
                    proposed_seat_no = current_seat_no

                    # 已绑定学生只要当前人框仍与原座位区有重叠, 就继续保持原座位,
                    # 避免在座位附近小范围移动时误切到邻座。
                    if bound_seat_no is not None:
                        bound_zone = seat_no_to_zone.get(bound_seat_no)
                        if bound_zone is not None:
                            keep_iou, keep_area = convex_polygon_iou(
                                box_polygon(person["xyxy"]),
                                bound_zone["polygon"],
                            )
                            anchor_inside_bound = point_in_polygon(
                                anchor, bound_zone["polygon"]
                            )
                            if anchor_inside_bound or keep_area > 1.0:
                                proposed_seat_no = bound_seat_no

                    should_update_binding = ((inside_classroom and not is_teacher_like)
                                             or stable_manager.has_active_binding(
                        tid, frame_idx
                    ))
                    if should_update_binding:
                        display_seat_id = stable_manager.update(
                            tid, frame_idx, proposed_seat_no
                        )
                    else:
                        display_seat_id = None

                    display_physical_desk = (
                        seat_no_to_desk.get(display_seat_id)
                        if display_seat_id is not None else None
                    )
                    display_desk = display_physical_desk
                    if display_seat_id is not None:
                        track_physical_desk[tid] = display_physical_desk
                        track_seat_votes.setdefault(tid, Counter())[display_seat_id] += 1
                        seat_track_votes.setdefault(display_seat_id, Counter())[tid] += 1

                    is_bound = display_seat_id is not None
                    if is_bound:
                        r = "student"
                    elif is_teacher_like:
                        r = "teacher"
                    elif inside_classroom:
                        r = "unknown"
                    else:
                        r = "teacher"
                    track_region_role[tid] = r

                    if r == "student" and is_bound:
                        student_count += 1
                    elif r == "teacher":
                        teacher_count += 1
                    else:
                        unknown_count += 1

                    if annotated is not None:
                        self._draw_person(
                            annotated, person, anchor, r, display_desk,
                            display_physical_desk,
                            display_seat_id,
                            is_bound, assignment, zone_lookup,
                        )

                    if r == "student":
                        public_track_id = display_seat_id
                        aligned_student_id = display_seat_id
                    else:
                        public_track_id = f"T{tid}"
                        aligned_student_id = None

                    frame_row = {
                        "frame_idx": frame_idx,
                        "track_id": public_track_id,
                        "track_id_raw": tid,
                        "role": r,
                        "is_bound": is_bound,
                        "is_confirmed": is_bound,
                        "person_conf": round(person["conf"], 4),
                        "seat_no_current": current_seat_no,
                        "seat_no_display": display_seat_id,
                        "desk_id_current": current_desk_id,
                        "desk_id_confirmed": display_physical_desk,
                        "desk_id_display": display_desk,
                        "bind_track_id": tid,
                        "physical_desk_id_current": current_desk_id,
                        "physical_desk_id_display": display_physical_desk,
                        "student_id_aligned": aligned_student_id,
                        "seat_id_same_as_track": display_seat_id,
                        "inside_classroom": inside_classroom,
                        "in_zone": assignment is not None,
                        "overlap_iou": (
                            assignment.get("overlap_iou") if assignment else None
                        ),
                        "overlap_area": (
                            assignment.get("overlap_area") if assignment else None
                        ),
                        "fallback_grid": bool(
                            assignment
                            and (
                                assignment.get("fallback_grid")
                                or assignment.get("column_fallback")
                            )
                        ),
                        "anchor_x": round(float(anchor[0]), 2),
                        "anchor_y": round(float(anchor[1]), 2),
                        "zone_cx": assignment["zone_cx"] if assignment else None,
                        "zone_cy": assignment["zone_cy"] if assignment else None,
                        "x1": round(person["xyxy"][0], 2),
                        "y1": round(person["xyxy"][1], 2),
                        "x2": round(person["xyxy"][2], 2),
                        "y2": round(person["xyxy"][3], 2),
                        "total_move": 0.0,
                    }
                    frame_rows.append(frame_row)
                    if csv_writer is not None:
                        csv_writer.writerow(frame_row)

                info = (
                    f"Frame {frame_idx}/{total_frames or '?'} | "
                    f"S={student_count} T={teacher_count} ?={unknown_count}"
                )
                if annotated is not None:
                    draw_text(annotated, info, (10, 28),
                              (255, 255, 255), scale=0.8)
                    if video_writer is not None:
                        video_writer.write(annotated)

                callback_frame = None
                if callable(self.frame_callback):
                    callback_frame = (
                        annotated.copy() if annotated is not None else frame.copy()
                    )
                    self._emit_frame({
                        "source": self.source,
                        "frame_idx": frame_idx,
                        "total_frames": total_frames,
                        "fps": fps,
                        "frame_width": width,
                        "frame_height": height,
                        "student_count": student_count,
                        "teacher_count": teacher_count,
                        "unknown_count": unknown_count,
                        "active_bindings": sorted(
                            int(seat_no) for seat_no in stable_manager.active_seat_numbers(frame_idx)
                        ),
                        "rows": frame_rows,
                        "frame_bgr": callback_frame,
                    })

                if self.display and annotated is not None:
                    cv2.imshow("V3 Zone Binding", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("用户中断")
                        break

                if frame_idx % 30 == 0 and total_frames > 0:
                    pct = 100.0 * frame_idx / total_frames
                    total_assigned = len(assignments)
                    print(f"  {frame_idx}/{total_frames} ({pct:.1f}%) "
                          f"S={student_count} T={teacher_count} ?={unknown_count} | "
                          f"当前重叠绑定={total_assigned}/30")

                frame_idx += 1

        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if csv_file is not None:
                csv_file.close()
            if self.display:
                cv2.destroyAllWindows()

        # ── 后处理：按整段视频汇总每个座位的最终 track ───────
        print("=" * 60)
        print("后处理：按每个座位的累计重叠票数生成最终绑定")

        final_seat_bindings = {}
        for seat_no in sorted(seat_no_to_desk):
            raw_votes = seat_track_votes.get(seat_no, Counter())
            if not raw_votes:
                continue
            raw_tid, vote_frames = raw_votes.most_common(1)[0]
            desk_id = seat_no_to_desk[seat_no]
            final_seat_bindings[seat_no] = {
                "original_track_id": int(raw_tid),
                "unified_track_id": int(seat_no),
                "physical_desk_id": desk_id,
                "vote_frames": int(vote_frames),
            }
            print(f"  座位 {seat_no:02d} -> raw track {raw_tid} ({vote_frames} 帧)")

        occupied_seat_nos = set(final_seat_bindings.keys())
        occupied_bind_ids = {seat_no_to_desk[seat_no] for seat_no in occupied_seat_nos}
        print(f"  最终绑定: {len(occupied_bind_ids)}/30 个座位")

        # ── 输出 JSON ────────────────────────────────────────
        print("=" * 60)
        print("输出汇总:")

        tracks_list = []
        for seat_no in sorted(final_seat_bindings):
            binding = final_seat_bindings[seat_no]
            raw_tid = binding["original_track_id"]
            seat_votes = track_seat_votes.get(raw_tid, Counter())
            tracks_list.append({
                "track_id": int(seat_no),
                "seat_no": int(seat_no),
                "confirmed_desk_id": binding["physical_desk_id"],
                "role": "student",
                "observed_frames": int(sum(seat_votes.values())),
                "seat_vote_frames": int(binding["vote_frames"]),
                "student_id_aligned": int(seat_no),
                "seat_id_unified": int(seat_no),
                "physical_desk_id": binding["physical_desk_id"],
                "original_track_id": raw_tid,
            })

        print(f"  生成学生tracks: {len(tracks_list)}/30")

        summary = {
            "source": self.source,
            "version": "v3_zone_binding_iou",
            "reference_frame_idx": reference["frame_idx"],
            "desk_count": len(desks),
            "zone_count": len(zones),
            "occupied_seats": len(occupied_bind_ids),
            "config": {
                "binding_method": "inter_desk_zone_iou",
                "anchor_point": "bbox_polygon",
                "teacher_rule": "outside_classroom_polygon",
                "student_rule": "inside_classroom_polygon",
                "matching_rule": "bbox_zone_iou_greedy",
                "initial_bind_seconds": round(
                    stable_manager.initial_confirm_frames / fps, 3
                ),
                "switch_confirm_seconds": round(
                    stable_manager.switch_confirm_frames / fps, 3
                ),
                "miss_hold_seconds": round(
                    stable_manager.miss_hold_frames / fps, 3
                ),
                "teacher_phase_start_seconds": round(
                    teacher_phase_start_frame / fps, 3
                ),
                "teacher_detect_observe_seconds": round(
                    teacher_detector.observe_frames / fps, 3
                ),
                "teacher_move_window_seconds": round(
                    teacher_detector.move_window_frames / fps, 3
                ),
                "teacher_move_distance_thresh": teacher_detector.move_distance_thresh,
                "teacher_net_displacement_thresh": teacher_detector.net_displacement_thresh,
                "first_extend_ratio": self.first_extend_ratio,
                "last_extend_ratio": self.last_extend_ratio,
                "simple_vis": self.simple_vis,
            },
            "classroom_polygon": (classroom_polygon.tolist()
                                  if classroom_polygon is not None else None),
            "desks": desks,
            "zones": [
                {
                    "desk_id": z["desk_id"],
                    "desk_no": z["desk_no"],
                    "column_index": z["column_index"],
                    "row_index": z["row_index"],
                    "desk_xyxy": z["desk_xyxy"],
                    "polygon": z["polygon"].tolist(),
                    "zone_center": z["zone_center"],
                }
                for z in zones
            ],
            "column_lines": column_lines,
            "tracks": tracks_list,
            "seat_track_bindings": {
                str(seat_no): {
                    **binding,
                    "unified_student_id": int(seat_no),
                }
                for seat_no, binding in sorted(final_seat_bindings.items(), key=lambda kv: kv[0])
            },
            "seat_track_bindings_by_desk_id": {
                binding["physical_desk_id"]: {
                    **binding,
                    "unified_student_id": int(seat_no),
                }
                for seat_no, binding in sorted(final_seat_bindings.items(), key=lambda kv: kv[0])
            },
            "seat_occupancy": {
                str(z["desk_no"]): int(z["desk_no"]) in occupied_seat_nos
                for z in zones
            },
            "seat_occupancy_by_desk_id": {
                z["desk_id"]: int(z["desk_no"]) in occupied_seat_nos
                for z in zones
            },
            "saved_outputs": {},
        }

        if "binding_video" in self.outputs:
            summary["saved_outputs"]["binding_video"] = output_video_path
            print(f"  绑定视频: {output_video_path}")
        if "zone_map" in self.outputs:
            summary["saved_outputs"]["zone_map"] = output_zone_map_path
            print(f"  区间可视化: {output_zone_map_path}")
        if "csv" in self.outputs:
            summary["saved_outputs"]["csv"] = output_csv_path
            print(f"  CSV: {output_csv_path}")
        if "json" in self.outputs:
            output_json_dir = os.path.dirname(output_json_path)
            if output_json_dir:
                os.makedirs(output_json_dir, exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            summary["saved_outputs"]["json"] = output_json_path
            print(f"  JSON: {output_json_path}")

        print("=" * 60)
        print(f"完成! {frame_idx} 帧, "
              f"已绑定座位 {len(occupied_bind_ids)}/{len(zones)} 个")
        self._emit_finish(summary)
        return summary


# ─── CLI ─────────────────────────────────────────────────────

def parse_classes(s: str | None):
    if not s:
        return None
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def parse_outputs(raw: str | None):
    valid = {"binding_video", "zone_map", "csv", "json"}
    if not raw:
        return valid
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    if "all" in parts:
        return valid
    unknown = parts - valid
    if unknown:
        raise ValueError(
            f"未知输出: {', '.join(unknown)}。"
            f"可选: {', '.join(sorted(valid))}")
    return parts


def main():
    p = argparse.ArgumentParser(
        "人桌绑定 V3 (相邻桌子区间绑定)")

    p.add_argument("--source", required=True, help="输入视频路径")
    p.add_argument("--weights",
                   default="exam_seat_binding/weight/yolo11speopel.pt",
                   help="人物模型权重")
    p.add_argument("--desk-weights",
                   default="exam_seat_binding/weight/yolo11desk.pt",
                   help="桌子模型权重")
    p.add_argument("--output", default="exam_seat_binding/output",
                   help="输出目录")
    p.add_argument("--outputs",
                   default="binding_video,zone_map,csv,json",
                   help="输出: binding_video,zone_map,csv,json 或 all")

    # 检测参数
    p.add_argument("--conf", type=float, default=0.18)
    p.add_argument("--iou", type=float, default=0.60)
    p.add_argument("--desk-conf", type=float, default=0.7)
    p.add_argument("--desk-iou", type=float, default=0.45)
    p.add_argument("--device", default="")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--half", action="store_true")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--classes", default=None)
    p.add_argument("--display", action="store_true")

    # 桌子布局参数
    p.add_argument("--desk-mode", default="auto",
                   choices=["normal", "scheme1", "scheme2", "auto"])
    p.add_argument("--desk-num-cols", type=int, default=5)
    p.add_argument("--desk-required-per-col", type=int, default=6)
    p.add_argument("--reference-max-frames", type=int, default=120)
    p.add_argument("--reference-sample-step", type=int, default=5)

    # V3 核心: 区间构建参数
    p.add_argument("--first-extend", type=float, default=0.6,
                   help="第一排座位向下(近相机)延伸比例 (基于间距)")
    p.add_argument("--last-extend", type=float, default=1.2,
                   help="最后排座位向上(远相机)延伸比例 (基于间距)")

    # 时间确认参数
    p.add_argument("--confirm-seconds", type=float, default=3.0)
    p.add_argument("--confirm-ratio", type=float, default=0.50)
    p.add_argument("--hold-seconds", type=float, default=5.0)
    p.add_argument("--re-confirm-seconds", type=float, default=5.0,
                   help="已确认后切换座位所需连续帧时间")
    p.add_argument("--lock-seconds", type=float, default=20.0,
                   help="确认后绑定锁定时长，锁定期内不允许换座")
    p.add_argument("--release-miss-seconds", type=float, default=12.0,
                   help="连续丢失超过该时长才释放绑定")
    p.add_argument("--pending-switch-seconds", type=float, default=1.2,
                   help="未确认显示座位切换所需持续时间(去抖)")
    p.add_argument("--pending-hold-seconds", type=float, default=2.0,
                   help="未确认短时丢失时显示保留时长")
    p.add_argument("--simple-vis", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="是否使用简洁可视化（默认开启）")

    # 教室规模
    p.add_argument("--max-students", type=int, default=30)
    p.add_argument("--max-teachers", type=int, default=2)

    args = p.parse_args()

    try:
        outputs = parse_outputs(args.outputs)
    except ValueError as exc:
        p.error(str(exc))

    pipeline = PersonDeskBindingPipelineV3(
        source=args.source,
        person_weights=args.weights,
        desk_weights=args.desk_weights,
        output_dir=args.output,
        person_conf=args.conf,
        person_iou=args.iou,
        desk_conf=args.desk_conf,
        desk_iou=args.desk_iou,
        device=args.device,
        img_size=args.img_size,
        half=args.half,
        tracker=args.tracker,
        person_classes=parse_classes(args.classes),
        desk_mode=args.desk_mode,
        desk_num_cols=args.desk_num_cols,
        desk_required_per_col=args.desk_required_per_col,
        reference_max_frames=args.reference_max_frames,
        reference_sample_step=args.reference_sample_step,
        confirm_seconds=args.confirm_seconds,
        confirm_ratio=args.confirm_ratio,
        hold_seconds=args.hold_seconds,
        re_confirm_seconds=args.re_confirm_seconds,
        lock_seconds=args.lock_seconds,
        release_miss_seconds=args.release_miss_seconds,
        pending_switch_seconds=args.pending_switch_seconds,
        pending_hold_seconds=args.pending_hold_seconds,
        first_extend_ratio=args.first_extend,
        last_extend_ratio=args.last_extend,
        outputs=outputs,
        display=args.display,
        simple_vis=args.simple_vis,
        max_students=args.max_students,
        max_teachers=args.max_teachers,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
