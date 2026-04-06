"""
人桌绑定 V2 —— 基于头部投影 + 列直线 + 时间确认

核心改进（相对 V1）:
1. 锚点: 使用检测框顶部中心 (头部位置), 而非 72% 位置
2. 列归属: 将头部锚点投影到最近的列直线上, 而非基于桌子框重叠
3. 行归属: 根据投影点在列上的位置, 与该列各桌子中心 y 坐标比较
4. 时间确认: 投影点在同一座位上稳定停留 N 秒后才确认绑定 (默认 10s)
5. 教师检测: 一段时间观察后仍无稳定绑定的人识别为教师

用法:
python exam_seat_binding/person_desk_binding_v2.py --source data/ideotest/merged_output.mp4
python exam_seat_binding/person_desk_binding_v2.py --source data/ideotest/merged_output.mp4 --confirm-seconds 10 --display
"""

import argparse
import csv
import importlib.util
import json
import math
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

def head_point(box):
    """返回检测框顶部中心 (近似头部位置)."""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, y1], dtype=np.float32)


def xyxy_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def point_to_line_distance_kb(px, py, k, b):
    """
    计算图像坐标点 (px, py) 到列直线的距离。
    列直线定义: y_up = k * x + b, 其中 y_up = -y_img。
    标准式: k*x - y_up + b = 0
    """
    y_up = -py
    return abs(k * px - y_up + b) / (math.sqrt(k * k + 1.0) + 1e-8)


def project_point_onto_line_kb(px, py, k, b):
    """
    将图像坐标点 (px, py) 投影到列直线 y_up = k*x + b 上。
    返回投影后的图像坐标 (proj_x, proj_y_img)。
    """
    y_up = -py
    # 标准式: k*x - y_up + b = 0,  法向量 (k, -1)
    denom = k * k + 1.0
    t = (k * px - y_up + b) / denom
    proj_x = px - k * t
    proj_y_up = y_up + t
    proj_y_img = -proj_y_up
    return float(proj_x), float(proj_y_img)


def l2(a, b):
    return float(np.linalg.norm(
        np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    ))


# ─── 头部投影绑定器 ──────────────────────────────────────────

class HeadProjectionBinder:
    """
    基于头部投影的人-座位绑定器。

    流程:
    1. 计算头部点到每条列直线的距离, 取最近列
    2. 将头部点投影到该列直线上
    3. 根据投影点的 y 坐标, 在该列的桌子中找最近的一个
    4. 距离检查: 若投影距离或行距离超过阈值, 视为不在座位上
    """

    def __init__(
        self,
        desks: list,
        column_lines: list,
        max_col_distance_ratio: float = 0.45,
        max_row_distance_ratio: float = 0.55,
    ):
        """
        Args:
            desks: 桌子列表, 每项含 desk_id, column_index, row_index, center, xyxy
            column_lines: 列直线列表, 每项含 column_index, line_kb, desk_ids, avg_step 等
            max_col_distance_ratio: 头部到列直线的最大距离 / avg_step, 超过则不绑定
            max_row_distance_ratio: 投影点到最近桌子中心的距离 / avg_step, 超过则不绑定
        """
        self.desks = desks
        self.column_lines = sorted(column_lines, key=lambda c: c["column_index"])
        self.max_col_dist_ratio = max_col_distance_ratio
        self.max_row_dist_ratio = max_row_distance_ratio

        # 为每列预排序桌子 (按 row_index 升序, 即按 y 从大到小 / 近到远)
        self.col_desks = {}
        for col_line in self.column_lines:
            col_idx = col_line["column_index"]
            col_desk_list = [
                d for d in self.desks if d["column_index"] == col_idx
            ]
            col_desk_list.sort(key=lambda d: d["row_index"])
            self.col_desks[col_idx] = col_desk_list

        # desk_id → desk 映射
        self.desk_lookup = {d["desk_id"]: d for d in self.desks}

    def assign_one(self, hx, hy):
        """
        给定头部点 (hx, hy) 的图像坐标, 返回最匹配的座位。

        Returns:
            dict with keys: desk_id, col_idx, row_idx, col_distance, row_distance, proj_x, proj_y
            如果无匹配返回 None
        """
        best = None

        for col_line in self.column_lines:
            col_idx = col_line["column_index"]
            k, b = col_line["line_kb"]
            avg_step = col_line["avg_step"]

            # 1. 头部点到列直线的距离
            col_dist = point_to_line_distance_kb(hx, hy, k, b)
            max_col_dist = avg_step * self.max_col_dist_ratio
            if col_dist > max_col_dist:
                continue

            # 2. 投影
            proj_x, proj_y = project_point_onto_line_kb(hx, hy, k, b)

            # 3. 在该列的桌子中找最近的 (基于图像 y 坐标距离)
            col_desk_list = self.col_desks.get(col_idx, [])
            if not col_desk_list:
                continue

            min_row_dist = float("inf")
            nearest_desk = None
            for desk in col_desk_list:
                desk_cy = desk["center"][1]
                # 使用投影点到桌子中心的欧氏距离来确定行
                row_dist = math.sqrt(
                    (proj_x - desk["center"][0]) ** 2
                    + (proj_y - desk_cy) ** 2
                )
                if row_dist < min_row_dist:
                    min_row_dist = row_dist
                    nearest_desk = desk

            if nearest_desk is None:
                continue

            max_row_dist = avg_step * self.max_row_dist_ratio
            if min_row_dist > max_row_dist:
                continue

            # 综合代价: 列距离权重 + 行距离权重
            cost = col_dist / max(1.0, avg_step) + 0.5 * min_row_dist / max(1.0, avg_step)

            if best is None or cost < best["cost"]:
                best = {
                    "desk_id": nearest_desk["desk_id"],
                    "desk_no": nearest_desk["desk_no"],
                    "col_idx": col_idx,
                    "row_idx": nearest_desk["row_index"],
                    "col_distance": round(col_dist, 2),
                    "row_distance": round(min_row_dist, 2),
                    "proj_x": round(proj_x, 2),
                    "proj_y": round(proj_y, 2),
                    "cost": round(cost, 4),
                }

        return best

    def assign_batch(self, people):
        """
        批量分配, 保证一个座位只绑定一个人 (代价最小优先)。

        Args:
            people: 列表, 每项含 track_id, xyxy

        Returns:
            assignments: {track_id: assignment_dict}
            head_points: {track_id: (hx, hy)}
        """
        candidates = []
        head_pts = {}

        for person in people:
            tid = person["track_id"]
            hp = head_point(person["xyxy"])
            head_pts[tid] = hp
            result = self.assign_one(float(hp[0]), float(hp[1]))
            if result is not None:
                candidates.append((result["cost"], tid, result))

        # 按代价排序, 贪心分配 (一桌一人)
        candidates.sort(key=lambda x: x[0])
        used_desks = set()
        used_people = set()
        assignments = {}

        for cost, tid, result in candidates:
            if tid in used_people or result["desk_id"] in used_desks:
                continue
            assignments[tid] = result
            used_desks.add(result["desk_id"])
            used_people.add(tid)

        return assignments, head_pts


# ─── 时间确认器 ──────────────────────────────────────────────

class TemporalConfirmer:
    """
    基于时间窗口的绑定确认。

    规则:
    - 维护每个 track_id 最近 window_seconds 内的帧级座位分配
    - 如果同一座位在窗口内出现比例 >= confirm_ratio, 则确认绑定
    - 确认后即使短暂丢失, 在 hold_seconds 内仍保持绑定
    - 超过 hold_seconds 无检测则释放绑定
    """

    def __init__(
        self,
        fps: float,
        confirm_seconds: float = 10.0,
        confirm_ratio: float = 0.65,
        hold_seconds: float = 5.0,
    ):
        self.fps = max(1.0, fps)
        self.confirm_frames = int(confirm_seconds * self.fps)
        self.confirm_ratio = confirm_ratio
        self.hold_frames = int(hold_seconds * self.fps)

        # track_id -> state dict
        self.states = {}

    def _get_state(self, track_id):
        if track_id not in self.states:
            self.states[track_id] = {
                "history": deque(maxlen=self.confirm_frames),
                "confirmed_desk_id": None,
                "last_seen_frame": -1,
                "total_move": 0.0,
                "last_center": None,
            }
        return self.states[track_id]

    def update(self, track_id, frame_idx, desk_id, box):
        """
        每帧调用, 更新 track_id 的分配状态。

        Args:
            track_id: 跟踪 ID
            frame_idx: 当前帧号
            desk_id: 本帧分配结果 (str 或 None)
            box: 检测框 xyxy
        """
        st = self._get_state(track_id)
        st["history"].append(desk_id)
        st["last_seen_frame"] = frame_idx

        center = xyxy_center(box)
        if st["last_center"] is not None:
            st["total_move"] += l2(center, st["last_center"])
        st["last_center"] = center

        # 检查确认条件
        history = st["history"]
        if len(history) >= min(self.confirm_frames, int(self.fps * 3)):
            # 统计非 None 的投票
            votes = [d for d in history if d is not None]
            if votes:
                most_common_desk, count = Counter(votes).most_common(1)[0]
                ratio = count / len(history)
                if ratio >= self.confirm_ratio:
                    st["confirmed_desk_id"] = most_common_desk

    def get_confirmed(self, track_id, current_frame):
        """返回确认的 desk_id, 考虑 hold 逻辑。"""
        if track_id not in self.states:
            return None

        st = self.states[track_id]
        if st["confirmed_desk_id"] is None:
            return None

        # 如果人已经消失超过 hold_frames, 释放
        if current_frame - st["last_seen_frame"] > self.hold_frames:
            st["confirmed_desk_id"] = None
            return None

        return st["confirmed_desk_id"]

    def get_current_vote(self, track_id):
        """返回当前窗口内多数投票结果 (未经确认)."""
        if track_id not in self.states:
            return None
        history = self.states[track_id]["history"]
        votes = [d for d in history if d is not None]
        if not votes:
            return None
        return Counter(votes).most_common(1)[0][0]

    def get_display_desk_id(self, track_id, current_frame):
        """返回用于显示的 desk_id: 优先确认, 其次投票结果。"""
        confirmed = self.get_confirmed(track_id, current_frame)
        if confirmed is not None:
            return confirmed
        return self.get_current_vote(track_id)

    def is_confirmed(self, track_id, current_frame):
        return self.get_confirmed(track_id, current_frame) is not None

    def role(self, track_id, current_frame, min_observe_frames=None):
        """
        判断角色:
        - 有确认绑定 → student
        - 观察时间不足 → unknown
        - 观察足够仍无绑定 → teacher
        """
        if min_observe_frames is None:
            min_observe_frames = self.confirm_frames

        if self.is_confirmed(track_id, current_frame):
            return "student"

        if track_id not in self.states:
            return "unknown"

        st = self.states[track_id]
        if len(st["history"]) >= min_observe_frames:
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
    """在图像上画列直线"""
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


# ─── 主管线 V2 ───────────────────────────────────────────────

class PersonDeskBindingPipelineV2:
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
        max_col_distance_ratio: float,
        max_row_distance_ratio: float,
        outputs: set[str],
        display: bool,
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
        self.max_col_distance_ratio = max_col_distance_ratio
        self.max_row_distance_ratio = max_row_distance_ratio
        self.outputs = set(outputs)
        self.display = display

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
                    col_lines = result["layout"].get("column_lines", [])
                    score = sum(d["conf"] for d in desks)
                    if desks:
                        if (
                            best is None
                            or len(desks) > best["desk_count"]
                            or (len(desks) == best["desk_count"] and score > best["score"])
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

    def _draw_desks(self, frame, desks):
        annotated = frame.copy()
        for desk in desks:
            x1, y1, x2, y2 = map(int, desk["xyxy"])
            color = color_for_desk(desk["desk_no"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            draw_text(annotated, desk["desk_id"],
                      (x1, max(20, y1 - 6)), color, scale=0.55)
        return annotated

    def _draw_person(self, frame, person, hp, role, display_desk_id,
                     is_confirmed, assignment, desk_lookup):
        x1, y1, x2, y2 = map(int, person["xyxy"])

        # 颜色: 确认学生=绿, 未确认候选=黄, 教师=红, 未知=灰
        if is_confirmed:
            color = (0, 255, 0)
        elif display_desk_id is not None:
            color = (0, 255, 255)
        elif role == "teacher":
            color = (0, 0, 255)
        else:
            color = (180, 180, 180)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 画头部点
        hx, hy = int(hp[0]), int(hp[1])
        cv2.circle(frame, (hx, hy), 5, (0, 255, 255), -1)

        # 画投影点和连线
        if assignment is not None:
            proj_x, proj_y = int(assignment["proj_x"]), int(assignment["proj_y"])
            cv2.circle(frame, (proj_x, proj_y), 4, (255, 0, 255), -1)
            cv2.line(frame, (hx, hy), (proj_x, proj_y), (255, 0, 255), 1)

        # 画到绑定桌子的连线
        if display_desk_id and display_desk_id in desk_lookup:
            desk_center = desk_lookup[display_desk_id]["center"]
            dcx, dcy = int(desk_center[0]), int(desk_center[1])
            cv2.line(frame, (hx, hy), (dcx, dcy), color, 2)

        # 标签
        tid = person["track_id"]
        status = "✓" if is_confirmed else "?"
        label1 = f"ID{tid} {role}"
        label2 = f"{display_desk_id or '-'}{status}"
        draw_text(frame, label1, (x1, max(20, y1 - 24)), color, scale=0.55)
        draw_text(frame, label2, (x1, max(40, y1 - 4)), color, scale=0.50)

    # ── 主运行 ───────────────────────────────────────────────

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        video_name = Path(self.source).stem

        output_video_path = os.path.join(self.output_dir, f"bound_v2_{video_name}.mp4")
        output_csv_path = os.path.join(self.output_dir, f"binding_v2_{video_name}.csv")
        output_json_path = os.path.join(self.output_dir, f"binding_v2_{video_name}.json")
        output_desk_map_path = os.path.join(self.output_dir, f"desk_layout_v2_{video_name}.jpg")

        # ── 桌子检测 ─────────────────────────────────────────
        print("正在建立桌子布局...")
        reference = self._select_reference_desks()
        layout = reference["layout"]
        desks = layout["desks"]
        column_lines = layout.get("column_lines", [])
        desk_lookup = {d["desk_id"]: d for d in desks}

        if not column_lines:
            raise RuntimeError("未能获取列直线信息，请检查桌子检测结果。")

        print(f"桌子参考帧: {reference['frame_idx']} | 检测到桌子 {len(desks)} 个 | 列直线 {len(column_lines)} 条")

        if "desk_map" in self.outputs:
            desk_frame = self._draw_desks(reference["frame"], desks)
            desk_frame = draw_column_lines(desk_frame, column_lines, reference["frame"].shape[0])
            cv2.imwrite(output_desk_map_path, desk_frame)
            print(f"座位图: {output_desk_map_path}")

        # ── 构建绑定器 ───────────────────────────────────────
        binder = HeadProjectionBinder(
            desks=desks,
            column_lines=column_lines,
            max_col_distance_ratio=self.max_col_distance_ratio,
            max_row_distance_ratio=self.max_row_distance_ratio,
        )

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

        # ── 时间确认器 ───────────────────────────────────────
        confirmer = TemporalConfirmer(
            fps=fps,
            confirm_seconds=self.confirm_seconds,
            confirm_ratio=self.confirm_ratio,
            hold_seconds=self.hold_seconds,
        )

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
            csv_file = open(output_csv_path, "w", newline="", encoding="utf-8-sig")
            csv_writer = csv.DictWriter(csv_file, fieldnames=[
                "frame_idx", "track_id", "role", "is_confirmed", "person_conf",
                "desk_id_current", "desk_id_confirmed", "desk_id_display",
                "binding_cost", "col_distance", "row_distance",
                "head_x", "head_y", "proj_x", "proj_y",
                "x1", "y1", "x2", "y2", "total_move",
            ])
            csv_writer.writeheader()

        # ── 逐帧处理 ─────────────────────────────────────────
        frame_idx = 0
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

                # 头部投影绑定
                assignments, head_pts = binder.assign_batch(people)

                # 更新时间确认器
                for person in people:
                    tid = person["track_id"]
                    desk_id = assignments[tid]["desk_id"] if tid in assignments else None
                    confirmer.update(tid, frame_idx, desk_id, person["xyxy"])

                # 绘制
                annotated = None
                if video_writer is not None or self.display:
                    annotated = self._draw_desks(frame, desks)
                    annotated = draw_column_lines(annotated, column_lines, height)

                student_count = 0
                teacher_count = 0
                unknown_count = 0

                for person in people:
                    tid = person["track_id"]
                    hp = head_pts[tid]
                    assignment = assignments.get(tid)
                    display_desk = confirmer.get_display_desk_id(tid, frame_idx)
                    is_conf = confirmer.is_confirmed(tid, frame_idx)
                    r = confirmer.role(tid, frame_idx)

                    if r == "student":
                        student_count += 1
                    elif r == "teacher":
                        teacher_count += 1
                    else:
                        unknown_count += 1

                    if annotated is not None:
                        self._draw_person(
                            annotated, person, hp, r, display_desk,
                            is_conf, assignment, desk_lookup,
                        )

                    if csv_writer is not None:
                        st = confirmer.states.get(tid, {})
                        csv_writer.writerow({
                            "frame_idx": frame_idx,
                            "track_id": tid,
                            "role": r,
                            "is_confirmed": is_conf,
                            "person_conf": round(person["conf"], 4),
                            "desk_id_current": assignment["desk_id"] if assignment else None,
                            "desk_id_confirmed": confirmer.get_confirmed(tid, frame_idx),
                            "desk_id_display": display_desk,
                            "binding_cost": assignment["cost"] if assignment else None,
                            "col_distance": assignment["col_distance"] if assignment else None,
                            "row_distance": assignment["row_distance"] if assignment else None,
                            "head_x": round(float(hp[0]), 2),
                            "head_y": round(float(hp[1]), 2),
                            "proj_x": assignment["proj_x"] if assignment else None,
                            "proj_y": assignment["proj_y"] if assignment else None,
                            "x1": round(person["xyxy"][0], 2),
                            "y1": round(person["xyxy"][1], 2),
                            "x2": round(person["xyxy"][2], 2),
                            "y2": round(person["xyxy"][3], 2),
                            "total_move": round(st.get("total_move", 0.0), 2),
                        })

                info = (
                    f"Frame {frame_idx}/{total_frames if total_frames > 0 else '?'} | "
                    f"Students {student_count} | Teachers {teacher_count} | ? {unknown_count}"
                )
                if annotated is not None:
                    draw_text(annotated, info, (10, 28), (255, 255, 255), scale=0.8)
                    if video_writer is not None:
                        video_writer.write(annotated)

                if self.display and annotated is not None:
                    cv2.imshow("Person-Desk Binding V2", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("用户中断")
                        break

                if frame_idx % 30 == 0 and total_frames > 0:
                    progress = 100.0 * frame_idx / total_frames
                    print(f"进度: {frame_idx}/{total_frames} ({progress:.1f}%) | "
                          f"S={student_count} T={teacher_count}")

                frame_idx += 1

        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if csv_file is not None:
                csv_file.close()
            if self.display:
                cv2.destroyAllWindows()

        # ── 输出 JSON ────────────────────────────────────────
        summary = {
            "source": self.source,
            "version": "v2_head_projection",
            "reference_frame_idx": reference["frame_idx"],
            "desk_count": len(desks),
            "config": {
                "confirm_seconds": self.confirm_seconds,
                "confirm_ratio": self.confirm_ratio,
                "hold_seconds": self.hold_seconds,
                "max_col_distance_ratio": self.max_col_distance_ratio,
                "max_row_distance_ratio": self.max_row_distance_ratio,
                "binding_point": "head_top_center",
            },
            "desks": desks,
            "column_lines": column_lines,
            "tracks": [
                confirmer.summary(tid, frame_idx)
                for tid in sorted(confirmer.states.keys())
            ],
            "saved_outputs": {},
        }

        if "binding_video" in self.outputs:
            summary["saved_outputs"]["binding_video"] = output_video_path
        if "desk_map" in self.outputs:
            summary["saved_outputs"]["desk_map"] = output_desk_map_path
        if "csv" in self.outputs:
            summary["saved_outputs"]["csv"] = output_csv_path

        if "json" in self.outputs:
            output_json_dir = os.path.dirname(output_json_path)
            if output_json_dir:
                os.makedirs(output_json_dir, exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            summary["saved_outputs"]["json"] = output_json_path

        for name, path in summary["saved_outputs"].items():
            print(f"{name}: {path}")


# ─── CLI ─────────────────────────────────────────────────────

def parse_classes(s: str | None):
    if not s:
        return None
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def parse_outputs(raw: str | None):
    valid = {"binding_video", "desk_map", "csv", "json"}
    if not raw:
        return {"binding_video", "desk_map", "csv", "json"}
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    if "all" in parts:
        return valid
    unknown = parts - valid
    if unknown:
        raise ValueError(f"未知输出: {', '.join(unknown)}。可选: {', '.join(sorted(valid))}")
    return parts


def main():
    p = argparse.ArgumentParser("人桌绑定 V2 (头部投影 + 列直线)")
    p.add_argument("--source", required=True, help="输入视频路径")
    p.add_argument("--weights", default="exam_seat_binding/weight/yolo11speopel.pt",
                   help="人物模型权重")
    p.add_argument("--desk-weights", default="exam_seat_binding/weight/yolo11desk.pt",
                   help="桌子模型权重")
    p.add_argument("--output", default="exam_seat_binding/output", help="输出目录")
    p.add_argument("--outputs", default="binding_video,desk_map,csv,json",
                   help="输出内容: binding_video,desk_map,csv,json 或 all")

    # 检测参数
    p.add_argument("--conf", type=float, default=0.18, help="人物检测置信度")
    p.add_argument("--iou", type=float, default=0.60, help="人物 NMS IOU")
    p.add_argument("--desk-conf", type=float, default=0.7, help="桌子检测置信度")
    p.add_argument("--desk-iou", type=float, default=0.45, help="桌子检测 NMS IOU")
    p.add_argument("--device", default="", help="推理设备 (cpu/0/1)")
    p.add_argument("--img-size", type=int, default=None, help="推理尺寸")
    p.add_argument("--half", action="store_true", help="FP16 推理")
    p.add_argument("--tracker", default="bytetrack.yaml", help="跟踪器配置")
    p.add_argument("--classes", default=None, help="人物类别过滤")
    p.add_argument("--display", action="store_true", help="实时预览")

    # 桌子布局参数
    p.add_argument("--desk-mode", default="auto",
                   choices=["normal", "scheme1", "scheme2", "auto"])
    p.add_argument("--desk-num-cols", type=int, default=5)
    p.add_argument("--desk-required-per-col", type=int, default=6)
    p.add_argument("--reference-max-frames", type=int, default=120)
    p.add_argument("--reference-sample-step", type=int, default=5)

    # V2 核心参数: 时间确认
    p.add_argument("--confirm-seconds", type=float, default=10.0,
                   help="确认绑定所需稳定停留时间 (秒)")
    p.add_argument("--confirm-ratio", type=float, default=0.65,
                   help="窗口内同一座位出现比例阈值")
    p.add_argument("--hold-seconds", type=float, default=5.0,
                   help="人消失后保持绑定的时间 (秒)")

    # V2 核心参数: 投影距离
    p.add_argument("--max-col-dist-ratio", type=float, default=0.45,
                   help="头部到列直线最大距离 / avg_step")
    p.add_argument("--max-row-dist-ratio", type=float, default=0.55,
                   help="投影点到最近桌子最大距离 / avg_step")

    args = p.parse_args()

    try:
        outputs = parse_outputs(args.outputs)
    except ValueError as exc:
        p.error(str(exc))

    pipeline = PersonDeskBindingPipelineV2(
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
        max_col_distance_ratio=args.max_col_dist_ratio,
        max_row_distance_ratio=args.max_row_dist_ratio,
        outputs=outputs,
        display=args.display,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
