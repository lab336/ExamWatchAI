"""
桌子布局检测器。

职责：
1. 使用 YOLO 检测桌子
2. 通过两种分列方案完成桌子编号
3. 支持 normal / scheme1 / scheme2 / auto 四种模式

示例：
1. 普通检测:
   python exam_seat_binding/desk_layout_detector.py --source data/1.10/clip/clip_desk.mp4 --weights exam_seat_binding/weight/yolo11desk.pt --mode normal
2. 分列方案1:
   python exam_seat_binding/desk_layout_detector.py --source data/1.10/clipleft/clipped_testdata1.mp4 --weights exam_seat_binding/weight/yolo11desk.pt --mode scheme1   --output output/video
3. 分列方案2:
   python exam_seat_binding/desk_layout_detector.py --source data/1.10/clip/clip_desk.mp4 --weights exam_seat_binding/weight/yolo11desk.pt --mode scheme2
4. 自动选择方案:
   python exam_seat_binding/desk_layout_detector.py --source data/1.10/clip/clip_desk.mp4 --weights exam_seat_binding/weight/yolo11desk.pt --mode auto
"""

import argparse
import itertools
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def box_xyxy(box):
    return tuple(map(float, box.xyxy[0].tolist()))


def box_center(box):
    x1, y1, x2, y2 = box_xyxy(box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-8)


def order_column_indices(boxes, columns, style=1):
    ordered_columns = []
    boxes_list = list(boxes)

    for idxs in columns:
        centers_list = []
        for idx in idxs:
            cx, cy = box_center(boxes_list[idx])
            centers_list.append((idx, cx, cy))

        if style == 1:
            sorted_pairs = sorted(centers_list, key=lambda p: p[2], reverse=True)
        else:
            sorted_pairs = sorted(centers_list, key=lambda p: (p[2], p[1]), reverse=True)
        ordered_columns.append([p[0] for p in sorted_pairs])

    return ordered_columns


def build_layout_entries(boxes, columns, style=1, num_per_col=6):
    boxes_list = list(boxes)
    ordered_columns = order_column_indices(boxes_list, columns, style=style)
    entries = []

    for col_id, sorted_idxs in enumerate(ordered_columns):
        for row_id, idx in enumerate(sorted_idxs):
            box = boxes_list[idx]
            x1, y1, x2, y2 = box_xyxy(box)
            cx, cy = box_center(box)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            desk_no = col_id * num_per_col + (row_id + 1)
            entries.append(
                {
                    "index": idx,
                    "desk_no": desk_no,
                    "desk_id": f"D{desk_no:02d}",
                    "column_index": col_id,
                    "row_index": row_id,
                    "xyxy": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "conf": conf,
                    "cls": cls,
                }
            )

    entries.sort(key=lambda item: item["desk_no"])
    return ordered_columns, entries


def build_column_line_entries(boxes, ordered_columns, layout_entries):
    boxes_list = list(boxes)
    column_lines = []

    for col_id, sorted_idxs in enumerate(ordered_columns):
        if not sorted_idxs:
            continue

        points = []
        widths = []
        heights = []
        ordered_centers = []
        for idx in sorted_idxs:
            box = boxes_list[idx]
            x1, y1, x2, y2 = box_xyxy(box)
            center = list(box_center(box))
            points.append(center)
            ordered_centers.append(center)
            widths.append(x2 - x1)
            heights.append(y2 - y1)

        points_np = np.asarray(points, dtype=np.float32)
        line_kb = vd1.fit_line_kb_positive(points_np, min_k=0.02)
        column_desks = sorted(
            [item for item in layout_entries if item["column_index"] == col_id],
            key=lambda item: item["row_index"],
        )
        step_lengths = []
        for point_a, point_b in zip(ordered_centers[:-1], ordered_centers[1:]):
            step_lengths.append(float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b))))
        avg_step = float(np.mean(step_lengths)) if step_lengths else float(np.mean(heights))

        column_lines.append(
            {
                "column_index": col_id,
                "line_kb": [float(line_kb[0]), float(line_kb[1])],
                "x_min": float(np.min(points_np[:, 0])),
                "x_max": float(np.max(points_np[:, 0])),
                "y_min": float(np.min(points_np[:, 1])),
                "y_max": float(np.max(points_np[:, 1])),
                "avg_box_width": float(np.mean(widths)),
                "avg_box_height": float(np.mean(heights)),
                "avg_step": avg_step,
                "segment_start": [float(ordered_centers[0][0]), float(ordered_centers[0][1])],
                "segment_end": [float(ordered_centers[-1][0]), float(ordered_centers[-1][1])],
                "desk_ids": [item["desk_id"] for item in column_desks],
            }
        )

    return column_lines


def _fit_line_from_centers(points):
    if len(points) == 0:
        return (0.02, 0.0)
    return vd1.fit_line_kb_positive(np.asarray(points, dtype=np.float32), min_k=0.02)


def _x_on_line_at_y(line_kb, y, fallback_x):
    k, b = line_kb
    if abs(k) < 1e-6:
        return float(fallback_x)
    return float((-float(y) - b) / k)


def _line_crosses_box(line_kb, xyxy, margin=0.0):
    k, b = line_kb
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 -= margin
    y1 -= margin
    x2 += margin
    y2 += margin
    xs = np.linspace(x1, x2, num=5, dtype=np.float32)
    for x in xs:
        y = -float(k) * float(x) - float(b)
        if y1 <= y <= y2:
            return True
    return False


def _three_point_angle_deg(prev_point, mid_point, next_point):
    a = np.asarray(prev_point, dtype=np.float32) - np.asarray(mid_point, dtype=np.float32)
    b = np.asarray(next_point, dtype=np.float32) - np.asarray(mid_point, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 180.0
    cos_value = float(np.dot(a, b) / denom)
    cos_value = max(-1.0, min(1.0, cos_value))
    return float(np.degrees(np.arccos(cos_value)))


def column_angle_stats_from_desks(desks):
    ordered = sorted(desks, key=lambda item: item["row_index"])
    if len(ordered) < 3:
        return {"angles": [], "min_angle": None, "mean_angle": None, "sharp_turns": 0}

    angles = []
    for prev, mid, nxt in zip(ordered[:-2], ordered[1:-1], ordered[2:]):
        angles.append(_three_point_angle_deg(prev["center"], mid["center"], nxt["center"]))

    return {
        "angles": [float(v) for v in angles],
        "min_angle": float(np.min(angles)),
        "mean_angle": float(np.mean(angles)),
        "sharp_turns": int(sum(1 for v in angles if v < 170.0)),
    }


def build_column_line_entries_from_desks(desks, num_cols):
    column_lines = []
    for col_id in range(num_cols):
        column_desks = sorted(
            [item for item in desks if item["column_index"] == col_id],
            key=lambda item: item["row_index"],
        )
        if not column_desks:
            continue

        centers = [item["center"] for item in column_desks]
        widths = [item["xyxy"][2] - item["xyxy"][0] for item in column_desks]
        heights = [item["xyxy"][3] - item["xyxy"][1] for item in column_desks]
        points_np = np.asarray(centers, dtype=np.float32)
        line_kb = _fit_line_from_centers(points_np)
        step_lengths = [
            float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
            for a, b in zip(centers[:-1], centers[1:])
        ]
        avg_step = float(np.mean(step_lengths)) if step_lengths else float(np.mean(heights))

        column_lines.append(
            {
                "column_index": col_id,
                "line_kb": [float(line_kb[0]), float(line_kb[1])],
                "x_min": float(np.min(points_np[:, 0])),
                "x_max": float(np.max(points_np[:, 0])),
                "y_min": float(np.min(points_np[:, 1])),
                "y_max": float(np.max(points_np[:, 1])),
                "avg_box_width": float(np.mean(widths)),
                "avg_box_height": float(np.mean(heights)),
                "avg_step": avg_step,
                "segment_start": [float(centers[0][0]), float(centers[0][1])],
                "segment_end": [float(centers[-1][0]), float(centers[-1][1])],
                "desk_ids": [item["desk_id"] for item in column_desks],
            }
        )

    return column_lines


def _load_module_from_file(name: str, filename: str):
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    from exam_seat_binding import desk_layout_scheme1 as vd1
    from exam_seat_binding import desk_layout_scheme2 as vd2
except Exception:
    vd1 = _load_module_from_file("desk_layout_scheme1", "desk_layout_scheme1.py")
    vd2 = _load_module_from_file("desk_layout_scheme2", "desk_layout_scheme2.py")


def avg_point_line_distance(centers: np.ndarray, columns, lines) -> float:
    if len(centers) == 0:
        return float("inf")
    total = 0.0
    cnt = 0
    for c, col in enumerate(columns):
        if len(col) == 0:
            continue
        line = lines[c] if c < len(lines) else None
        if line is None:
            continue
        for idx in col:
            total += vd1.point_line_distance_kb(centers[idx], line)
            cnt += 1
    return total / max(1, cnt)


def fitted_columns_straightness_score(
    centers: np.ndarray,
    columns,
    required_per_col: int = 6,
) -> tuple:
    """评估“拟合列直线度”。

    规则：每列取前 required_per_col 个点拟合直线，再计算这些点到拟合线的平均距离。
    返回 (score, fitted_lines)。score 越小表示越直；无法完成所有列拟合则返回 inf。
    """
    if len(columns) == 0:
        return float("inf"), []

    fitted_lines = []
    per_col_scores = []
    for col in columns:
        if len(col) < required_per_col:
            return float("inf"), []
        idxs = np.array(col[:required_per_col], dtype=np.int32)
        pts = centers[idxs]
        line_kb = vd1.fit_line_kb_positive(pts, min_k=0.02)
        fitted_lines.append(line_kb)
        dists = [vd1.point_line_distance_kb(centers[i], line_kb) for i in idxs]
        per_col_scores.append(float(np.mean(dists)))

    return float(np.mean(per_col_scores)), fitted_lines


def _scheme_name(scheme: int) -> str:
    return "scheme1" if int(scheme) == 1 else "scheme2"


def draw_normal(img, boxes, model_names):
    annotated = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if isinstance(model_names, dict):
            class_name = model_names.get(cls, str(cls))
        elif isinstance(model_names, (list, tuple)) and 0 <= cls < len(model_names):
            class_name = str(model_names[cls])
        else:
            class_name = str(cls)

        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return annotated


def draw_layout(img, boxes, columns, style=1, num_per_col=6):
    annotated = img.copy()
    col_colors = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 80, 255),
        (255, 200, 80),
        (220, 80, 255),
    ]
    boxes_list = list(boxes)
    ordered_columns, layout_entries = build_layout_entries(
        boxes_list,
        columns,
        style=style,
        num_per_col=num_per_col,
    )

    for box in boxes_list:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (120, 120, 120), 1)

    for col_id, sorted_idxs in enumerate(ordered_columns):
        if len(sorted_idxs) == 0:
            continue
        color = col_colors[col_id % len(col_colors)]

        col_entries = [item for item in layout_entries if item["column_index"] == col_id]
        for item in col_entries:
            box = boxes_list[item["index"]]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = item["conf"]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{item['desk_no']}: {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        for j in range(len(sorted_idxs) - 1):
            idx_a = sorted_idxs[j]
            idx_b = sorted_idxs[j + 1]
            box_a = boxes_list[idx_a]
            box_b = boxes_list[idx_b]
            ax1, ay1, ax2, ay2 = map(int, box_a.xyxy[0].tolist())
            bx1, by1, bx2, by2 = map(int, box_b.xyxy[0].tolist())

            if style == 1:
                cv2.line(annotated, (ax1, ay1), (bx1, by1), color, 2)
                cv2.line(annotated, (ax2, ay2), (bx2, by2), color, 2)
            else:
                cv2.line(annotated, (ax1, ay2), (bx1, by2), color, 2)
                cv2.line(annotated, (ax2, ay1), (bx2, by1), color, 2)

        top_idx = sorted_idxs[-1]
        box_top = boxes_list[top_idx]
        x1, y1, x2, y2 = map(int, box_top.xyxy[0].tolist())
        tx = (x1 + x2) // 2
        ty = y1
        label = f"C{col_id + 1}({len(sorted_idxs)})"
        cv2.putText(annotated, label, (tx + 8, max(20, ty - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return annotated


def draw_layout_entries(img, desks, num_cols=5, required_per_col=6, title="Reliable seat layout"):
    annotated = img.copy()
    col_colors = [
        (255, 80, 80),
        (80, 210, 80),
        (80, 120, 255),
        (255, 200, 80),
        (220, 80, 255),
    ]

    for desk in sorted(desks, key=lambda d: d["desk_no"]):
        x1, y1, x2, y2 = [int(round(v)) for v in desk["xyxy"]]
        col_id = int(desk["column_index"])
        color = col_colors[col_id % len(col_colors)]
        is_virtual = bool(desk.get("is_virtual", False))
        thickness = 2 if not is_virtual else 1
        line_type = cv2.LINE_AA

        if is_virtual:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, line_type)
            cv2.line(annotated, (x1, y1), (x2, y2), color, 1, line_type)
            cv2.line(annotated, (x1, y2), (x2, y1), color, 1, line_type)
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, line_type)

        label = f"{desk['desk_id']} {float(desk.get('conf', 0.0)):.2f}"
        if is_virtual:
            label += " inferred"
        elif desk.get("is_low_conf_recovered", False):
            label += " recovered"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_y = max(label_h + baseline + 4, y1)
        cv2.rectangle(annotated, (x1, label_y - label_h - baseline - 4), (x1 + label_w + 4, label_y), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    for col_id in range(num_cols):
        col_desks = sorted(
            [d for d in desks if int(d["column_index"]) == col_id],
            key=lambda d: int(d["row_index"]),
        )
        color = col_colors[col_id % len(col_colors)]
        for prev, cur in zip(col_desks[:-1], col_desks[1:]):
            p1 = tuple(int(round(v)) for v in prev["center"])
            p2 = tuple(int(round(v)) for v in cur["center"])
            cv2.line(annotated, p1, p2, color, 2, cv2.LINE_AA)
        if col_desks:
            top = col_desks[-1]
            tx, ty = [int(round(v)) for v in top["center"]]
            locked = "locked" if all(not d.get("is_virtual", False) for d in col_desks) else "repaired"
            cv2.putText(
                annotated,
                f"C{col_id + 1} {len(col_desks)}/{required_per_col} {locked}",
                (tx + 8, max(24, ty - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    cv2.rectangle(annotated, (8, 8), (500, 42), (0, 0, 0), -1)
    cv2.putText(annotated, title, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def draw_sequence_best_frame(img, layout, title="Best sampled frame layout"):
    desks = [item for item in layout.get("desks", []) if not item.get("is_virtual", False)]
    annotated = draw_layout_entries(
        img,
        desks,
        num_cols=max(1, len(layout.get("columns", [])) or 5),
        required_per_col=6,
        title=title,
    )
    metrics = layout.get("layout_metrics", {})
    repair = layout.get("layout_repair", {})
    lines = [
        f"mode={layout.get('chosen_mode', 'unknown')}",
        f"actual={len(desks)}",
    ]
    if metrics:
        lines.append(f"score={float(metrics.get('score', 0.0)):.1f}")
        lines.append(f"cols={metrics.get('column_counts')}")
    if repair:
        lines.append(f"low_conf={repair.get('sequence_low_conf_recovered_desks', repair.get('low_conf_recovered_desks', 0))}")
        lines.append(f"virtual={repair.get('virtual_desks', 0)}")

    y = 62
    for text in lines:
        cv2.putText(annotated, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 24
    return annotated


class DeskLayoutDetector:
    def __init__(
        self,
        weights_path,
        conf_threshold=0.7,
        iou_threshold=0.45,
        device="",
        img_size=None,
        half=False,
        mode="auto",
        num_cols=5,
        required_per_col=6,
        repair_conf_threshold=None,
    ):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

        print(f"正在加载模型: {weights_path}")
        if self._check_cuda():
            self._clear_cuda_cache()

        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device if device else ("cuda" if self._check_cuda() else "cpu")
        self.img_size = img_size
        self.half = half and "cuda" in str(self.device)
        self.mode = mode
        self.num_cols = num_cols
        self.required_per_col = required_per_col
        self.repair_conf_threshold = repair_conf_threshold

        if "cuda" in str(self.device):
            self._clear_cuda_cache()

        print("模型加载成功!")
        print(f"使用设备: {self.device}")
        print(f"运行模式: {self.mode}")
        if self.img_size:
            print(f"推理图像尺寸: {self.img_size}")
        if self.half:
            print("使用半精度(FP16)推理")
        if "cuda" in str(self.device):
            self._print_gpu_info()

    def _check_cuda(self):
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def _clear_cuda_cache(self):
        try:
            import gc
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                print("已清理CUDA缓存")
        except Exception:
            pass

    def _print_gpu_info(self):
        try:
            import torch

            if torch.cuda.is_available():
                gpu_id = 0 if self.device == "cuda" else int(self.device)
                gpu_name = torch.cuda.get_device_name(gpu_id)
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                cached = torch.cuda.memory_reserved(gpu_id) / 1024**3
                print(f"GPU: {gpu_name}")
                print(f"总显存: {total_memory:.2f} GB")
                print(f"已分配: {allocated:.2f} GB")
                print(f"已缓存: {cached:.2f} GB")
        except Exception:
            pass

    def _predict(self, source):
        return self._predict_with_conf(source, self.conf_threshold)

    def _predict_with_conf(self, source, conf_threshold):
        return self.model.predict(
            source=source,
            conf=conf_threshold,
            iou=self.iou_threshold,
            save=False,
            verbose=False,
            device=self.device,
            imgsz=self.img_size or 640,
            half=self.half,
        )

    def _effective_repair_conf_threshold(self):
        if self.repair_conf_threshold is not None:
            return float(self.repair_conf_threshold)
        return max(0.15, min(0.45, float(self.conf_threshold) - 0.30))

    def _extract_centers(self, boxes):
        centers = []
        for box in boxes:
            centers.append(list(box_center(box)))
        return np.array(centers, dtype=np.float32)

    def _split_columns_for_scheme(self, centers_np: np.ndarray, scheme: int):
        if int(scheme) == 1:
            return vd1.split_into_columns_by_origin_walk(
                centers_np,
                num_cols=self.num_cols,
                expected_per_col=self.required_per_col,
            )
        return vd2.split_into_columns_by_origin_walk(
            centers_np,
            num_cols=self.num_cols,
            expected_per_col=self.required_per_col,
        )

    def _build_layout_for_scheme(self, boxes, scheme: int):
        centers_np = self._extract_centers(boxes)
        columns, lines, _ = self._split_columns_for_scheme(centers_np, scheme)
        style = 1 if int(scheme) == 1 else 2
        ordered_columns, desks = build_layout_entries(
            boxes,
            columns,
            style=style,
            num_per_col=self.required_per_col,
        )
        column_lines = build_column_line_entries(boxes, ordered_columns, desks)
        return {
            "chosen_scheme": int(scheme),
            "chosen_mode": _scheme_name(scheme),
            "style": style,
            "columns": columns,
            "ordered_columns": ordered_columns,
            "column_lines": column_lines,
            "desks": desks,
            "_raw_lines": lines,
        }

    def _pick_best_column_six(self, desks):
        ordered = sorted(desks, key=lambda item: item["row_index"])
        if len(ordered) <= self.required_per_col:
            return self._prune_sharp_turns_in_column(ordered)

        best = None
        candidates = itertools.combinations(ordered, self.required_per_col) if len(ordered) <= 12 else []
        if not candidates:
            candidates = [ordered[i : i + self.required_per_col] for i in range(0, len(ordered) - self.required_per_col + 1)]

        for group in candidates:
            centers = np.asarray([item["center"] for item in group], dtype=np.float32)
            line = _fit_line_from_centers(centers)
            straight = float(np.mean([vd1.point_line_distance_kb(pt, line) for pt in centers]))
            mean_conf = float(np.mean([float(item.get("conf", 0.0)) for item in group]))
            row_span = max(item["row_index"] for item in group) - min(item["row_index"] for item in group)
            angle_stats = column_angle_stats_from_desks(group)
            min_angle = angle_stats["min_angle"] if angle_stats["min_angle"] is not None else 180.0
            angle_penalty = max(0.0, 170.0 - min_angle) * 10.0 + angle_stats["sharp_turns"] * 160.0
            score = straight - mean_conf * 12.0 + abs(row_span - (self.required_per_col - 1)) * 1.5 + angle_penalty
            if best is None or score < best[0]:
                best = (score, list(group))

        selected = best[1] if best else ordered[: self.required_per_col]
        return self._prune_sharp_turns_in_column(sorted(selected, key=lambda item: item["row_index"]))

    def _prune_sharp_turns_in_column(self, desks, min_keep=2, angle_threshold=160.0):
        selected = sorted(desks, key=lambda item: item["row_index"])
        removed = []
        while len(selected) > min_keep:
            best_bad = None
            for i in range(1, len(selected) - 1):
                angle = _three_point_angle_deg(
                    selected[i - 1]["center"],
                    selected[i]["center"],
                    selected[i + 1]["center"],
                )
                if angle >= angle_threshold:
                    continue
                if best_bad is None or angle < best_bad[0]:
                    best_bad = (angle, i)

            if best_bad is None:
                break

            _, mid_idx = best_bad
            candidates = sorted(set([mid_idx - 1, mid_idx, mid_idx + 1]))
            best_remove = None
            for idx in candidates:
                trial = [item for j, item in enumerate(selected) if j != idx]
                stats = column_angle_stats_from_desks(trial)
                min_angle = stats["min_angle"] if stats["min_angle"] is not None else 180.0
                centers = np.asarray([item["center"] for item in trial], dtype=np.float32)
                line = _fit_line_from_centers(centers)
                straight = float(np.mean([vd1.point_line_distance_kb(pt, line) for pt in centers]))
                conf = float(selected[idx].get("conf", 0.0))
                score = max(0.0, angle_threshold - min_angle) * 8.0 + stats["sharp_turns"] * 100.0 + straight - conf * 10.0
                if best_remove is None or score < best_remove[0]:
                    best_remove = (score, idx)

            remove_idx = best_remove[1]
            removed_item = dict(selected.pop(remove_idx))
            removed_item["removed_reason"] = "sharp_column_angle"
            removed.append(removed_item)

        for item in selected:
            item.pop("removed_reason", None)
        return selected

    def _make_low_conf_candidates(self, boxes, existing_desks, frame_idx=None):
        candidates = []
        existing_xyxy = [desk["xyxy"] for desk in existing_desks]
        boxes_list = list(boxes) if boxes is not None else []
        for idx, box in enumerate(boxes_list):
            if isinstance(box, dict):
                xyxy = [float(v) for v in box["xyxy"]]
                if any(box_iou_xyxy(xyxy, prev) > 0.45 for prev in existing_xyxy):
                    continue
                cand = dict(box)
                cand.setdefault("low_index", idx)
                cand.setdefault("index", -1)
                cand.setdefault("is_low_conf_recovered", True)
                cand.setdefault("source", "low_conf_column_recovery")
                if frame_idx is not None:
                    cand["source_frame_idx"] = int(frame_idx)
                candidates.append(cand)
                continue

            x1, y1, x2, y2 = box_xyxy(box)
            xyxy = [x1, y1, x2, y2]
            if any(box_iou_xyxy(xyxy, prev) > 0.45 for prev in existing_xyxy):
                continue
            cx, cy = box_center(box)
            candidates.append(
                {
                    "low_index": idx,
                    "index": -1,
                    "xyxy": xyxy,
                    "center": [cx, cy],
                    "conf": float(box.conf[0]),
                    "cls": int(box.cls[0]),
                    "is_low_conf_recovered": True,
                    "source": "low_conf_column_recovery",
                    "source_frame_idx": int(frame_idx) if frame_idx is not None else None,
                }
            )
        return candidates

    def _recover_low_conf_desks_for_columns(
        self,
        pruned_columns,
        low_conf_candidates,
        row_y,
        row_w,
        row_h,
        fallback_w,
        fallback_h,
    ):
        if not low_conf_candidates:
            return 0

        used_candidate_ids = set()
        recovered_count = 0
        row_step_values = [
            abs(float(row_y[r]) - float(row_y[r + 1]))
            for r in range(self.required_per_col - 1)
            if row_y.get(r) is not None and row_y.get(r + 1) is not None
        ]
        row_step = float(np.median(row_step_values)) if row_step_values else fallback_h * 1.8

        for col_id in range(self.num_cols):
            selected = pruned_columns[col_id]
            if len(selected) >= self.required_per_col:
                continue
            if len(selected) < 2:
                continue

            line = _fit_line_from_centers([item["center"] for item in selected])
            present_rows = {int(item["row_index"]) for item in selected}
            missing_rows = [r for r in range(self.required_per_col) if r not in present_rows]

            for row_id in missing_rows:
                expected_y = float(row_y[row_id])
                fallback_x = float(np.median([item["center"][0] for item in selected]))
                expected_x = _x_on_line_at_y(line, expected_y, fallback_x)
                expected = np.asarray([expected_x, expected_y], dtype=np.float32)

                best = None
                for cand_id, cand in enumerate(low_conf_candidates):
                    if cand_id in used_candidate_ids:
                        continue
                    center = np.asarray(cand["center"], dtype=np.float32)
                    line_dist = vd1.point_line_distance_kb(center, line)
                    center_dist = float(np.linalg.norm(center - expected))
                    y_dist = abs(float(center[1]) - expected_y)
                    cand_w = float(cand["xyxy"][2] - cand["xyxy"][0])
                    cand_h = float(cand["xyxy"][3] - cand["xyxy"][1])
                    target_w = float(row_w.get(row_id) or fallback_w)
                    target_h = float(row_h.get(row_id) or fallback_h)
                    size_ratio = max(cand_w / (target_w + 1e-8), target_w / (cand_w + 1e-8))
                    size_ratio += max(cand_h / (target_h + 1e-8), target_h / (cand_h + 1e-8))
                    line_crosses_box = _line_crosses_box(line, cand["xyxy"], margin=max(4.0, target_w * 0.08))

                    max_line_dist = max(target_w * 0.42, 28.0)
                    max_center_dist = max(row_step * 0.72, target_w * 1.05, 58.0)
                    max_y_dist = max(row_step * 0.58, target_h * 1.35, 45.0)
                    if line_dist > max_line_dist or center_dist > max_center_dist or y_dist > max_y_dist:
                        if not (line_crosses_box and center_dist <= max_center_dist * 1.15 and y_dist <= max_y_dist * 1.15):
                            continue
                    if size_ratio > 5.2:
                        continue

                    score = (
                        line_dist * 1.5
                        + center_dist
                        + y_dist * 0.7
                        + size_ratio * 8.0
                        - (35.0 if line_crosses_box else 0.0)
                        - float(cand.get("conf", 0.0)) * 80.0
                    )
                    if best is None or score < best[0]:
                        best = (score, cand_id, cand)

                if best is None:
                    continue

                _, cand_id, cand = best
                used_candidate_ids.add(cand_id)
                desk_no = col_id * self.required_per_col + row_id + 1
                recovered = dict(cand)
                recovered.update(
                    {
                        "desk_no": desk_no,
                        "desk_id": f"D{desk_no:02d}",
                        "column_index": col_id,
                        "row_index": row_id,
                        "is_virtual": False,
                    }
                )
                selected.append(recovered)
                selected.sort(key=lambda item: item["row_index"])
                recovered_count += 1

        return recovered_count

    def _repair_layout_to_grid(self, layout, image_shape=None, low_conf_boxes=None):
        """优化序列选出的布局，但不凭空生成桌位。

        完整列被视为稳定列，不再被其他列的点改写；缺失位置只允许按列线恢复低置信检测框。
        若没有真实检测框，就保持缺失，不再创建 inferred/virtual 桌子。
        """
        if self.mode == "normal":
            return layout

        desks = [dict(item) for item in layout.get("desks", [])]
        if not desks:
            return layout

        expected_total = self.num_cols * self.required_per_col

        columns = {c: [] for c in range(self.num_cols)}
        for desk in desks:
            col = int(desk.get("column_index", -1))
            if 0 <= col < self.num_cols:
                columns[col].append(desk)

        pruned_columns = {}
        locked_columns = []
        all_widths = []
        all_heights = []
        for col_id in range(self.num_cols):
            selected = self._pick_best_column_six(columns[col_id])
            for row_id, desk in enumerate(selected):
                desk["column_index"] = col_id
                desk["row_index"] = row_id
                desk["desk_no"] = col_id * self.required_per_col + row_id + 1
                desk["desk_id"] = f"D{desk['desk_no']:02d}"
                desk["is_virtual"] = bool(desk.get("is_virtual", False))
                all_widths.append(float(desk["xyxy"][2] - desk["xyxy"][0]))
                all_heights.append(float(desk["xyxy"][3] - desk["xyxy"][1]))
            pruned_columns[col_id] = selected
            if len(selected) >= self.required_per_col and np.mean([d.get("conf", 0.0) for d in selected]) >= max(0.5, self.conf_threshold - 0.15):
                locked_columns.append(col_id)

        fallback_w = float(np.median(all_widths)) if all_widths else 60.0
        fallback_h = float(np.median(all_heights)) if all_heights else 40.0

        row_y_samples = {r: [] for r in range(self.required_per_col)}
        row_w_samples = {r: [] for r in range(self.required_per_col)}
        row_h_samples = {r: [] for r in range(self.required_per_col)}
        for selected in pruned_columns.values():
            if len(selected) >= self.required_per_col:
                for row_id, desk in enumerate(selected[: self.required_per_col]):
                    row_y_samples[row_id].append(float(desk["center"][1]))
                    row_w_samples[row_id].append(float(desk["xyxy"][2] - desk["xyxy"][0]))
                    row_h_samples[row_id].append(float(desk["xyxy"][3] - desk["xyxy"][1]))

        if not any(row_y_samples.values()):
            sorted_all = sorted(desks, key=lambda item: item["center"][1], reverse=True)
            for row_id, group in enumerate(np.array_split(sorted_all, self.required_per_col)):
                for desk in group:
                    row_y_samples[row_id].append(float(desk["center"][1]))

        row_y = {}
        row_w = {}
        row_h = {}
        for row_id in range(self.required_per_col):
            row_y[row_id] = float(np.median(row_y_samples[row_id])) if row_y_samples[row_id] else None
            row_w[row_id] = float(np.median(row_w_samples[row_id])) if row_w_samples[row_id] else fallback_w
            row_h[row_id] = float(np.median(row_h_samples[row_id])) if row_h_samples[row_id] else fallback_h

        known_row_ys = [v for v in row_y.values() if v is not None]
        if known_row_ys:
            y_desc = sorted(known_row_ys, reverse=True)
            step = float(np.median([abs(a - b) for a, b in zip(y_desc[:-1], y_desc[1:])])) if len(y_desc) > 1 else fallback_h * 1.8
            for row_id in range(self.required_per_col):
                if row_y[row_id] is None:
                    if row_id > 0 and row_y[row_id - 1] is not None:
                        row_y[row_id] = row_y[row_id - 1] - step
                    else:
                        row_y[row_id] = y_desc[0] - row_id * step
        else:
            row_y = {row_id: fallback_h * (self.required_per_col - row_id) for row_id in range(self.required_per_col)}

        for col_id, selected in pruned_columns.items():
            if len(selected) >= self.required_per_col:
                continue
            available_rows = set(range(self.required_per_col))
            reassigned = []
            for desk in sorted(selected, key=lambda item: float(item.get("conf", 0.0)), reverse=True):
                row_id = min(available_rows, key=lambda r: abs(float(desk["center"][1]) - float(row_y[r])))
                available_rows.remove(row_id)
                desk["row_index"] = int(row_id)
                desk["desk_no"] = col_id * self.required_per_col + row_id + 1
                desk["desk_id"] = f"D{desk['desk_no']:02d}"
                reassigned.append(desk)
            pruned_columns[col_id] = sorted(reassigned, key=lambda item: item["row_index"])

        low_conf_candidates = self._make_low_conf_candidates(low_conf_boxes, desks)
        low_conf_recovered = self._recover_low_conf_desks_for_columns(
            pruned_columns,
            low_conf_candidates,
            row_y,
            row_w,
            row_h,
            fallback_w,
            fallback_h,
        )

        repaired = []
        for col_id in range(self.num_cols):
            selected = pruned_columns[col_id]
            selected = self._prune_sharp_turns_in_column(selected)
            for row_id, desk in enumerate(sorted(selected, key=lambda item: item["row_index"])):
                desk["row_index"] = row_id
                desk_no = col_id * self.required_per_col + row_id + 1
                desk["desk_no"] = desk_no
                desk["desk_id"] = f"D{desk_no:02d}"
                desk["is_virtual"] = False
                repaired.append(desk)

        repaired.sort(key=lambda item: item["desk_no"])
        ordered_columns = [
            [desk["index"] for desk in repaired if desk["column_index"] == col_id]
            for col_id in range(self.num_cols)
        ]
        layout = dict(layout)
        layout["desks"] = repaired
        layout["ordered_columns"] = ordered_columns
        layout["columns"] = ordered_columns
        layout["column_lines"] = build_column_line_entries_from_desks(repaired, self.num_cols)
        layout["layout_repair"] = {
            "method": "angle_pruned_low_conf_recovery_no_virtual",
            "expected_total": int(expected_total),
            "final_total": int(len(repaired)),
            "virtual_desks": 0,
            "low_conf_recovered_desks": int(low_conf_recovered),
            "repair_conf_threshold": float(self._effective_repair_conf_threshold()),
            "locked_columns": [int(c) for c in locked_columns],
            "column_counts_before": [int(len(columns[c])) for c in range(self.num_cols)],
            "column_counts_after": [
                int(sum(1 for desk in repaired if desk["column_index"] == c))
                for c in range(self.num_cols)
            ],
            "column_angle_stats": [
                column_angle_stats_from_desks(
                    [desk for desk in repaired if desk["column_index"] == c]
                )
                for c in range(self.num_cols)
            ],
        }
        return layout

    def _optimize_layout_across_sampled_frames(self, layout, sampled_frames):
        """用多帧低置信检测沿当前列线持续优化布局。"""
        current_layout = layout
        repair_conf = self._effective_repair_conf_threshold()
        iterations = []
        total_recovered = 0

        if repair_conf >= self.conf_threshold:
            current_layout = self._repair_layout_to_grid(current_layout)
            current_layout.setdefault("layout_repair", {})["sequence_iterations"] = iterations
            current_layout["layout_repair"]["sequence_low_conf_recovered_desks"] = 0
            return current_layout

        for item in sampled_frames:
            frame_idx = int(item["frame_idx"])
            frame = item["frame"]
            low_conf_results = self._predict_with_conf(frame, repair_conf)
            low_conf_boxes = low_conf_results[0].boxes
            before_count = len(current_layout.get("desks", []))
            current_layout = self._repair_layout_to_grid(
                current_layout,
                frame.shape,
                low_conf_boxes=low_conf_boxes,
            )
            repair_info = current_layout.get("layout_repair", {})
            recovered = int(repair_info.get("low_conf_recovered_desks", 0))
            total_recovered += recovered
            iterations.append(
                {
                    "frame_idx": frame_idx,
                    "before_count": int(before_count),
                    "after_count": int(len(current_layout.get("desks", []))),
                    "low_conf_recovered_desks": recovered,
                    "column_counts_after": repair_info.get("column_counts_after", []),
                    "column_angle_stats": repair_info.get("column_angle_stats", []),
                }
            )

        current_layout.setdefault("layout_repair", {})["sequence_iterations"] = iterations
        current_layout["layout_repair"]["sequence_low_conf_recovered_desks"] = int(total_recovered)
        current_layout["layout_repair"]["repair_conf_threshold"] = float(repair_conf)
        return current_layout

    def _score_layout_for_scheme(self, boxes, scheme: int):
        """给单帧的某个分列方案打分，分数越高越适合作为固定座位模型。"""
        if len(boxes) == 0:
            return None

        layout = self._build_layout_for_scheme(boxes, scheme)
        centers_np = self._extract_centers(boxes)
        columns = layout["columns"]
        raw_lines = layout.pop("_raw_lines", [])

        expected_total = self.num_cols * self.required_per_col
        desk_count = len(layout["desks"])
        col_counts = [len(col) for col in columns]
        complete_cols = sum(1 for count in col_counts if count >= self.required_per_col)
        count_penalty = abs(desk_count - expected_total)
        col_balance_penalty = sum(abs(count - self.required_per_col) for count in col_counts)
        column_angle_stats = []
        sharp_turns = 0
        min_angles = []
        for col_id in range(self.num_cols):
            col_desks = [d for d in layout["desks"] if d["column_index"] == col_id]
            stats = column_angle_stats_from_desks(col_desks)
            column_angle_stats.append(stats)
            sharp_turns += int(stats["sharp_turns"])
            if stats["min_angle"] is not None:
                min_angles.append(float(stats["min_angle"]))
        min_column_angle = float(np.min(min_angles)) if min_angles else None
        angle_penalty = (
            max(0.0, 170.0 - min_column_angle) * 35.0
            if min_column_angle is not None else 0.0
        ) + sharp_turns * 900.0

        straight_score, _ = fitted_columns_straightness_score(
            centers_np,
            columns,
            required_per_col=self.required_per_col,
        )
        if not np.isfinite(straight_score):
            straight_score = avg_point_line_distance(centers_np, columns, raw_lines)
        straight_penalty = straight_score if np.isfinite(straight_score) else 10000.0

        conf_sum = sum(float(d["conf"]) for d in layout["desks"])
        mean_conf = conf_sum / max(1, desk_count)

        score = (
            desk_count * 1000.0
            + complete_cols * 250.0
            + mean_conf * 100.0
            - count_penalty * 300.0
            - col_balance_penalty * 80.0
            - straight_penalty * 5.0
            - angle_penalty
        )
        layout["layout_score"] = float(score)
        layout["layout_metrics"] = {
            "desk_count": int(desk_count),
            "expected_total": int(expected_total),
            "column_counts": [int(v) for v in col_counts],
            "complete_columns": int(complete_cols),
            "count_penalty": int(count_penalty),
            "column_balance_penalty": int(col_balance_penalty),
            "straightness": float(straight_score) if np.isfinite(straight_score) else None,
            "min_column_angle": min_column_angle,
            "sharp_column_turns": int(sharp_turns),
            "column_angle_stats": column_angle_stats,
            "confidence_sum": float(conf_sum),
            "mean_confidence": float(mean_conf),
            "score": float(score),
        }
        return layout

    def _select_layout(self, centers_np: np.ndarray, tag="", log=True):
        if self.mode == "scheme1":
            cols, _, _ = self._split_columns_for_scheme(centers_np, 1)
            return 1, cols

        if self.mode == "scheme2":
            cols, _, _ = self._split_columns_for_scheme(centers_np, 2)
            return 2, cols

        cols1, lines1, _ = self._split_columns_for_scheme(centers_np, 1)
        cols2, lines2, _ = self._split_columns_for_scheme(centers_np, 2)
        straight1, fitted1 = fitted_columns_straightness_score(
            centers_np, cols1, required_per_col=self.required_per_col
        )
        straight2, fitted2 = fitted_columns_straightness_score(
            centers_np, cols2, required_per_col=self.required_per_col
        )

        finite1 = np.isfinite(straight1)
        finite2 = np.isfinite(straight2)
        if finite1 or finite2:
            if finite1 and finite2:
                chosen = 1 if straight1 <= straight2 else 2
                if log:
                    print(
                        f"[AUTO] 按五列拟合直线度选择: s1={straight1:.4f}, s2={straight2:.4f}, 选择方案{chosen}: {tag}"
                    )
            else:
                chosen = 1 if finite1 else 2
                if log:
                    print(f"[AUTO] 仅方案{chosen}可完成五列拟合，选择方案{chosen}: {tag}")
            return chosen, (cols1 if chosen == 1 else cols2)

        score1 = avg_point_line_distance(centers_np, cols1, lines1)
        score2 = avg_point_line_distance(centers_np, cols2, lines2)
        chosen = 1 if score1 <= score2 else 2
        if log:
            print(f"[AUTO] 五列拟合不足，回退整体误差比较: s1={score1:.4f}, s2={score2:.4f}, 选择方案{chosen}: {tag}")
        return chosen, (cols1 if chosen == 1 else cols2)

    def _candidate_schemes(self):
        if self.mode == "scheme1":
            return [1]
        if self.mode == "scheme2":
            return [2]
        if self.mode == "auto":
            return [1, 2]
        return []

    def select_best_layout_from_video(
        self,
        video_path: str,
        max_frames: int = 120,
        sample_step: int = 5,
        save_dir: str | None = None,
        log: bool = True,
    ):
        """在视频前若干帧中采样检测，选择最稳定、最完整的座位布局模型。

        返回结构与旧版参考帧选择兼容:
        {
            frame_idx, frame, layout, desk_count, score
        }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        sample_step = max(1, int(sample_step))
        max_frames = max(1, int(max_frames))

        if self.mode == "normal":
            best = None
            frame_idx = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame_idx >= max_frames:
                        break
                    if frame_idx % sample_step == 0:
                        result = self.detect_desks(
                            frame,
                            tag=f"seq:{frame_idx}",
                            annotate=False,
                            log=False,
                        )
                        desks = result["layout"]["desks"]
                        score = sum(d["conf"] for d in desks)
                        if desks and (
                            best is None
                            or len(desks) > best["desk_count"]
                            or (len(desks) == best["desk_count"] and score > best["score"])
                        ):
                            best = {
                                "frame_idx": frame_idx,
                                "frame": frame.copy(),
                                "layout": result["layout"],
                                "desk_count": len(desks),
                                "score": float(score),
                            }
                    frame_idx += 1
            finally:
                cap.release()

            if best is None:
                raise RuntimeError("桌子检测失败，未找到可用桌子布局。")
            return best

        schemes = self._candidate_schemes()
        if not schemes:
            raise RuntimeError(f"不支持的桌子布局模式: {self.mode}")

        stats = {
            scheme: {
                "sampled_frames": 0,
                "usable_frames": 0,
                "full_layout_frames": 0,
                "score_sum": 0.0,
                "best": None,
            }
            for scheme in schemes
        }

        frame_idx = 0
        sampled_frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= max_frames:
                    break
                if frame_idx % sample_step != 0:
                    frame_idx += 1
                    continue

                sampled_frames.append({"frame_idx": frame_idx, "frame": frame.copy()})
                results = self._predict(frame)
                boxes = results[0].boxes
                for scheme in schemes:
                    stats[scheme]["sampled_frames"] += 1
                    layout = self._score_layout_for_scheme(boxes, scheme)
                    if layout is None:
                        continue
                    metrics = layout["layout_metrics"]
                    stats[scheme]["usable_frames"] += 1
                    stats[scheme]["score_sum"] += layout["layout_score"]
                    if metrics["desk_count"] >= metrics["expected_total"]:
                        stats[scheme]["full_layout_frames"] += 1

                    best = stats[scheme]["best"]
                    if best is None or layout["layout_score"] > best["score"]:
                        stats[scheme]["best"] = {
                            "frame_idx": frame_idx,
                            "frame": frame.copy(),
                            "layout": layout,
                            "desk_count": len(layout["desks"]),
                            "score": float(layout["layout_score"]),
                        }

                frame_idx += 1
        finally:
            cap.release()

        candidates = []
        for scheme, item in stats.items():
            if item["best"] is None or item["usable_frames"] == 0:
                continue
            avg_score = item["score_sum"] / max(1, item["usable_frames"])
            aggregate_score = (
                avg_score
                + item["full_layout_frames"] * 500.0
                + item["usable_frames"] * 25.0
            )
            candidates.append((aggregate_score, item["best"]["score"], scheme))

        if not candidates:
            raise RuntimeError("桌子检测失败，未找到可用桌子布局。")

        candidates.sort(reverse=True)
        _, _, chosen_scheme = candidates[0]
        chosen = stats[chosen_scheme]["best"]
        chosen_raw_layout = {
            key: value
            for key, value in chosen["layout"].items()
            if key not in {"_raw_lines"}
        }
        chosen["layout"] = self._optimize_layout_across_sampled_frames(
            chosen["layout"],
            sampled_frames,
        )
        chosen["desk_count"] = len(chosen["layout"].get("desks", []))

        sequence_summary = {}
        for scheme, item in stats.items():
            avg_score = (
                item["score_sum"] / max(1, item["usable_frames"])
                if item["usable_frames"] > 0 else None
            )
            sequence_summary[_scheme_name(scheme)] = {
                "sampled_frames": int(item["sampled_frames"]),
                "usable_frames": int(item["usable_frames"]),
                "full_layout_frames": int(item["full_layout_frames"]),
                "average_score": float(avg_score) if avg_score is not None else None,
                "best_score": (
                    float(item["best"]["score"]) if item["best"] is not None else None
                ),
                "best_frame_idx": (
                    int(item["best"]["frame_idx"]) if item["best"] is not None else None
                ),
            }

        chosen["layout"]["sequence_selection"] = {
            "method": "sampled_video_layout_score",
            "mode": self.mode,
            "chosen_scheme": int(chosen_scheme),
            "chosen_mode": _scheme_name(chosen_scheme),
            "max_frames": int(max_frames),
            "sample_step": int(sample_step),
            "stats": sequence_summary,
        }
        repair_info = chosen["layout"].get("layout_repair", {})
        if log:
            print("[SEQUENCE_LAYOUT] 视频序列布局评估:")
            for name, info in sequence_summary.items():
                print(
                    f"  {name}: usable={info['usable_frames']}/"
                    f"{info['sampled_frames']}, full={info['full_layout_frames']}, "
                    f"avg={info['average_score']}, best={info['best_score']} "
                    f"@frame={info['best_frame_idx']}"
                )
            print(
                f"  选择: {_scheme_name(chosen_scheme)} "
                f"@ frame {chosen['frame_idx']} score={chosen['score']:.2f}"
            )
            if repair_info:
                print(
                    "  布局优化: "
                    f"final={repair_info.get('final_total')}/"
                    f"{repair_info.get('expected_total')}, "
                    f"low_conf={repair_info.get('sequence_low_conf_recovered_desks', repair_info.get('low_conf_recovered_desks'))}, "
                    f"virtual=0, "
                    f"locked_cols={repair_info.get('locked_columns')}"
                )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            video_name = Path(str(video_path)).stem if not str(video_path).isdigit() else f"camera_{video_path}"
            raw_layout_img = draw_sequence_best_frame(
                chosen["frame"],
                chosen_raw_layout,
                title=f"Best sampled frame before repair | frame {chosen['frame_idx']}",
            )
            raw_output_path = os.path.join(save_dir, f"best_sampled_frame_{video_name}.jpg")
            cv2.imwrite(raw_output_path, raw_layout_img)

            layout_img = draw_layout_entries(
                chosen["frame"],
                chosen["layout"].get("desks", []),
                num_cols=self.num_cols,
                required_per_col=self.required_per_col,
                title=f"Best sampled frame after repair | frame {chosen['frame_idx']}",
            )
            output_path = os.path.join(save_dir, f"best_sampled_frame_repaired_{video_name}.jpg")
            cv2.imwrite(output_path, layout_img)
            chosen["layout"]["best_sampled_frame_image"] = raw_output_path
            chosen["layout"]["best_sampled_frame_repaired_image"] = output_path
            chosen["layout"]["reliable_layout_image"] = raw_output_path
            if log:
                print(f"  最佳具体帧图(未补齐): {raw_output_path}")
                print(f"  最佳具体帧图(补齐后): {output_path}")
        return chosen

    def get_layout_info(self, boxes, tag="", log=True):
        if len(boxes) == 0:
            return {
                "chosen_scheme": 0,
                "chosen_mode": "normal",
                "style": 0,
                "columns": [],
                "ordered_columns": [],
                "column_lines": [],
                "desks": [],
            }

        if self.mode == "normal":
            desks = []
            for idx, box in enumerate(list(boxes), 1):
                x1, y1, x2, y2 = box_xyxy(box)
                cx, cy = box_center(box)
                desks.append(
                    {
                        "index": idx - 1,
                        "desk_no": idx,
                        "desk_id": f"D{idx:02d}",
                        "column_index": -1,
                        "row_index": idx - 1,
                        "xyxy": [x1, y1, x2, y2],
                        "center": [cx, cy],
                        "conf": float(box.conf[0]),
                        "cls": int(box.cls[0]),
                    }
                )
            return {
                "chosen_scheme": 0,
                "chosen_mode": "normal",
                "style": 0,
                "columns": [],
                "ordered_columns": [],
                "column_lines": [],
                "desks": desks,
            }

        centers_np = self._extract_centers(boxes)
        chosen, columns = self._select_layout(centers_np, tag=tag, log=log)
        style = 1 if chosen == 1 else 2
        ordered_columns, desks = build_layout_entries(
            boxes,
            columns,
            style=style,
            num_per_col=self.required_per_col,
        )
        column_lines = build_column_line_entries(boxes, ordered_columns, desks)
        chosen_mode = "scheme1" if chosen == 1 else "scheme2"
        layout = {
            "chosen_scheme": chosen,
            "chosen_mode": chosen_mode,
            "style": style,
            "columns": columns,
            "ordered_columns": ordered_columns,
            "column_lines": column_lines,
            "desks": desks,
        }
        return self._repair_layout_to_grid(layout)

    def detect_desks(self, source, tag="", annotate=False, log=True):
        results = self._predict(source)
        boxes = results[0].boxes
        layout_info = self.get_layout_info(boxes, tag=tag, log=log)

        annotated = None
        if annotate:
            if isinstance(source, str):
                image = cv2.imread(source)
            else:
                image = source.copy()
            annotated = self._annotate(image, boxes, tag=tag, log=log)

        return {
            "boxes": boxes,
            "layout": layout_info,
            "annotated": annotated,
        }

    def _annotate(self, img, boxes, tag="", log=True):
        if len(boxes) == 0:
            return img.copy()

        if self.mode == "normal":
            return draw_normal(img, boxes, self.model.names if hasattr(self.model, "names") else None)

        layout_info = self.get_layout_info(boxes, tag=tag, log=log)
        return draw_layout_entries(
            img,
            layout_info.get("desks", []),
            num_cols=self.num_cols,
            required_per_col=self.required_per_col,
            title=f"Reliable seat layout | {layout_info.get('chosen_mode', 'unknown')}",
        )

    def _output_suffix(self):
        mapping = {
            "normal": "",
            "scheme1": "_layout_s1",
            "scheme2": "_layout_s2",
            "auto": "_layout_auto",
        }
        return mapping.get(self.mode, "")

    def detect_image(self, image_path, save_dir="output"):
        if not os.path.exists(image_path):
            print(f"警告: 图片不存在 - {image_path}")
            return None

        os.makedirs(save_dir, exist_ok=True)
        results = self._predict(image_path)
        boxes = results[0].boxes
        img = cv2.imread(image_path)
        annotated = self._annotate(img, boxes, tag=image_path, log=True)

        filename = Path(image_path).name
        output_path = os.path.join(save_dir, f"detected{self._output_suffix()}_{filename}")
        cv2.imwrite(output_path, annotated)
        print(f"检测到 {len(boxes)} 个目标 - 保存至: {output_path}")
        return annotated, boxes

    def detect_folder(self, folder_path, save_dir="output"):
        if not os.path.exists(folder_path):
            print(f"错误: 文件夹不存在 - {folder_path}")
            return

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        image_files = []
        folder = Path(folder_path)
        for ext in image_extensions:
            image_files.extend(sorted(folder.glob(f"*{ext}")))
            image_files.extend(sorted(folder.glob(f"*{ext.upper()}")))

        if not image_files:
            print(f"警告: 文件夹中没有找到图片文件 - {folder_path}")
            return

        print(f"找到 {len(image_files)} 张图片")
        for idx, image_file in enumerate(image_files, 1):
            print(f"\n处理 [{idx}/{len(image_files)}]: {image_file.name}")
            self.detect_image(str(image_file), save_dir)
        print(f"\n所有图片检测完成! 结果保存至: {save_dir}")

    def detect_video(self, video_path, save_dir="output", display=False):
        if isinstance(video_path, int) or str(video_path).isdigit():
            cap = cv2.VideoCapture(int(video_path))
            video_name = f"camera_{video_path}"
        else:
            if not os.path.exists(video_path):
                print(f"警告: 视频不存在 - {video_path}")
                return
            cap = cv2.VideoCapture(video_path)
            video_name = Path(video_path).stem

        if not cap.isOpened():
            print(f"错误: 无法打开视频 - {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 25

        print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")
        os.makedirs(save_dir, exist_ok=True)

        output_path = os.path.join(save_dir, f"detected{self._output_suffix()}_{video_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        fixed_reference = None
        fixed_layout = None
        if self.mode != "normal":
            try:
                fixed_reference = self.select_best_layout_from_video(
                    video_path,
                    max_frames=min(180, total_frames) if total_frames > 0 else 180,
                    sample_step=5,
                    save_dir=save_dir,
                    log=True,
                )
                fixed_layout = fixed_reference["layout"]
                print(
                    "固定可靠布局: "
                    f"frame={fixed_reference['frame_idx']}, "
                    f"desks={len(fixed_layout.get('desks', []))}, "
                    f"image={fixed_layout.get('reliable_layout_image')}"
                )
            except Exception as e:
                print(f"警告: 固定可靠布局建立失败，将回退逐帧布局: {e}")

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                try:
                    results = self._predict(frame)
                except Exception as e:
                    print(f"\n检测失败 (Frame {frame_count}): {e}")
                    print("提示: 如果是CUDA内存错误,请尝试 --device cpu 或 --img-size 640")
                    raise

                boxes = results[0].boxes
                if fixed_layout is not None:
                    annotated = draw_layout_entries(
                        frame,
                        fixed_layout.get("desks", []),
                        num_cols=self.num_cols,
                        required_per_col=self.required_per_col,
                        title=f"Reliable fixed layout | ref frame {fixed_reference['frame_idx']}",
                    )
                else:
                    annotated = self._annotate(frame, boxes, tag=f"{video_name}:{frame_count}", log=False)

                info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(boxes)}"
                cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(annotated)

                if display:
                    cv2.imshow("Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("用户中断检测")
                        break

                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"处理进度: {frame_count}/{total_frames} ({progress:.1f}%)")
        finally:
            cap.release()
            out.release()
            if display:
                cv2.destroyAllWindows()

        print(f"视频检测完成! 保存至: {output_path}")


def _resolve_mode(args):
    mode_cli_specified = "--mode" in sys.argv
    if mode_cli_specified:
        return args.mode
    if args.enable_desk_layout:
        return args.layout_scheme
    return "normal"


def main():
    parser = argparse.ArgumentParser(description="桌子布局检测脚本")
    parser.add_argument("--source", type=str, required=True, help="检测源: 图片/视频/文件夹/摄像头ID(0)")
    parser.add_argument("--weights", type=str, default="exam_seat_binding/weight/yolo11desk.pt", help="模型权重文件路径")
    parser.add_argument("--conf", type=float, default=0.7, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IOU阈值")
    parser.add_argument("--output", type=str, default="output", help="结果保存目录")
    parser.add_argument("--display", action="store_true", help="实时显示检测结果(仅视频)")
    parser.add_argument("--device", type=str, default="", help="运行设备: cpu/cuda/0/1 等")
    parser.add_argument("--img-size", type=int, default=None, help="推理图像尺寸，如 640/1280")
    parser.add_argument("--half", action="store_true", help="使用FP16半精度推理(仅GPU)")

    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["normal", "scheme1", "scheme2", "auto"],
        help="运行模式: normal/scheme1/scheme2/auto",
    )
    parser.add_argument("--num-cols", type=int, default=5, help="列数")
    parser.add_argument("--required-per-col", type=int, default=6, help="AUTO模式每列拟合所需点数")
    parser.add_argument(
        "--repair-conf",
        type=float,
        default=None,
        help="缺失桌位按列补检的低置信度阈值，默认比 --conf 低 0.30，最低 0.15",
    )

    # 兼容旧参数风格
    parser.add_argument("--enable-desk-layout", action="store_true", help="兼容参数：启用分列布局")
    parser.add_argument(
        "--layout-scheme",
        type=str,
        default="scheme1",
        choices=["scheme1", "scheme2", "auto"],
        help="兼容参数：启用布局后的方案选择",
    )

    args = parser.parse_args()
    run_mode = _resolve_mode(args)

    if not os.path.exists(args.weights):
        print(f"错误: 权重文件不存在 - {args.weights}")
        print(f"示例: python {sys.argv[0]} --source image.jpg --weights /path/to/best.pt")
        return

    try:
        detector = DeskLayoutDetector(
            weights_path=args.weights,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            img_size=args.img_size,
            half=args.half,
            mode=run_mode,
            num_cols=args.num_cols,
            required_per_col=args.required_per_col,
            repair_conf_threshold=args.repair_conf,
        )
    except Exception as e:
        print(f"\n初始化检测器失败: {e}")
        return

    source = args.source
    if source.isdigit():
        print(f"\n开始检测摄像头: {source}")
        detector.detect_video(source, args.output, args.display)
    elif os.path.isdir(source):
        print(f"\n开始检测文件夹: {source}")
        detector.detect_folder(source, args.output)
    elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")):
        print(f"\n开始检测视频: {source}")
        detector.detect_video(source, args.output, args.display)
    elif os.path.isfile(source):
        print(f"\n开始检测图片: {source}")
        detector.detect_image(source, args.output)
    else:
        print(f"错误: 无法识别的输入源 - {source}")
        print("支持: 图片/视频/文件夹/摄像头ID")


if __name__ == "__main__":
    main()


UnifiedDeskDetector = DeskLayoutDetector
