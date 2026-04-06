"""
桌子布局检测器。

职责：
1. 使用 YOLO 检测桌子
2. 通过两种分列方案完成桌子编号
3. 支持 normal / scheme1 / scheme2 / auto 四种模式

示例：
1. 普通检测:
   python exam_seat_binding/desk_layout_detector.py --source data/my_screenshots --weights exam_seat_binding/weight/yolo11desk.pt --mode normal
2. 分列方案1:
   python exam_seat_binding/desk_layout_detector.py --source data/my_screenshots --weights exam_seat_binding/weight/yolo11desk.pt --mode scheme1
3. 分列方案2:
   python exam_seat_binding/desk_layout_detector.py --source data/my_screenshots --weights exam_seat_binding/weight/yolo11desk.pt --mode scheme2
4. 自动选择方案:
   python exam_seat_binding/desk_layout_detector.py --source data/my_screenshots --weights exam_seat_binding/weight/yolo11desk.pt --mode auto
"""

import argparse
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
        return self.model.predict(
            source=source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=False,
            verbose=False,
            device=self.device,
            imgsz=self.img_size or 640,
            half=self.half,
        )

    def _extract_centers(self, boxes):
        centers = []
        for box in boxes:
            centers.append(list(box_center(box)))
        return np.array(centers, dtype=np.float32)

    def _select_layout(self, centers_np: np.ndarray, tag="", log=True):
        if self.mode == "scheme1":
            cols, _, _ = vd1.split_into_columns_by_origin_walk(centers_np, num_cols=self.num_cols)
            return 1, cols

        if self.mode == "scheme2":
            cols, _, _ = vd2.split_into_columns_by_origin_walk(centers_np, num_cols=self.num_cols)
            return 2, cols

        cols1, lines1, _ = vd1.split_into_columns_by_origin_walk(centers_np, num_cols=self.num_cols)
        cols2, lines2, _ = vd2.split_into_columns_by_origin_walk(centers_np, num_cols=self.num_cols)
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
        return {
            "chosen_scheme": chosen,
            "chosen_mode": chosen_mode,
            "style": style,
            "columns": columns,
            "ordered_columns": ordered_columns,
            "column_lines": column_lines,
            "desks": desks,
        }

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
        return draw_layout(
            img,
            boxes,
            layout_info["columns"],
            style=layout_info["style"],
            num_per_col=self.required_per_col,
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
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU阈值")
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
