"""基于两套分列方案自动选择拟合更好的方案并绘制结果。

用法示例:
python exam_seat_binding/visualize_desk_detection_auto.py --source data/my_screenshots --weights exam_seat_binding/weight/yolo11desk.pt --enable-desk-layout
"""
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import importlib.util
import sys

# 尝试正常包导入；若作为脚本直接运行导致找不到包，则按文件路径动态加载模块
def _load_module_from_file(name: str, filename: str):
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    from exam_seat_binding import visualize_desk_detection as vd1
    from exam_seat_binding import visualize_desk_detection2 as vd2
except Exception:
    vd1 = _load_module_from_file('visualize_desk_detection', 'visualize_desk_detection.py')
    vd2 = _load_module_from_file('visualize_desk_detection2', 'visualize_desk_detection2.py')


def avg_point_line_distance(centers: np.ndarray, columns, lines) -> float:
    if len(centers) == 0:
        return float('inf')
    total = 0.0
    cnt = 0
    for c, col in enumerate(columns):
        if len(col) == 0:
            continue
        line = lines[c]
        for idx in col:
            total += vd1.point_line_distance_kb(centers[idx], line)
            cnt += 1
    return total / max(1, cnt)


def scheme_can_fit_all_columns(centers: np.ndarray, columns, required_per_col: int = 6) -> tuple:
    """检查每列是否至少有 required_per_col 个点并用这些点拟合直线。

    返回 (can_fit_all, fitted_lines)。fitted_lines 为每列的拟合直线（长度与 columns 相同），
    若某列点不足则返回 False 且对应行放 None。
    """
    fitted_lines = []
    for col in columns:
        if len(col) < required_per_col:
            fitted_lines.append(None)
            return False, fitted_lines
        # 取列中前 required_per_col 个点进行拟合（假设列内顺序已为从底到顶）
        idxs = np.array(col[:required_per_col], dtype=np.int32)
        pts = centers[idxs]
        line_kb = vd1.fit_line_kb_positive(pts, min_k=0.02)
        fitted_lines.append(line_kb)
    return True, fitted_lines


def draw_using_style(img, boxes, centers, columns, lines, style=1, model_names=None, num_per_col=6):
    annotated = img.copy()
    h, w = annotated.shape[:2]
    col_colors = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 80, 255),
        (255, 200, 80),
        (220, 80, 255),
    ]

    boxes_list = list(boxes)

    # 灰色底框
    for box in boxes_list:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (120, 120, 120), 1)

    # 绘制拟合直线
    for c, line in enumerate(lines):
        if len(columns[c]) > 0:
            color = col_colors[c % len(col_colors)]
            pt1, pt2 = vd1.clip_line_kb_to_image(line, w, h)
            cv2.line(annotated, pt1, pt2, color, 1)

    # 每列绘制与连线
    for col_id, idxs in enumerate(columns):
        if len(idxs) == 0:
            continue
        color = col_colors[col_id % len(col_colors)]

        # 计算中心点列表用于排序
        centers_list = []
        for idx in idxs:
            box = boxes_list[idx]
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers_list.append((idx, cx, cy))

        # 排序：从底到顶；style=2 保持 (y, x) 次序，style=1 只按 y
        if style == 1:
            sorted_pairs = sorted(centers_list, key=lambda p: p[2], reverse=True)
        else:
            sorted_pairs = sorted(centers_list, key=lambda p: (p[2], p[1]), reverse=True)

        sorted_idxs = [p[0] for p in sorted_pairs]

        # 绘制编号与置信度
        for row_id, idx in enumerate(sorted_idxs):
            seat_no = col_id * num_per_col + (row_id + 1)
            box = boxes_list[idx]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model_names[cls] if model_names is not None and cls in model_names else str(cls)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{seat_no}: {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 相邻连线：根据 style 决定端点
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

        # 列名
        if len(sorted_idxs) > 0:
            top_idx = sorted_idxs[-1]
            box_top = boxes_list[top_idx]
            x1, y1, x2, y2 = map(int, box_top.xyxy[0].tolist())
            tx = (x1 + x2) // 2
            ty = y1
            label = f"C{col_id + 1}({len(sorted_idxs)})"
            cv2.putText(annotated, label, (tx + 8, max(20, ty - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default="exam_seat_binding/weight/yolo11desk.pt")
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", default="output")
    parser.add_argument("--num-cols", type=int, default=5)
    parser.add_argument("--enable-desk-layout", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"权重不存在: {args.weights}")
        return

    model = YOLO(args.weights)

    # 支持文件夹或单张图片输入
    source = args.source
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    def process_single_image(image_path: str):
        results = model.predict(source=image_path, conf=args.conf, iou=args.iou, verbose=False)
        result = results[0]
        boxes = result.boxes
        if len(boxes) == 0:
            print(f"未检测到目标: {image_path}")
            return None

        img = cv2.imread(image_path)

        # 提取中心
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            centers.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        centers_np = np.array(centers, dtype=np.float32)

        # 执行两套分列算法
        cols1, lines1, seeds1 = vd1.split_into_columns_by_origin_walk(centers_np, num_cols=args.num_cols)
        cols2, lines2, seeds2 = vd2.split_into_columns_by_origin_walk(centers_np, num_cols=args.num_cols)

        # 先检查每列是否能用前6个点拟合直线
        can1, fitted1 = scheme_can_fit_all_columns(centers_np, cols1, required_per_col=6)
        can2, fitted2 = scheme_can_fit_all_columns(centers_np, cols2, required_per_col=6)

        chosen = None
        chosen_lines = None

        if can1 and not can2:
            chosen = 1
            chosen_lines = fitted1
            print(f"方案1: 每列均可用6点拟合，选择方案1 ({image_path})")
        elif can2 and not can1:
            chosen = 2
            chosen_lines = fitted2
            print(f"方案2: 每列均可用6点拟合，选择方案2 ({image_path})")
        elif can1 and can2:
            # 两者均可拟合，比较拟合平均误差
            score1 = avg_point_line_distance(centers_np, cols1, fitted1)
            score2 = avg_point_line_distance(centers_np, cols2, fitted2)
            print(f"两方案均可拟合 ({image_path})，方案1 平均点线距={score1:.4f}, 方案2 平均点线距={score2:.4f}")
            chosen = 1 if score1 <= score2 else 2
            chosen_lines = fitted1 if chosen == 1 else fitted2
            print(f"选择方案: {chosen}")
        else:
            # 若两者都无法满足每列6点，则回退到原先的平均点线距比较
            score1 = avg_point_line_distance(centers_np, cols1, lines1)
            score2 = avg_point_line_distance(centers_np, cols2, lines2)
            print(f"均无法用6点拟合 ({image_path})，回退比较平均点线距: 方案1={score1:.4f}, 方案2={score2:.4f}")
            chosen = 1 if score1 <= score2 else 2
            chosen_lines = lines1 if chosen == 1 else lines2
            print(f"选择方案: {chosen}")

        os.makedirs(args.output, exist_ok=True)
        filename = Path(image_path).name

        if chosen == 1:
            annotated = draw_using_style(img, boxes, centers_np, cols1, chosen_lines, style=1, model_names=model.names)
        else:
            annotated = draw_using_style(img, boxes, centers_np, cols2, chosen_lines, style=2, model_names=model.names)

        out_path = os.path.join(args.output, f"auto_choice_{filename}")
        cv2.imwrite(out_path, annotated)
        print(f"保存至: {out_path}")
        return out_path

    # 如果是文件夹则批量处理
    if os.path.isdir(source):
        files = []
        p = Path(source)
        for ext in image_extensions:
            files.extend(sorted(p.glob(f"*{ext}")))
            files.extend(sorted(p.glob(f"*{ext.upper()}")))
        if not files:
            print(f"文件夹中未找到图片: {source}")
            return
        for f in files:
            process_single_image(str(f))
        return

    # 否则按单张图片处理
    if os.path.isfile(source):
        process_single_image(source)
        return

    print("无效的 --source 输入: 既不是文件也不是文件夹")


if __name__ == '__main__':
    main()
