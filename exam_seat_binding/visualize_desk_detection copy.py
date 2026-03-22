# -*- coding: utf-8 -*-
"""
桌子检测可视化脚本（5列中心点连线）
=================================
python exam_seat_binding/visualize_desk_detection_copy.py --image data/desk/originimages/room_first_frame.png --output-dir data/desk/output
功能：
1) 使用 exam_seat_binding/weight/yolov11_desk_model.pt 检测桌子
2) 画出每个检测框与中心点
3) 将中心点按 5 列分组
4) 在每一列内按 y 从小到大连接中心点

运行方式（推荐）：
python exam_seat_binding/visualize_desk_detection.py \
  --image exam_seat_binding/weight/screenshot_01m57s719ms_frame002943.png \
  --output-dir exam_seat_binding/outputs
"""

import os
import cv2
import argparse
import logging
import importlib
from typing import List, Dict, Tuple

import numpy as np
from ultralytics import YOLO

try:
    _sk_cluster = importlib.import_module("sklearn.cluster")
    _sk_linear = importlib.import_module("sklearn.linear_model")
    KMeans = _sk_cluster.KMeans
    RANSACRegressor = _sk_linear.RANSACRegressor
    LinearRegression = _sk_linear.LinearRegression
    SKLEARN_OK = True
except Exception:
    KMeans = None
    RANSACRegressor = None
    LinearRegression = None
    SKLEARN_OK = False


LOGGER = logging.getLogger(__name__)


def ensure_dir(path: str):
    if path and (not os.path.exists(path)):
        os.makedirs(path, exist_ok=True)


def detect_desks(
    image,
    model_path: str,
    conf: float = 0.6,
    iou: float = 0.45,
    device: str = "0",
    imgsz: int = 1280,
) -> List[Dict]:
    """使用 YOLO 检测桌子，返回标准化结果。"""
    model = YOLO(model_path)
    results = model(image, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)

    dets: List[Dict] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
            score = float(boxes.conf[i].cpu().numpy())
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dets.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": score,
                    "center": [cx, cy],
                }
            )
    return dets


def split_into_columns_by_x(centers: np.ndarray, num_cols: int = 5) -> List[List[int]]:
    """
    按 x 坐标从左到右分列（等人数切分）。
    返回每列对应的点索引列表。
    """
    n = len(centers)
    if n == 0:
        return [[] for _ in range(num_cols)]

    order = np.argsort(centers[:, 0])
    chunks = np.array_split(order, num_cols)
    columns: List[List[int]] = [chunk.tolist() for chunk in chunks]

    while len(columns) < num_cols:
        columns.append([])
    return columns[:num_cols]


def fit_line_kb_positive(points: np.ndarray, min_k: float = 0.02) -> Tuple[float, float]:
    """
    拟合 y_up = kx + b（数学坐标，y_up=-y_img），并强制 k>0。
    返回 (k, b)。
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return 0.2, 0.0

    x = pts[:, 0]
    y_up = -pts[:, 1]

    if len(pts) == 1:
        k = max(min_k, 0.2)
        b = float(y_up[0] - k * x[0])
        return float(k), float(b)

    A = np.vstack([x, np.ones_like(x)]).T
    k, b = np.linalg.lstsq(A, y_up, rcond=None)[0]
    k = float(max(float(k), min_k))
    b = float(np.mean(y_up - k * x))
    return k, b


def point_line_distance_kb(point: np.ndarray, line_kb: Tuple[float, float]) -> float:
    """点到 y_up=kx+b 直线距离。"""
    x = float(point[0])
    y_up = float(-point[1])
    k, b = line_kb
    return abs(k * x - y_up + b) / (np.sqrt(k * k + 1.0) + 1e-8)


def clip_line_kb_to_image(line_kb: Tuple[float, float], w: int, h: int):
    """
    将 y_up=kx+b 转到图像坐标 y_img=-kx-b，并裁剪到图像边界。
    """
    k, b = line_kb
    pts = []

    # x=0, x=w-1
    y0 = -k * 0.0 - b
    yw = -k * (w - 1) - b
    if 0 <= y0 <= h - 1:
        pts.append((0, int(round(y0))))
    if 0 <= yw <= h - 1:
        pts.append((w - 1, int(round(yw))))

    # y=0, y=h-1
    if abs(k) > 1e-8:
        x_top = (-0.0 - b) / k
        x_bottom = (-(h - 1) - b) / k
        if 0 <= x_top <= w - 1:
            pts.append((int(round(x_top)), 0))
        if 0 <= x_bottom <= w - 1:
            pts.append((int(round(x_bottom)), h - 1))

    uniq = []
    for p in pts:
        if p not in uniq:
            uniq.append(p)
    if len(uniq) >= 2:
        return uniq[0], uniq[1]

    # fallback
    x1, x2 = 0, w - 1
    y1 = int(round(-k * x1 - b))
    y2 = int(round(-k * x2 - b))
    return (x1, y1), (x2, y2)


def find_bottom_row_seeds(centers: np.ndarray, num_cols: int = 5) -> List[int]:
    """先找底排（y 最大）的5个点，按 x 从左到右作为5列种子。"""
    n = len(centers)
    if n == 0:
        return []
    k = min(num_cols, n)
    bottom_ids = np.argsort(centers[:, 1])[::-1][:k]  # y 大 -> 近处底排
    bottom_ids = sorted(bottom_ids.tolist(), key=lambda i: centers[i, 0])
    return bottom_ids


def estimate_global_slope_positive(centers: np.ndarray) -> float:
    """估计全局正斜率 k（数学坐标 y_up = kx + b）。"""
    n = len(centers)
    if n < 4:
        return 0.2

    k = max(2, n // 5)
    bottom = centers[np.argsort(centers[:, 1])[::-1][:k]]
    top = centers[np.argsort(centers[:, 1])[:k]]

    all_pts = np.vstack([bottom, top]).astype(np.float32)
    k_est, _ = fit_line_kb_positive(all_pts, min_k=0.02)
    return float(max(0.02, k_est))


def split_into_columns_by_fitted_lines(
    centers: np.ndarray,
    num_cols: int = 5,
    num_iters: int = 10,
    expected_per_col: int = 6,
) -> Tuple[List[List[int]], List[Tuple[float, float]], List[int]]:
    """
    先用底排5个点初始化5条列线，再迭代重分配并拟合。
    返回: columns, lines, seeds
    """
    n = len(centers)
    if n == 0:
        return [[] for _ in range(num_cols)], [], []

    seeds = find_bottom_row_seeds(centers, num_cols=num_cols)
    if len(seeds) < num_cols:
        return split_into_columns_by_x(centers, num_cols=num_cols), [], seeds

    k_global = estimate_global_slope_positive(centers)

    # 初始化5条直线（都过底排点，形式 y_up = kx + b）
    lines = []
    for sid in seeds:
        x0, y0 = centers[sid]
        y_up0 = -float(y0)
        b0 = y_up0 - k_global * float(x0)
        lines.append((float(k_global), float(b0)))

    labels = np.full((n,), -1, dtype=np.int32)

    for _ in range(num_iters):
        # 1) 按“点到线距离 + 容量惩罚”分配
        counts = [0 for _ in range(num_cols)]
        process_order = np.argsort(centers[:, 1])[::-1]  # 先分配底排

        for idx in process_order:
            p = centers[idx]

            # 底排种子固定到对应列
            if idx in seeds:
                c_fix = seeds.index(int(idx))
                labels[idx] = c_fix
                counts[c_fix] += 1
                continue

            best_c = 0
            best_cost = 1e18
            for c in range(num_cols):
                d = point_line_distance_kb(p, lines[c])
                penalty = 6.0 * max(0, counts[c] - expected_per_col + 1) ** 2
                cost = d + penalty
                if cost < best_cost:
                    best_cost = cost
                    best_c = c

            labels[idx] = best_c
            counts[best_c] += 1

        # 2) 重拟合每列直线
        new_lines = []
        for c in range(num_cols):
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                sid = seeds[c]
                x0, y0 = centers[sid]
                y_up0 = -float(y0)
                b0 = y_up0 - k_global * float(x0)
                new_lines.append((float(k_global), float(b0)))
            else:
                new_lines.append(fit_line_kb_positive(centers[idxs], min_k=0.02))

        # 若列顺序错位，按底部交点 x 重新排序为 C1..C5
        y_ref = float(np.max(centers[:, 1]))
        y_ref_up = -y_ref
        x_at_ref = []
        for line in new_lines:
            k_line, b_line = line
            if abs(k_line) < 1e-8:
                x_at_ref.append(1e9)
            else:
                x_at_ref.append((y_ref_up - b_line) / k_line)
        order = np.argsort(np.array(x_at_ref))

        remap = {int(old): int(new) for new, old in enumerate(order.tolist())}
        labels = np.array([remap[int(l)] for l in labels], dtype=np.int32)
        lines = [new_lines[i] for i in order]

    columns = []
    for c in range(num_cols):
        idxs = np.where(labels == c)[0].tolist()
        # 列内按 y 从大到小（底排 -> 顶排）
        idxs = sorted(idxs, key=lambda i: centers[i, 1], reverse=True)
        columns.append(idxs)

    # 强制每列尽量 6 点：超出的列把“离本列直线最远点”转移到不足列
    target = expected_per_col
    for _ in range(200):
        counts = [len(c) for c in columns]
        over_cols = [i for i, cnt in enumerate(counts) if cnt > target]
        under_cols = [i for i, cnt in enumerate(counts) if cnt < target]
        if not over_cols or not under_cols:
            break

        src = over_cols[0]
        dst = under_cols[0]

        # src 中选一个最适合搬到 dst 的点
        best_idx = None
        best_gain = -1e18
        for idx in columns[src]:
            p = centers[idx]
            d_src = point_line_distance_kb(p, lines[src])
            d_dst = point_line_distance_kb(p, lines[dst])
            gain = d_src - d_dst
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx is None:
            break

        columns[src].remove(best_idx)
        columns[dst].append(best_idx)

        # 更新两列直线
        if len(columns[src]) > 0:
            lines[src] = fit_line_kb_positive(centers[np.array(columns[src], dtype=np.int32)], min_k=0.02)
        if len(columns[dst]) > 0:
            lines[dst] = fit_line_kb_positive(centers[np.array(columns[dst], dtype=np.int32)], min_k=0.02)

    # 最终截断/排序，保证每列最多6点
    for c in range(num_cols):
        columns[c] = sorted(columns[c], key=lambda i: centers[i, 1], reverse=True)
        if len(columns[c]) > target:
            columns[c] = columns[c][:target]

    # 最终根据列点重拟合一次，保证输出线与展示一致
    for c in range(num_cols):
        if len(columns[c]) > 0:
            lines[c] = fit_line_kb_positive(centers[np.array(columns[c], dtype=np.int32)], min_k=0.02)

    return columns, lines, seeds


def _rebalance_columns_to_target(
    centers: np.ndarray,
    columns: List[List[int]],
    lines: List[Tuple[float, float]],
    target: int = 6,
):
    """把各列点数尽量平衡为 target（通常=6）。"""
    num_cols = len(columns)

    for _ in range(300):
        counts = [len(c) for c in columns]
        over_cols = [i for i, cnt in enumerate(counts) if cnt > target]
        under_cols = [i for i, cnt in enumerate(counts) if cnt < target]
        if not over_cols or not under_cols:
            break

        src = over_cols[0]
        dst = under_cols[0]

        best_idx = None
        best_gain = -1e18
        for idx in columns[src]:
            p = centers[idx]
            d_src = point_line_distance_kb(p, lines[src])
            d_dst = point_line_distance_kb(p, lines[dst])
            gain = d_src - d_dst
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx is None:
            break

        columns[src].remove(best_idx)
        columns[dst].append(best_idx)

        if len(columns[src]) > 0:
            lines[src] = fit_line_kb_positive(centers[np.array(columns[src], dtype=np.int32)], min_k=0.02)
        if len(columns[dst]) > 0:
            lines[dst] = fit_line_kb_positive(centers[np.array(columns[dst], dtype=np.int32)], min_k=0.02)

    # 截断并排序（底->顶）
    for c in range(num_cols):
        columns[c] = sorted(columns[c], key=lambda i: centers[i, 1], reverse=True)
        if len(columns[c]) > target:
            columns[c] = columns[c][:target]
        if len(columns[c]) > 0:
            lines[c] = fit_line_kb_positive(centers[np.array(columns[c], dtype=np.int32)], min_k=0.02)


def split_into_columns_by_kmeans_x(
    centers: np.ndarray,
    num_cols: int = 5,
    expected_per_col: int = 6,
) -> Tuple[List[List[int]], List[Tuple[float, float]], List[int]]:
    """
    方法1：KMeans 对中心点 x 做聚类分列（K=5），然后每列拟合直线。
    """
    n = len(centers)
    if n == 0:
        return [[] for _ in range(num_cols)], [], []

    x_feat = centers[:, 0].reshape(-1, 1).astype(np.float32)

    if SKLEARN_OK:
        km = KMeans(n_clusters=num_cols, random_state=0, n_init=20)
        labels = km.fit_predict(x_feat)
        cxs = km.cluster_centers_.reshape(-1)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, lbl, ctr = cv2.kmeans(x_feat, num_cols, None, criteria, 20, cv2.KMEANS_PP_CENTERS)
        labels = lbl.reshape(-1)
        cxs = ctr.reshape(-1)

    order = np.argsort(cxs)
    remap = {int(old): int(new) for new, old in enumerate(order.tolist())}
    labels = np.array([remap[int(l)] for l in labels], dtype=np.int32)

    columns = []
    lines = []
    seeds = []

    for c in range(num_cols):
        idxs = np.where(labels == c)[0].tolist()
        idxs = sorted(idxs, key=lambda i: centers[i, 1], reverse=True)
        columns.append(idxs)

        if len(idxs) > 0:
            line = fit_line_kb_positive(centers[np.array(idxs, dtype=np.int32)], min_k=0.02)
            lines.append(line)
            seeds.append(idxs[0])  # 底排点
        else:
            lines.append((0.02, 0.0))

    _rebalance_columns_to_target(centers, columns, lines, target=expected_per_col)

    # 更新 seeds
    seeds = []
    for c in range(num_cols):
        seeds.append(columns[c][0] if len(columns[c]) > 0 else 0)

    return columns, lines, seeds


def _projection_t_from_seed(point: np.ndarray, line_kb: Tuple[float, float], seed_point: np.ndarray) -> float:
    """沿列线方向的投影参数 t（用于列内排序）。"""
    k, _ = line_kb
    d = np.array([1.0, k], dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-8)

    p = np.array([float(point[0]), -float(point[1])], dtype=np.float64)
    p0 = np.array([float(seed_point[0]), -float(seed_point[1])], dtype=np.float64)
    return float(np.dot(p - p0, d))


def split_into_columns_by_ransac_projection(
    centers: np.ndarray,
    num_cols: int = 5,
    num_iters: int = 12,
    expected_per_col: int = 6,
) -> Tuple[List[List[int]], List[Tuple[float, float]], List[int]]:
    """
    方法2：RANSAC 拟合5条列线 + 点投影排序。
    """
    n = len(centers)
    if n == 0:
        return [[] for _ in range(num_cols)], [], []

    seeds = find_bottom_row_seeds(centers, num_cols=num_cols)
    if len(seeds) < num_cols:
        return split_into_columns_by_kmeans_x(centers, num_cols=num_cols, expected_per_col=expected_per_col)

    k_global = estimate_global_slope_positive(centers)
    lines: List[Tuple[float, float]] = []

    # 初始化线
    for sid in seeds:
        x0, y0 = centers[sid]
        b0 = -float(y0) - k_global * float(x0)
        lines.append((float(k_global), float(b0)))

    labels = np.full((n,), -1, dtype=np.int32)

    for _ in range(num_iters):
        # 分配
        counts = [0 for _ in range(num_cols)]
        order_points = np.argsort(centers[:, 1])[::-1]
        for idx in order_points:
            p = centers[idx]
            if idx in seeds:
                c_fix = seeds.index(int(idx))
                labels[idx] = c_fix
                counts[c_fix] += 1
                continue

            best_c = 0
            best_cost = 1e18
            for c in range(num_cols):
                d = point_line_distance_kb(p, lines[c])
                penalty = 6.0 * max(0, counts[c] - expected_per_col + 1) ** 2
                cost = d + penalty
                if cost < best_cost:
                    best_cost = cost
                    best_c = c
            labels[idx] = best_c
            counts[best_c] += 1

        # RANSAC 重拟合
        new_lines: List[Tuple[float, float]] = []
        for c in range(num_cols):
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                sid = seeds[c]
                x0, y0 = centers[sid]
                b0 = -float(y0) - k_global * float(x0)
                new_lines.append((float(k_global), float(b0)))
                continue

            pts = centers[idxs]
            x = pts[:, 0].reshape(-1, 1)
            y_up = -pts[:, 1]

            if SKLEARN_OK and len(idxs) >= 3:
                try:
                    ransac = RANSACRegressor(
                        estimator=LinearRegression(),
                        min_samples=max(2, len(idxs) // 2),
                        residual_threshold=12.0,
                        random_state=0,
                    )
                    ransac.fit(x, y_up)
                    k = float(ransac.estimator_.coef_[0])
                    b = float(ransac.estimator_.intercept_)
                    k = max(0.02, k)
                    b = float(np.mean(y_up - k * pts[:, 0]))
                    new_lines.append((k, b))
                except Exception:
                    new_lines.append(fit_line_kb_positive(pts, min_k=0.02))
            else:
                new_lines.append(fit_line_kb_positive(pts, min_k=0.02))

        # 左到右重排
        y_ref = float(np.max(centers[:, 1]))
        y_ref_up = -y_ref
        x_at_ref = []
        for k_line, b_line in new_lines:
            if abs(k_line) < 1e-8:
                x_at_ref.append(1e9)
            else:
                x_at_ref.append((y_ref_up - b_line) / k_line)
        order_cols = np.argsort(np.array(x_at_ref))

        remap = {int(old): int(new) for new, old in enumerate(order_cols.tolist())}
        labels = np.array([remap[int(l)] for l in labels], dtype=np.int32)
        lines = [new_lines[i] for i in order_cols]

    columns = []
    for c in range(num_cols):
        idxs = np.where(labels == c)[0].tolist()
        if len(idxs) == 0:
            columns.append([])
            continue

        # 投影排序（底->顶）
        seed_idx = max(idxs, key=lambda i: centers[i, 1])
        seed_p = centers[seed_idx]
        idxs = sorted(idxs, key=lambda i: _projection_t_from_seed(centers[i], lines[c], seed_p))
        columns.append(idxs)

    _rebalance_columns_to_target(centers, columns, lines, target=expected_per_col)

    # 最终按投影排序并更新 seeds
    seeds = []
    for c in range(num_cols):
        if len(columns[c]) == 0:
            seeds.append(0)
            continue
        seed_idx = max(columns[c], key=lambda i: centers[i, 1])
        seed_p = centers[seed_idx]
        columns[c] = sorted(columns[c], key=lambda i: _projection_t_from_seed(centers[i], lines[c], seed_p))
        columns[c] = sorted(columns[c], key=lambda i: centers[i, 1], reverse=True)
        seeds.append(columns[c][0])

    return columns, lines, seeds


def _dist_to_origin(pt: np.ndarray) -> float:
    return float(np.sqrt(float(pt[0]) * float(pt[0]) + float(pt[1]) * float(pt[1])))


def _pick_first_column_head(centers: np.ndarray, remaining: set) -> int:
    """第一列头点：x 最小，且距离原点最近。"""
    cand = list(remaining)
    cand_sorted = sorted(cand, key=lambda i: (centers[i, 0], _dist_to_origin(centers[i])))
    return cand_sorted[0]


def _pick_next_in_same_column(centers: np.ndarray, remaining: set, cur_idx: int) -> int:
    """同列找下一个点：优先 x 变小且 y 变小，并且离原点近。"""
    cx, cy = centers[cur_idx]

    # 严格条件：x<cx 且 y<cy
    cand = [i for i in remaining if centers[i, 0] < cx and centers[i, 1] < cy]
    if len(cand) > 0:
        cand = sorted(cand, key=lambda i: (_dist_to_origin(centers[i]), np.linalg.norm(centers[i] - centers[cur_idx])))
        return cand[0]

    # 放宽1：y<cy 且 x<=cx
    cand = [i for i in remaining if centers[i, 1] < cy and centers[i, 0] <= cx]
    if len(cand) > 0:
        cand = sorted(cand, key=lambda i: (_dist_to_origin(centers[i]), np.linalg.norm(centers[i] - centers[cur_idx])))
        return cand[0]

    # 放宽2：只要求 y<cy
    cand = [i for i in remaining if centers[i, 1] < cy]
    if len(cand) > 0:
        cand = sorted(cand, key=lambda i: (_dist_to_origin(centers[i]), np.linalg.norm(centers[i] - centers[cur_idx])))
        return cand[0]

    return -1


def _pick_next_column_head(centers: np.ndarray, remaining: set, prev_head_idx: int) -> int:
    """下一列头点：相对上一列头点，x 增大且 y 增大，且离原点最近。"""
    px, py = centers[prev_head_idx]

    # 严格条件：x>px 且 y>py
    cand = [i for i in remaining if centers[i, 0] > px and centers[i, 1] > py]
    if len(cand) > 0:
        cand = sorted(cand, key=lambda i: _dist_to_origin(centers[i]))
        return cand[0]

    # 放宽1：只要求 x>px
    cand = [i for i in remaining if centers[i, 0] > px]
    if len(cand) > 0:
        cand = sorted(cand, key=lambda i: (_dist_to_origin(centers[i]), abs(float(centers[i, 1] - py))))
        return cand[0]

    # 放宽2：剩余里离原点最近
    if len(remaining) > 0:
        cand = sorted(list(remaining), key=lambda i: _dist_to_origin(centers[i]))
        return cand[0]

    return -1


def split_into_columns_by_origin_walk(
    centers: np.ndarray,
    num_cols: int = 5,
    expected_per_col: int = 6,
) -> Tuple[List[List[int]], List[Tuple[float, float]], List[int]]:
    """
    方法3（用户指定）：
    1) 第一列头点：x 最小且距离原点最近
    2) 同列向上找点：x 变小 + y 变小 + 距离原点近
    3) 下一列头点：相对上一列头点 x 变大 + y 变大 且距离原点最近
    4) 重复得到 5 列，每列目标 6 点
    """
    n = len(centers)
    if n == 0:
        return [[] for _ in range(num_cols)], [], []

    remaining = set(range(n))
    columns: List[List[int]] = [[] for _ in range(num_cols)]
    seeds: List[int] = []

    # 第一列头点
    head = _pick_first_column_head(centers, remaining)

    for c in range(num_cols):
        if head < 0:
            break

        col = [head]
        if head in remaining:
            remaining.remove(head)

        cur = head
        # 同列补到 6 点
        for _ in range(expected_per_col - 1):
            nxt = _pick_next_in_same_column(centers, remaining, cur)
            if nxt < 0:
                break
            col.append(nxt)
            remaining.remove(nxt)
            cur = nxt

        columns[c] = col
        seeds.append(col[0])

        # 下一列头点
        if c < num_cols - 1:
            head = _pick_next_column_head(centers, remaining, seeds[-1])

    # 将剩余点分配到最近列线（若某列不足6）
    lines: List[Tuple[float, float]] = []
    for c in range(num_cols):
        if len(columns[c]) > 0:
            pts = centers[np.array(columns[c], dtype=np.int32)]
            lines.append(fit_line_kb_positive(pts, min_k=0.02))
        else:
            lines.append((0.02, 0.0))

    # 先把剩余点按最近直线分配
    for idx in list(remaining):
        p = centers[idx]
        best_c = 0
        best_d = 1e18
        for c in range(num_cols):
            d = point_line_distance_kb(p, lines[c])
            if d < best_d:
                best_d = d
                best_c = c
        columns[best_c].append(idx)

    # 平衡到每列 6 点
    _rebalance_columns_to_target(centers, columns, lines, target=expected_per_col)

    # 列内排序：底->顶
    for c in range(num_cols):
        columns[c] = sorted(columns[c], key=lambda i: centers[i, 1], reverse=True)

    # 左到右重排（按底部点 x）
    bottoms = []
    for c in range(num_cols):
        if len(columns[c]) > 0:
            b = columns[c][0]
            bottoms.append((centers[b, 0], c))
        else:
            bottoms.append((1e9, c))
    order = [c for _, c in sorted(bottoms, key=lambda t: t[0])]
    columns = [columns[c] for c in order]
    lines = [lines[c] for c in order]

    seeds = []
    for c in range(num_cols):
        if len(columns[c]) > 0:
            seeds.append(columns[c][0])
            lines[c] = fit_line_kb_positive(centers[np.array(columns[c], dtype=np.int32)], min_k=0.02)
        else:
            seeds.append(0)

    return columns, lines, seeds


def draw_visualization(
    image,
    dets: List[Dict],
    columns: List[List[int]],
    line_thickness: int = 2,
):
    """绘制方框、座位编号、列内方框左上/右下角连线。"""
    vis = image.copy()

    col_colors = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 80, 255),
        (255, 200, 80),
        (220, 80, 255),
    ]

    # 先画灰色底框
    for i, det in enumerate(dets):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (120, 120, 120), 1)

    # 按列绘制
    for col_id, idxs in enumerate(columns):
        if len(idxs) == 0:
            continue
        color = col_colors[col_id % len(col_colors)]
        sorted_idxs = sorted(idxs, key=lambda idx: dets[idx]["center"][1], reverse=True)

        # 座位编号：第1列 1~6；第2列 7~12；...
        for row_id, idx in enumerate(sorted_idxs):
            seat_no = col_id * 6 + (row_id + 1)
            x1, y1, x2, y2 = [int(v) for v in dets[idx]["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis,
                f"{seat_no}",
                (x1 + 3, max(16, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        # 相邻方框：左上角->左上角、右下角->右下角
        for j in range(len(sorted_idxs) - 1):
            idx_a = sorted_idxs[j]
            idx_b = sorted_idxs[j + 1]

            ax1, ay1, ax2, ay2 = [int(v) for v in dets[idx_a]["bbox"]]
            bx1, by1, bx2, by2 = [int(v) for v in dets[idx_b]["bbox"]]

            cv2.line(vis, (ax1, ay1), (bx1, by1), color, line_thickness)
            cv2.line(vis, (ax2, ay2), (bx2, by2), color, line_thickness)

        # 列名标注
        top_idx = sorted_idxs[-1]
        tx, ty = [int(v) for v in dets[top_idx]["center"]]
        cv2.putText(
            vis,
            f"C{col_id + 1}({len(sorted_idxs)})",
            (tx + 8, max(20, ty - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return vis


def run(
    image_path: str,
    output_dir: str,
    model_path: str,
    device: str,
    conf: float,
    iou: float,
    imgsz: int,
    num_cols: int,
    show: bool,
):
    ensure_dir(output_dir)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    dets = detect_desks(
        image=image,
        model_path=model_path,
        conf=conf,
        iou=iou,
        device=device,
        imgsz=imgsz,
    )
    LOGGER.info("检测到桌子数量: %d", len(dets))

    centers = np.array([d["center"] for d in dets], dtype=np.float32) if len(dets) > 0 else np.zeros((0, 2), dtype=np.float32)

    columns, _, _ = split_into_columns_by_origin_walk(
        centers,
        num_cols=num_cols,
        expected_per_col=6,
    )

    for c, idxs in enumerate(columns):
        LOGGER.info("C%d 点数: %d", c + 1, len(idxs))

    vis = draw_visualization(image, dets, columns)

    out_img = os.path.join(output_dir, "desk_detection_origin_walk_numbered.jpg")
    cv2.imwrite(out_img, vis)
    LOGGER.info("已保存可视化结果: %s", out_img)

    if show:
        cv2.imshow("desk detection origin_walk numbered", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_args():
    default_model = os.path.join(os.path.dirname(__file__), "weight", "yolov11_desk_model.pt")
    default_output = os.path.join(os.path.dirname(__file__), "outputs")

    parser = argparse.ArgumentParser(description="桌子检测可视化（原点距离+方向约束，含座位编号）")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--model", type=str, default=default_model, help="桌子模型路径")
    parser.add_argument("--output-dir", type=str, default=default_output, help="输出目录")
    parser.add_argument("--device", type=str, default="0", help="推理设备，如 0/cpu")
    parser.add_argument("--conf", type=float, default=0.7, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=1280, help="推理尺寸")
    parser.add_argument("--num-cols", type=int, default=5, help="列数（默认5）")
    parser.add_argument("--show", action="store_true", help="显示窗口")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    run(
        image_path=args.image,
        output_dir=args.output_dir,
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        num_cols=args.num_cols,
        show=args.show,
    )