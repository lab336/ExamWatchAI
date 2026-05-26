"""
桌子布局分列方案 1。

这里只保留桌子布局算法本身，供其他脚本导入复用：
- 直线拟合
- 点到直线距离
- 按列分组
"""

from typing import List, Tuple

import numpy as np


def fit_line_kb_positive(points: np.ndarray, min_k: float = 0.02) -> Tuple[float, float]:
    """拟合 y_up = kx + b（数学坐标，y_up = -y_img），并强制 k > 0。"""
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
    """点到 y_up = kx + b 直线的距离。"""
    x = float(point[0])
    y_up = float(-point[1])
    k, b = line_kb
    return abs(k * x - y_up + b) / (np.sqrt(k * k + 1.0) + 1e-8)


def _dist_to_origin(pt: np.ndarray) -> float:
    return float(np.sqrt(float(pt[0]) * float(pt[0]) + float(pt[1]) * float(pt[1])))


def _pick_first_column_head(centers: np.ndarray, remaining: set) -> int:
    """第一列头点：x 最小，且距离原点最近。"""
    candidates = list(remaining)
    candidates = sorted(candidates, key=lambda idx: (centers[idx, 0], _dist_to_origin(centers[idx])))
    return candidates[0]


def _pick_next_in_same_column(centers: np.ndarray, remaining: set, cur_idx: int) -> int:
    """同列找下一个点：优先 x 变小且 y 变小。"""
    cx, cy = centers[cur_idx]

    candidates = [idx for idx in remaining if centers[idx, 0] < cx and centers[idx, 1] < cy]
    if candidates:
        candidates = sorted(
            candidates,
            key=lambda idx: (_dist_to_origin(centers[idx]), np.linalg.norm(centers[idx] - centers[cur_idx])),
        )
        return candidates[0]

    candidates = [idx for idx in remaining if centers[idx, 1] < cy and centers[idx, 0] <= cx]
    if candidates:
        candidates = sorted(
            candidates,
            key=lambda idx: (_dist_to_origin(centers[idx]), np.linalg.norm(centers[idx] - centers[cur_idx])),
        )
        return candidates[0]

    candidates = [idx for idx in remaining if centers[idx, 1] < cy]
    if candidates:
        candidates = sorted(
            candidates,
            key=lambda idx: (_dist_to_origin(centers[idx]), np.linalg.norm(centers[idx] - centers[cur_idx])),
        )
        return candidates[0]

    return -1


def _pick_next_column_head(centers: np.ndarray, remaining: set, prev_head_idx: int) -> int:
    """下一列头点：相对上一列头点，x 增大且 y 增大。"""
    px, py = centers[prev_head_idx]

    candidates = [idx for idx in remaining if centers[idx, 0] > px and centers[idx, 1] > py]
    if candidates:
        candidates = sorted(candidates, key=lambda idx: _dist_to_origin(centers[idx]))
        return candidates[0]

    candidates = [idx for idx in remaining if centers[idx, 0] > px]
    if candidates:
        candidates = sorted(candidates, key=lambda idx: (_dist_to_origin(centers[idx]), abs(float(centers[idx, 1] - py))))
        return candidates[0]

    if remaining:
        candidates = sorted(list(remaining), key=lambda idx: _dist_to_origin(centers[idx]))
        return candidates[0]

    return -1


def split_into_columns_by_origin_walk(
    centers: np.ndarray,
    num_cols: int = 5,
    expected_per_col: int = 6,
) -> Tuple[List[List[int]], List[Tuple[float, float]], List[int]]:
    """桌子分列算法（原点距离 + 方向约束）。"""
    n = len(centers)
    if n == 0:
        return [[] for _ in range(num_cols)], [], []

    remaining = set(range(n))
    columns: List[List[int]] = [[] for _ in range(num_cols)]
    seeds: List[int] = []

    head = _pick_first_column_head(centers, remaining)

    for c in range(num_cols):
        if head < 0:
            break

        col = [head]
        remaining.discard(head)

        cur = head
        for _ in range(expected_per_col - 1):
            nxt = _pick_next_in_same_column(centers, remaining, cur)
            if nxt < 0:
                break
            col.append(nxt)
            remaining.remove(nxt)
            cur = nxt

        columns[c] = col
        seeds.append(col[0])

        if c < num_cols - 1:
            head = _pick_next_column_head(centers, remaining, seeds[-1])

    lines: List[Tuple[float, float]] = []
    for c in range(num_cols):
        if columns[c]:
            pts = centers[np.array(columns[c], dtype=np.int32)]
            lines.append(fit_line_kb_positive(pts, min_k=0.02))
        else:
            lines.append((0.02, 0.0))

    for idx in list(remaining):
        point = centers[idx]
        best_col = 0
        best_dist = 1e18
        for c in range(num_cols):
            dist = point_line_distance_kb(point, lines[c])
            if dist < best_dist:
                best_dist = dist
                best_col = c
        columns[best_col].append(idx)

    for c in range(num_cols):
        columns[c] = sorted(columns[c], key=lambda idx: centers[idx, 1], reverse=True)

    bottoms = []
    for c in range(num_cols):
        if columns[c]:
            bottoms.append((centers[columns[c][0], 0], c))
        else:
            bottoms.append((1e9, c))

    order = [c for _, c in sorted(bottoms, key=lambda item: item[0])]
    columns = [columns[c] for c in order]
    lines = [lines[c] for c in order]

    seeds = []
    for c in range(num_cols):
        if columns[c]:
            seeds.append(columns[c][0])
            pts = centers[np.array(columns[c], dtype=np.int32)]
            lines[c] = fit_line_kb_positive(pts, min_k=0.02)
        else:
            seeds.append(0)

    return columns, lines, seeds


__all__ = [
    "fit_line_kb_positive",
    "point_line_distance_kb",
    "split_into_columns_by_origin_walk",
]
