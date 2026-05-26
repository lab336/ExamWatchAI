"""
桌子布局分列方案 2。

这里只保留桌子布局算法本身，供其他脚本导入复用：
- 直线拟合
- 点到直线距离
- 按列分组
"""

from typing import List, Tuple

import numpy as np

try:
    from exam_seat_binding.desk_layout_scheme1 import fit_line_kb_positive, point_line_distance_kb
except Exception:
    from desk_layout_scheme1 import fit_line_kb_positive, point_line_distance_kb


def _pick_max_y_head(centers: np.ndarray, remaining: set) -> int:
    """按 y 值降序选择列头，y 最大者优先。"""
    if not remaining:
        return -1
    candidates = list(remaining)
    candidates = sorted(candidates, key=lambda idx: (-float(centers[idx, 1]), float(centers[idx, 0])))
    return candidates[0]


def _pick_next_in_same_column(centers: np.ndarray, remaining: set, cur_idx: int) -> int:
    """同列找下一个点：严格要求 x 变小且 y 变小。"""
    cx, cy = centers[cur_idx]
    candidates = [idx for idx in remaining if centers[idx, 0] < cx and centers[idx, 1] < cy]
    if not candidates:
        return -1

    candidates = sorted(
        candidates,
        key=lambda idx: (
            float(cy - centers[idx, 1]),
            np.linalg.norm(centers[idx] - centers[cur_idx]),
            float(centers[idx, 0]),
        ),
    )
    return candidates[0]


def split_into_columns_by_origin_walk(
    centers: np.ndarray,
    num_cols: int = 5,
    expected_per_col: int = 6,
) -> Tuple[List[List[int]], List[Tuple[float, float]], List[int]]:
    """桌子分列算法（底部优先 + 方向约束）。"""
    n = len(centers)
    if n == 0:
        return [[] for _ in range(num_cols)], [], []

    remaining = set(range(n))
    columns: List[List[int]] = [[] for _ in range(num_cols)]
    seeds: List[int] = []

    head = _pick_max_y_head(centers, remaining)

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
            head = _pick_max_y_head(centers, remaining)

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
        columns[c] = sorted(columns[c], key=lambda idx: (centers[idx, 1], centers[idx, 0]), reverse=True)

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
