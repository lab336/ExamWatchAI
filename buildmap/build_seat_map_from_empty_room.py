import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


COCO_DINING_TABLE = 60


# =========================================================
# Utils
# =========================================================

def draw_box(img, box, color=(0, 255, 0), label=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            img, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
        )


def box_center(box):
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def box_size(box):
    x1, y1, x2, y2 = box
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def rect_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def nms_boxes(objs, iou_th=0.5):
    if not objs:
        return []

    objs = sorted(objs, key=lambda x: x["conf"], reverse=True)
    kept = []

    for o in objs:
        keep = True
        for k in kept:
            if rect_iou(o["bbox"], k["bbox"]) > iou_th:
                keep = False
                break
        if keep:
            kept.append(o)
    return kept


def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


# =========================================================
# YOLO COCO detector
# =========================================================

class CocoDetector:
    def __init__(self, weights="yolo11s.pt", device="0", half=False):
        self.model = YOLO(weights)
        self.device = device
        self.half = half

    def predict(self, image, conf=0.20, iou=0.6, imgsz=1280, classes=None):
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device if self.device else None,
            half=self.half,
            verbose=False,
            classes=classes,
            save=False,
        )
        return results[0].boxes


def boxes_to_objects(boxes, names=None):
    objs = []
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = float(b.conf[0]) if b.conf is not None else 1.0
        cls = int(b.cls[0]) if b.cls is not None else -1
        objs.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "conf": conf,
            "cls": cls,
            "name": names[cls] if names and cls in names else str(cls)
        })
    return objs


# =========================================================
# Perspective grid fitter from desk centers
# Scheme A: quadrilateral + bilinear interpolation
# =========================================================

def bilinear_point(P00, P04, P50, P54, u, v):
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


class DeskPerspectiveGridFitter:
    def __init__(self, num_rows=6, num_cols=5):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.corners = None
        self.grid_points = None
        self.row_expected_w = None
        self.row_expected_h = None

    def _init_corners_from_points(self, pts):
        arr = np.asarray([[p[0], p[1], p[2], p[3]] for p in pts], dtype=np.float32)
        ys = arr[:, 1]
        idx = np.argsort(ys)
        arr_sorted = arr[idx]

        n = len(arr_sorted)
        top_group = arr_sorted[:max(4, n // 4)]
        bottom_group = arr_sorted[-max(4, n // 4):]

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
            if cost[i, j] < 2.0:
                pairs.append((i, j))
                costs.append(cost[i, j])

        mean_cost = float(np.mean(costs)) if len(costs) > 0 else 1e9
        return pairs, mean_cost

    def _refine_corners(self, pts, corners, steps=40, step_scale=8.0):
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
            for cid in range(4):
                for d in directions:
                    trial = [c.copy() for c in best_corners]
                    trial[cid] = trial[cid] + scale * d

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
        pts: list[(cx, cy, w, h)]
        """
        if len(pts) < 6:
            raise ValueError("可用于拟合桌位网格的桌子点太少")

        current_pts = list(pts)
        corners = self._init_corners_from_points(current_pts)

        best_cost = 1e9
        best_corners = corners

        for _ in range(num_iters):
            refined_corners, cost = self._refine_corners(current_pts, corners)
            grid = self._build_grid(refined_corners)

            pairs, _ = self._match_points_to_grid(current_pts, grid)
            if len(pairs) < 4:
                break

            errors = []
            for pt_idx, slot_idx in pairs:
                px, py, _, _ = current_pts[pt_idx]
                gx = grid[slot_idx]["center_x"]
                gy = grid[slot_idx]["center_y"]
                err = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                errors.append((err, pt_idx))

            errors.sort(key=lambda x: x[0])
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

        row_w = defaultdict(list)
        row_h = defaultdict(list)
        pairs, _ = self._match_points_to_grid(pts, self.grid_points)
        for pt_idx, slot_idx in pairs:
            r = self.grid_points[slot_idx]["row_id"]
            row_w[r].append(pts[pt_idx][2])
            row_h[r].append(pts[pt_idx][3])

        self.row_expected_w = {}
        self.row_expected_h = {}
        for r in range(self.num_rows):
            self.row_expected_w[r] = float(np.median(row_w[r])) if len(row_w[r]) > 0 else 120.0
            self.row_expected_h[r] = float(np.median(row_h[r])) if len(row_h[r]) > 0 else 80.0

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
                "expected_w": self.row_expected_w.get(r, 120.0),
                "expected_h": self.row_expected_h.get(r, 80.0),
            })
        return slots


# =========================================================
# Single-seat map builder from desks only
# =========================================================

class SingleSeatMapBuilderFromDesks:
    def __init__(self, image_w, image_h, expected_rows=6, expected_cols=5):
        self.image_w = image_w
        self.image_h = image_h
        self.expected_rows = expected_rows
        self.expected_cols = expected_cols

    # -----------------------------
    # Stage 1. Filter raw desk proposals
    # -----------------------------
    def in_teacher_zone(self, cx, cy):
        """
        简单教师区/讲台区过滤。
        针对你给的空教室图，这里把左下前景区域过滤掉。
        后续可按考场再调。
        """
        return (cx < 0.45 * self.image_w) and (cy > 0.52 * self.image_h)

    def filter_desks(self, desks):
        filtered = []
        for d in desks:
            x1, y1, x2, y2 = d["bbox"]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            cx, cy = box_center(d["bbox"])

            # 基础尺寸过滤
            if area < 1500:
                continue
            if area > 250000:
                continue

            # 讲台 / 教师区过滤
            if self.in_teacher_zone(cx, cy):
                continue

            # 过高过窄/不太像桌子的形状过滤
            if w < 35 or h < 15:
                continue

            filtered.append(d)

        return filtered

    # -----------------------------
    # Stage 2. Isolated proposal removal
    # -----------------------------
    def remove_isolated_desks(self, desks):
        """
        只保留位于规则阵列中的 desk：
        - 周围至少有若干“相似尺度”的邻居
        """
        if len(desks) <= 2:
            return desks

        centers = np.array([box_center(d["bbox"]) for d in desks], dtype=np.float32)
        sizes = np.array([box_size(d["bbox"]) for d in desks], dtype=np.float32)

        kept = []
        for i, d in enumerate(desks):
            cx, cy = centers[i]
            w, h = sizes[i]

            neigh_cnt = 0
            for j in range(len(desks)):
                if i == j:
                    continue

                dx = abs(centers[j][0] - cx)
                dy = abs(centers[j][1] - cy)

                # 允许的邻居范围：考虑透视下的稀疏排列
                if dx < 4.0 * w and dy < 3.5 * h:
                    wj, hj = sizes[j]
                    # 尺度差太大不算同类桌位邻居
                    if 0.35 * w < wj < 2.5 * w and 0.35 * h < hj < 2.5 * h:
                        neigh_cnt += 1

            if neigh_cnt >= 1:
                kept.append(d)

        return kept

    # -----------------------------
    # Stage 3. Prepare points for fitting
    # -----------------------------
    def proposals_to_points(self, desks):
        pts = []
        for d in desks:
            cx, cy = box_center(d["bbox"])
            w, h = box_size(d["bbox"])
            pts.append((cx, cy, w, h))
        return pts

    # -----------------------------
    # Stage 4. Build standardized seat cells
    # -----------------------------
    def build_seat_map_from_slots(self, slots):
        seat_map = []

        for s in slots:
            cx = s["center_x"]
            cy = s["center_y"]
            w = s["expected_w"]
            h = s["expected_h"]

            # 单人 seat bbox：
            # 以 desk 中心为锚，向上下左右扩展
            # 这里 bbox 更像 seat cell / occupancy 区，而不是 desk 原始框
            x1 = cx - 0.65 * w
            x2 = cx + 0.65 * w
            y1 = cy - 0.40 * h
            y2 = cy + 1.15 * h

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.image_w - 1, x2)
            y2 = min(self.image_h - 1, y2)

            seat_map.append({
                "seat_id": s["slot_id"],
                "row_id": s["row_id"],
                "col_id": s["col_id"],
                "center_x": cx,
                "center_y": cy,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "expected_w": w,
                "expected_h": h,
            })

        return seat_map


# =========================================================
# Visualization
# =========================================================

def draw_grid_lines(img, seat_map, num_rows, num_cols, color=(0, 165, 255)):
    vis = img.copy()
    seat_lookup = {(s["row_id"], s["col_id"]): s for s in seat_map}

    # 行线
    for r in range(num_rows):
        pts = []
        for c in range(num_cols):
            s = seat_lookup.get((r, c), None)
            if s is not None:
                pts.append((int(s["center_x"]), int(s["center_y"])))
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(vis, pts[i], pts[i + 1], color, 2)

    # 列线
    for c in range(num_cols):
        pts = []
        for r in range(num_rows):
            s = seat_lookup.get((r, c), None)
            if s is not None:
                pts.append((int(s["center_x"]), int(s["center_y"])))
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(vis, pts[i], pts[i + 1], color, 2)

    return vis


def draw_result(img, raw_desks, filtered_desks, seat_map, corners=None, num_rows=6, num_cols=5):
    vis = img.copy()

    # 原始 desk proposal
    for d in raw_desks:
        draw_box(vis, d["bbox"], color=(255, 0, 0), label="desk_raw", thickness=1)

    # 过滤后的 desk
    for d in filtered_desks:
        draw_box(vis, d["bbox"], color=(255, 255, 0), label="desk_keep", thickness=2)

    # seat map
    for s in seat_map:
        draw_box(vis, s["bbox"], color=(0, 255, 0),
                 label=f"S{s['seat_id']} R{s['row_id']}C{s['col_id']}", thickness=2)
        cx = int(s["center_x"])
        cy = int(s["center_y"])
        cv2.circle(vis, (cx, cy), 5, (255, 0, 255), -1)

    vis = draw_grid_lines(vis, seat_map, num_rows, num_cols)

    # 角点
    if corners is not None:
        P00, P04, P50, P54 = corners
        for p, name in zip([P00, P04, P50, P54], ["P00", "P04", "P50", "P54"]):
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0, 165, 255), -1)
            cv2.putText(vis, name, (int(p[0]) + 6, int(p[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        pts = np.array([P00, P04, P54, P50], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 165, 255), thickness=2)

    return vis


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser("Build single-seat map from desks only (YOLOv11 COCO)")
    ap.add_argument("--image", type=str, required=True, help="空教室图像")
    ap.add_argument("--weights", type=str, default="yolo11s.pt", help="YOLOv11 COCO 权重")
    ap.add_argument("--output-dir", type=str, default="detect/output", help="输出目录")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--expected-rows", type=int, default=6)
    ap.add_argument("--expected-cols", type=int, default=5)
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(args.image)

    os.makedirs(args.output_dir, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"无法读取图像: {args.image}")

    H, W = img.shape[:2]

    detector = CocoDetector(weights=args.weights, device=args.device, half=args.half)
    names = detector.model.names

    # 1) 只检测 dining table
    boxes = detector.predict(
        img,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=[COCO_DINING_TABLE]
    )
    objs = boxes_to_objects(boxes, names)

    raw_desks = nms_boxes(objs, iou_th=0.5)

    builder = SingleSeatMapBuilderFromDesks(
        image_w=W,
        image_h=H,
        expected_rows=args.expected_rows,
        expected_cols=args.expected_cols
    )

    filtered_desks = builder.filter_desks(raw_desks)
    filtered_desks = builder.remove_isolated_desks(filtered_desks)

    pts = builder.proposals_to_points(filtered_desks)

    if len(pts) < 6:
        raise RuntimeError(
            f"过滤后 desk 点太少，无法建模。raw={len(raw_desks)}, filtered={len(filtered_desks)}"
        )

    fitter = DeskPerspectiveGridFitter(
        num_rows=args.expected_rows,
        num_cols=args.expected_cols
    )
    grid_points, best_cost = fitter.fit(pts)
    slots = fitter.get_slots()
    seat_map = builder.build_seat_map_from_slots(slots)

    # 保存 json
    out_json = os.path.join(args.output_dir, f"{Path(args.image).stem}_desk_seat_map.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "image_path": args.image,
            "image_width": W,
            "image_height": H,
            "num_rows": args.expected_rows,
            "num_cols": args.expected_cols,
            "num_seats": len(seat_map),
            "fit_cost": best_cost,
            "corners": [c.tolist() for c in fitter.corners],
            "seat_map": seat_map
        }, f, ensure_ascii=False, indent=2)

    # 可视化
    vis = draw_result(
        img,
        raw_desks=raw_desks,
        filtered_desks=filtered_desks,
        seat_map=seat_map,
        corners=fitter.corners,
        num_rows=args.expected_rows,
        num_cols=args.expected_cols
    )
    out_img = os.path.join(args.output_dir, f"{Path(args.image).stem}_desk_seat_map_vis.jpg")
    cv2.imwrite(out_img, vis)

    print("完成!")
    print(f"raw_desks={len(raw_desks)}")
    print(f"filtered_desks={len(filtered_desks)}")
    print(f"seat_map={len(seat_map)}")
    print(f"fit_cost={best_cost:.4f}")
    print(f"seat_map json: {out_json}")
    print(f"visualization: {out_img}")


if __name__ == "__main__":
    main()