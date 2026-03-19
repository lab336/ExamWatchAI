import os
import cv2
import math
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque, Counter
from ultralytics import YOLO


# =========================
# 配置
# =========================
CONFIG = {
    "desk_model_path": "model/yolo11desk.pt",
    "person_model_path": "model/yolo11speopel.pt",
    "video_path": "data/clip_video/clipped_00000000028000000.mp4",
    "output_video_path": "output/detect/result.mp4",
    "output_csv_path": "output/detect/tracking.csv",
    "output_seatmap_path": "output/detect/seat_map.jpg",

    # 教室先验
    "num_rows": 6,
    "num_cols": 5,

    # 用前多少帧来建立座位网格
    "build_grid_frames": 120,
    "build_grid_sample_step": 5,

    # 检测参数
    "desk_conf": 0.25,
    "person_conf": 0.25,
    "imgsz": 1280,

    # 跟踪
    "tracker_cfg": "bytetrack.yaml",

    # 人-座位匹配阈值（相对于局部座位间距）
    "seat_match_ratio": 0.75,

    # 稳定绑定窗口
    "seat_vote_window": 20,
    "seat_vote_min_count": 8,

    # 判断老师的规则
    "teacher_unassigned_frames": 30,
    "teacher_move_threshold": 80.0,  # 轨迹中心累计移动阈值
}


# =========================
# 工具函数
# =========================
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def bbox_xyxy_to_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def bbox_bottom_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


def draw_text(img, text, org, color=(0, 255, 0), scale=0.6, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def sort_points_by_x(points):
    return sorted(points, key=lambda p: p[0])


def safe_mean(points):
    if len(points) == 0:
        return None
    return np.mean(np.asarray(points), axis=0)


def l2(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def cv_kmeans_1d(values, k):
    """
    对一维数据做 kmeans，返回每个值对应的 cluster index（按中心从小到大重排后）
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if len(values) < k:
        # 数据不足时简单平均分桶
        idx = np.argsort(values[:, 0])
        labels = np.zeros(len(values), dtype=np.int32)
        step = max(1, len(values) // k)
        for i, ii in enumerate(idx):
            labels[ii] = min(i // step, k - 1)
        return labels

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    compactness, labels, centers = cv2.kmeans(
        values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()
    centers = centers.flatten()

    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order)}
    new_labels = np.array([remap[x] for x in labels], dtype=np.int32)
    return new_labels


def bilinear_grid(tl, tr, bl, br, rows, cols):
    """
    根据四角点生成 rows x cols 的双线性中心网格
    """
    grid = []
    for r in range(rows):
        vr = 0 if rows == 1 else r / (rows - 1)
        left = (1 - vr) * tl + vr * bl
        right = (1 - vr) * tr + vr * br
        row_pts = []
        for c in range(cols):
            uc = 0 if cols == 1 else c / (cols - 1)
            p = (1 - uc) * left + uc * right
            row_pts.append(p.astype(np.float32))
        grid.append(row_pts)
    return np.array(grid, dtype=np.float32)  # [rows, cols, 2]


def best_frame_desk_boxes(video_path, desk_model, conf=0.25, imgsz=1280,
                          max_frames=120, sample_step=5):
    """
    在前 max_frames 帧中，找桌子检测数量最多的一帧作为建网格参考帧
    """
    cap = cv2.VideoCapture(video_path)
    best = {
        "count": -1,
        "frame": None,
        "boxes": None,
        "frame_idx": -1
    }
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= max_frames:
            break

        if idx % sample_step == 0:
            res = desk_model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            boxes = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for b, s in zip(xyxy, confs):
                    boxes.append([float(x) for x in b] + [float(s)])

            if len(boxes) > best["count"]:
                best["count"] = len(boxes)
                best["frame"] = frame.copy()
                best["boxes"] = boxes
                best["frame_idx"] = idx

        idx += 1

    cap.release()
    return best


# =========================
# 座位网格构建
# =========================
class SeatGridBuilder:
    def __init__(self, num_rows=6, num_cols=5):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid_centers = None  # [R, C, 2]
        self.cell_w = None
        self.cell_h = None

    def build_from_desk_boxes(self, desk_boxes):
        """
        desk_boxes: [[x1,y1,x2,y2,score], ...]
        """
        if desk_boxes is None or len(desk_boxes) < 4:
            raise ValueError("桌子检测数量太少，无法建立座位网格。至少需要4个桌子。")

        centers = np.array([bbox_xyxy_to_center(b[:4]) for b in desk_boxes], dtype=np.float32)

        # 1) 按 y 聚成 6 行
        row_labels = cv_kmeans_1d(centers[:, 1], self.num_rows)
        rows = [[] for _ in range(self.num_rows)]
        for p, rl in zip(centers, row_labels):
            rows[rl].append(p)

        # 2) 每行按 x 排序
        rows = [sort_points_by_x(r) for r in rows]

        # 3) 提取四个角点（鲁棒补全）
        def leftmost(row):
            return np.array(row[0], dtype=np.float32) if len(row) > 0 else None

        def rightmost(row):
            return np.array(row[-1], dtype=np.float32) if len(row) > 0 else None

        # 顶行和底行可能聚类不完美，做一个容错搜索
        top_candidates = [r for r in rows[:max(2, self.num_rows // 3)] if len(r) > 0]
        bottom_candidates = [r for r in rows[-max(2, self.num_rows // 3):] if len(r) > 0]

        if len(top_candidates) == 0 or len(bottom_candidates) == 0:
            raise ValueError("行聚类失败，无法提取角点。")

        top_row = min(top_candidates, key=lambda r: np.mean(np.asarray(r)[:, 1]))
        bottom_row = max(bottom_candidates, key=lambda r: np.mean(np.asarray(r)[:, 1]))

        tl = leftmost(top_row)
        tr = rightmost(top_row)
        bl = leftmost(bottom_row)
        br = rightmost(bottom_row)

        if tl is None or tr is None or bl is None or br is None:
            raise ValueError("四角点提取失败，无法构建完整座位网格。")

        # 4) 双线性插值补出 6x5 网格中心
        self.grid_centers = bilinear_grid(
            tl, tr, bl, br,
            rows=self.num_rows,
            cols=self.num_cols
        )

        # 5) 估计局部网格尺度
        self.cell_w, self.cell_h = self._estimate_cell_size()

        return self.grid_centers

    def _estimate_cell_size(self):
        ws, hs = [], []

        for r in range(self.num_rows):
            for c in range(self.num_cols - 1):
                ws.append(l2(self.grid_centers[r, c], self.grid_centers[r, c + 1]))
        for r in range(self.num_rows - 1):
            for c in range(self.num_cols):
                hs.append(l2(self.grid_centers[r, c], self.grid_centers[r + 1, c]))

        cell_w = float(np.median(ws)) if len(ws) > 0 else 80.0
        cell_h = float(np.median(hs)) if len(hs) > 0 else 60.0
        return cell_w, cell_h

    def get_all_seats(self):
        seats = []
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                seats.append({
                    "seat_id": f"R{r+1}C{c+1}",
                    "row": r + 1,
                    "col": c + 1,
                    "center": self.grid_centers[r, c].copy()
                })
        return seats

    def nearest_seat(self, point, match_ratio=0.75):
        """
        point: [x, y]
        返回最近 seat_id, dist, 是否有效匹配
        """
        p = np.asarray(point, dtype=np.float32)
        min_dist = 1e9
        min_info = None

        for r in range(self.num_rows):
            for c in range(self.num_cols):
                center = self.grid_centers[r, c]
                d = l2(p, center)
                if d < min_dist:
                    min_dist = d
                    min_info = (r, c, center)

        th = match_ratio * max(self.cell_w, self.cell_h)
        valid = min_dist <= th
        r, c, center = min_info
        seat_id = f"R{r+1}C{c+1}"
        return seat_id, min_dist, valid

    def draw(self, frame):
        vis = frame.copy()
        # 画网格线
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                x, y = self.grid_centers[r, c].astype(int)
                cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)
                draw_text(vis, f"{r+1}-{c+1}", (x + 4, y - 4), color=(0, 255, 255), scale=0.5, thickness=1)

        for r in range(self.num_rows):
            for c in range(self.num_cols - 1):
                p1 = tuple(self.grid_centers[r, c].astype(int))
                p2 = tuple(self.grid_centers[r, c + 1].astype(int))
                cv2.line(vis, p1, p2, (255, 0, 0), 2)

        for r in range(self.num_rows - 1):
            for c in range(self.num_cols):
                p1 = tuple(self.grid_centers[r, c].astype(int))
                p2 = tuple(self.grid_centers[r + 1, c].astype(int))
                cv2.line(vis, p1, p2, (255, 0, 0), 2)

        return vis


# =========================
# 轨迹状态
# =========================
class TrackState:
    def __init__(self, track_id, vote_window=20):
        self.track_id = int(track_id)
        self.history_points = deque(maxlen=100)
        self.seat_votes = deque(maxlen=vote_window)
        self.last_box = None
        self.total_move = 0.0
        self.last_center = None
        self.stable_seat = None
        self.label = "unknown"
        self.unassigned_count = 0

    def update_box(self, box):
        self.last_box = box
        center = bbox_xyxy_to_center(box)
        self.history_points.append(center)

        if self.last_center is not None:
            self.total_move += l2(center, self.last_center)
        self.last_center = center

    def vote_seat(self, seat_id_or_none):
        self.seat_votes.append(seat_id_or_none)
        if seat_id_or_none is None:
            self.unassigned_count += 1
        else:
            self.unassigned_count = 0

    def update_stable_seat(self, min_count=8):
        votes = [v for v in self.seat_votes if v is not None]
        if len(votes) == 0:
            return None
        seat, count = Counter(votes).most_common(1)[0]
        if count >= min_count:
            self.stable_seat = seat
        return self.stable_seat

    def infer_role(self, teacher_unassigned_frames=30, teacher_move_threshold=80.0):
        """
        规则：
        - 稳定绑定某座位：student
        - 长时间没座位，且总移动很大：teacher
        """
        if self.stable_seat is not None:
            self.label = "student"
        else:
            if self.unassigned_count >= teacher_unassigned_frames and self.total_move >= teacher_move_threshold:
                self.label = "teacher"
            else:
                self.label = "unknown"
        return self.label


# =========================
# 主系统
# =========================
class SeatPersonRelationSystem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.desk_model = YOLO(cfg["desk_model_path"])
        self.person_model = YOLO(cfg["person_model_path"])

        self.grid_builder = SeatGridBuilder(
            num_rows=cfg["num_rows"],
            num_cols=cfg["num_cols"]
        )

        self.tracks = dict()
        self.records = []

    def build_seat_grid(self):
        print("[INFO] 正在建立座位网格...")
        best = best_frame_desk_boxes(
            video_path=self.cfg["video_path"],
            desk_model=self.desk_model,
            conf=self.cfg["desk_conf"],
            imgsz=self.cfg["imgsz"],
            max_frames=self.cfg["build_grid_frames"],
            sample_step=self.cfg["build_grid_sample_step"]
        )

        if best["frame"] is None or best["boxes"] is None or len(best["boxes"]) == 0:
            raise RuntimeError("未找到合适的桌子检测帧，无法建立座位网格。")

        print(f"[INFO] 建网格使用第 {best['frame_idx']} 帧，检测到桌子 {best['count']} 个")

        self.grid_builder.build_from_desk_boxes(best["boxes"])

        seat_map = self.grid_builder.draw(best["frame"])
        ensure_dir(self.cfg["output_seatmap_path"])
        cv2.imwrite(self.cfg["output_seatmap_path"], seat_map)
        print(f"[INFO] 座位图已保存到: {self.cfg['output_seatmap_path']}")

    def run(self):
        self.build_seat_grid()

        cap = cv2.VideoCapture(self.cfg["video_path"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ensure_dir(self.cfg["output_video_path"])
        writer = cv2.VideoWriter(
            self.cfg["output_video_path"],
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            vis = frame.copy()

            # 先把座位网格画上
            vis = self.grid_builder.draw(vis)

            # 人检测 + 跟踪
            results = self.person_model.track(
                frame,
                conf=self.cfg["person_conf"],
                imgsz=self.cfg["imgsz"],
                persist=True,
                tracker=self.cfg["tracker_cfg"],
                verbose=False
            )

            det = results[0]

            if det.boxes is not None and len(det.boxes) > 0:
                xyxy = det.boxes.xyxy.cpu().numpy()
                confs = det.boxes.conf.cpu().numpy()

                if det.boxes.id is not None:
                    ids = det.boxes.id.cpu().numpy().astype(int)
                else:
                    ids = np.arange(len(xyxy))

                for box, score, tid in zip(xyxy, confs, ids):
                    box = [float(v) for v in box]
                    x1, y1, x2, y2 = map(int, box)

                    if tid not in self.tracks:
                        self.tracks[tid] = TrackState(
                            track_id=tid,
                            vote_window=self.cfg["seat_vote_window"]
                        )

                    ts = self.tracks[tid]
                    ts.update_box(box)

                    # 用人体底部中心点去匹配座位
                    foot = bbox_bottom_center(box)
                    seat_id, dist, valid = self.grid_builder.nearest_seat(
                        foot,
                        match_ratio=self.cfg["seat_match_ratio"]
                    )

                    assigned_seat = seat_id if valid else None
                    ts.vote_seat(assigned_seat)
                    ts.update_stable_seat(min_count=self.cfg["seat_vote_min_count"])
                    role = ts.infer_role(
                        teacher_unassigned_frames=self.cfg["teacher_unassigned_frames"],
                        teacher_move_threshold=self.cfg["teacher_move_threshold"]
                    )

                    # 可视化颜色
                    if role == "student":
                        color = (0, 255, 0)
                    elif role == "teacher":
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 255)

                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(vis, tuple(foot.astype(int)), 4, color, -1)

                    text1 = f"ID:{tid} {role}"
                    text2 = f"seat:{ts.stable_seat if ts.stable_seat else 'None'}"

                    draw_text(vis, text1, (x1, max(20, y1 - 22)), color=color, scale=0.6, thickness=2)
                    draw_text(vis, text2, (x1, max(40, y1 - 2)), color=color, scale=0.55, thickness=2)

                    # 如果已经绑定座位，把人和座位中心连线
                    if ts.stable_seat is not None:
                        rr = int(ts.stable_seat.split("C")[0][1:]) - 1
                        cc = int(ts.stable_seat.split("C")[1]) - 1
                        center = self.grid_builder.grid_centers[rr, cc].astype(int)
                        cv2.line(vis, tuple(foot.astype(int)), tuple(center), color, 2)

                    # 记录结果
                    self.records.append({
                        "frame_idx": frame_idx,
                        "track_id": int(tid),
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                        "det_conf": float(score),
                        "foot_x": float(foot[0]),
                        "foot_y": float(foot[1]),
                        "seat_id_current": assigned_seat,
                        "seat_id_stable": ts.stable_seat,
                        "role": role,
                        "total_move": ts.total_move
                    })

            # 显示统计信息
            student_count = sum(1 for t in self.tracks.values() if t.label == "student")
            teacher_count = sum(1 for t in self.tracks.values() if t.label == "teacher")
            draw_text(vis, f"Frame: {frame_idx}", (20, 30), color=(255, 255, 255), scale=0.8, thickness=2)
            draw_text(vis, f"Students: {student_count}", (20, 60), color=(0, 255, 0), scale=0.8, thickness=2)
            draw_text(vis, f"Teachers: {teacher_count}", (20, 90), color=(0, 0, 255), scale=0.8, thickness=2)

            writer.write(vis)
            frame_idx += 1

        cap.release()
        writer.release()

        # 保存 CSV
        ensure_dir(self.cfg["output_csv_path"])
        df = pd.DataFrame(self.records)
        df.to_csv(self.cfg["output_csv_path"], index=False, encoding="utf-8-sig")

        print(f"[INFO] 结果视频已保存到: {self.cfg['output_video_path']}")
        print(f"[INFO] 跟踪结果 CSV 已保存到: {self.cfg['output_csv_path']}")


if __name__ == "__main__":
    system = SeatPersonRelationSystem(CONFIG)
    system.run()