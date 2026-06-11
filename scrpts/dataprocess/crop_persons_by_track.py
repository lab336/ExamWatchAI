"""
crop_persons_by_track.py
─────────────────────────
利用 YOLO 检测 + 自研稳定追踪器，将视频中每个人裁剪成独立视频。
每个稳定 ID 对应一个输出视频，内容为该人当前帧的裁剪区域（固定画布，居中贴图）。

核心设计：
  StableTracker —— 基于空间锚点的稳定追踪（多人场景优化）
    1. 预热阶段（前 --warmup-seconds 秒，默认5秒）：用 YOLO 追踪建立各人的初始锚点位置
    2. 锚定阶段：每帧检测结果与已知锚点做匈牙利匹配（基于中心距离 + 尺寸相似度）
       - 匹配距离 > --max-drift 倍 box 高度 → 拒绝，不更新该 ID
       - 匹配成功 → EMA 缓慢更新锚点，保持稳定
    3. **防重复建锚点**：新检测如果离任何已有锚点 < min_new_anchor_dist，
       拒绝建新 ID（防止一个人被拆分成多个 ID）
    4. 置信度系统：连续匹配 confidence +1，缺失 -2，高置信度 ID 不容易被抢走

用法:
    # 基础用法
    python scrpts/dataprocess/crop_persons_by_track.py \
        --source data/1.10/clip/clip_desk.mp4

    # 截取时间段（支持 HH:MM:SS / MM:SS / 秒数）
    python scrpts/dataprocess/crop_persons_by_track.py \
        --source data/1.10/clip/clip_desk.mp4 \
        --start 0:30 --end 3:00
    
    # 多人场景（30人+）推荐参数
    python scrpts/dataprocess/crop_persons_by_track.py \
        --source data/1.10/clipleft/clipped_testdata2.mp4 \
        --warmup-seconds 8 \
        --max-drift 0.2 \
        --min-new-anchor-dist 0.5 \
        --conf 0.28 \
        --display \
        --start 0:30 --end 3:00
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def parse_time(s: str) -> float:
    """将 'HH:MM:SS' / 'MM:SS' / '秒数' 解析为浮点秒数。"""
    parts = s.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    else:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def expand_box(x1, y1, x2, y2, pad, W, H):
    return (
        clamp(int(x1) - pad, 0, W - 1),
        clamp(int(y1) - pad, 0, H - 1),
        clamp(int(x2) + pad, 0, W),
        clamp(int(y2) + pad, 0, H),
    )


def box_center(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def place_on_canvas(crop: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    """将 crop 居中贴入固定尺寸黑色画布，不缩放。"""
    ch, cw = crop.shape[:2]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    off_y = (canvas_h - ch) // 2
    off_x = (canvas_w - cw) // 2
    paste_h = min(ch, canvas_h - off_y)
    paste_w = min(cw, canvas_w - off_x)
    canvas[off_y:off_y + paste_h, off_x:off_x + paste_w] = crop[:paste_h, :paste_w]
    return canvas


# ═══════════════════════════════════════════════════════════════
# 稳定追踪器
# ═══════════════════════════════════════════════════════════════

class StableTracker:
    """
    基于空间锚点的稳定追踪器（增强版：严格防 ID 漂移 + 防重复建锚点）。
    
    改进点：
      1. 代价函数同时考虑中心距离 + 尺寸相似度
      2. 锚点置信度机制：连续匹配次数越多，越不容易被"抢走"
      3. **防重复建锚点**：新检测如果离任何已有锚点太近（包括暂时缺失的），拒绝建新 ID
      4. 更严格的默认 max_drift（0.25 倍 box 高度）
    """

    def __init__(self, max_drift: float = 0.25, anchor_ema: float = 0.03,
                 max_missing_frames: int = 50, size_weight: float = 0.3,
                 min_new_anchor_dist: float = 0.4):
        """
        max_drift: 允许的最大偏移（相对于 box 高度的倍数，默认 0.25）
        anchor_ema: 锚点更新速率（0=完全固定, 1=每帧重置，默认 0.03）
        max_missing_frames: 连续缺失超过此帧数则认为该人已离开
        size_weight: 尺寸相似度在代价函数中的权重（默认 0.3）
        min_new_anchor_dist: 新锚点与已有锚点的最小距离（默认 0.4 倍 box 高度）
        """
        self.max_drift = max_drift
        self.ema = anchor_ema
        self.max_missing = max_missing_frames
        self.size_weight = size_weight
        self.min_new_anchor_dist = min_new_anchor_dist
        # stable_id -> {cx, cy, w, h, missing, confidence}
        self.anchors: dict[int, dict] = {}
        self._next_id = 1
        # 统计信息
        self.rejected_new_anchors = 0  # 被拒绝的新锚点数（太接近已有锚点）

    def _new_anchor(self, cx, cy, w, h) -> int:
        sid = self._next_id
        self._next_id += 1
        self.anchors[sid] = dict(
            cx=float(cx), cy=float(cy),
            w=float(w), h=float(h),
            missing=0,
            confidence=0,  # 连续匹配次数，越高越稳定
        )
        return sid

    def _cost(self, anchor: dict, cx, cy, w, h) -> float:
        """
        混合代价 = 归一化中心距离 + size_weight × 尺寸差异。
        
        尺寸差异 = |Δw/w_anchor| + |Δh/h_anchor|
        如果尺寸变化超过 50%，说明可能是另一个人。
        """
        # 中心距离（归一化到 box 高度）
        dx = anchor["cx"] - cx
        dy = anchor["cy"] - cy
        dist = (dx * dx + dy * dy) ** 0.5
        ref_h = max(anchor["h"], float(h), 1.0)
        norm_dist = dist / ref_h
        
        # 尺寸相似度（宽高比变化）
        aw, ah = anchor["w"], anchor["h"]
        dw = abs(float(w) - aw) / max(aw, 1.0)
        dh = abs(float(h) - ah) / max(ah, 1.0)
        size_diff = dw + dh  # 0~2+ 范围，越小越相似
        
        # 置信度惩罚：高置信度锚点（稳定跟踪）提高代价，不容易被"抢走"
        conf_penalty = 0.0
        if anchor["confidence"] > 20:  # 已连续跟踪 >20 帧
            conf_penalty = 0.1  # 增加一点代价，让新检测更难匹配到它
        
        return norm_dist + self.size_weight * size_diff + conf_penalty

    def update(self, detections: list) -> list:
        """
        detections: [(x1,y1,x2,y2), ...]
        返回:       [(stable_id, x1,y1,x2,y2), ...]
        """
        if not detections:
            for a in self.anchors.values():
                a["missing"] += 1
                a["confidence"] = max(0, a["confidence"] - 2)  # 缺失时降低置信度
            self._prune()
            return []

        det_info = []
        for d in detections:
            cx, cy = box_center(*d)
            w, h = d[2] - d[0], d[3] - d[1]
            det_info.append((cx, cy, w, h))

        sids = list(self.anchors.keys())
        results = []

        if not sids:
            for d, (cx, cy, w, h) in zip(detections, det_info):
                sid = self._new_anchor(cx, cy, w, h)
                results.append((sid, *d))
            return results

        # 构建代价矩阵 [#anchors × #detections]
        cost_matrix = np.full((len(sids), len(detections)), fill_value=1e6)
        for i, sid in enumerate(sids):
            for j, (cx, cy, w, h) in enumerate(det_info):
                cost_matrix[i, j] = self._cost(self.anchors[sid], cx, cy, w, h)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_anchor_set = set()
        matched_det_set = set()

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] > self.max_drift:
                continue
            sid = sids[i]
            cx, cy, w, h = det_info[j]
            a = self.anchors[sid]
            
            # EMA 更新锚点
            a["cx"] = (1 - self.ema) * a["cx"] + self.ema * cx
            a["cy"] = (1 - self.ema) * a["cy"] + self.ema * cy
            a["w"]  = (1 - self.ema) * a["w"]  + self.ema * w
            a["h"]  = (1 - self.ema) * a["h"]  + self.ema * h
            a["missing"] = 0
            a["confidence"] = min(100, a["confidence"] + 1)  # 连续匹配，置信度 +1（上限 100）
            
            results.append((sid, *detections[j]))
            matched_anchor_set.add(i)
            matched_det_set.add(j)

        # 未被匹配的锚点 missing+1，置信度下降
        for i, sid in enumerate(sids):
            if i not in matched_anchor_set:
                self.anchors[sid]["missing"] += 1
                self.anchors[sid]["confidence"] = max(0, self.anchors[sid]["confidence"] - 2)

        # 未被匹配的检测 → 检查是否应建新锚点
        for j, d in enumerate(detections):
            if j not in matched_det_set:
                cx, cy, w, h = det_info[j]
                
                # **关键防护**：如果离任何已有锚点太近（包括暂时缺失的），拒绝建新 ID
                too_close = False
                for a in self.anchors.values():
                    dx = a["cx"] - cx
                    dy = a["cy"] - cy
                    dist = (dx * dx + dy * dy) ** 0.5
                    ref_h = max(a["h"], float(h), 1.0)
                    if dist / ref_h < self.min_new_anchor_dist:
                        too_close = True
                        break
                
                if not too_close:
                    sid = self._new_anchor(cx, cy, w, h)
                    results.append((sid, *d))
                else:
                    self.rejected_new_anchors += 1
                # 否则：这个检测被忽略（可能是某个暂时缺失的锚点的真实检测，
                #       但因为距离稍远没匹配上；下一帧会重新尝试）

        self._prune()
        return results

    def _prune(self):
        dead = [sid for sid, a in self.anchors.items()
                if a["missing"] > self.max_missing]
        for sid in dead:
            del self.anchors[sid]

    def seed_from_yolo(self, yolo_result) -> list:
        """预热阶段：用 YOLO track_id 直接初始化/更新锚点。"""
        boxes = yolo_result.boxes
        if boxes is None or boxes.id is None:
            return []
        ids = boxes.id.cpu().numpy().astype(int)
        xyxys = boxes.xyxy.cpu().numpy()
        results = []
        for tid, (x1, y1, x2, y2) in zip(ids, xyxys):
            cx, cy = box_center(x1, y1, x2, y2)
            w, h = x2 - x1, y2 - y1
            if tid not in self.anchors:
                self.anchors[tid] = dict(
                    cx=float(cx), cy=float(cy),
                    w=float(w), h=float(h),
                    missing=0,
                    confidence=1,  # 预热阶段初始置信度
                )
                if tid >= self._next_id:
                    self._next_id = tid + 1
            else:
                a = self.anchors[tid]
                a["cx"] = (1 - self.ema) * a["cx"] + self.ema * cx
                a["cy"] = (1 - self.ema) * a["cy"] + self.ema * cy
                a["w"]  = (1 - self.ema) * a["w"]  + self.ema * w
                a["h"]  = (1 - self.ema) * a["h"]  + self.ema * h
                a["missing"] = 0
                a["confidence"] = min(100, a["confidence"] + 1)
            results.append((int(tid), int(x1), int(y1), int(x2), int(y2)))
        return results


# ═══════════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════════

def run(args):
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"找不到输入文件: {source}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    device = args.device if args.device else None
    print(f"[INFO] 已加载模型: {args.weights}")

    cap_tmp = cv2.VideoCapture(str(source))
    W           = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    H           = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_tmp.release()

    start_sec   = parse_time(args.start) if args.start else 0.0
    end_sec     = parse_time(args.end)   if args.end   else total_frames / fps
    
    # 校验时间范围
    video_duration = total_frames / fps
    if start_sec >= video_duration:
        raise ValueError(
            f"开始时间 {start_sec:.1f}s 超出视频长度 {video_duration:.1f}s！"
        )
    if end_sec > video_duration:
        print(f"[警告] 结束时间 {end_sec:.1f}s 超出视频长度，已自动调整为 {video_duration:.1f}s")
        end_sec = video_duration
    if start_sec >= end_sec:
        raise ValueError(
            f"开始时间 {start_sec:.1f}s >= 结束时间 {end_sec:.1f}s，无效范围！"
        )
    
    start_frame = max(0, int(start_sec * fps))
    end_frame   = min(total_frames, int(end_sec * fps))
    warmup_frames = int(args.warmup_seconds * fps)
    span = end_frame - start_frame

    print(f"[INFO] 视频: {W}x{H} @ {fps:.1f}fps，共 {total_frames} 帧")
    print(f"[INFO] 处理范围: {start_sec:.1f}s ~ {end_sec:.1f}s "
          f"（第 {start_frame} ~ {end_frame} 帧，共 {span} 帧）")
    print(f"[INFO] 预热帧数: {warmup_frames}（{args.warmup_seconds:.1f}s）")

    pad = args.padding
    tracker = StableTracker(
        max_drift=args.max_drift,
        anchor_ema=args.anchor_ema,
        max_missing_frames=int(fps * 2),
        min_new_anchor_dist=args.min_new_anchor_dist,
    )
    track_boxes: dict[int, list] = defaultdict(list)

    # ── 第一遍：稳定追踪，收集坐标 ──────────────────────────────
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print("[第一遍] 收集稳定轨迹 ...")
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        local_idx = frame_idx - start_frame
        in_warmup = local_idx < warmup_frames

        result = model.track(
            frame,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            classes=[0],
            device=device,
            verbose=False,
        )[0]

        if in_warmup:
            assigned = tracker.seed_from_yolo(result)
        else:
            boxes = result.boxes
            dets = []
            if boxes is not None and len(boxes):
                dets = [(int(x1), int(y1), int(x2), int(y2))
                        for x1, y1, x2, y2 in boxes.xyxy.cpu().numpy()]
            assigned = tracker.update(dets)

        for sid, x1, y1, x2, y2 in assigned:
            ex = expand_box(x1, y1, x2, y2, pad, W, H)
            if ex[2] - ex[0] > 0 and ex[3] - ex[1] > 0:
                track_boxes[sid].append((frame_idx, *ex))

        # 实时显示（第一遍）
        if args.display:
            vis = frame.copy()
            for sid, x1, y1, x2, y2 in assigned:
                conf = tracker.anchors.get(sid, {}).get("confidence", 0)
                # 置信度越高，颜色越绿；越低越红
                if conf > 50:
                    color = (0, 255, 0)      # 绿：高置信度
                elif conf > 20:
                    color = (0, 255, 255)    # 黄：中置信度
                else:
                    color = (0, 128, 255)    # 橙：低置信度
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"ID{sid} C{conf}"
                cv2.putText(vis, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            phase = "预热" if in_warmup else "追踪"
            cv2.putText(vis, f"[{phase}] Frame {local_idx+1}/{span}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("StableTracker", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if (local_idx + 1) % 100 == 0:
            # 统计置信度分布
            high_conf = sum(1 for a in tracker.anchors.values() if a["confidence"] > 50)
            med_conf  = sum(1 for a in tracker.anchors.values() if 20 < a["confidence"] <= 50)
            low_conf  = sum(1 for a in tracker.anchors.values() if a["confidence"] <= 20)
            # 统计已积累帧数
            valid_tracks = sum(1 for b in track_boxes.values() if len(b) >= args.min_frames)
            print(f"  {local_idx + 1}/{span} 帧 "
                  f"({(local_idx+1)/span*100:.0f}%)，"
                  f"活跃ID: {len(tracker.anchors)} "
                  f"[高信度:{high_conf} 中:{med_conf} 低:{low_conf}]，"
                  f"有效轨迹: {valid_tracks}/{len(track_boxes)}")

    cap.release()

    # 统计信息
    print(f"[统计] 拒绝建立新锚点次数: {tracker.rejected_new_anchors}（防止重复 ID）")
    
    # 过滤短轨迹
    valid_ids = {tid for tid, b in track_boxes.items() if len(b) >= args.min_frames}
    print(f"[INFO] 有效轨迹（≥{args.min_frames}帧）: {len(valid_ids)} 个，"
          f"丢弃 {len(track_boxes) - len(valid_ids)} 个短轨迹")

    # 固定画布尺寸
    canvas_size: dict[int, tuple] = {}
    for tid in valid_ids:
        boxes = track_boxes[tid]
        mw = max(x2 - x1 for _, x1, y1, x2, y2 in boxes)
        mh = max(y2 - y1 for _, x1, y1, x2, y2 in boxes)
        canvas_size[tid] = (mw + mw % 2, mh + mh % 2)

    # ── 第二遍：按坐标裁剪写入（无需重跑推理）──────────────────
    frame_lookup: dict[int, list] = defaultdict(list)
    for tid in valid_ids:
        for fidx, x1, y1, x2, y2 in track_boxes[tid]:
            frame_lookup[fidx].append((tid, x1, y1, x2, y2))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {tid: cv2.VideoWriter(
                    str(output_dir / f"person_{tid:04d}.mp4"),
                    fourcc, fps, canvas_size[tid])
               for tid in valid_ids}
    frame_count: dict[int, int] = defaultdict(int)

    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print("[第二遍] 裁剪写入视频 ...")
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        local_idx = frame_idx - start_frame

        if args.display:
            vis = frame.copy()
            for tid, x1, y1, x2, y2 in frame_lookup.get(frame_idx, []):
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"ID{tid}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis, f"[裁剪] Frame {local_idx+1}/{span}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("StableTracker", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        for tid, x1, y1, x2, y2 in frame_lookup.get(frame_idx, []):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cw, ch = canvas_size[tid]
            writers[tid].write(place_on_canvas(crop, cw, ch))
            frame_count[tid] += 1

        if (local_idx + 1) % 100 == 0:
            print(f"  {local_idx + 1}/{span} 帧 ({(local_idx+1)/span*100:.0f}%)")

    cap.release()
    if args.display:
        cv2.destroyAllWindows()
    for w in writers.values():
        w.release()

    removed = 0
    for tid in valid_ids:
        if frame_count[tid] < args.min_frames:
            (output_dir / f"person_{tid:04d}.mp4").unlink(missing_ok=True)
            removed += 1

    kept = len(valid_ids) - removed
    print(f"\n[完成] 保留 {kept} 个人物视频，输出目录: {output_dir.resolve()}")


# ═══════════════════════════════════════════════════════════════
# 参数解析 & 入口
# ═══════════════════════════════════════════════════════════════

def parse_args():
    project_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(
        description="YOLO + 稳定追踪器：将视频中每个人裁剪为独立视频（防 ID 漂移）"
    )
    parser.add_argument("--source", required=True, help="输入视频路径")
    parser.add_argument("--weights",
                        default=str(project_root / "model" / "best2.pt"))
    parser.add_argument("--output",
                        default=str(project_root / "output" / "cropped_persons"))
    parser.add_argument("--start", default="",
                        help="开始时间 HH:MM:SS / MM:SS / 秒（默认0）")
    parser.add_argument("--end", default="",
                        help="结束时间，格式同上（默认视频末尾）")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLO 检测置信度阈值（默认0.3，多人场景建议0.25-0.3）")
    parser.add_argument("--iou",  type=float, default=0.5)
    parser.add_argument("--padding", type=int, default=20, help="边框扩展像素")
    parser.add_argument("--warmup-seconds", type=float, default=5.0,
                        help="预热时长（秒），用 YOLO 原生 ID 建立锚点（默认5.0，多人场景建议≥5）")
    parser.add_argument("--max-drift", type=float, default=0.25,
                        help="最大允许漂移（box 高度倍数，默认0.25，越小越严格）")
    parser.add_argument("--anchor-ema", type=float, default=0.03,
                        help="锚点 EMA 更新系数，越小越稳定（默认0.03）")
    parser.add_argument("--min-new-anchor-dist", type=float, default=0.4,
                        help="新锚点与已有锚点最小距离（box 高度倍数，默认0.4，防止重复建锚点）")
    parser.add_argument("--min-frames", type=int, default=15,
                        help="最少帧数阈值（默认15）")
    parser.add_argument("--device", default="", help="推理设备 cpu/0/cuda:0")
    parser.add_argument("--display", action="store_true", help="实时显示画面")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
