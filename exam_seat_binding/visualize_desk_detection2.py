"""
独立的YOLO11人群检测脚本
可以将此脚本和模型权重移动到任何位置使用
只需要: pip install ultralytics opencv-python

使用方法:
1. 检测图片: python exam_seat_binding/visualize_desk_detection2.py --source data/my_screenshots/screenshot_00m00s400ms_frame000010.png --weights exam_seat_binding/weight/yolo11desk.pt
2. 检测视频: python exam_seat_binding/standalone_detect.py --source video.mp4 --weights exam_seat_binding/weight/yolo11desk.pt
3. 检测摄像头: python exam_seat_binding/standalone_detect.py --source 0 --weights exam_seat_binding/weight/yolo11desk.pt
4. 检测文件夹: python exam_seat_binding/standalone_detect.py --source data/desk/originimages --weights exam_seat_binding/weight/yolo11desk.pt
5. 使用CPU: python exam_seat_binding/standalone_detect.py --source video.mp4 --weights exam_seat_binding/weight/yolo11desk.pt --device cpu
6. 缩放图像: python exam_seat_binding/standalone_detect.py --source video.mp4 --weights exam_seat_binding/weight/yolo11desk.pt --img-size 1280
7. 启用桌子排列策略(自动分列编号): python exam_seat_binding/visualize_desk_detection.py --source data/my_screenshots --weights exam_seat_binding/weight/yolo11desk.pt --enable-desk-layout
8. 指定列数: python exam_seat_binding/standalone_detect.py --source data/desk/originimages --weights exam_seat_binding/weight/yolo11desk.pt --enable-desk-layout --num-cols 5
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple


# ========== 桌子排列策略相关函数 ==========

def fit_line_kb_positive(points: np.ndarray, min_k: float = 0.02) -> Tuple[float, float]:
    """拟合 y_up = kx + b（数学坐标，y_up=-y_img），并强制 k>0。返回 (k, b)。"""
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
    """将 y_up=kx+b 转到图像坐标 y_img=-kx-b，并裁剪到图像边界。"""
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


def _pick_max_y_head(centers: np.ndarray, remaining: set) -> int:
    """按 y 值降序选择列头（y 最大为底部），y 相同时 x 最小作为备选。"""
    if len(remaining) == 0:
        return -1
    cand = list(remaining)
    # 按 y 降序（底部优先），y 相等时按 x 升序（更左优先）
    cand_sorted = sorted(cand, key=lambda i: (-float(centers[i, 1]), float(centers[i, 0])))
    return cand_sorted[0]


def _pick_next_in_same_column(centers: np.ndarray, remaining: set, cur_idx: int) -> int:
    """同列找下一个点：严格要求 x 变小且 y 变小。

    选择规则更新为：在满足 x<cx 且 y<cy 的候选中，优先选择 y 与当前点差值最小的点（即垂直方向上距离最近的上方点），
    若 y 差值相同再按欧氏距离排序，最后按 x 值升序（更左优先）。"""
    cx, cy = centers[cur_idx]

    # 严格条件：x<cx 且 y<cy
    cand = [i for i in remaining if centers[i, 0] < cx and centers[i, 1] < cy]
    if len(cand) > 0:
        # 计算 y 差值 dy = cy - yi（>0），优先 dy 最小；再按距离；再按 x 更小优先
        cand = sorted(
            cand,
            key=lambda i: (
                float(cy - centers[i, 1]),
                np.linalg.norm(centers[i] - centers[cur_idx]),
                float(centers[i, 0]),
            ),
        )
        return cand[0]

    return -1


def split_into_columns_by_origin_walk(
    centers: np.ndarray,
    num_cols: int = 5,
    expected_per_col: int = 6,
) -> Tuple[List[List[int]], List[Tuple[float, float]], List[int]]:
    """桌子分列算法（原点距离+方向约束）。"""
    n = len(centers)
    if n == 0:
        return [[] for _ in range(num_cols)], [], []

    remaining = set(range(n))
    columns: List[List[int]] = [[] for _ in range(num_cols)]
    seeds: List[int] = []

    # 第一列头点：从所有点中取 y 最大者（画面最底端的点）
    head = _pick_max_y_head(centers, remaining)

    for c in range(num_cols):
        if head < 0:
            break

        col = [head]
        if head in remaining:
            remaining.remove(head)

        cur = head
        # 同列补到 expected_per_col 点 (查找方向：y 减小，x 减小)
        for _ in range(expected_per_col - 1):
            nxt = _pick_next_in_same_column(centers, remaining, cur)
            if nxt < 0:
                break
            col.append(nxt)
            remaining.remove(nxt)
            cur = nxt

        columns[c] = col
        seeds.append(col[0])

        # 下一列头点：从剩余点中继续选出 y 最大者
        if c < num_cols - 1:
            head = _pick_max_y_head(centers, remaining)

    # 将剩余点分配到最近列线
    lines: List[Tuple[float, float]] = []
    for c in range(num_cols):
        if len(columns[c]) > 0:
            pts = centers[np.array(columns[c], dtype=np.int32)]
            lines.append(fit_line_kb_positive(pts, min_k=0.02))
        else:
            lines.append((0.02, 0.0))

    # 分配剩余点
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

    # 列内排序：底->顶（当 y 减小时，x 也应随之减小）
    # 优先按 y 降序（底->顶），同 y 时按 x 降序，保证从底到顶 x 也呈减小趋势
    for c in range(num_cols):
        columns[c] = sorted(columns[c], key=lambda i: (centers[i, 1], centers[i, 0]), reverse=True)

    # 保持列顺序：按“列头 y 从大到小”构建出来的自然顺序，不再按 x 二次重排

    seeds = []
    for c in range(num_cols):
        if len(columns[c]) > 0:
            seeds.append(columns[c][0])
            lines[c] = fit_line_kb_positive(centers[np.array(columns[c], dtype=np.int32)], min_k=0.02)
        else:
            seeds.append(0)

    return columns, lines, seeds


class StandaloneDetector:
    """独立的YOLO检测器（含桌子排列策略）"""
    
    def __init__(self, weights_path, conf_threshold=0.25, iou_threshold=0.45, device='', img_size=None, half=False, enable_desk_layout=False, num_cols=5):
        """
        初始化检测器
        
        Args:
            weights_path: 模型权重文件路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IOU阈值
            device: 设备选择 ('cpu', 'cuda', '0', '1' 等)
            img_size: 推理图像尺寸，如果为None则使用原始尺寸
            half: 是否使用半精度(FP16)推理，可以减少显存并避免某些CUDA错误
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")
        
        print(f"正在加载模型: {weights_path}")
        
        # 清理CUDA缓存
        if self._check_cuda():
            self._clear_cuda_cache()
        
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device if device else ('cuda' if self._check_cuda() else 'cpu')
        self.img_size = img_size
        self.half = half and 'cuda' in str(self.device)  # 只在GPU上使用半精度
        self.enable_desk_layout = enable_desk_layout  # 是否启用桌子排列策略
        self.num_cols = num_cols  # 列数
        
        # 再次清理CUDA缓存
        if 'cuda' in str(self.device):
            self._clear_cuda_cache()
        
        print(f"模型加载成功!")
        print(f"使用设备: {self.device}")
        if self.img_size:
            print(f"推理图像尺寸: {self.img_size}")
        if self.half:
            print(f"使用半精度(FP16)推理")
        
        # 显示GPU信息
        if 'cuda' in str(self.device):
            self._print_gpu_info()
    
    def _check_cuda(self):
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _clear_cuda_cache(self):
        """清理CUDA缓存"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                print("已清理CUDA缓存")
        except Exception:
            pass
    
    def _print_gpu_info(self):
        """打印GPU信息"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_id = 0 if self.device == 'cuda' else int(self.device)
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
        
    def detect_image(self, image_path, save_dir="output"):
        """
        检测单张图片
        
        Args:
            image_path: 图片路径
            save_dir: 结果保存目录
        """
        if not os.path.exists(image_path):
            print(f"警告: 图片不存在 - {image_path}")
            return None
        
        # 创建输出目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 运行检测
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=False,
            verbose=False,
            device=self.device,
            imgsz=self.img_size or 640,
            half=self.half
        )
        
        # 获取结果
        result = results[0]
        boxes = result.boxes
        
        # 读取原图
        img = cv2.imread(image_path)
        
        # 如果启用桌子排列策略
        if self.enable_desk_layout and len(boxes) > 0:
            # 提取中心点
            centers = []
            for box in boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                centers.append([cx, cy])
            centers = np.array(centers, dtype=np.float32)
            
            # 执行分列算法
            columns, lines, _ = split_into_columns_by_origin_walk(centers, num_cols=self.num_cols)
            
            # 绘制带排列策略的结果
            annotated_img = self._draw_boxes_with_layout(img, boxes, columns, lines)
            
            # 打印分列信息
            for c, idxs in enumerate(columns):
                print(f"第{c+1}列: {len(idxs)}个桌子")
        else:
            # 绘制普通检测框
            annotated_img = self._draw_boxes(img, boxes)
        
        # 保存结果
        filename = Path(image_path).name
        suffix = "_layout" if self.enable_desk_layout else ""
        output_path = os.path.join(save_dir, f"detected{suffix}_{filename}")
        cv2.imwrite(output_path, annotated_img)
        
        # 打印检测信息
        num_detections = len(boxes)
        print(f"检测到 {num_detections} 个目标 - 保存至: {output_path}")
        
        return annotated_img, boxes
    
    def detect_video(self, video_path, save_dir="output", display=False):
        """
        检测视频
        
        Args:
            video_path: 视频路径 (或摄像头ID, 如0)
            save_dir: 结果保存目录
            display: 是否实时显示
        """
        # 打开视频
        if isinstance(video_path, int) or video_path.isdigit():
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
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")
        
        # 创建输出目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置视频写入器
        output_path = os.path.join(save_dir, f"detected_{video_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 运行检测
                try:
                    results = self.model.predict(
                        source=frame,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        save=False,
                        verbose=False,
                        device=self.device,
                        imgsz=self.img_size or 640,
                        half=self.half
                    )
                except Exception as e:
                    print(f"\n检测失败 (Frame {frame_count}): {e}")
                    print("提示: 如果是CUDA内存错误,请尝试:")
                    print("  1. 使用 --device cpu 切换到CPU")
                    print("  2. 使用 --img-size 640 或更小的值来缩放图像")
                    print("  3. 使用 --half 启用FP16半精度推理")
                    raise
                
                # 获取结果并绘制
                result = results[0]
                boxes = result.boxes
                annotated_frame = self._draw_boxes(frame, boxes)
                
                # 添加帧信息
                num_detections = len(boxes)
                info_text = f"Frame: {frame_count}/{total_frames} | Detections: {num_detections}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 写入视频
                out.write(annotated_frame)
                
                # 显示
                if display:
                    cv2.imshow('Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户中断检测")
                        break
                
                # 打印进度
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"处理进度: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        finally:
            cap.release()
            out.release()
            if display:
                cv2.destroyAllWindows()
        
        print(f"视频检测完成! 保存至: {output_path}")
    
    def detect_folder(self, folder_path, save_dir="output"):
        """
        检测文件夹中的所有图片
        
        Args:
            folder_path: 图片文件夹路径
            save_dir: 结果保存目录
        """
        if not os.path.exists(folder_path):
            print(f"错误: 文件夹不存在 - {folder_path}")
            return
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 获取所有图片
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f"*{ext}"))
            image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"警告: 文件夹中没有找到图片文件 - {folder_path}")
            return
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 批量检测
        for idx, image_file in enumerate(image_files, 1):
            print(f"\n处理 [{idx}/{len(image_files)}]: {image_file.name}")
            self.detect_image(str(image_file), save_dir)
        
        print(f"\n所有图片检测完成! 结果保存至: {save_dir}")
    
    def _draw_boxes(self, img, boxes):
        """
        在图片上绘制检测框
        
        Args:
            img: 原始图片
            boxes: 检测框
        """
        annotated_img = img.copy()
        
        for box in boxes:
            # 获取坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # 获取类别名称
            class_name = self.model.names[cls] if hasattr(self.model, 'names') else str(cls)
            
            # 绘制框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 标签背景
            cv2.rectangle(
                annotated_img,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # 标签文字
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return annotated_img
    
    def _draw_boxes_with_layout(self, img, boxes, columns, lines):
        """绘制带桌子排列策略的检测框"""
        annotated_img = img.copy()
        h, w = annotated_img.shape[:2]
        
        col_colors = [
            (255, 80, 80),   # 红色
            (80, 255, 80),   # 绿色
            (80, 80, 255),   # 蓝色
            (255, 200, 80),  # 橙色
            (220, 80, 255),  # 紫色
        ]
        
        # 先画灰色底框
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (120, 120, 120), 1)
        
        # 绘制拟合直线
        for c, line_kb in enumerate(lines):
            if len(columns[c]) > 0:
                color = col_colors[c % len(col_colors)]
                pt1, pt2 = clip_line_kb_to_image(line_kb, w, h)
                cv2.line(annotated_img, pt1, pt2, color, 1)
        
        # 按列绘制
        for col_id, idxs in enumerate(columns):
            if len(idxs) == 0:
                continue
            color = col_colors[col_id % len(col_colors)]
            
            # 获取排序后的索引（从底到顶）
            boxes_list = list(boxes)
            centers = []
            for idx in idxs:
                box = boxes_list[idx]
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                # 存为 (cy, cx)，便于按 (y, x) 组合排序
                centers.append((cy, cx))

            # 按 y 降序（底->顶），同 y 时按 x 降序，保证 y 减小时 x 也随之减小
            sorted_pairs = sorted(zip(idxs, centers), key=lambda p: (p[1][0], p[1][1]), reverse=True)
            sorted_idxs = [p[0] for p in sorted_pairs]
            
            # 座位编号：第1列 1~6；第2列 7~12；...
            for row_id, idx in enumerate(sorted_idxs):
                seat_no = col_id * 6 + (row_id + 1)
                box = boxes_list[idx]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                
                # 绘制框
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
                # 绘制座位编号和置信度
                label = f"{seat_no}: {conf:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # 标签背景
                cv2.rectangle(
                    annotated_img,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                
                # 标签文字
                cv2.putText(
                    annotated_img,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
            
            # 相邻方框连线（前框左下角 -> 后框右上角）
            for j in range(len(sorted_idxs) - 1):
                idx_a = sorted_idxs[j]
                idx_b = sorted_idxs[j + 1]
                
                box_a = boxes_list[idx_a]
                box_b = boxes_list[idx_b]
                
                ax1, ay1, ax2, ay2 = map(int, box_a.xyxy[0].tolist())
                bx1, by1, bx2, by2 = map(int, box_b.xyxy[0].tolist())
                
                # 从当前框的左下角连到下一个框的左下角
                cv2.line(annotated_img, (ax1, ay2), (bx1, by2), color, 2)
                # 从当前框的右上角连到下一个框的右上角
                cv2.line(annotated_img, (ax2, ay1), (bx2, by1), color, 2)
            
            # 列名标注
            if len(sorted_idxs) > 0:
                top_idx = sorted_idxs[-1]
                box_top = boxes_list[top_idx]
                x1, y1, x2, y2 = map(int, box_top.xyxy[0].tolist())
                tx = (x1 + x2) // 2
                ty = y1
                label = f"C{col_id + 1}({len(sorted_idxs)})"
                cv2.putText(
                    annotated_img,
                    label,
                    (tx + 8, max(20, ty - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        
        return annotated_img


def main():
    parser = argparse.ArgumentParser(description="YOLO11独立检测脚本")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="检测源: 图片路径/视频路径/文件夹路径/摄像头ID(0)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best.pt",
        help="模型权重文件路径 (默认: best.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.7,
        help="置信度阈值 (默认: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS的IOU阈值 (默认: 0.45)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="结果保存目录 (默认: output)"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="实时显示检测结果(仅视频)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="运行设备: cpu, cuda, 0, 1 等 (默认: 自动选择)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="推理图像尺寸，缩放以节省内存 (如: 640, 1280)"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="使用FP16半精度推理(仅GPU,可减少显存并可能避免某些CUDA错误)"
    )
    parser.add_argument(
        "--enable-desk-layout",
        action="store_true",
        help="启用桌子排列策略（自动分列、编号和连线）"
    )
    parser.add_argument(
        "--num-cols",
        type=int,
        default=5,
        help="桌子列数 (默认: 5)"
    )
    
    args = parser.parse_args()
    
    # 检查权重文件
    if not os.path.exists(args.weights):
        print(f"错误: 权重文件不存在 - {args.weights}")
        print(f"\n提示: 请确保权重文件在当前目录或提供完整路径")
        print(f"例如: python {sys.argv[0]} --source image.jpg --weights /path/to/best.pt")
        return
    
    # 创建检测器
    try:
        detector = StandaloneDetector(
            weights_path=args.weights,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            img_size=args.img_size,
            half=args.half,
            enable_desk_layout=args.enable_desk_layout,
            num_cols=args.num_cols
        )
    except Exception as e:
        print(f"\n初始化检测器失败: {e}")
        return
    
    # 判断输入类型并执行检测
    source = args.source
    
    # 检查是否为摄像头
    if source.isdigit():
        print(f"\n开始检测摄像头: {source}")
        detector.detect_video(source, args.output, args.display)
    
    # 检查是否为文件夹
    elif os.path.isdir(source):
        print(f"\n开始检测文件夹: {source}")
        detector.detect_folder(source, args.output)
    
    # 检查是否为视频文件
    elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
        print(f"\n开始检测视频: {source}")
        detector.detect_video(source, args.output, args.display)
    
    # 否则作为图片处理
    elif os.path.isfile(source):
        print(f"\n开始检测图片: {source}")
        detector.detect_image(source, args.output)
    
    else:
        print(f"错误: 无法识别的输入源 - {source}")
        print("支持的输入:")
        print("  - 图片文件: .jpg, .png, .bmp 等")
        print("  - 视频文件: .mp4, .avi, .mov 等")
        print("  - 文件夹: 包含图片的文件夹")
        print("  - 摄像头: 数字ID (如 0)")


if __name__ == "__main__":
    main()
