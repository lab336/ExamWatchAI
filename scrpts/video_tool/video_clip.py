"""
视频剪辑工具
支持设置开始时间和持续时间对视频进行剪辑
# 基本使用（默认从0秒开始，裁剪10秒）
python detect/video_clip.py

# 自定义参数
# 从第2分30秒开始，剪辑30秒
python scrpts/video_tool/video_clip.py --start "2:30" --duration "0:30"

# 从第1分开始，剪辑1分钟
python scrpts/video_tool/video_clip.py --start "1:00" --duration "1:00"

# 混合使用 - 从120秒开始，剪辑2分30秒
python scrpts/video_tool/video_clip.py --start 120 --duration "2:30"

# 秒数格式（原有方式）
python scrpts/video_tool/video_clip.py --start 150 --duration 30
"""

import cv2
import argparse
import os
from pathlib import Path


def parse_time(time_str):
    """
    解析时间字符串
    支持格式：
    - 秒数: "120" 或 "120.5"
    - 分:秒: "2:30" (2分30秒) 或 "2:30.5"
    
    返回：总秒数（float）
    """
    if isinstance(time_str, (int, float)):
        return float(time_str)
    
    time_str = str(time_str).strip()
    
    # 检查是否包含冒号（分:秒格式）
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) != 2:
            raise ValueError(f"时间格式错误: {time_str}，应为 'MM:SS' 格式")
        
        try:
            minutes = int(parts[0])
            seconds = float(parts[1])
            total_seconds = minutes * 60 + seconds
            return total_seconds
        except ValueError:
            raise ValueError(f"时间格式错误: {time_str}，应为数字")
    else:
        # 纯秒数格式
        try:
            return float(time_str)
        except ValueError:
            raise ValueError(f"时间格式错误: {time_str}，应为秒数或 'MM:SS' 格式")


def clip_video(input_video, output_video, start_time, duration):
    """
    对视频进行剪辑
    
    Args:
        input_video: 输入视频路径
        output_video: 输出视频路径
        start_time: 开始时间（秒）
        duration: 持续时间（秒）
    """
    
    # 打开视频
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {input_video}")
        return False
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算开始帧和结束帧
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps)
    
    # 确保帧数在合理范围内
    start_frame = max(0, start_frame)
    end_frame = min(frame_count, end_frame)
    
    if start_frame >= frame_count:
        print(f"错误：开始时间超过视频总长度")
        return False
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_video) or '.', exist_ok=True)
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"错误：无法创建输出视频文件")
        return False
    
    # 设置起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = start_frame
    print(f"视频信息:")
    print(f"  总帧数: {frame_count}")
    print(f"  帧率: {fps}")
    print(f"  分辨率: {width}x{height}")
    print(f"  开始帧: {start_frame} ({start_time}s)")
    print(f"  结束帧: {end_frame} ({start_time + duration}s)")
    print(f"  处理帧数: {end_frame - start_frame}")
    print()
    
    # 逐帧写入
    print("正在处理视频...")
    while frame_idx < end_frame:
        ret, frame = cap.read()
        
        if not ret:
            print("警告：提前到达视频末尾")
            break
        
        out.write(frame)
        frame_idx += 1
        
        # 显示进度
        if (frame_idx - start_frame) % 30 == 0:
            progress = ((frame_idx - start_frame) / (end_frame - start_frame)) * 100
            print(f"进度: {progress:.1f}% ({frame_idx - start_frame}/{end_frame - start_frame})")
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"✓ 剪辑完成！已保存到: {output_video}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='视频剪辑工具',
        epilog='时间格式说明: --start "1:30" (1分30秒) 或 --start 90 (90秒)'
    )
    parser.add_argument('--input', type=str, default='data/origin_video/testdata.mp4',
                        help='输入视频路径')
    parser.add_argument('--output', type=str, default='data/ideotest/clipped_testdata2.mp4',
                        help='输出视频路径')
    parser.add_argument('--start', type=str, default='34:50',
                        help='开始时间 - 支持两种格式: 秒数(如120)或分:秒(如2:30)')
    parser.add_argument('--duration', type=str, default='0:30',
                        help='持续时间 - 支持两种格式: 秒数(如30)或分:秒(如0:30)')
    
    args = parser.parse_args()
    
    # 验证输入文件存在
    if not os.path.exists(args.input):
        print(f"错误：输入视频文件不存在: {args.input}")
        return
    
    # 解析时间参数
    try:
        start_time = parse_time(args.start)
        duration = parse_time(args.duration)
    except ValueError as e:
        print(f"错误：{e}")
        return
    
    # 验证参数
    if start_time < 0:
        print("错误：开始时间不能为负数")
        return
    
    if duration <= 0:
        print("错误：持续时间必须大于0")
        return
    
    # 执行剪辑
    clip_video(args.input, args.output, start_time, duration)


if __name__ == '__main__':
    main()
