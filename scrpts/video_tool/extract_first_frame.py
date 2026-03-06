"""
视频首帧提取工具
从视频中提取第一帧并保存为图片
"""

import cv2
import argparse
import os
from pathlib import Path


def extract_first_frame(video_path, output_path):
    """
    提取视频的第一帧
    
    Args:
        video_path: 输入视频路径
        output_path: 输出图片路径
    """
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 错误：无法打开视频文件 {video_path}")
        return False
    
    # 读取第一帧
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ 错误：无法读取视频帧")
        return False
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # 保存第一帧
    success = cv2.imwrite(output_path, frame)
    
    if success:
        height, width = frame.shape[:2]
        print(f"✓ 成功提取首帧！")
        print(f"  分辨率: {width}x{height}")
        print(f"  保存位置: {output_path}")
        return True
    else:
        print(f"❌ 错误：无法保存图片到 {output_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description='视频首帧提取工具')
    parser.add_argument('--input', type=str, default='data/clip_video/clipped_00000000028000000.mp4',
                        help='输入视频路径')
    parser.add_argument('--output', type=str, default='data/first_frame.png',
                        help='输出图片路径')
    
    args = parser.parse_args()
    
    # 验证输入文件存在
    if not os.path.exists(args.input):
        print(f"❌ 错误：输入视频文件不存在: {args.input}")
        return
    
    # 执行提取
    extract_first_frame(args.input, args.output)


if __name__ == '__main__':
    main()
