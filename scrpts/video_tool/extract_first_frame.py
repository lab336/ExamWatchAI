"""
视频首帧提取工具
从视频中提取第一帧并保存为图片
支持OpenCV和FFmpeg两种方案
"""

import cv2
import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def extract_first_frame_opencv(video_path, output_path):
    """
    使用OpenCV提取视频的第一帧
    
    Args:
        video_path: 输入视频路径
        output_path: 输出图片路径
    
    Returns:
        (success, message)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False, "OpenCV: 无法打开视频文件"
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False, "OpenCV: 无法读取视频帧"
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        success = cv2.imwrite(output_path, frame)
        
        if success:
            height, width = frame.shape[:2]
            return True, f"OpenCV: {width}x{height}"
        else:
            return False, "OpenCV: 无法保存图片"
    except Exception as e:
        return False, f"OpenCV异常: {str(e)}"


def extract_first_frame_ffmpeg(video_path, output_path):
    """
    使用FFmpeg提取视频的第一帧（后备方案）
    
    Args:
        video_path: 输入视频路径
        output_path: 输出图片路径
    
    Returns:
        (success, message)
    """
    try:
        # 检查ffmpeg是否可用
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "FFmpeg: ffmpeg命令不可用"
    
    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # 使用ffmpeg提取第一帧：在第0秒处提取，分辨率限制在合理范围
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', 'select=eq(n\\,0)',
            '-q:v', '2',
            '-y',
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return False, f"FFmpeg失败: {result.stderr[:100]}"
        
        if os.path.exists(output_path):
            img = cv2.imread(output_path)
            if img is not None:
                height, width = img.shape[:2]
                return True, f"FFmpeg: {width}x{height}"
        
        return False, "FFmpeg: 输出文件不存在"
    except subprocess.TimeoutExpired:
        return False, "FFmpeg: 超时"
    except Exception as e:
        return False, f"FFmpeg异常: {str(e)}"


def extract_first_frame(video_path, output_path, use_ffmpeg_first=False):
    """
    提取视频的第一帧，支持多种方案
    
    Args:
        video_path: 输入视频路径
        output_path: 输出图片路径
        use_ffmpeg_first: 是否优先使用FFmpeg
    
    Returns:
        success (bool)
    """
    if use_ffmpeg_first:
        # 优先FFmpeg，回退到OpenCV
        success, msg = extract_first_frame_ffmpeg(video_path, output_path)
        if success:
            print(f"✓ 成功提取首帧（使用FFmpeg）！")
            print(f"  {msg}")
            print(f"  保存位置: {output_path}")
            return True
        else:
            print(f"⚠ FFmpeg方案失败: {msg}")
            success, msg = extract_first_frame_opencv(video_path, output_path)
            if success:
                print(f"✓ 成功提取首帧（使用OpenCV后备）！")
                print(f"  {msg}")
                print(f"  保存位置: {output_path}")
                return True
            else:
                print(f"❌ OpenCV后备方案也失败: {msg}")
                return False
    else:
        # 优先OpenCV，回退到FFmpeg
        success, msg = extract_first_frame_opencv(video_path, output_path)
        if success:
            print(f"✓ 成功提取首帧（使用OpenCV）！")
            print(f"  {msg}")
            print(f"  保存位置: {output_path}")
            return True
        else:
            print(f"⚠ OpenCV方案失败: {msg}")
            success, msg = extract_first_frame_ffmpeg(video_path, output_path)
            if success:
                print(f"✓ 成功提取首帧（使用FFmpeg后备）！")
                print(f"  {msg}")
                print(f"  保存位置: {output_path}")
                return True
            else:
                print(f"❌ FFmpeg后备方案也失败: {msg}")
                return False


def main():
    parser = argparse.ArgumentParser(
        description='视频首帧提取工具',
        epilog='支持OpenCV和FFmpeg两种方案，自动回退'
    )
    parser.add_argument('--input', type=str, default='data/origin_video/gk7401273C2C88_1749551630_2.mp4',
                        help='输入视频路径')
    parser.add_argument('--output', type=str, default='data/room_first_frame.png',
                        help='输出图片路径')
    parser.add_argument('--use-ffmpeg-first', action='store_true',
                        help='优先使用FFmpeg（当OpenCV失败时）')
    
    args = parser.parse_args()
    
    # 验证输入文件存在
    if not os.path.exists(args.input):
        print(f"❌ 错误：输入视频文件不存在: {args.input}")
        return
    
    # 执行提取
    extract_first_frame(args.input, args.output, use_ffmpeg_first=args.use_ffmpeg_first)


if __name__ == '__main__':
    main()
