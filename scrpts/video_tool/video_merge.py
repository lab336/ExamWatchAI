"""
视频合并工具
支持将多个视频按顺序合并成一个视频

使用示例：
# 合并两个视频
python scrpts/video_tool/video_merge.py --input data/ideotest/clipped_testdata2.mp4 data/ideotest/clipped_testdata.mp4 --output data/ideotest/merged_output.mp4

# 从文本文件读取视频列表
python scrpts/video_tool/video_merge.py --list videos.txt --output merged.mp4

# 使用 OpenCV 方式合并（不推荐，会丢失音频）
python scrpts/video_tool/video_merge.py --input video1.mp4 video2.mp4 --output merged.mp4 --method opencv
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2


def check_ffmpeg():
    """检查 ffmpeg 是否可用"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def merge_with_ffmpeg(input_videos, output_path):
    """使用 ffmpeg 合并视频（推荐方式，保留音频）"""
    print(f"使用 ffmpeg 合并 {len(input_videos)} 个视频...")
    
    # 创建临时文件列表
    temp_list_file = "temp_video_list.txt"
    
    try:
        # 写入文件列表
        with open(temp_list_file, "w", encoding="utf-8") as f:
            for video in input_videos:
                # ffmpeg 需要绝对路径或相对路径，并且路径需要转义
                abs_path = os.path.abspath(video)
                f.write(f"file '{abs_path}'\n")
        
        # 构建 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", temp_list_file,
            "-c", "copy",  # 直接复制编码，不重新编码
            "-y",  # 覆盖输出文件
            output_path
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 执行合并
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ 视频合并完成: {output_path}")
            return True
        else:
            print(f"✗ ffmpeg 合并失败:")
            print(result.stderr)
            return False
            
    finally:
        # 清理临时文件
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)


def merge_with_opencv(input_videos, output_path):
    """使用 OpenCV 合并视频（会丢失音频）"""
    print(f"使用 OpenCV 合并 {len(input_videos)} 个视频...")
    print("⚠ 警告: OpenCV 方式会丢失音频，建议使用 ffmpeg 方式")
    
    # 获取第一个视频的参数
    first_cap = cv2.VideoCapture(input_videos[0])
    if not first_cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_videos[0]}")
    
    fps = first_cap.get(cv2.CAP_PROP_FPS)
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()
    
    print(f"输出参数: {width}x{height} @ {fps} fps")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output_path}")
    
    total_frames = 0
    
    try:
        for idx, video_path in enumerate(input_videos, 1):
            print(f"处理第 {idx}/{len(input_videos)} 个视频: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠ 警告: 无法打开视频 {video_path}，跳过")
                continue
            
            # 检查尺寸是否一致
            curr_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            curr_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if curr_width != width or curr_height != height:
                print(f"⚠ 警告: 视频尺寸不一致 ({curr_width}x{curr_height} vs {width}x{height})")
                print(f"   将进行缩放处理")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 如果尺寸不一致，进行缩放
                if curr_width != width or curr_height != height:
                    frame = cv2.resize(frame, (width, height))
                
                out.write(frame)
                frame_count += 1
                total_frames += 1
                
                if frame_count % 100 == 0:
                    print(f"  已处理 {frame_count} 帧...", end="\r")
            
            print(f"  完成: {frame_count} 帧              ")
            cap.release()
            
    finally:
        out.release()
    
    print(f"✓ 视频合并完成: {output_path}")
    print(f"  总帧数: {total_frames}")


def read_video_list_from_file(list_file):
    """从文本文件读取视频列表"""
    videos = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                videos.append(line)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="视频合并工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 合并两个视频
  python scrpts/video_tool/video_merge.py --input video1.mp4 video2.mp4 -o merged.mp4
  
  # 从文件列表合并
  python scrpts/video_tool/video_merge.py --list videos.txt -o merged.mp4
  
  # 指定使用 OpenCV 方式
  python scrpts/video_tool/video_merge.py --input video1.mp4 video2.mp4 -o merged.mp4 --method opencv
        """
    )
    
    # 输入方式
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        nargs="+",
        help="输入视频文件列表（空格分隔）"
    )
    input_group.add_argument(
        "--list", "-l",
        help="包含视频路径的文本文件（每行一个路径）"
    )
    
    # 输出
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出视频路径"
    )
    
    # 合并方式
    parser.add_argument(
        "--method", "-m",
        choices=["auto", "ffmpeg", "opencv"],
        default="auto",
        help="合并方式: auto(自动选择), ffmpeg(推荐), opencv(会丢失音频)"
    )
    
    args = parser.parse_args()
    
    # 获取输入视频列表
    if args.input:
        input_videos = args.input
    else:
        if not os.path.isfile(args.list):
            print(f"✗ 错误: 文件列表不存在: {args.list}")
            sys.exit(1)
        input_videos = read_video_list_from_file(args.list)
    
    # 验证输入视频
    if len(input_videos) < 2:
        print("✗ 错误: 至少需要两个视频文件进行合并")
        sys.exit(1)
    
    print(f"准备合并 {len(input_videos)} 个视频:")
    for idx, video in enumerate(input_videos, 1):
        if not os.path.isfile(video):
            print(f"✗ 错误: 视频文件不存在: {video}")
            sys.exit(1)
        
        # 获取文件大小
        size_mb = os.path.getsize(video) / (1024 * 1024)
        print(f"  {idx}. {video} ({size_mb:.2f} MB)")
    
    print(f"\n输出: {args.output}\n")
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 选择合并方式
    method = args.method
    if method == "auto":
        if check_ffmpeg():
            method = "ffmpeg"
            print("✓ 检测到 ffmpeg，使用 ffmpeg 方式合并（保留音频）\n")
        else:
            method = "opencv"
            print("⚠ 未检测到 ffmpeg，使用 OpenCV 方式合并（会丢失音频）")
            print("  提示: 安装 ffmpeg 以保留音频: sudo apt install ffmpeg\n")
    
    # 执行合并
    try:
        if method == "ffmpeg":
            if not check_ffmpeg():
                print("✗ 错误: ffmpeg 不可用，请安装 ffmpeg 或使用 --method opencv")
                sys.exit(1)
            success = merge_with_ffmpeg(input_videos, args.output)
            if not success:
                print("\n尝试使用 OpenCV 方式...")
                merge_with_opencv(input_videos, args.output)
        else:
            merge_with_opencv(input_videos, args.output)
        
        # 显示输出文件信息
        if os.path.isfile(args.output):
            output_size_mb = os.path.getsize(args.output) / (1024 * 1024)
            print(f"\n输出文件大小: {output_size_mb:.2f} MB")
            
    except Exception as e:
        print(f"\n✗ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
