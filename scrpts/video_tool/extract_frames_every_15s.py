"""
批量视频提帧工具

默认处理 data/考场数据视频 目录下的所有视频，每 30 秒保存一帧原始图像。
所有图像会保存到同一个输出目录，输出目录和文件名默认只使用英文、数字和下划线。

示例:
python scrpts/video_tool/extract_frames_every_15s.py
python scrpts/video_tool/extract_frames_every_15s.py --interval 10 --image-ext jpg
python scrpts/video_tool/extract_frames_every_15s.py --input-dir data/考场数据视频 --output-dir data/exam_frames_30s
"""

import argparse
import hashlib
import os
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpeg",
    ".mpg",
}


def import_cv2():
    """延迟导入 OpenCV，让 --help 在缺少依赖时也能正常显示。"""
    try:
        import cv2
    except ModuleNotFoundError:
        print("❌ 错误：当前 Python 环境未安装 OpenCV，请先安装 opencv-python 或切换到项目环境")
        return None
    return cv2


def make_video_prefix(video_path, video_index):
    """生成只包含 ASCII 字符的视频前缀，避免中文文件名写入输出图片。"""
    digest = hashlib.md5(str(video_path).encode("utf-8")).hexdigest()[:8]
    return f"video_{video_index:04d}_{digest}"


def find_videos(input_dir, recursive=False):
    """查找目录中的视频文件。"""
    input_path = Path(input_dir)
    pattern = "**/*" if recursive else "*"
    videos = [
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def save_frame(cv2, output_path, frame, image_ext, jpg_quality):
    """保存单帧图像。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_ext.lower() in {"jpg", "jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
    else:
        params = []

    return cv2.imwrite(str(output_path), frame, params)


def extract_frames(
    cv2,
    video_path,
    video_index,
    output_root,
    interval_seconds=15.0,
    image_ext="png",
    jpg_quality=95,
):
    """
    从单个视频按固定秒数间隔提帧。

    保存的是 OpenCV 读取到的原始帧，不做缩放、裁剪、标注。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        print(f"❌ 无法获取帧率，跳过: {video_path}")
        cap.release()
        return 0

    duration = frame_count / fps if frame_count > 0 else 0
    output_root = Path(output_root)
    interval_frames = max(1, int(round(interval_seconds * fps)))
    video_prefix = make_video_prefix(video_path, video_index)

    print(f"\n处理视频: {video_path.name}")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f}")
    print(f"  总帧数: {frame_count}")
    print(f"  时长: {duration:.2f}s")
    print(f"  输出目录: {output_root}")

    saved_count = 0
    frame_index = 0

    while frame_count <= 0 or frame_index < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_index / fps
        output_name = (
            f"{video_prefix}_"
            f"{int(timestamp):06d}s_"
            f"frame{frame_index:08d}.{image_ext}"
        )
        output_path = output_root / output_name

        if save_frame(cv2, output_path, frame, image_ext, jpg_quality):
            saved_count += 1
        else:
            print(f"  ❌ 保存失败: {output_path}")

        frame_index += interval_frames

    cap.release()
    print(f"  ✓ 已保存 {saved_count} 张图像")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="批量视频每隔固定时间提取原始帧")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/考场数据视频",
        help="输入视频目录，默认: data/考场数据视频",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/exam_video_30s_frames",
        help="输出图像目录，默认: data/exam_video_30s_frames",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="提帧间隔秒数，默认: 30",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="输出图片格式，默认: png",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPG质量，范围 1-100，默认: 95",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理输入目录下的子目录",
    )

    args = parser.parse_args()

    if args.interval <= 0:
        print("❌ 错误：--interval 必须大于 0")
        return

    if not os.path.isdir(args.input_dir):
        print(f"❌ 错误：输入目录不存在: {args.input_dir}")
        return

    jpg_quality = min(100, max(1, args.jpg_quality))
    videos = find_videos(args.input_dir, recursive=args.recursive)
    if not videos:
        print(f"❌ 没有找到视频文件: {args.input_dir}")
        return

    cv2 = import_cv2()
    if cv2 is None:
        return

    print(f"找到 {len(videos)} 个视频，开始每 {args.interval:g}s 提取一帧...")
    total_saved = 0
    for index, video_path in enumerate(videos, start=1):
        print(f"\n[{index}/{len(videos)}]")
        total_saved += extract_frames(
            cv2=cv2,
            video_path=video_path,
            video_index=index,
            output_root=args.output_dir,
            interval_seconds=args.interval,
            image_ext=args.image_ext,
            jpg_quality=jpg_quality,
        )

    print("\n全部处理完成")
    print(f"  视频数量: {len(videos)}")
    print(f"  图像数量: {total_saved}")
    print(f"  输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
