"""
视频截图工具
可以查看视频并按键截图保存
支持播放/暂停、帧跳转等功能
自动缩放视频适应屏幕，支持连续播放多个视频
"""

import cv2
import argparse
import os
from pathlib import Path


def get_screen_resolution():
    """
    获取屏幕分辨率
    """
    try:
        # 尝试使用screeninfo库获取屏幕信息
        import subprocess
        result = subprocess.run(['xrandr'], capture_output=True, text=True, timeout=5)
        
        # 简单解析：找出最大分辨率
        max_width = 1920
        max_height = 1080
        
        for line in result.stdout.split('\n'):
            if ' connected' in line and 'x' in line:
                # 格式: HDMI-1 connected primary 1920x1080+0+0
                parts = line.split()
                for part in parts:
                    if 'x' in part and '+' in part:
                        res = part.split('+')[0]
                        w, h = map(int, res.split('x'))
                        max_width = max(max_width, w)
                        max_height = max(max_height, h)
        
        return max_width, max_height
    except:
        # 默认16:9屏幕
        return 1920, 1080


def resize_frame_to_screen(frame, max_width=1920, max_height=1080, padding=120, scale_factor=0.7):
    """
    将视频帧缩放到屏幕大小，保持宽高比
    
    Args:
        frame: 输入帧
        max_width: 最大宽度
        max_height: 最大高度
        padding: 预留边距(像素)
        scale_factor: 额外缩放因子(0-1)，进一步缩小视频
    
    Returns:
        resized frame
    """
    h, w = frame.shape[:2]
    
    # 计算可用空间（留边距防止到屏幕边缘）
    available_width = (max_width - padding) * scale_factor
    available_height = (max_height - padding) * scale_factor
    
    # 计算缩放比例
    scale = min(available_width / w, available_height / h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return frame


class VideoScreenshot:
    def __init__(self, video_path, output_dir="data/screenshots", video_list=None, current_index=0):
        """
        初始化视频截图工具
        
        Args:
            video_path: 输入视频路径
            output_dir: 截图保存目录
            video_list: 视频文件列表（用于支持连续播放）
            current_index: 当前视频在列表中的索引
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.current_frame = 0
        self.is_paused = False
        self.screenshot_count = 0
        
        # 多视频支持
        self.video_list = video_list or [video_path]
        self.current_index = current_index
        
        # 屏幕分辨率
        self.screen_width, self.screen_height = get_screen_resolution()
        
        os.makedirs(output_dir, exist_ok=True)
        
    def open_video(self):
        """打开视频文件"""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            print(f"❌ 错误：无法打开视频文件 {self.video_path}")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ 视频打开成功")
        print(f"  分辨率: {self.width}x{self.height}")
        print(f"  总帧数: {self.total_frames}")
        print(f"  帧率: {self.fps:.2f}")
        return True
    
    def save_screenshot(self, frame):
        """保存截图"""
        self.screenshot_count += 1
        
        # 生成文件名：时间戳_帧号.png
        frame_time = self.current_frame / self.fps if self.fps > 0 else 0
        minutes = int(frame_time) // 60
        seconds = int(frame_time) % 60
        milliseconds = int((frame_time % 1) * 1000)
        
        filename = f"screenshot_{minutes:02d}m{seconds:02d}s{milliseconds:03d}ms_frame{self.current_frame:06d}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        success = cv2.imwrite(filepath, frame)
        
        if success:
            print(f"✓ 第{self.screenshot_count}张截图已保存")
            print(f"  时间: {minutes:02d}:{seconds:02d}.{milliseconds:03d}")
            print(f"  帧号: {self.current_frame}/{self.total_frames}")
            print(f"  文件: {filename}")
        else:
            print(f"❌ 截图保存失败")
        
        return success
    
    def display_help(self):
        """显示帮助信息"""
        help_text = """
╔══════════════════════════════════════╗
║       视频截图工具 - 快捷键说明      ║
╠══════════════════════════════════════╣
║  SPACE/P    播放/暂停                ║
║  S          截图保存                ║
║  →          快进 5 帧                ║
║  ←          快退 5 帧                ║
║  ↑          快进 30 帧               ║
║  ↓          快退 30 帧               ║
║  1-9        跳转到指定进度(10%-90%)  ║
║  0          跳转到开始               ║
║  E          跳转到结束               ║
║  G          输入帧号跳转             ║
║  N          下一个视频               ║
║  B          上一个视频               ║
║  H          显示本帮助               ║
║  Q/ESC      退出                    ║
╚══════════════════════════════════════╝
        """
        print(help_text)
    
    def run(self):
        """运行交互式视频播放"""
        if not self.open_video():
            return
        
        print(f"\n正在播放: {os.path.basename(self.video_path)} ({self.current_index + 1}/{len(self.video_list)})")
        print("按 'H' 查看帮助信息，或直接开始操作...\n")
        
        delay = int(1000 / self.fps) if self.fps > 0 else 33
        
        while True:
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"\n✓ {os.path.basename(self.video_path)} 播放完毕")
                    
                    # 检查是否有下一个视频
                    if self.current_index + 1 < len(self.video_list):
                        print(f"\n➜ 自动加载下一个视频...\n")
                        self.cap.release()
                        cv2.destroyAllWindows()
                        return "next"
                    else:
                        print(f"✓ 所有视频播放完毕")
                        break
                self.current_frame += 1
            else:
                # 暂停状态，重新读取当前帧
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame - 1)
                ret, frame = self.cap.read()
                if not ret:
                    break
            
            # 添加状态信息到画面
            frame_copy = frame.copy()
            self._draw_info(frame_copy)
            
            # 缩放到屏幕大小
            frame_resized = resize_frame_to_screen(
                frame_copy,
                self.screen_width,
                self.screen_height,
                padding=120,
                scale_factor=0.45
            )
            
            # 显示视频
            cv2.imshow("Video Screenshot Tool", frame_resized)
            
            # 获取键盘输入
            key = cv2.waitKey(delay if not self.is_paused else 0) & 0xFF
            
            if key == ord('q') or key == 27:  # Q 或 ESC
                print("\n✓ 已退出")
                self.cap.release()
                cv2.destroyAllWindows()
                return "quit"
            elif key == ord(' ') or key == ord('p'):  # 空格或P：播放/暂停
                self.is_paused = not self.is_paused
                status = "暂停" if self.is_paused else "播放"
                print(f"➜ {status}")
            elif key == ord('s'):  # S：截图
                self.save_screenshot(frame)
            elif key == 83 or key == 2555904:  # 右箭头：快进5帧
                self.current_frame = min(self.current_frame + 5, self.total_frames)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame - 1)
            elif key == 81 or key == 2424832:  # 左箭头：快退5帧
                self.current_frame = max(self.current_frame - 5, 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame - 1)
            elif key == 82 or key == 2490368:  # 上箭头：快进30帧
                self.current_frame = min(self.current_frame + 30, self.total_frames)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame - 1)
            elif key == 84 or key == 2621440:  # 下箭头：快退30帧
                self.current_frame = max(self.current_frame - 30, 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame - 1)
            elif ord('0') <= key <= ord('9'):  # 数字键：跳转到指定进度
                if key == ord('0'):
                    self.current_frame = 1
                else:
                    progress = (key - ord('1') + 1) / 9.0
                    self.current_frame = max(1, int(self.total_frames * progress))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame - 1)
                minutes = int(self.current_frame / self.fps / 60) if self.fps > 0 else 0
                seconds = int((self.current_frame / self.fps) % 60) if self.fps > 0 else 0
                print(f"➜ 跳转到 {minutes:02d}:{seconds:02d}")
            elif key == ord('e'):  # E：跳转到结束
                self.current_frame = self.total_frames
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame - 1)
                print(f"➜ 跳转到末尾")
            elif key == ord('g'):  # G：输入帧号跳转
                try:
                    frame_num = int(input("\n输入目标帧号 (0-{}): ".format(self.total_frames)))
                    if 0 <= frame_num <= self.total_frames:
                        self.current_frame = frame_num
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                        print(f"➜ 跳转到帧 {self.current_frame}")
                    else:
                        print(f"❌ 帧号超出范围")
                except ValueError:
                    print(f"❌ 输入无效")
            elif key == ord('n'):  # N：下一个视频
                if self.current_index + 1 < len(self.video_list):
                    print(f"\n➜ 加载下一个视频...\n")
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return "next"
                else:
                    print(f"⚠ 已是最后一个视频")
            elif key == ord('b'):  # B：上一个视频
                if self.current_index > 0:
                    print(f"\n➜ 加载上一个视频...\n")
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return "prev"
                else:
                    print(f"⚠ 已是第一个视频")
            elif key == ord('h'):  # H：显示帮助
                self.display_help()
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n总计保存 {self.screenshot_count} 张截图")
        print(f"保存目录: {os.path.abspath(self.output_dir)}")
        return "end"
    
    def _draw_info(self, frame):
        """在画面上绘制状态信息"""
        minutes = int(self.current_frame / self.fps / 60) if self.fps > 0 else 0
        seconds = int((self.current_frame / self.fps) % 60) if self.fps > 0 else 0
        milliseconds = int(((self.current_frame / self.fps) % 1) * 1000) if self.fps > 0 else 0
        
        progress = (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0
        
        status = "暂停" if self.is_paused else "播放"
        
        info_text = f"{status} | {minutes:02d}:{seconds:02d}.{milliseconds:03d} | 帧 {self.current_frame}/{self.total_frames} ({progress:.1f}%)"
        
        # 背景
        cv2.rectangle(frame, (0, 0), (len(info_text) * 8 + 20, 30), (0, 0, 0), -1)
        
        # 文本
        cv2.putText(frame, info_text, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 进度条
        bar_width = frame.shape[1]
        bar_height = 5
        bar_y = frame.shape[0] - bar_height
        filled_width = int(bar_width * (self.current_frame / self.total_frames)) if self.total_frames > 0 else 0
        
        cv2.rectangle(frame, (0, bar_y), (bar_width, frame.shape[0]), (100, 100, 100), -1)
        cv2.rectangle(frame, (0, bar_y), (filled_width, frame.shape[0]), (0, 255, 0), -1)


def get_video_files(path):
    """
    获取目录或文件中的视频文件列表
    
    Args:
        path: 文件路径或目录路径
    
    Returns:
        list of video file paths
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
    video_files = []
    
    if os.path.isfile(path):
        # 如果是文件，直接返回
        if path.lower().endswith(video_extensions):
            return [path]
        else:
            return []
    elif os.path.isdir(path):
        # 如果是目录，列出所有视频文件
        try:
            files = sorted(os.listdir(path))
            for file in files:
                filepath = os.path.join(path, file)
                if os.path.isfile(filepath) and file.lower().endswith(video_extensions):
                    video_files.append(filepath)
        except Exception as e:
            print(f"❌ 错误：无法读取目录 {path}: {e}")
            return []
    
    return video_files


def select_video_from_list(video_files):
    """
    从列表中选择视频文件
    
    Args:
        video_files: 视频文件列表
    
    Returns:
        selected video file path or None
    """
    if not video_files:
        print("❌ 错误：未找到视频文件")
        return None
    
    if len(video_files) == 1:
        return video_files[0]
    
    print(f"\n✓ 找到 {len(video_files)} 个视频文件：\n")
    for i, file in enumerate(video_files, 1):
        filename = os.path.basename(file)
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"  [{i}] {filename} ({size_mb:.1f} MB)")
    
    while True:
        try:
            choice = input("\n请选择视频 (输入数字): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(video_files):
                print(f"\n✓ 已选择: {os.path.basename(video_files[idx])}\n")
                return video_files[idx]
            else:
                print(f"❌ 输入无效，请输入 1-{len(video_files)} 之间的数字")
        except ValueError:
            print(f"❌ 输入无效，请输入数字")


def main():
    parser = argparse.ArgumentParser(
        description='视频截图工具',
        epilog='支持单个视频或文件夹批量加载'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入视频路径或目录')
    parser.add_argument('--output', '-o', type=str, default='data/screenshots',
                        help='截图保存目录 (默认: data/screenshots)')
    
    args = parser.parse_args()
    
    # 验证输入路径
    if not os.path.exists(args.input):
        print(f"❌ 错误：输入路径不存在: {args.input}")
        return
    
    # 获取视频文件列表
    video_files = get_video_files(args.input)
    
    if not video_files:
        print(f"❌ 错误：未找到视频文件")
        return
    
    # 选择视频
    selected_video = select_video_from_list(video_files)
    if not selected_video:
        return
    
    # 查找选中视频的索引
    current_index = video_files.index(selected_video)
    
    # 连续播放视频
    while True:
        tool = VideoScreenshot(
            video_files[current_index],
            args.output,
            video_list=video_files,
            current_index=current_index
        )
        result = tool.run()
        
        if result == "next":
            current_index = min(current_index + 1, len(video_files) - 1)
        elif result == "prev":
            current_index = max(current_index - 1, 0)
        else:  # "quit" 或 "end"
            break


if __name__ == '__main__':
    main()
