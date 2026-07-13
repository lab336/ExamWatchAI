"""可视化裁剪并合并同一视频中的多个片段。

示例：
    python scrpts/video_tool/video_clip_merge_ui.py
    python scrpts/video_tool/video_clip_merge_ui.py \
        --input data/origin_video/gk7401273C2C88_1749551630_2.mp4

依赖：系统安装 ffmpeg（同时包含 ffprobe）。Tkinter 通常随 Python 一起安装。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - dependency check at runtime
    Image = None
    ImageTk = None


VIDEO_FILETYPES = [
    ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm"),
    ("所有文件", "*.*"),
]


def format_time(seconds: float) -> str:
    total_milliseconds = max(0, int(round(float(seconds) * 1000)))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def parse_time(value: str) -> float:
    """Accept seconds, MM:SS(.mmm), or HH:MM:SS(.mmm)."""
    text = value.strip()
    if not text:
        raise ValueError("时间不能为空")
    parts = text.split(":")
    try:
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = minutes * 60 + float(parts[1])
        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = hours * 3600 + minutes * 60 + float(parts[2])
        else:
            raise ValueError
    except ValueError as exc:
        raise ValueError("时间格式应为秒数、MM:SS 或 HH:MM:SS") from exc
    if seconds < 0:
        raise ValueError("时间不能为负数")
    return seconds


@dataclass(frozen=True)
class ClipSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=True)


def probe_video(path: Path) -> dict[str, Any]:
    result = run_checked(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration:stream=index,codec_type,width,height,avg_frame_rate",
            "-of", "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    streams = payload.get("streams") or []
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    if duration <= 0 or video_stream is None:
        raise RuntimeError("无法从文件中读取有效的视频时长或视频流")
    return {
        "duration": duration,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "has_audio": any(item.get("codec_type") == "audio" for item in streams),
    }


def preview_frame(path: Path, seconds: float) -> "Image.Image":
    if Image is None:
        raise RuntimeError("预览需要 Pillow：pip install pillow")
    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(path),
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    image = Image.open(BytesIO(result.stdout)).convert("RGB")
    image.load()
    return image


def build_export_command(
    source: Path,
    output: Path,
    segments: list[ClipSegment],
    has_audio: bool,
) -> list[str]:
    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for index, segment in enumerate(segments):
        filter_parts.append(
            f"[0:v]trim=start={segment.start:.6f}:end={segment.end:.6f},setpts=PTS-STARTPTS[v{index}]"
        )
        concat_inputs.append(f"[v{index}]")
        if has_audio:
            filter_parts.append(
                f"[0:a]atrim=start={segment.start:.6f}:end={segment.end:.6f},asetpts=PTS-STARTPTS[a{index}]"
            )
            concat_inputs.append(f"[a{index}]")

    if has_audio:
        filter_parts.append(
            f"{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=1[outv][outa]"
        )
    else:
        filter_parts.append(
            f"{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=0[outv]"
        )

    command = [
        "ffmpeg", "-y", "-hide_banner", "-progress", "pipe:1", "-nostats",
        "-i", str(source), "-filter_complex", ";".join(filter_parts),
        "-map", "[outv]", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-movflags", "+faststart",
    ]
    if has_audio:
        command.extend(["-map", "[outa]", "-c:a", "aac", "-b:a", "192k"])
    command.append(str(output))
    return command


class ClipMergeApp(tk.Tk):
    def __init__(self, initial_input: str = ""):
        super().__init__()
        self.title("视频裁剪并合并")
        self.ui_scale = self._configure_screen_adaptive_style()

        self.source_path: Path | None = None
        self.metadata: dict[str, Any] = {}
        self.segments: list[ClipSegment] = []
        self.preview_photo: Any = None
        self.preview_job: str | None = None
        self.export_process: subprocess.Popen[str] | None = None
        self.events: Queue[tuple[str, Any]] = Queue()

        self.source_var = tk.StringVar(value=initial_input)
        self.output_var = tk.StringVar()
        self.start_var = tk.StringVar(value="00:00.000")
        self.end_var = tk.StringVar(value="00:10.000")
        self.playhead_var = tk.DoubleVar(value=0.0)
        self.time_label_var = tk.StringVar(value="未加载视频")
        self.status_var = tk.StringVar(value="请选择一个视频文件")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self.after(100, self._poll_events)
        if initial_input:
            self.after(100, self.load_source)

    def _configure_screen_adaptive_style(self) -> float:
        """Scale the window and all Tk/ttk controls for the active display."""
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        dpi = float(self.winfo_fpixels("1i") or 96.0)

        # 1440×900 is a comfortable baseline. DPI is included because a 4K
        # display may report a large pixel size while physical text is still
        # too small. The cap keeps controls usable on very dense displays.
        resolution_scale = min(screen_width / 1440.0, screen_height / 900.0)
        dpi_scale = dpi / 96.0
        scale = max(1.10, min(1.75, resolution_scale * dpi_scale))

        window_width = min(screen_width - 48, max(1080, int(1120 * scale)))
        window_height = min(screen_height - 72, max(720, int(760 * scale)))
        self.minsize(min(window_width, int(900 * scale)), min(window_height, int(620 * scale)))
        self.geometry(f"{window_width}x{window_height}")

        default_size = max(11, int(round(10.5 * scale)))
        text_size = max(10, int(round(9.5 * scale)))
        heading_size = max(12, int(round(11.5 * scale)))
        for font_name, size in (
            ("TkDefaultFont", default_size),
            ("TkTextFont", default_size),
            ("TkMenuFont", default_size),
            ("TkHeadingFont", heading_size),
            ("TkFixedFont", text_size),
        ):
            try:
                tkfont.nametofont(font_name).configure(size=size)
            except tk.TclError:
                pass

        style = ttk.Style(self)
        style.configure("TButton", padding=(int(9 * scale), int(5 * scale)))
        style.configure("TEntry", padding=(int(5 * scale), int(3 * scale)))
        style.configure("TLabelframe", padding=int(2 * scale))
        style.configure("TLabelframe.Label", font=("TkDefaultFont", default_size, "bold"))
        style.configure("Treeview", font=("TkDefaultFont", text_size), rowheight=int(25 * scale))
        style.configure("Treeview.Heading", font=("TkDefaultFont", default_size, "bold"))
        return scale

    def _build_ui(self) -> None:
        pad = int(12 * self.ui_scale)
        root = ttk.Frame(self, padding=pad)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        source_frame = ttk.LabelFrame(root, text="视频与导出", padding=int(10 * self.ui_scale))
        source_frame.grid(row=0, column=0, sticky="ew")
        source_frame.columnconfigure(1, weight=1)
        source_frame.columnconfigure(4, weight=1)
        ttk.Label(source_frame, text="输入视频").grid(row=0, column=0, sticky="w")
        ttk.Entry(source_frame, textvariable=self.source_var).grid(row=0, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Button(source_frame, text="选择…", command=self.choose_source).grid(row=0, column=4, padx=(0, 6))
        ttk.Button(source_frame, text="加载", command=self.load_source).grid(row=0, column=5)
        ttk.Label(source_frame, text="导出文件").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(source_frame, textvariable=self.output_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=(8, 0))
        ttk.Button(source_frame, text="另存为…", command=self.choose_output).grid(row=1, column=4, columnspan=2, pady=(8, 0))

        preview_frame_widget = ttk.LabelFrame(root, text="定位与预览", padding=int(10 * self.ui_scale))
        preview_frame_widget.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        preview_frame_widget.columnconfigure(0, weight=1)
        self.preview_label = ttk.Label(
            preview_frame_widget,
            text="加载视频后可拖动时间轴预览画面",
            anchor="center",
            relief="sunken",
        )
        self.preview_label.grid(row=0, column=0, columnspan=5, sticky="ew")
        self.timeline = ttk.Scale(
            preview_frame_widget,
            from_=0.0,
            to=1.0,
            variable=self.playhead_var,
            command=self._on_timeline_changed,
        )
        self.timeline.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(10, 0))
        ttk.Label(preview_frame_widget, textvariable=self.time_label_var).grid(row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Button(preview_frame_widget, text="设为开始", command=lambda: self._set_time_from_playhead(self.start_var)).grid(row=2, column=2, pady=(5, 0))
        ttk.Button(preview_frame_widget, text="设为结束", command=lambda: self._set_time_from_playhead(self.end_var)).grid(row=2, column=3, padx=6, pady=(5, 0))
        ttk.Button(preview_frame_widget, text="刷新预览", command=self.refresh_preview).grid(row=2, column=4, pady=(5, 0))

        content = ttk.Frame(root)
        content.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        edit_frame = ttk.LabelFrame(content, text="添加裁剪片段", padding=int(10 * self.ui_scale))
        edit_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        edit_frame.columnconfigure(1, weight=1)
        ttk.Label(edit_frame, text="开始").grid(row=0, column=0, sticky="w")
        ttk.Entry(edit_frame, textvariable=self.start_var, width=16).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(edit_frame, text="结束").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(edit_frame, textvariable=self.end_var, width=16).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Label(edit_frame, text="格式：秒数、MM:SS 或 HH:MM:SS", foreground="#666").grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(edit_frame, text="添加到合并列表", command=self.add_segment).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Separator(edit_frame).grid(row=4, column=0, columnspan=2, sticky="ew", pady=14)
        ttk.Label(
            edit_frame,
            text="使用方式",
            font=("TkDefaultFont", max(12, int(round(11 * self.ui_scale))), "bold"),
        ).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Label(
            edit_frame,
            text="1. 拖动时间轴定位\n2. 设定开始和结束\n3. 添加多个片段\n4. 按列表顺序导出",
            justify="left",
        ).grid(row=6, column=0, columnspan=2, sticky="nw", pady=(6, 0))

        list_frame = ttk.LabelFrame(content, text="待合并片段（由上到下决定播放顺序）", padding=int(10 * self.ui_scale))
        list_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(list_frame, columns=("number", "start", "end", "duration"), show="headings", selectmode="browse")
        for column, title, width in (("number", "序号", 60), ("start", "开始", 120), ("end", "结束", 120), ("duration", "时长", 120)):
            self.tree.heading(column, text=title)
            self.tree.column(
                column,
                width=int(width * self.ui_scale),
                anchor="center",
                stretch=column == "duration",
            )
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        controls = ttk.Frame(list_frame)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(controls, text="预览所选", command=self.preview_selected).pack(side="left")
        ttk.Button(controls, text="上移", command=lambda: self.move_segment(-1)).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="下移", command=lambda: self.move_segment(1)).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="删除", command=self.remove_segment).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="清空", command=self.clear_segments).pack(side="left", padx=(6, 0))

        export_frame = ttk.Frame(root)
        export_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        export_frame.columnconfigure(0, weight=1)
        ttk.Label(export_frame, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Progressbar(export_frame, maximum=100.0, variable=self.progress_var).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.export_button = ttk.Button(export_frame, text="裁剪并合并导出", command=self.export)
        self.export_button.grid(row=0, column=1, rowspan=2, padx=(12, 0))
        self.cancel_button = ttk.Button(export_frame, text="取消", command=self.cancel_export, state="disabled")
        self.cancel_button.grid(row=0, column=2, rowspan=2, padx=(6, 0))

    def choose_source(self) -> None:
        filename = filedialog.askopenfilename(title="选择视频", filetypes=VIDEO_FILETYPES)
        if filename:
            self.source_var.set(filename)
            self.load_source()

    def choose_output(self) -> None:
        initial = self.output_var.get() or "merged_clips.mp4"
        filename = filedialog.asksaveasfilename(
            title="导出合并后的视频",
            initialfile=Path(initial).name,
            defaultextension=".mp4",
            filetypes=[("MP4 视频", "*.mp4")],
        )
        if filename:
            self.output_var.set(filename)

    def load_source(self) -> None:
        if not self._ensure_ffmpeg():
            return
        candidate = Path(self.source_var.get().strip()).expanduser()
        if not candidate.is_file():
            messagebox.showerror("无法加载", "请选择存在的视频文件。")
            return
        try:
            metadata = probe_video(candidate)
        except (subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
            messagebox.showerror("无法读取视频", str(exc))
            return
        self.source_path = candidate.resolve()
        self.metadata = metadata
        self.timeline.configure(to=metadata["duration"])
        self.playhead_var.set(0.0)
        self.start_var.set("00:00.000")
        self.end_var.set(format_time(min(10.0, metadata["duration"])))
        if not self.output_var.get().strip():
            self.output_var.set(str(self.source_path.with_name(f"{self.source_path.stem}_clips_merged.mp4")))
        audio_text = "含音频" if metadata["has_audio"] else "无音频"
        self.time_label_var.set(
            f"0: {format_time(metadata['duration'])} | {metadata['width']}×{metadata['height']} | {audio_text}"
        )
        self.status_var.set(f"已加载：{self.source_path.name}")
        self.refresh_preview()

    def _ensure_ffmpeg(self) -> bool:
        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            return True
        messagebox.showerror("缺少 FFmpeg", "未找到 ffmpeg 或 ffprobe。请安装 FFmpeg 并确保其在 PATH 中。")
        return False

    def _on_timeline_changed(self, _value: str) -> None:
        seconds = float(self.playhead_var.get())
        duration = float(self.metadata.get("duration", 0.0))
        self.time_label_var.set(f"当前位置：{format_time(seconds)} / {format_time(duration)}")
        if self.source_path is not None:
            if self.preview_job is not None:
                self.after_cancel(self.preview_job)
            self.preview_job = self.after(250, self.refresh_preview)

    def _set_time_from_playhead(self, variable: tk.StringVar) -> None:
        variable.set(format_time(float(self.playhead_var.get())))

    def refresh_preview(self) -> None:
        self.preview_job = None
        if self.source_path is None or Image is None:
            return
        seconds = float(self.playhead_var.get())
        self.preview_label.configure(text="正在读取预览…", image="")
        self.update_idletasks()
        try:
            image = preview_frame(self.source_path, seconds)
        except (subprocess.CalledProcessError, OSError) as exc:
            self.preview_label.configure(text=f"预览失败：{exc}")
            return
        image.thumbnail(
            (int(720 * self.ui_scale), int(320 * self.ui_scale)),
            Image.Resampling.LANCZOS,
        )
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_photo, text="")

    def add_segment(self) -> None:
        if self.source_path is None:
            messagebox.showwarning("尚未加载视频", "请先选择并加载视频。")
            return
        try:
            start = parse_time(self.start_var.get())
            end = parse_time(self.end_var.get())
        except ValueError as exc:
            messagebox.showerror("时间格式错误", str(exc))
            return
        duration = float(self.metadata["duration"])
        if start >= end:
            messagebox.showerror("片段无效", "结束时间必须大于开始时间。")
            return
        if end > duration + 0.001:
            messagebox.showerror("片段无效", f"结束时间超过视频总时长：{format_time(duration)}")
            return
        self.segments.append(ClipSegment(start=start, end=min(end, duration)))
        self._refresh_segments()
        self.status_var.set(f"已添加片段 {len(self.segments)}，累计导出 {format_time(sum(item.duration for item in self.segments))}")

    def _refresh_segments(self, select_index: int | None = None) -> None:
        self.tree.delete(*self.tree.get_children())
        for index, segment in enumerate(self.segments):
            item_id = self.tree.insert(
                "", "end", iid=str(index),
                values=(index + 1, format_time(segment.start), format_time(segment.end), format_time(segment.duration)),
            )
            if select_index == index:
                self.tree.selection_set(item_id)
                self.tree.focus(item_id)

    def _selected_index(self) -> int | None:
        selection = self.tree.selection()
        return int(selection[0]) if selection else None

    def preview_selected(self) -> None:
        index = self._selected_index()
        if index is None:
            messagebox.showinfo("未选择片段", "请先在列表中选择一个片段。")
            return
        segment = self.segments[index]
        self.playhead_var.set(segment.start)
        self.start_var.set(format_time(segment.start))
        self.end_var.set(format_time(segment.end))
        self.refresh_preview()

    def move_segment(self, direction: int) -> None:
        index = self._selected_index()
        if index is None:
            return
        target = index + direction
        if not 0 <= target < len(self.segments):
            return
        self.segments[index], self.segments[target] = self.segments[target], self.segments[index]
        self._refresh_segments(select_index=target)

    def remove_segment(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        del self.segments[index]
        self._refresh_segments(select_index=min(index, len(self.segments) - 1) if self.segments else None)

    def clear_segments(self) -> None:
        if self.segments and messagebox.askyesno("清空片段", "确定清空所有待导出片段吗？"):
            self.segments.clear()
            self._refresh_segments()

    def export(self) -> None:
        if self.source_path is None or not self.segments:
            messagebox.showwarning("无法导出", "请先加载视频并至少添加一个片段。")
            return
        output = Path(self.output_var.get().strip()).expanduser()
        if not output.name:
            messagebox.showwarning("无法导出", "请指定输出文件。")
            return
        if output.suffix.lower() != ".mp4":
            output = output.with_suffix(".mp4")
            self.output_var.set(str(output))
        output = output.resolve()
        if output == self.source_path:
            messagebox.showerror("无法导出", "输出文件不能覆盖输入视频。")
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not messagebox.askyesno("覆盖文件", f"文件已存在，是否覆盖？\n{output}"):
            return
        command = build_export_command(
            self.source_path, output, self.segments, bool(self.metadata.get("has_audio")),
        )
        self.progress_var.set(0.0)
        self.export_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status_var.set("正在裁剪并合并…")
        total_seconds = sum(segment.duration for segment in self.segments)
        threading.Thread(target=self._run_export, args=(command, output, total_seconds), daemon=True).start()

    def _run_export(self, command: list[str], output: Path, total_seconds: float) -> None:
        try:
            self.export_process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
            )
            assert self.export_process.stdout is not None
            for line in self.export_process.stdout:
                key, _, value = line.strip().partition("=")
                if key == "out_time_ms":
                    progress = min(99.0, float(value) / 1_000_000.0 / max(0.001, total_seconds) * 100.0)
                    self.events.put(("progress", progress))
            stderr = self.export_process.stderr.read() if self.export_process.stderr else ""
            code = self.export_process.wait()
            if code == 0 and output.is_file():
                # ffmpeg may return zero even when a damaged source yields an
                # audio-only output. Verify that the requested video stream
                # was actually written before reporting success.
                try:
                    probe_video(output)
                except (subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
                    self.events.put(("error", f"导出文件没有有效视频流：{exc}\n\n{stderr[-2000:]}"))
                else:
                    self.events.put(("done", output))
            else:
                self.events.put(("error", stderr[-3000:] or f"ffmpeg 退出码：{code}"))
        except Exception as exc:  # pragma: no cover - runtime/system dependent
            self.events.put(("error", str(exc)))
        finally:
            self.export_process = None

    def cancel_export(self) -> None:
        if self.export_process is not None:
            self.export_process.terminate()
            self.status_var.set("正在取消导出…")

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "progress":
                    self.progress_var.set(float(payload))
                    self.status_var.set(f"正在裁剪并合并：{float(payload):.1f}%")
                elif event == "done":
                    self.progress_var.set(100.0)
                    self.status_var.set(f"导出完成：{payload}")
                    messagebox.showinfo("导出完成", f"已生成合并视频：\n{payload}")
                    self._set_export_idle()
                elif event == "error":
                    self.status_var.set("导出失败或已取消")
                    messagebox.showerror("导出失败", str(payload))
                    self._set_export_idle()
        except Empty:
            pass
        self.after(100, self._poll_events)

    def _set_export_idle(self) -> None:
        self.export_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="视频裁剪并合并桌面工具")
    parser.add_argument(
        "--input",
        default="data/origin_video/gk7401273C2C88_1749551630_2.mp4",
        help="启动时自动加载的视频路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = ClipMergeApp(initial_input=args.input if Path(args.input).is_file() else "")
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except tk.TclError as exc:
        print(f"无法启动 Tkinter 界面：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
