from __future__ import annotations

import argparse
import json
import random
import threading
import time
import traceback
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter import font as tkfont
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional dependency
    imageio = None

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
RESAMPLING_LANCZOS = (
    Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
)

APP_BG = "#15182B"
PANEL_BG = "#252A46"
PANEL_HEADER_BG = "#2F3556"
PANEL_MUTED_BG = "#1D2140"
PANEL_BORDER = "#3A4270"
TEXT_PRIMARY = "#F1F4FF"
TEXT_SECONDARY = "#AAB3D6"
ACCENT = "#5EB5FF"
STATUS_NORMAL = "#334068"
STATUS_NORMAL_TEXT = "#66E0B3"
STATUS_EMPTY = "#11BA86"
STATUS_WARNING = "#FF8A3D"
STATUS_CRITICAL = "#FF5E62"
STATUS_EMPTY_TEXT = "#E9FFF8"

STATUS_LABELS = {
    "normal": "正常",
    "empty": "空位",
    "warning": "异常预警",
    "critical": "重点预警",
}

LEVEL_LABELS = {
    "warning": "IV级（一般）",
    "critical": "III级（较重）",
}

ALERT_BEHAVIORS = [
    "双手桌下",
    "低头停留",
    "左顾右盼",
    "疑似交流",
    "手部遮挡",
]

REAL_STATUS_NOTE = {
    "normal": "已确认",
    "empty": "空位",
    "warning": "待确认",
    "critical": "重点关注",
}


@dataclass(slots=True)
class SeatBox:
    seat_no: int
    x1: float
    y1: float
    x2: float
    y2: float
    col_index: int = 0
    row_index: int = 0

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass(slots=True)
class AlertRecord:
    timestamp: str
    room_name: str
    seat_no: int
    level_label: str
    behavior: str


@dataclass(slots=True)
class InspectionSnapshot:
    status_by_seat: dict[int, str]
    behavior_by_seat: dict[int, str]
    occupied_count: int
    alert_count: int
    empty_count: int
    focus_seat: int | None
    events: list[AlertRecord]


@dataclass(slots=True)
class LiveBindingConfig:
    source: str
    person_weights: str
    desk_weights: str
    output_dir: str
    person_conf: float = 0.18
    person_iou: float = 0.60
    desk_conf: float = 0.70
    desk_iou: float = 0.45
    device: str = ""
    img_size: int | None = None
    half: bool = False
    tracker: str = "bytetrack.yaml"
    person_classes: list[int] | None = None
    desk_mode: str = "auto"
    desk_num_cols: int = 5
    desk_required_per_col: int = 6
    reference_max_frames: int = 120
    reference_sample_step: int = 5
    confirm_seconds: float = 3.0
    confirm_ratio: float = 0.50
    hold_seconds: float = 5.0
    re_confirm_seconds: float = 5.0
    lock_seconds: float = 20.0
    release_miss_seconds: float = 12.0
    pending_switch_seconds: float = 1.2
    pending_hold_seconds: float = 2.0
    first_extend_ratio: float = 0.6
    last_extend_ratio: float = 1.2
    outputs: set[str] = field(default_factory=set)
    simple_vis: bool = True
    max_students: int = 30
    max_teachers: int = 2


def chunk_sizes(total: int, count: int) -> list[int]:
    if count <= 0:
        return []
    base, rem = divmod(total, count)
    return [base + (1 if idx < rem else 0) for idx in range(count)]


def parse_classes(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_outputs(raw: str | None) -> set[str]:
    valid = {"binding_video", "zone_map", "csv", "json"}
    if not raw:
        return set()
    parts = {item.strip() for item in raw.split(",") if item.strip()}
    if "all" in parts:
        return valid
    unknown = parts - valid
    if unknown:
        raise ValueError(f"未知输出: {', '.join(sorted(unknown))}")
    return parts


def to_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def format_video_time(frame_idx: int, fps: float) -> str:
    if fps <= 1e-6:
        fps = 25.0
    total_seconds = max(0.0, float(frame_idx) / float(fps))
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    centiseconds = int(round((total_seconds - int(total_seconds)) * 100))
    return f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def pick_pil_font_spec(weight: str = "regular") -> tuple[str | None, int]:
    candidates = {
        "regular": [
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc", 2),
            ("C:/Windows/Fonts/msyh.ttc", 0),
            ("/System/Library/Fonts/PingFang.ttc", 0),
            ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0),
        ],
        "medium": [
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc", 2),
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
            ("C:/Windows/Fonts/msyhbd.ttc", 0),
            ("C:/Windows/Fonts/msyh.ttc", 0),
            ("/System/Library/Fonts/PingFang.ttc", 0),
        ],
        "bold": [
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 2),
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc", 2),
            ("C:/Windows/Fonts/msyhbd.ttc", 0),
            ("C:/Windows/Fonts/msyh.ttc", 0),
            ("/System/Library/Fonts/PingFang.ttc", 0),
        ],
    }
    for candidate, index in candidates.get(weight, candidates["regular"]):
        if Path(candidate).exists():
            return candidate, index
    return None, 0


def load_pil_font(
    size: int,
    weight: str = "regular",
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path, font_index = pick_pil_font_spec(weight=weight)
    if font_path:
        try:
            return ImageFont.truetype(font_path, size, index=font_index)
        except Exception:
            pass
    return ImageFont.load_default()


def render_placeholder_image(
    size: tuple[int, int],
    title: str,
    message: str,
    detail: str | None = None,
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", (width, height), "#161A2F")
    draw = ImageDraw.Draw(image)
    for y in range(height):
        ratio = y / max(1, height)
        color = (
            int(18 + 22 * ratio),
            int(24 + 18 * ratio),
            int(40 + 34 * ratio),
        )
        draw.line((0, y, width, y), fill=color)

    panel = (
        int(width * 0.11),
        int(height * 0.17),
        int(width * 0.89),
        int(height * 0.82),
    )
    draw.rounded_rectangle(panel, radius=34, outline="#364071", width=3)

    title_font = load_pil_font(max(24, int(height * 0.04)), weight="bold")
    text_font = load_pil_font(max(18, int(height * 0.025)), weight="medium")
    detail_font = load_pil_font(max(15, int(height * 0.018)), weight="regular")

    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_x = (width - (title_bbox[2] - title_bbox[0])) / 2.0
    title_y = panel[1] + (panel[3] - panel[1]) * 0.36
    draw.text((title_x, title_y), title, fill="#F1F4FF", font=title_font)

    msg_bbox = draw.textbbox((0, 0), message, font=text_font)
    msg_x = (width - (msg_bbox[2] - msg_bbox[0])) / 2.0
    msg_y = title_y + 72
    draw.text((msg_x, msg_y), message, fill="#AAB3D6", font=text_font)

    if detail:
        detail_bbox = draw.textbbox((0, 0), detail, font=detail_font)
        detail_x = (width - (detail_bbox[2] - detail_bbox[0])) / 2.0
        draw.text((detail_x, msg_y + 56), detail[:96], fill="#FFB089", font=detail_font)

    return image


def resolve_demo_source(project_root: Path) -> Path | None:
    candidates = [
        project_root / "data" / "ideotest" / "merged_output.mp4",
        project_root / "exam_seat_binding" / "output" / "bound_v3_merged_output.mp4",
        project_root / "data" / "my_screenshots",
        project_root / "output",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            if any(path.suffix.lower() in IMAGE_SUFFIXES for path in candidate.iterdir()):
                return candidate
        elif candidate.is_file():
            return candidate
    return None


def build_synthetic_layout(
    num_cols: int = 5,
    num_rows: int = 6,
) -> tuple[list[SeatBox], tuple[int, int]]:
    base_w, base_h = 1920, 1080
    seats: list[SeatBox] = []
    left_far = (620.0, 250.0)
    right_far = (1280.0, 250.0)
    left_near = (360.0, 760.0)
    right_near = (1570.0, 760.0)

    seat_no = 1
    for col_idx in range(num_cols):
        v = 0.0 if num_cols == 1 else col_idx / (num_cols - 1)
        top_x = left_far[0] + (right_far[0] - left_far[0]) * v
        top_y = left_far[1] + (right_far[1] - left_far[1]) * v
        bottom_x = left_near[0] + (right_near[0] - left_near[0]) * v
        bottom_y = left_near[1] + (right_near[1] - left_near[1]) * v

        for row_idx in range(num_rows):
            u = 0.0 if num_rows == 1 else row_idx / (num_rows - 1)
            cx = top_x + (bottom_x - top_x) * u
            cy = top_y + (bottom_y - top_y) * u
            scale = 0.68 + 0.82 * u
            box_w = 86.0 * scale
            box_h = 44.0 * scale
            seats.append(
                SeatBox(
                    seat_no=seat_no,
                    x1=cx - box_w / 2.0,
                    y1=cy - box_h / 2.0,
                    x2=cx + box_w / 2.0,
                    y2=cy + box_h / 2.0,
                    col_index=col_idx,
                    row_index=row_idx,
                )
            )
            seat_no += 1

    return seats, (base_w, base_h)


def load_seat_layout(
    layout_path: Path | None,
    num_cols: int = 5,
) -> tuple[list[SeatBox], tuple[int, int]]:
    if layout_path is None or not layout_path.exists():
        return build_synthetic_layout(num_cols=num_cols)

    with layout_path.open("r", encoding="utf-8") as fp:
        layout_data = json.load(fp)

    raw_seats = layout_data.get("seats") or []
    if not raw_seats:
        return build_synthetic_layout(num_cols=num_cols)

    xs: list[float] = []
    ys: list[float] = []
    seats: list[SeatBox] = []
    for idx, item in enumerate(raw_seats, start=1):
        seat_no = int(item.get("id", idx - 1)) + 1
        x1 = float(item["x1"])
        y1 = float(item["y1"])
        x2 = float(item["x2"])
        y2 = float(item["y2"])
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        seats.append(SeatBox(seat_no=seat_no, x1=x1, y1=y1, x2=x2, y2=y2))

    image_path = Path(layout_data.get("image", ""))
    if image_path.exists():
        with Image.open(image_path) as image:
            base_size = image.size
    else:
        base_size = (2560, 1440) if max(xs) > 1920 or max(ys) > 1080 else (1920, 1080)

    ordered_by_x = sorted(seats, key=lambda seat: seat.center_x)
    sizes = chunk_sizes(len(ordered_by_x), num_cols)
    cursor = 0
    for col_idx, size in enumerate(sizes):
        column = ordered_by_x[cursor:cursor + size]
        cursor += size
        for row_idx, seat in enumerate(sorted(column, key=lambda item: item.center_y)):
            seat.col_index = col_idx
            seat.row_index = row_idx

    seats.sort(key=lambda item: item.seat_no)
    return seats, base_size


def build_layout_from_zones(
    zones: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> tuple[list[SeatBox], tuple[int, int]]:
    if not zones:
        return build_synthetic_layout()

    max_row = max(int(zone.get("row_index", 0)) for zone in zones)
    seats: list[SeatBox] = []
    for zone in zones:
        desk_xyxy = zone.get("desk_xyxy")
        if desk_xyxy and len(desk_xyxy) == 4:
            x1, y1, x2, y2 = [float(value) for value in desk_xyxy]
        else:
            polygon = zone.get("polygon") or []
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        seats.append(
            SeatBox(
                seat_no=int(zone["desk_no"]),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                col_index=int(zone.get("column_index", 0)),
                row_index=max_row - int(zone.get("row_index", 0)),
            )
        )
    seats.sort(key=lambda seat: seat.seat_no)
    return seats, (int(frame_width), int(frame_height))


def build_snapshot_from_binding_rows(
    rows: list[dict[str, Any]],
    seat_numbers: list[int],
    room_name: str,
    frame_idx: int,
    fps: float,
    previous_status: dict[int, str] | None,
) -> tuple[InspectionSnapshot, dict[int, str]]:
    status_by_seat = {seat_no: "empty" for seat_no in seat_numbers}
    behavior_by_seat = {seat_no: REAL_STATUS_NOTE["empty"] for seat_no in seat_numbers}

    confirmed_seats: set[int] = set()
    pending_seats: set[int] = set()
    focus_seat: int | None = None

    for row in rows:
        role = str(row.get("role", "unknown"))
        seat_display = to_int(row.get("seat_no_display"))
        seat_current = to_int(row.get("seat_no_current"))
        if role == "student" and seat_display is not None:
            confirmed_seats.add(seat_display)
            status_by_seat[seat_display] = "normal"
            behavior_by_seat[seat_display] = REAL_STATUS_NOTE["normal"]
            continue

        if role in {"student", "unknown"} and seat_current is not None:
            pending_seats.add(seat_current)
            if status_by_seat.get(seat_current) != "normal":
                status_by_seat[seat_current] = "warning"
                behavior_by_seat[seat_current] = REAL_STATUS_NOTE["warning"]
                if focus_seat is None:
                    focus_seat = seat_current

    if focus_seat is None and pending_seats:
        focus_seat = sorted(pending_seats)[0]
    if focus_seat is None and confirmed_seats:
        focus_seat = sorted(confirmed_seats)[0]

    events: list[AlertRecord] = []
    timestamp = format_video_time(frame_idx, fps)
    if previous_status is not None:
        for seat_no in seat_numbers:
            prev = previous_status.get(seat_no)
            curr = status_by_seat[seat_no]
            if prev == curr:
                continue
            if curr == "warning":
                events.append(
                    AlertRecord(
                        timestamp=timestamp,
                        room_name=room_name,
                        seat_no=seat_no,
                        level_label=LEVEL_LABELS["warning"],
                        behavior="待确认",
                    )
                )
            elif prev == "normal" and curr == "empty":
                events.append(
                    AlertRecord(
                        timestamp=timestamp,
                        room_name=room_name,
                        seat_no=seat_no,
                        level_label=LEVEL_LABELS["warning"],
                        behavior="座位空出",
                    )
                )

    snapshot = InspectionSnapshot(
        status_by_seat=status_by_seat,
        behavior_by_seat=behavior_by_seat,
        occupied_count=len(confirmed_seats),
        alert_count=sum(1 for seat_no in seat_numbers if status_by_seat[seat_no] == "warning"),
        empty_count=sum(1 for seat_no in seat_numbers if status_by_seat[seat_no] == "empty"),
        focus_seat=focus_seat,
        events=events,
    )
    return snapshot, status_by_seat


class RealBindingSession:
    def __init__(self, config: LiveBindingConfig):
        self.config = config
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.pending_layout: dict[str, Any] | None = None
        self.pending_frame: dict[str, Any] | None = None
        self.pending_finish: dict[str, Any] | None = None
        self.pending_error: dict[str, Any] | None = None
        self.status_message = "等待启动真实检测"
        self.started = False
        self.finished = False

    def description(self) -> str:
        return self.config.source

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _set_status(self, text: str) -> None:
        with self.lock:
            self.status_message = text

    def _store_latest(self, attr_name: str, payload: dict[str, Any]) -> None:
        with self.lock:
            setattr(self, attr_name, payload)

    def _run(self) -> None:
        self.started = True
        self._set_status("正在初始化检测模型...")
        try:
            from exam_seat_binding.person_desk_binding_v3 import PersonDeskBindingPipelineV3

            pipeline = PersonDeskBindingPipelineV3(
                source=self.config.source,
                person_weights=self.config.person_weights,
                desk_weights=self.config.desk_weights,
                output_dir=self.config.output_dir,
                person_conf=self.config.person_conf,
                person_iou=self.config.person_iou,
                desk_conf=self.config.desk_conf,
                desk_iou=self.config.desk_iou,
                device=self.config.device,
                img_size=self.config.img_size,
                half=self.config.half,
                tracker=self.config.tracker,
                person_classes=self.config.person_classes,
                desk_mode=self.config.desk_mode,
                desk_num_cols=self.config.desk_num_cols,
                desk_required_per_col=self.config.desk_required_per_col,
                reference_max_frames=self.config.reference_max_frames,
                reference_sample_step=self.config.reference_sample_step,
                confirm_seconds=self.config.confirm_seconds,
                confirm_ratio=self.config.confirm_ratio,
                hold_seconds=self.config.hold_seconds,
                re_confirm_seconds=self.config.re_confirm_seconds,
                lock_seconds=self.config.lock_seconds,
                release_miss_seconds=self.config.release_miss_seconds,
                pending_switch_seconds=self.config.pending_switch_seconds,
                pending_hold_seconds=self.config.pending_hold_seconds,
                first_extend_ratio=self.config.first_extend_ratio,
                last_extend_ratio=self.config.last_extend_ratio,
                outputs=self.config.outputs,
                display=False,
                simple_vis=self.config.simple_vis,
                max_students=self.config.max_students,
                max_teachers=self.config.max_teachers,
                layout_callback=lambda payload: self._store_latest("pending_layout", payload),
                frame_callback=lambda payload: self._store_latest("pending_frame", payload),
                finish_callback=lambda payload: self._store_latest("pending_finish", payload),
            )
            self._set_status("正在逐帧检测...")
            pipeline.run()
            self._set_status("检测完成")
        except Exception as exc:
            self._store_latest(
                "pending_error",
                {
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            self._set_status(f"检测失败: {exc}")
        finally:
            self.finished = True

    def poll(self) -> dict[str, Any]:
        with self.lock:
            payload = {
                "layout": self.pending_layout,
                "frame": self.pending_frame,
                "finish": self.pending_finish,
                "error": self.pending_error,
                "status": self.status_message,
                "finished": self.finished,
            }
            self.pending_layout = None
            self.pending_frame = None
            self.pending_finish = None
            self.pending_error = None
        return payload


class FrameProvider:
    def __init__(self, source: str | Path | None):
        self.source = Path(source).expanduser().resolve() if source else None
        self.mode = "placeholder"
        self.error: str | None = None
        self.image_paths: list[Path] = []
        self.image_index = 0
        self.video_index = 0
        self.video_capture = None
        self.video_reader = None
        self._configure()

    def _configure(self) -> None:
        if self.source is None:
            self.mode = "placeholder"
            return

        if self.source.is_dir():
            self.image_paths = sorted(
                [
                    path for path in self.source.iterdir()
                    if path.suffix.lower() in IMAGE_SUFFIXES
                ]
            )
            if self.image_paths:
                self.mode = "image_sequence"
                return
            self.error = f"目录中没有可用图片: {self.source}"
            self.mode = "placeholder"
            return

        if self.source.is_file() and self.source.suffix.lower() in IMAGE_SUFFIXES:
            self.image_paths = [self.source]
            self.mode = "single_image"
            return

        if self.source.is_file() and self.source.suffix.lower() in VIDEO_SUFFIXES:
            self._configure_video()
            return

        self.error = f"暂不支持的素材类型: {self.source}"
        self.mode = "placeholder"

    def _configure_video(self) -> None:
        if cv2 is not None:
            capture = cv2.VideoCapture(str(self.source))
            if capture.isOpened():
                self.video_capture = capture
                self.mode = "opencv_video"
                return
            capture.release()

        if imageio is not None:
            try:
                self.video_reader = imageio.get_reader(str(self.source))
                self.mode = "imageio_video"
                return
            except Exception as exc:  # pragma: no cover - backend dependent
                self.error = f"视频后端不可用: {exc}"
                self.mode = "placeholder"
                return

        self.error = "当前环境没有可用的视频解码后端，请先安装 OpenCV 或 imageio[ffmpeg]。"
        self.mode = "placeholder"

    def reset(self) -> None:
        self.image_index = 0
        self.video_index = 0
        if self.mode == "opencv_video" and self.video_capture is not None:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def description(self) -> str:
        if self.source is None:
            return "模拟巡检"
        return str(self.source)

    def read_next(self) -> Image.Image:
        if self.mode in {"image_sequence", "single_image"} and self.image_paths:
            image_path = self.image_paths[self.image_index]
            if self.mode == "image_sequence":
                self.image_index = (self.image_index + 1) % len(self.image_paths)
            with Image.open(image_path) as image:
                return image.convert("RGB")

        if self.mode == "opencv_video" and self.video_capture is not None:
            ok, frame = self.video_capture.read()
            if not ok:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.video_capture.read()
            if ok:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_frame)

        if self.mode == "imageio_video" and self.video_reader is not None:
            try:
                frame = self.video_reader.get_data(self.video_index)
            except Exception:  # pragma: no cover - backend dependent
                self.video_index = 0
                frame = self.video_reader.get_data(self.video_index)
            self.video_index += 1
            return Image.fromarray(frame)

        return self.placeholder_image()

    def placeholder_image(self) -> Image.Image:
        return render_placeholder_image(
            (1920, 1080),
            "ExamWatchAI",
            "未检测到可用画面，当前展示为占位巡检界面",
            self.error,
        )


class MockInspectionEngine:
    def __init__(self, seats: list[SeatBox], room_name: str = "第一考场"):
        self.seat_numbers = [seat.seat_no for seat in seats]
        self.room_name = room_name
        self.random = random.Random(20260506)
        self.start_time = time.time()
        self.tick_count = 0
        self.active_alerts: dict[int, dict[str, object]] = {}
        self.empty_seats: set[int] = set()
        self.latest_focus: int | None = None
        self.reset()

    def reset(self) -> None:
        self.tick_count = 0
        self.active_alerts.clear()
        self.empty_seats = set(
            self.random.sample(
                self.seat_numbers,
                k=min(3, max(1, len(self.seat_numbers) // 10)),
            )
        )
        self.latest_focus = None

    def clear_alerts(self) -> None:
        self.active_alerts.clear()
        self.latest_focus = None

    def _timestamp(self) -> str:
        return time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(self.start_time + self.tick_count),
        )

    def _spawn_alert(self, level: str | None = None) -> AlertRecord | None:
        candidates = [
            seat_no for seat_no in self.seat_numbers
            if seat_no not in self.empty_seats and seat_no not in self.active_alerts
        ]
        if not candidates:
            return None

        seat_no = self.random.choice(candidates)
        alert_level = level or ("critical" if self.random.random() < 0.28 else "warning")
        behavior = self.random.choice(ALERT_BEHAVIORS)
        duration = self.random.randint(5, 10) if alert_level == "warning" else self.random.randint(6, 12)
        self.active_alerts[seat_no] = {
            "level": alert_level,
            "behavior": behavior,
            "remaining": duration,
        }
        self.latest_focus = seat_no
        return AlertRecord(
            timestamp=self._timestamp(),
            room_name=self.room_name,
            seat_no=seat_no,
            level_label=LEVEL_LABELS[alert_level],
            behavior=behavior,
        )

    def _rotate_empty_seats(self) -> None:
        if not self.seat_numbers or self.tick_count % 15 != 0:
            return

        current_empty = set(self.empty_seats)
        if current_empty:
            current_empty.pop()

        occupied_candidates = [
            seat_no for seat_no in self.seat_numbers
            if seat_no not in current_empty and seat_no not in self.active_alerts
        ]
        if occupied_candidates:
            current_empty.add(self.random.choice(occupied_candidates))
        self.empty_seats = current_empty

    def step(self) -> InspectionSnapshot:
        self.tick_count += 1
        events: list[AlertRecord] = []

        self._rotate_empty_seats()

        for seat_no in list(self.active_alerts):
            alert = self.active_alerts[seat_no]
            alert["remaining"] = int(alert["remaining"]) - 1
            if (
                alert["level"] == "warning"
                and int(alert["remaining"]) in {2, 3}
                and self.random.random() < 0.16
            ):
                alert["level"] = "critical"
                alert["remaining"] = int(alert["remaining"]) + 2
                self.latest_focus = seat_no
                events.append(
                    AlertRecord(
                        timestamp=self._timestamp(),
                        room_name=self.room_name,
                        seat_no=seat_no,
                        level_label=LEVEL_LABELS["critical"],
                        behavior=str(alert["behavior"]),
                    )
                )
            if int(alert["remaining"]) <= 0:
                del self.active_alerts[seat_no]

        desired_alerts = 2 if self.tick_count < 8 else 2 + ((self.tick_count // 12) % 2)
        while len(self.active_alerts) < desired_alerts:
            event = self._spawn_alert()
            if event is None:
                break
            events.append(event)

        if self.tick_count % 9 == 0 and len(self.active_alerts) < 4:
            event = self._spawn_alert(level="warning")
            if event is not None:
                events.append(event)

        status_by_seat = {seat_no: "normal" for seat_no in self.seat_numbers}
        behavior_by_seat = {seat_no: "" for seat_no in self.seat_numbers}

        for seat_no in self.empty_seats:
            status_by_seat[seat_no] = "empty"
            behavior_by_seat[seat_no] = "空位"

        for seat_no, alert in self.active_alerts.items():
            status_by_seat[seat_no] = str(alert["level"])
            behavior_by_seat[seat_no] = str(alert["behavior"])

        occupied_count = len(self.seat_numbers) - len(self.empty_seats)
        alert_count = len(self.active_alerts)
        focus_seat = self.latest_focus
        if focus_seat not in self.active_alerts and focus_seat not in self.empty_seats:
            focus_seat = next(iter(self.active_alerts), None)

        return InspectionSnapshot(
            status_by_seat=status_by_seat,
            behavior_by_seat=behavior_by_seat,
            occupied_count=occupied_count,
            alert_count=alert_count,
            empty_count=len(self.empty_seats),
            focus_seat=focus_seat,
            events=events,
        )


class DetectionDashboard:
    def __init__(
        self,
        root: tk.Tk,
        source: str | Path | None,
        layout_path: str | Path | None,
        room_name: str = "第一考场",
        refresh_ms: int = 850,
        mode: str = "auto",
        live_config: LiveBindingConfig | None = None,
    ):
        self.root = root
        self.root.title("ExamWatchAI 检测巡检界面")
        self.root.geometry("1680x960")
        self.root.minsize(1360, 860)
        self.root.configure(bg=APP_BG)

        self.room_name = room_name
        self.refresh_ms = max(280, int(refresh_ms))
        self.mode = mode
        self.live_config = live_config
        self.live_session: RealBindingSession | None = None
        self.live_status_map: dict[int, str] | None = None
        self.latest_progress_text = "等待巡检数据"
        self.runtime_message = ""
        self.source_error: str | None = None

        self.seats, self.layout_base_size = load_seat_layout(
            Path(layout_path) if layout_path else None
        )
        self.seat_lookup = {seat.seat_no: seat for seat in self.seats}

        self.provider: FrameProvider | None = None
        self.engine: MockInspectionEngine | None = None
        self._configure_data_source(source, room_name)

        self.playing = True
        self.selected_seat: int | None = None
        self.fullscreen = False
        self.records: list[AlertRecord] = []
        self.main_photo = None
        self.detail_photo = None

        self.ui_font_family = self._pick_ui_font_family()
        self.pil_font_small, self.pil_font_medium = self._build_pil_fonts()

        self.metric_vars = {
            "occupied": tk.StringVar(value="0"),
            "alerts": tk.StringVar(value="0"),
            "empty": tk.StringVar(value="0"),
            "rate": tk.StringVar(value="0.0%"),
        }
        self.source_var = tk.StringVar(value=self._source_text())
        self.detail_var = tk.StringVar(value="等待巡检数据")
        self.table_summary_var = tk.StringVar(value="事件 0 条")
        self.search_var = tk.StringVar()

        self._configure_styles()
        self._build_layout()
        self.root.bind("<Escape>", self._leave_fullscreen)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.update_idletasks()
        self._refresh_dashboard()
        self.root.after(self.refresh_ms, self._poll_loop)

    def _source_text(self) -> str:
        if self.live_session is not None:
            text = f"真实检测: {self.live_session.description()} | {self.runtime_message or self.live_session.status_message}"
        elif self.provider is not None:
            text = f"素材源: {self.provider.description()}"
        else:
            text = "素材源: 模拟巡检"
        if self.source_error:
            text += f" | {self.source_error}"
        return text

    def _configure_data_source(self, source: str | Path | None, room_name: str) -> None:
        source_path = Path(source).expanduser().resolve() if source else None
        default_frame = render_placeholder_image(
            (1920, 1080),
            "ExamWatchAI",
            "正在准备检测界面",
            None,
        )

        wants_live = self.mode == "live"
        auto_live = (
            self.mode == "auto"
            and source_path is not None
            and source_path.is_file()
            and source_path.suffix.lower() in VIDEO_SUFFIXES
            and self.live_config is not None
        )

        if wants_live or auto_live:
            if self.live_config is not None:
                self.mode = "live"
                self.live_session = RealBindingSession(self.live_config)
                self.live_session.start()
                self.current_frame = render_placeholder_image(
                    (1920, 1080),
                    "ExamWatchAI",
                    "正在启动真实检测流水线",
                    str(source_path) if source_path else None,
                )
                self.snapshot = InspectionSnapshot(
                    status_by_seat={seat.seat_no: "empty" for seat in self.seats},
                    behavior_by_seat={seat.seat_no: REAL_STATUS_NOTE["empty"] for seat in self.seats},
                    occupied_count=0,
                    alert_count=0,
                    empty_count=len(self.seats),
                    focus_seat=None,
                    events=[],
                )
                self.runtime_message = "模型启动中"
                return
            if wants_live:
                self.source_error = "未提供真实检测配置，已回退到模拟巡检。"

        self.provider = FrameProvider(source)
        self.mode = "mock"
        self.source_error = self.provider.error
        self.engine = MockInspectionEngine(self.seats, room_name=room_name)
        self.current_frame = self.provider.read_next()
        self.snapshot = self.engine.step()
        self.runtime_message = "模拟巡检中"
        if self.provider.error and source is None:
            self.current_frame = default_frame

    def _pick_ui_font_family(self) -> str:
        candidates = [
            "Noto Sans CJK SC",
            "Microsoft YaHei UI",
            "Microsoft YaHei",
            "PingFang SC",
            "WenQuanYi Micro Hei",
            "DejaVu Sans",
        ]
        available = set(tkfont.families())
        for candidate in candidates:
            if candidate in available:
                return candidate
        return "TkDefaultFont"

    def _build_pil_fonts(
        self,
    ) -> tuple[
        ImageFont.FreeTypeFont | ImageFont.ImageFont,
        ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ]:
        return (
            load_pil_font(22, weight="medium"),
            load_pil_font(28, weight="bold"),
        )

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Dashboard.Treeview",
            background=PANEL_MUTED_BG,
            foreground=TEXT_PRIMARY,
            fieldbackground=PANEL_MUTED_BG,
            bordercolor=PANEL_BORDER,
            rowheight=32,
            font=(self.ui_font_family, 11),
        )
        style.configure(
            "Dashboard.Treeview.Heading",
            background=PANEL_HEADER_BG,
            foreground=TEXT_PRIMARY,
            bordercolor=PANEL_BORDER,
            relief="flat",
            font=(self.ui_font_family, 11, "bold"),
        )
        style.map(
            "Dashboard.Treeview",
            background=[("selected", "#355284")],
            foreground=[("selected", TEXT_PRIMARY)],
        )

    def _build_layout(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        shell = tk.Frame(self.root, bg=APP_BG)
        shell.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        shell.grid_rowconfigure(0, weight=5)
        shell.grid_rowconfigure(1, weight=4)
        shell.grid_rowconfigure(2, weight=0)
        shell.grid_columnconfigure(0, weight=3)
        shell.grid_columnconfigure(1, weight=2)

        main_panel = self._create_panel(
            shell,
            "实时检测画面",
            "左侧主监控 + 座位框高亮",
        )
        main_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
        self._build_main_panel(main_panel)

        seat_panel = self._create_panel(
            shell,
            "座位态势图",
            "5列 × 6排 座位状态总览",
        )
        seat_panel.grid(row=0, column=1, sticky="nsew", pady=(0, 12))
        self._build_seat_panel(seat_panel)

        table_panel = self._create_panel(
            shell,
            "状态记录",
            "实时绑定与席位变化",
        )
        table_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        self._build_table_panel(table_panel)

        detail_panel = self._create_panel(
            shell,
            "局部预览",
            "当前聚焦座位特写",
        )
        detail_panel.grid(row=1, column=1, sticky="nsew")
        self._build_detail_panel(detail_panel)

        footer = tk.Frame(shell, bg=APP_BG)
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        footer.grid_columnconfigure(0, weight=1)
        self._build_footer(footer)

    def _create_panel(self, parent: tk.Widget, title: str, subtitle: str) -> tk.Frame:
        frame = tk.Frame(
            parent,
            bg=PANEL_BG,
            highlightbackground=PANEL_BORDER,
            highlightthickness=1,
        )
        header = tk.Frame(frame, bg=PANEL_HEADER_BG, height=46)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text=title,
            bg=PANEL_HEADER_BG,
            fg=TEXT_PRIMARY,
            font=(self.ui_font_family, 13, "bold"),
        ).pack(side="left", padx=16)
        tk.Label(
            header,
            text=subtitle,
            bg=PANEL_HEADER_BG,
            fg=TEXT_SECONDARY,
            font=(self.ui_font_family, 10),
        ).pack(side="right", padx=16)
        body = tk.Frame(frame, bg=PANEL_BG)
        body.pack(fill="both", expand=True, padx=14, pady=14)
        frame.body = body  # type: ignore[attr-defined]
        return frame

    def _build_main_panel(self, panel: tk.Frame) -> None:
        body = panel.body  # type: ignore[attr-defined]
        metrics = tk.Frame(body, bg=PANEL_BG)
        metrics.pack(fill="x")

        self._create_metric_card(metrics, "在座人数", self.metric_vars["occupied"], STATUS_EMPTY).pack(
            side="left", fill="x", expand=True, padx=(0, 10)
        )
        self._create_metric_card(metrics, "预警席位", self.metric_vars["alerts"], STATUS_WARNING).pack(
            side="left", fill="x", expand=True, padx=(0, 10)
        )
        self._create_metric_card(metrics, "空位数", self.metric_vars["empty"], STATUS_EMPTY).pack(
            side="left", fill="x", expand=True, padx=(0, 10)
        )
        self._create_metric_card(metrics, "异常率", self.metric_vars["rate"], ACCENT).pack(
            side="left", fill="x", expand=True
        )

        self.main_image_label = tk.Label(
            body,
            bg="#0F1327",
            bd=0,
            relief="flat",
        )
        self.main_image_label.pack(fill="both", expand=True, pady=(14, 0))

    def _create_metric_card(
        self,
        parent: tk.Widget,
        title: str,
        variable: tk.StringVar,
        accent_color: str,
    ) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg=PANEL_MUTED_BG,
            highlightbackground=PANEL_BORDER,
            highlightthickness=1,
            padx=14,
            pady=10,
        )
        indicator = tk.Frame(card, bg=accent_color, width=6, height=44)
        indicator.pack(side="left", fill="y", padx=(0, 12))
        indicator.pack_propagate(False)
        text_wrap = tk.Frame(card, bg=PANEL_MUTED_BG)
        text_wrap.pack(side="left", fill="both", expand=True)
        tk.Label(
            text_wrap,
            text=title,
            bg=PANEL_MUTED_BG,
            fg=TEXT_SECONDARY,
            font=(self.ui_font_family, 10),
        ).pack(anchor="w")
        tk.Label(
            text_wrap,
            textvariable=variable,
            bg=PANEL_MUTED_BG,
            fg=TEXT_PRIMARY,
            font=(self.ui_font_family, 18, "bold"),
        ).pack(anchor="w", pady=(4, 0))
        return card

    def _build_seat_panel(self, panel: tk.Frame) -> None:
        body = panel.body  # type: ignore[attr-defined]
        legend = tk.Frame(body, bg=PANEL_BG)
        legend.pack(fill="x")
        self._create_legend_badge(legend, "正常", STATUS_NORMAL, STATUS_NORMAL_TEXT).pack(side="left", padx=(0, 10))
        self._create_legend_badge(legend, "空位", STATUS_EMPTY, STATUS_EMPTY_TEXT).pack(side="left", padx=(0, 10))
        self._create_legend_badge(legend, "预警", STATUS_WARNING, "#22170F").pack(side="left", padx=(0, 10))
        self._create_legend_badge(legend, "重点", STATUS_CRITICAL, "#2B0F12").pack(side="left")

        self.seat_canvas = tk.Canvas(
            body,
            bg=PANEL_MUTED_BG,
            highlightthickness=0,
            bd=0,
        )
        self.seat_canvas.pack(fill="both", expand=True, pady=(14, 0))
        self.seat_canvas.bind("<Configure>", lambda _event: self._render_seat_map())

    def _create_legend_badge(self, parent: tk.Widget, text: str, bg_color: str, fg_color: str) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            bg=bg_color,
            fg=fg_color,
            padx=12,
            pady=4,
            font=(self.ui_font_family, 10, "bold"),
        )

    def _build_table_panel(self, panel: tk.Frame) -> None:
        body = panel.body  # type: ignore[attr-defined]
        table_wrap = tk.Frame(body, bg=PANEL_BG)
        table_wrap.pack(fill="both", expand=True)
        columns = ("time", "room", "seat", "level", "behavior")
        self.alert_tree = ttk.Treeview(
            table_wrap,
            columns=columns,
            show="headings",
            style="Dashboard.Treeview",
        )
        headings = {
            "time": "检测时刻",
            "room": "考场名称",
            "seat": "座位号",
            "level": "状态等级",
            "behavior": "状态说明",
        }
        widths = {
            "time": 168,
            "room": 190,
            "seat": 96,
            "level": 140,
            "behavior": 180,
        }
        for key in columns:
            self.alert_tree.heading(key, text=headings[key])
            self.alert_tree.column(key, width=widths[key], anchor="center")

        scrollbar = ttk.Scrollbar(table_wrap, orient="vertical", command=self.alert_tree.yview)
        self.alert_tree.configure(yscrollcommand=scrollbar.set)
        self.alert_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.alert_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        footer = tk.Frame(body, bg=PANEL_BG)
        footer.pack(fill="x", pady=(10, 0))
        tk.Label(
            footer,
            textvariable=self.table_summary_var,
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            font=(self.ui_font_family, 10),
        ).pack(side="left")

    def _build_detail_panel(self, panel: tk.Frame) -> None:
        body = panel.body  # type: ignore[attr-defined]
        self.detail_image_label = tk.Label(body, bg="#0F1327", bd=0)
        self.detail_image_label.pack(fill="both", expand=True)
        tk.Label(
            body,
            textvariable=self.detail_var,
            bg=PANEL_BG,
            fg=TEXT_SECONDARY,
            font=(self.ui_font_family, 11),
            anchor="w",
            justify="left",
        ).pack(fill="x", pady=(12, 0))

    def _build_footer(self, footer: tk.Frame) -> None:
        left = tk.Frame(footer, bg=APP_BG)
        left.grid(row=0, column=0, sticky="w")
        right = tk.Frame(footer, bg=APP_BG)
        right.grid(row=0, column=1, sticky="e")

        play_label = "冻结画面" if self.mode == "live" else "暂停巡检"
        self.play_button = self._make_button(left, play_label, self._toggle_playing, filled=True)
        self.play_button.pack(side="left")
        self._make_button(left, "复位界面", self._reset_dashboard).pack(side="left", padx=(10, 0))
        self._make_button(left, "清空记录", self._clear_records).pack(side="left", padx=(10, 0))
        self.load_button = self._make_button(left, "加载素材", self._load_source)
        self.load_button.pack(side="left", padx=(10, 0))
        if self.mode == "live":
            self.load_button.configure(state="disabled", disabledforeground="#7E86AD")

        tk.Label(
            right,
            text="定位座位",
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=(self.ui_font_family, 10),
        ).pack(side="left", padx=(0, 8))
        entry = tk.Entry(
            right,
            textvariable=self.search_var,
            width=8,
            bg="#F3F6FF",
            fg="#1A1F33",
            relief="flat",
            font=(self.ui_font_family, 11),
            justify="center",
        )
        entry.pack(side="left", ipady=6)
        entry.bind("<Return>", lambda _event: self._focus_seat_from_input())
        self._make_button(right, "定位", self._focus_seat_from_input).pack(side="left", padx=(10, 0))
        self._make_button(right, "全屏", self._toggle_fullscreen).pack(side="left", padx=(10, 0))

        status_bar = tk.Frame(footer, bg=APP_BG)
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        tk.Label(
            status_bar,
            textvariable=self.source_var,
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=(self.ui_font_family, 10),
            anchor="w",
        ).pack(fill="x")

    def _make_button(self, parent: tk.Widget, text: str, command, filled: bool = False) -> tk.Button:
        bg = ACCENT if filled else PANEL_HEADER_BG
        fg = "#0F1327" if filled else TEXT_PRIMARY
        active_bg = "#7BC8FF" if filled else "#3B436E"
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=fg,
            relief="flat",
            padx=16,
            pady=7,
            cursor="hand2",
            font=(self.ui_font_family, 10, "bold"),
        )

    def _toggle_playing(self) -> None:
        self.playing = not self.playing
        if self.mode == "live":
            self.play_button.configure(text="冻结画面" if self.playing else "恢复画面")
        else:
            self.play_button.configure(text="暂停巡检" if self.playing else "继续巡检")

    def _reset_dashboard(self) -> None:
        self.records.clear()
        for item_id in self.alert_tree.get_children():
            self.alert_tree.delete(item_id)
        self.selected_seat = None
        self.live_status_map = None
        if self.mode == "mock" and self.provider is not None and self.engine is not None:
            self.provider.reset()
            self.engine.reset()
            self.snapshot = self.engine.step()
            self.current_frame = self.provider.read_next()
        self._refresh_dashboard()

    def _clear_records(self) -> None:
        self.records.clear()
        for item_id in self.alert_tree.get_children():
            self.alert_tree.delete(item_id)
        self.table_summary_var.set("事件 0 条")
        self._refresh_dashboard()

    def _load_source(self) -> None:
        if self.mode == "live":
            self.detail_var.set("真实检测模式下，请重新启动并通过 --source 指定视频。")
            return
        file_path = filedialog.askopenfilename(
            title="选择图片或视频素材",
            filetypes=[
                ("Media", "*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*"),
            ],
        )
        selected_path = file_path
        if not selected_path:
            selected_path = filedialog.askdirectory(title="选择截图序列目录")
        if not selected_path:
            return

        self.provider = FrameProvider(selected_path)
        self.provider.reset()
        self.source_error = self.provider.error
        self.current_frame = self.provider.read_next()
        self.source_var.set(self._source_text())
        self._refresh_dashboard()

    def _focus_seat_from_input(self) -> None:
        raw_value = self.search_var.get().strip()
        if not raw_value:
            return
        try:
            seat_no = int(raw_value)
        except ValueError:
            self.detail_var.set(f"无法定位: 输入的座位号 `{raw_value}` 不是数字。")
            return

        if seat_no not in self.seat_lookup:
            self.detail_var.set(f"无法定位: 当前布局中没有座位 {seat_no}。")
            return

        self.selected_seat = seat_no
        self._refresh_dashboard()

    def _toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def _leave_fullscreen(self, _event=None) -> None:
        if self.fullscreen:
            self.fullscreen = False
            self.root.attributes("-fullscreen", False)

    def _apply_live_layout(self, payload: dict[str, Any]) -> None:
        zones = payload.get("zones") or []
        frame_width = int(payload.get("frame_width", self.layout_base_size[0]))
        frame_height = int(payload.get("frame_height", self.layout_base_size[1]))
        self.seats, self.layout_base_size = build_layout_from_zones(
            zones,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        self.seat_lookup = {seat.seat_no: seat for seat in self.seats}
        if self.selected_seat is not None and self.selected_seat not in self.seat_lookup:
            self.selected_seat = None

    def _image_from_bgr_frame(self, frame_bgr: Any) -> Image.Image:
        if frame_bgr is None:
            return render_placeholder_image(
                (1920, 1080),
                "ExamWatchAI",
                "真实检测尚未返回画面",
                self.runtime_message or self.source_error,
            )
        if np is not None and isinstance(frame_bgr, np.ndarray):
            if frame_bgr.ndim == 3 and frame_bgr.shape[2] >= 3:
                rgb = frame_bgr[:, :, :3][:, :, ::-1]
                return Image.fromarray(rgb.copy())
            if frame_bgr.ndim == 2:
                return Image.fromarray(frame_bgr)
        return render_placeholder_image(
            (1920, 1080),
            "ExamWatchAI",
            "收到的画面格式暂不支持",
            None,
        )

    def _apply_live_frame(self, payload: dict[str, Any]) -> None:
        self.current_frame = self._image_from_bgr_frame(payload.get("frame_bgr"))
        frame_idx = int(payload.get("frame_idx", 0))
        fps = float(payload.get("fps", 25.0))
        rows = payload.get("rows") or []
        snapshot, self.live_status_map = build_snapshot_from_binding_rows(
            rows=rows,
            seat_numbers=[seat.seat_no for seat in self.seats],
            room_name=self.room_name,
            frame_idx=frame_idx,
            fps=fps,
            previous_status=self.live_status_map,
        )
        self.snapshot = snapshot
        total_frames = payload.get("total_frames") or "?"
        self.latest_progress_text = (
            f"帧 {frame_idx}/{total_frames} | "
            f"学生 {payload.get('student_count', 0)} | "
            f"教师 {payload.get('teacher_count', 0)} | "
            f"教室内未确认 {payload.get('unknown_count', 0)}"
        )
        self._append_events(snapshot.events)
        self._refresh_dashboard()

    def _poll_live_session(self) -> None:
        if self.live_session is None:
            return
        payload = self.live_session.poll()
        self.runtime_message = str(payload.get("status") or self.runtime_message)
        should_refresh = False
        if payload.get("layout") is not None:
            self._apply_live_layout(payload["layout"])
            should_refresh = True
        if payload.get("error") is not None:
            error_payload = payload["error"]
            self.source_error = str(error_payload.get("message", "未知错误"))
            self.detail_var.set(self.source_error)
            should_refresh = True
        if payload.get("frame") is not None and self.playing:
            self._apply_live_frame(payload["frame"])
        elif payload.get("frame") is not None:
            self.latest_progress_text = "画面已冻结，检测仍在后台继续。"
        if payload.get("finish") is not None:
            self.runtime_message = "检测完成"
            self.source_error = None
            should_refresh = True
        self.source_var.set(self._source_text())
        if should_refresh and payload.get("frame") is None:
            self._refresh_dashboard()

    def _poll_loop(self) -> None:
        if self.mode == "live":
            self._poll_live_session()
        elif self.playing and self.provider is not None and self.engine is not None:
            self.current_frame = self.provider.read_next()
            self.snapshot = self.engine.step()
            self._append_events(self.snapshot.events)
            self._refresh_dashboard()
        self.root.after(self.refresh_ms, self._poll_loop)

    def _append_events(self, events: list[AlertRecord]) -> None:
        if not events:
            return
        for event in reversed(events):
            self.records.insert(0, event)
            self.alert_tree.insert(
                "",
                0,
                values=(
                    event.timestamp,
                    event.room_name,
                    event.seat_no,
                    event.level_label,
                    event.behavior,
                ),
                tags=(f"seat-{event.seat_no}",),
            )
        self.records = self.records[:200]
        children = self.alert_tree.get_children()
        for item_id in children[200:]:
            self.alert_tree.delete(item_id)

    def _refresh_dashboard(self) -> None:
        occupied_count = self.snapshot.occupied_count
        alert_count = self.snapshot.alert_count
        empty_count = self.snapshot.empty_count
        alert_rate = 0.0 if occupied_count <= 0 else (alert_count / occupied_count) * 100.0

        self.metric_vars["occupied"].set(str(occupied_count))
        self.metric_vars["alerts"].set(str(alert_count))
        self.metric_vars["empty"].set(str(empty_count))
        self.metric_vars["rate"].set(f"{alert_rate:.1f}%")
        self.table_summary_var.set(f"事件 {len(self.records)} 条")
        self.source_var.set(self._source_text())

        self._render_main_view()
        self._render_seat_map()
        self._render_detail_view()

    def _scaled_box(self, seat: SeatBox, frame_size: tuple[int, int]) -> tuple[float, float, float, float]:
        base_w, base_h = self.layout_base_size
        frame_w, frame_h = frame_size
        sx = frame_w / max(1.0, float(base_w))
        sy = frame_h / max(1.0, float(base_h))
        return (
            seat.x1 * sx,
            seat.y1 * sy,
            seat.x2 * sx,
            seat.y2 * sy,
        )

    def _status_colors(self, status: str) -> tuple[str, tuple[int, int, int, int], tuple[int, int, int]]:
        if status == "critical":
            return STATUS_CRITICAL, (255, 94, 98, 78), (255, 94, 98)
        if status == "warning":
            return STATUS_WARNING, (255, 138, 61, 78), (255, 138, 61)
        if status == "empty":
            return STATUS_EMPTY, (17, 186, 134, 58), (17, 186, 134)
        return STATUS_NORMAL, (59, 76, 123, 34), (102, 224, 179)

    def _overlay_tag(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        text: str,
        rgb_color: tuple[int, int, int],
    ) -> None:
        font = self.pil_font_small
        left, top = xy
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad_x = 12
        pad_y = 8
        rect = (
            left,
            top,
            left + text_w + pad_x * 2,
            top + text_h + pad_y * 2,
        )
        draw.rounded_rectangle(rect, radius=10, fill=(18, 22, 40, 220), outline=rgb_color, width=2)
        draw.text((left + pad_x, top + pad_y - 2), text, fill=TEXT_PRIMARY, font=font)

    def _render_main_view(self) -> None:
        frame = self.current_frame.copy().convert("RGBA")
        draw = ImageDraw.Draw(frame, "RGBA")
        focus_seat = self.selected_seat or self.snapshot.focus_seat

        for seat in self.seats:
            box = self._scaled_box(seat, frame.size)
            status = self.snapshot.status_by_seat.get(seat.seat_no, "normal")
            behavior = self.snapshot.behavior_by_seat.get(seat.seat_no, "")
            _, fill_rgba, outline_rgb = self._status_colors(status)

            if status in {"warning", "critical"}:
                draw.rounded_rectangle(box, radius=12, fill=fill_rgba, outline=outline_rgb, width=3)
                tag_text = f"{seat.seat_no:02d}  {behavior or STATUS_LABELS[status]}"
                tag_y = max(16.0, box[1] - 42.0)
                self._overlay_tag(draw, (box[0], tag_y), tag_text, outline_rgb)

            if focus_seat == seat.seat_no:
                draw.rounded_rectangle(box, radius=14, outline=ACCENT, width=4)
                self._overlay_tag(draw, (box[0], box[3] + 10.0), f"聚焦座位 {seat.seat_no:02d}", (94, 181, 255))

        if self.provider is not None and self.provider.error:
            self._overlay_tag(draw, (24.0, 24.0), self.provider.error[:46], (255, 156, 109))

        target_size = self._widget_target_size(self.main_image_label, default=(940, 540))
        rendered = ImageOps.contain(frame.convert("RGB"), target_size, method=RESAMPLING_LANCZOS)
        self.main_photo = ImageTk.PhotoImage(rendered)
        self.main_image_label.configure(image=self.main_photo)

    def _render_seat_map(self) -> None:
        canvas = self.seat_canvas
        canvas.delete("all")
        width = max(200, canvas.winfo_width())
        height = max(240, canvas.winfo_height())

        num_cols = max((seat.col_index for seat in self.seats), default=0) + 1
        num_rows = max((seat.row_index for seat in self.seats), default=0) + 1
        if num_cols <= 0 or num_rows <= 0:
            return

        outer_pad_x = 20
        outer_pad_y = 18
        bottom_pad = 78
        gap_x = 14
        gap_y = 12

        usable_w = width - outer_pad_x * 2 - gap_x * (num_cols - 1)
        usable_h = height - outer_pad_y * 2 - bottom_pad - gap_y * (num_rows - 1)
        cell_w = usable_w / max(1, num_cols)
        cell_h = usable_h / max(1, num_rows)

        focus_seat = self.selected_seat or self.snapshot.focus_seat
        for seat in self.seats:
            x1 = outer_pad_x + seat.col_index * (cell_w + gap_x)
            y1 = outer_pad_y + seat.row_index * (cell_h + gap_y)
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            status = self.snapshot.status_by_seat.get(seat.seat_no, "normal")
            behavior = self.snapshot.behavior_by_seat.get(seat.seat_no, "")
            if status == "critical":
                fill = STATUS_CRITICAL
                text_color = "#2B0F12"
            elif status == "warning":
                fill = STATUS_WARNING
                text_color = "#2A170B"
            elif status == "empty":
                fill = STATUS_EMPTY
                text_color = STATUS_EMPTY_TEXT
            else:
                fill = STATUS_NORMAL
                text_color = STATUS_NORMAL_TEXT

            outline = ACCENT if focus_seat == seat.seat_no else PANEL_BORDER
            outline_width = 3 if focus_seat == seat.seat_no else 1
            canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=outline_width)
            canvas.create_text(
                x1 + 18,
                y1 + 20,
                text=f"{seat.seat_no:02d}",
                fill=text_color,
                anchor="w",
                font=(self.ui_font_family, 12, "bold"),
            )
            if status != "normal":
                canvas.create_text(
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0 + 5,
                    text=behavior or STATUS_LABELS[status],
                    fill=text_color,
                    width=max(70, cell_w - 14),
                    justify="center",
                    font=(self.ui_font_family, 9, "bold"),
                )

        podium_w = min(160, width * 0.34)
        podium_h = 42
        podium_x1 = (width - podium_w) / 2.0
        podium_y1 = height - 54
        canvas.create_rectangle(
            podium_x1,
            podium_y1,
            podium_x1 + podium_w,
            podium_y1 + podium_h,
            fill="#5C87CA",
            outline="#7DB0FF",
            width=2,
        )
        canvas.create_text(
            width / 2.0,
            podium_y1 + podium_h / 2.0,
            text="讲台",
            fill=TEXT_PRIMARY,
            font=(self.ui_font_family, 12, "bold"),
        )

    def _render_detail_view(self) -> None:
        focus_seat = self.selected_seat or self.snapshot.focus_seat
        if focus_seat is None and self.seats:
            focus_seat = self.seats[0].seat_no

        if focus_seat is None or focus_seat not in self.seat_lookup:
            self.detail_var.set("暂无可聚焦的座位。")
            return

        seat = self.seat_lookup[focus_seat]
        frame = self.current_frame.copy().convert("RGBA")
        draw = ImageDraw.Draw(frame, "RGBA")
        box = self._scaled_box(seat, frame.size)
        status = self.snapshot.status_by_seat.get(focus_seat, "normal")
        behavior = self.snapshot.behavior_by_seat.get(focus_seat, STATUS_LABELS[status])
        _, fill_rgba, outline_rgb = self._status_colors(status)
        draw.rounded_rectangle(box, radius=14, fill=fill_rgba, outline=outline_rgb, width=4)

        x1, y1, x2, y2 = box
        pad_x = max(120.0, (x2 - x1) * 1.3)
        pad_y = max(100.0, (y2 - y1) * 1.8)
        crop_box = (
            max(0.0, x1 - pad_x),
            max(0.0, y1 - pad_y),
            min(frame.size[0], x2 + pad_x),
            min(frame.size[1], y2 + pad_y),
        )
        crop = frame.crop(tuple(int(value) for value in crop_box))

        target_size = self._widget_target_size(self.detail_image_label, default=(620, 360))
        rendered = ImageOps.contain(crop.convert("RGB"), target_size, method=RESAMPLING_LANCZOS)
        self.detail_photo = ImageTk.PhotoImage(rendered)
        self.detail_image_label.configure(image=self.detail_photo)

        self.detail_var.set(
            f"聚焦座位 {focus_seat:02d} | 状态: {STATUS_LABELS.get(status, '正常')} | "
            f"行为: {behavior or '正常'} | {self.latest_progress_text}"
        )

    def _widget_target_size(self, widget: tk.Widget, default: tuple[int, int]) -> tuple[int, int]:
        width = widget.winfo_width()
        height = widget.winfo_height()
        if width < 40 or height < 40:
            return default
        return width, height

    def _on_tree_select(self, _event) -> None:
        selected = self.alert_tree.selection()
        if not selected:
            return
        values = self.alert_tree.item(selected[0], "values")
        if len(values) < 3:
            return
        try:
            seat_no = int(values[2])
        except (TypeError, ValueError):
            return
        self.selected_seat = seat_no
        self.search_var.set(str(seat_no))
        self._refresh_dashboard()

    def _on_close(self) -> None:
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_source = resolve_demo_source(project_root)
    default_layout = project_root / "detect" / "seats.json"
    default_person_weights = project_root / "exam_seat_binding" / "weight" / "yolo11speopel.pt"
    default_desk_weights = project_root / "exam_seat_binding" / "weight" / "yolo11desk.pt"
    default_output_dir = project_root / "exam_seat_binding" / "output"

    parser = argparse.ArgumentParser(description="ExamWatchAI 检测巡检 UI")
    parser.add_argument(
        "--mode",
        choices=["auto", "mock", "live"],
        default="auto",
        help="数据模式：auto 优先真实检测，mock 为演示模式，live 强制真实检测",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(default_source) if default_source else "",
        help="输入素材；mock 支持图片/目录/视频，live 建议传视频",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default=str(default_layout),
        help="座位布局 JSON，默认 detect/seats.json",
    )
    parser.add_argument(
        "--room-name",
        type=str,
        default="第一考场",
        help="界面中显示的考场名称",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=850,
        help="界面刷新间隔，单位毫秒",
    )
    parser.add_argument("--weights", type=str, default=str(default_person_weights), help="人物模型权重")
    parser.add_argument("--desk-weights", type=str, default=str(default_desk_weights), help="桌子模型权重")
    parser.add_argument("--output", type=str, default=str(default_output_dir), help="真实检测输出目录")
    parser.add_argument("--pipeline-outputs", type=str, default="csv,json,zone_map", help="真实检测附带保存的输出")
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--desk-conf", type=float, default=0.70)
    parser.add_argument("--desk-iou", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--classes", default=None)
    parser.add_argument("--desk-mode", default="auto", choices=["normal", "scheme1", "scheme2", "auto"])
    parser.add_argument("--desk-num-cols", type=int, default=5)
    parser.add_argument("--desk-required-per-col", type=int, default=6)
    parser.add_argument("--reference-max-frames", type=int, default=120)
    parser.add_argument("--reference-sample-step", type=int, default=5)
    parser.add_argument("--first-extend", type=float, default=0.6)
    parser.add_argument("--last-extend", type=float, default=1.2)
    parser.add_argument("--confirm-seconds", type=float, default=3.0)
    parser.add_argument("--confirm-ratio", type=float, default=0.50)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--re-confirm-seconds", type=float, default=5.0)
    parser.add_argument("--lock-seconds", type=float, default=20.0)
    parser.add_argument("--release-miss-seconds", type=float, default=12.0)
    parser.add_argument("--pending-switch-seconds", type=float, default=1.2)
    parser.add_argument("--pending-hold-seconds", type=float, default=2.0)
    parser.add_argument("--simple-vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-students", type=int, default=30)
    parser.add_argument("--max-teachers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        live_outputs = parse_outputs(args.pipeline_outputs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    live_config = LiveBindingConfig(
        source=args.source,
        person_weights=args.weights,
        desk_weights=args.desk_weights,
        output_dir=args.output,
        person_conf=args.conf,
        person_iou=args.iou,
        desk_conf=args.desk_conf,
        desk_iou=args.desk_iou,
        device=args.device,
        img_size=args.img_size,
        half=args.half,
        tracker=args.tracker,
        person_classes=parse_classes(args.classes),
        desk_mode=args.desk_mode,
        desk_num_cols=args.desk_num_cols,
        desk_required_per_col=args.desk_required_per_col,
        reference_max_frames=args.reference_max_frames,
        reference_sample_step=args.reference_sample_step,
        confirm_seconds=args.confirm_seconds,
        confirm_ratio=args.confirm_ratio,
        hold_seconds=args.hold_seconds,
        re_confirm_seconds=args.re_confirm_seconds,
        lock_seconds=args.lock_seconds,
        release_miss_seconds=args.release_miss_seconds,
        pending_switch_seconds=args.pending_switch_seconds,
        pending_hold_seconds=args.pending_hold_seconds,
        first_extend_ratio=args.first_extend,
        last_extend_ratio=args.last_extend,
        outputs=live_outputs,
        simple_vis=args.simple_vis,
        max_students=args.max_students,
        max_teachers=args.max_teachers,
    )
    root = tk.Tk()
    app = DetectionDashboard(
        root=root,
        source=args.source or None,
        layout_path=args.layout or None,
        room_name=args.room_name,
        refresh_ms=args.refresh_ms,
        mode=args.mode,
        live_config=live_config,
    )
    root._dashboard = app  # type: ignore[attr-defined]
    root.mainloop()


if __name__ == "__main__":
    main()
