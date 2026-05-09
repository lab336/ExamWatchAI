from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

QT_AVAILABLE = False
QT_IMPORT_ERROR = None
try:
    from PyQt6.QtCore import QRectF, QSize, Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QAbstractItemView,
        QComboBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QSlider,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency
    QT_IMPORT_ERROR = exc

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


def load_person_binding_pipeline_class():
    try:
        from exam_seat_binding.person_desk_binding_v3 import PersonDeskBindingPipelineV3

        return PersonDeskBindingPipelineV3
    except Exception as package_exc:
        module_path = Path(__file__).resolve().with_name("person_desk_binding_v3.py")
        spec = importlib.util.spec_from_file_location(
            "person_desk_binding_v3_fallback",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"无法加载人桌绑定模块: {module_path}"
            ) from package_exc
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pipeline_cls = getattr(module, "PersonDeskBindingPipelineV3", None)
        if pipeline_cls is None:
            raise RuntimeError(
                f"模块中未找到 PersonDeskBindingPipelineV3: {module_path}"
            ) from package_exc
        return pipeline_cls


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
    frame_callback_stride: int = 3
    frame_callback_max_width: int = 1280


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
        self.pending_frames: deque[dict[str, Any]] = deque(maxlen=24)
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

    def _store_frame(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.pending_frames.append(payload)

    def _run(self) -> None:
        self.started = True
        self._set_status("正在初始化检测模型...")
        try:
            pipeline_cls = load_person_binding_pipeline_class()

            pipeline = pipeline_cls(
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
                frame_callback_stride=self.config.frame_callback_stride,
                frame_callback_max_width=self.config.frame_callback_max_width,
                layout_callback=lambda payload: self._store_latest("pending_layout", payload),
                frame_callback=self._store_frame,
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
                "frames": list(self.pending_frames),
                "finish": self.pending_finish,
                "error": self.pending_error,
                "status": self.status_message,
                "finished": self.finished,
            }
            self.pending_layout = None
            self.pending_frames.clear()
            self.pending_finish = None
            self.pending_error = None
        return payload


class FrameProvider:
    def __init__(self, source: str | Path | None):
        self.source = Path(source).expanduser().resolve() if source else None
        self.mode = "placeholder"
        self.error: str | None = None
        self.fps = 25.0
        self.image_paths: list[Path] = []
        self.image_index = 0
        self.video_index = 0
        self.total_frames: int | None = None
        self.current_frame_number = 0
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
                self.fps = 2.0
                self.total_frames = len(self.image_paths)
                return
            self.error = f"目录中没有可用图片: {self.source}"
            self.mode = "placeholder"
            return

        if self.source.is_file() and self.source.suffix.lower() in IMAGE_SUFFIXES:
            self.image_paths = [self.source]
            self.mode = "single_image"
            self.fps = 1.0
            self.total_frames = 1
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
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                if fps > 1e-6:
                    self.fps = fps
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if total_frames > 0:
                    self.total_frames = total_frames
                self.video_capture = capture
                self.mode = "opencv_video"
                return
            capture.release()

        if imageio is not None:
            try:
                self.video_reader = imageio.get_reader(str(self.source))
                try:
                    meta = self.video_reader.get_meta_data()
                    fps = float(meta.get("fps") or 0.0)
                    if fps > 1e-6:
                        self.fps = fps
                    nframes = meta.get("nframes")
                    if isinstance(nframes, int) and nframes > 0:
                        self.total_frames = nframes
                except Exception:
                    pass
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
        self.current_frame_number = 0
        if self.mode == "opencv_video" and self.video_capture is not None:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def description(self) -> str:
        if self.source is None:
            return "模拟巡检"
        return str(self.source)

    def playback_fps(self) -> float:
        return max(0.1, float(self.fps))

    def read_next(self) -> Image.Image:
        if self.mode in {"image_sequence", "single_image"} and self.image_paths:
            current_index = self.image_index
            image_path = self.image_paths[current_index]
            if self.mode == "image_sequence":
                self.image_index = (self.image_index + 1) % len(self.image_paths)
                self.current_frame_number = current_index
            else:
                self.current_frame_number = 0
            with Image.open(image_path) as image:
                return image.convert("RGB")

        if self.mode == "opencv_video" and self.video_capture is not None:
            ok, frame = self.video_capture.read()
            if not ok:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.video_capture.read()
            if ok:
                self.current_frame_number = max(
                    0,
                    int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1,
                )
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_frame)

        if self.mode == "imageio_video" and self.video_reader is not None:
            try:
                frame = self.video_reader.get_data(self.video_index)
            except Exception:  # pragma: no cover - backend dependent
                self.video_index = 0
                frame = self.video_reader.get_data(self.video_index)
            self.current_frame_number = self.video_index
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

    def progress_state(self) -> tuple[int, int | None, float]:
        return self.current_frame_number, self.total_frames, self.playback_fps()

    def can_seek(self) -> bool:
        return self.mode in {"image_sequence", "single_image", "opencv_video", "imageio_video"}

    def seek(self, frame_idx: int) -> Image.Image:
        if not self.can_seek():
            return self.read_next()
        target = max(0, int(frame_idx))
        if self.mode in {"image_sequence", "single_image"} and self.image_paths:
            if self.mode == "single_image":
                self.image_index = 0
            else:
                max_index = max(0, len(self.image_paths) - 1)
                self.image_index = min(target, max_index)
            return self.read_next()
        if self.mode == "opencv_video" and self.video_capture is not None:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = self.video_capture.read()
            if ok:
                self.current_frame_number = target
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_frame)
            return self.placeholder_image()
        if self.mode == "imageio_video" and self.video_reader is not None:
            self.video_index = target
            try:
                frame = self.video_reader.get_data(self.video_index)
            except Exception:
                self.video_index = 0
                frame = self.video_reader.get_data(self.video_index)
            self.current_frame_number = self.video_index
            self.video_index += 1
            return Image.fromarray(frame)
        return self.placeholder_image()


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


if QT_AVAILABLE:
    QT_FONT_CANDIDATES = [
        "Noto Sans CJK SC",
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
        "SimHei",
        "DejaVu Sans",
    ]


    def pick_qt_font_families() -> list[str]:
        available = set(QFontDatabase.families())
        families = [candidate for candidate in QT_FONT_CANDIDATES if candidate in available]
        default_font = QApplication.font()
        if default_font and default_font.family() and default_font.family() not in families:
            families.append(default_font.family())
        if not families:
            families.append("Sans Serif")
        return families


    def pick_qt_font_family() -> str:
        return pick_qt_font_families()[0]


    def build_qt_font(
        point_size: int,
        *,
        bold: bool = False,
        families: list[str] | None = None,
    ) -> QFont:
        font = QFont()
        font.setPointSize(point_size)
        font.setFamilies(families or pick_qt_font_families())
        font.setBold(bold)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        return font


    def qt_font_stack(families: list[str]) -> str:
        ordered = families + QT_FONT_CANDIDATES + ["Sans Serif"]
        seen: set[str] = set()
        stack: list[str] = []
        for family in ordered:
            if not family or family in seen:
                continue
            seen.add(family)
            if family == "Sans Serif":
                stack.append("sans-serif")
            elif " " in family:
                stack.append(f'"{family}"')
            else:
                stack.append(family)
        return ", ".join(stack)


    def pil_image_to_qpixmap(image: Image.Image) -> QPixmap:
        rgba = image.convert("RGBA")
        raw = rgba.tobytes("raw", "RGBA")
        qimage = QImage(
            raw,
            rgba.width,
            rgba.height,
            rgba.width * 4,
            QImage.Format.Format_RGBA8888,
        )
        return QPixmap.fromImage(qimage.copy())


    class AspectRatioImageLabel(QLabel):
        def __init__(
            self,
            minimum_size: tuple[int, int] = (320, 220),
            scale_mode: str = "fit",
        ):
            super().__init__()
            self._pixmap: QPixmap | None = None
            self.scale_mode = scale_mode
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setMinimumSize(*minimum_size)
            self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        def set_pil_image(self, image: Image.Image) -> None:
            self._pixmap = pil_image_to_qpixmap(image)
            self._sync_pixmap()

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            self._sync_pixmap()

        def _sync_pixmap(self) -> None:
            if self._pixmap is None:
                return
            size = self.size()
            if size.width() <= 1 or size.height() <= 1:
                return
            aspect_mode = (
                Qt.AspectRatioMode.KeepAspectRatioByExpanding
                if self.scale_mode == "fill"
                else Qt.AspectRatioMode.KeepAspectRatio
            )
            scaled = self._pixmap.scaled(
                size,
                aspect_mode,
                Qt.TransformationMode.SmoothTransformation,
            )
            super().setPixmap(scaled)


    class MetricCard(QFrame):
        def __init__(self, title: str, accent_color: str, font_families: list[str]):
            super().__init__()
            self.setObjectName("metricCard")
            self.setMinimumHeight(82)
            self.setMaximumHeight(82)
            layout = QHBoxLayout(self)
            layout.setContentsMargins(14, 10, 14, 10)
            layout.setSpacing(12)

            bar = QFrame()
            bar.setFixedWidth(6)
            bar.setStyleSheet(
                f"background:{accent_color}; border-radius:3px;"
            )
            layout.addWidget(bar)

            text_box = QVBoxLayout()
            text_box.setContentsMargins(0, 0, 0, 0)
            text_box.setSpacing(4)

            title_label = QLabel(title)
            title_label.setStyleSheet(f"color:{TEXT_SECONDARY};")
            title_label.setFont(build_qt_font(10, families=font_families))

            self.value_label = QLabel("0")
            self.value_label.setFont(build_qt_font(18, bold=True, families=font_families))
            self.value_label.setStyleSheet(f"color:{TEXT_PRIMARY};")

            text_box.addWidget(title_label)
            text_box.addWidget(self.value_label)
            layout.addLayout(text_box, 1)

        def set_value(self, value: str) -> None:
            self.value_label.setText(value)


    class SeatMapWidget(QWidget):
        seat_clicked = pyqtSignal(int)

        def __init__(self, font_families: list[str]):
            super().__init__()
            self.font_families = font_families
            self.seats: list[SeatBox] = []
            self.snapshot: InspectionSnapshot | None = None
            self.focus_seat: int | None = None
            self._seat_rects: dict[int, QRectF] = {}
            self.setMinimumHeight(360)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        def set_state(
            self,
            seats: list[SeatBox],
            snapshot: InspectionSnapshot,
            focus_seat: int | None,
        ) -> None:
            self.seats = seats
            self.snapshot = snapshot
            self.focus_seat = focus_seat
            self.update()

        def mousePressEvent(self, event) -> None:
            position = event.position()
            for seat_no, rect in self._seat_rects.items():
                if rect.contains(position):
                    self.seat_clicked.emit(seat_no)
                    break
            super().mousePressEvent(event)

        def paintEvent(self, _event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QColor(PANEL_MUTED_BG))
            self._seat_rects.clear()

            if not self.seats or self.snapshot is None:
                painter.end()
                return

            width = max(240.0, float(self.width()))
            height = max(260.0, float(self.height()))
            num_cols = max((seat.col_index for seat in self.seats), default=0) + 1
            num_rows = max((seat.row_index for seat in self.seats), default=0) + 1

            outer_pad_x = 20.0
            outer_pad_y = 18.0
            bottom_pad = 78.0
            gap_x = 14.0
            gap_y = 12.0
            usable_w = width - outer_pad_x * 2.0 - gap_x * max(0, num_cols - 1)
            usable_h = height - outer_pad_y * 2.0 - bottom_pad - gap_y * max(0, num_rows - 1)
            cell_w = usable_w / max(1, num_cols)
            cell_h = usable_h / max(1, num_rows)

            main_font = build_qt_font(12, bold=True, families=self.font_families)
            sub_font = build_qt_font(9, bold=True, families=self.font_families)

            for seat in self.seats:
                x1 = outer_pad_x + seat.col_index * (cell_w + gap_x)
                y1 = outer_pad_y + seat.row_index * (cell_h + gap_y)
                rect = QRectF(x1, y1, cell_w, cell_h)
                self._seat_rects[seat.seat_no] = rect

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

                outline = ACCENT if self.focus_seat == seat.seat_no else PANEL_BORDER
                pen = QPen(QColor(outline), 3 if self.focus_seat == seat.seat_no else 1)
                painter.setPen(pen)
                painter.setBrush(QColor(fill))
                painter.drawRoundedRect(rect, 10.0, 10.0)

                painter.setPen(QColor(text_color))
                painter.setFont(main_font)
                painter.drawText(
                    QRectF(rect.left() + 14.0, rect.top() + 8.0, rect.width() - 20.0, 24.0),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                    f"{seat.seat_no:02d}",
                )
                if status != "normal":
                    painter.setFont(sub_font)
                    painter.drawText(
                        QRectF(rect.left() + 8.0, rect.center().y() - 8.0, rect.width() - 16.0, rect.height() / 2.0),
                        int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
                        behavior or STATUS_LABELS.get(status, ""),
                    )

            podium_w = min(160.0, width * 0.34)
            podium_h = 42.0
            podium_x = (width - podium_w) / 2.0
            podium_y = height - 54.0
            painter.setPen(QPen(QColor("#7DB0FF"), 2))
            painter.setBrush(QColor("#5C87CA"))
            painter.drawRoundedRect(QRectF(podium_x, podium_y, podium_w, podium_h), 8.0, 8.0)
            podium_font = build_qt_font(12, bold=True, families=self.font_families)
            painter.setFont(podium_font)
            painter.setPen(QColor(TEXT_PRIMARY))
            painter.drawText(
                QRectF(podium_x, podium_y, podium_w, podium_h),
                int(Qt.AlignmentFlag.AlignCenter),
                "讲台",
            )
            painter.end()


    class DetectionDashboard(QMainWindow):
        def __init__(
            self,
            source: str | Path | None,
            layout_path: str | Path | None,
            room_name: str = "第一考场",
            refresh_ms: int = 850,
            mode: str = "auto",
            live_config: LiveBindingConfig | None = None,
        ):
            super().__init__()
            self.setWindowTitle("ExamWatchAI 检测巡检界面")
            self.resize(1680, 960)
            self.setMinimumSize(1180, 760)

            self.room_name = room_name
            self.refresh_ms = max(120, int(refresh_ms))
            self.mode = mode
            self.live_config = live_config
            self.base_refresh_ms = max(120, int(refresh_ms))
            self.playback_speed = 0.75
            self.live_session: RealBindingSession | None = None
            self.live_status_map: dict[int, str] | None = None
            self.live_frame_received = False
            self.latest_progress_text = "等待巡检数据"
            self.runtime_message = ""
            self.source_error: str | None = None
            self.source_traceback: str | None = None
            self.playing = True
            self.selected_seat: int | None = None
            self.fullscreen = False
            self.records: list[AlertRecord] = []
            self.latest_live_rows: list[dict[str, Any]] = []
            self.pending_live_frame_queue: deque[dict[str, Any]] = deque(maxlen=32)
            self.last_presented_live_at = 0.0
            self.last_presented_mock_at = 0.0
            self.latest_live_fps = 25.0
            self.stop_mode = False
            self.live_finished = False
            self.replay_rows_by_frame: dict[int, list[dict[str, Any]]] = {}
            self.slider_is_scrubbing = False
            self.live_buffer_target = 3

            self.seats, self.layout_base_size = load_seat_layout(
                Path(layout_path) if layout_path else None
            )
            self.latest_live_source_size = self.layout_base_size
            self.seat_lookup = {seat.seat_no: seat for seat in self.seats}
            self.snapshot = self._empty_snapshot()
            self.current_frame = render_placeholder_image(
                (1920, 1080),
                "ExamWatchAI",
                "正在准备检测界面",
                None,
            )
            self.current_clean_frame = self.current_frame.copy()
            self.current_frame_idx = 0
            self.current_total_frames: int | None = None
            self.current_fps = 25.0
            self.provider: FrameProvider | None = None
            self.engine: MockInspectionEngine | None = None

            self.ui_font_families = pick_qt_font_families()
            self.ui_font_family = self.ui_font_families[0]
            self.ui_font_stack = qt_font_stack(self.ui_font_families)
            self.pil_font_small = load_pil_font(22, weight="medium")
            self.pil_font_medium = load_pil_font(28, weight="bold")

            self._configure_window_style()
            self._build_layout()
            self._configure_data_source(source, room_name)
            self._refresh_dashboard()

            self.resize_refresh_timer = QTimer(self)
            self.resize_refresh_timer.setSingleShot(True)
            self.resize_refresh_timer.timeout.connect(self._refresh_dashboard)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._poll_loop)
            self._update_poll_timer()
            self.timer.start()

        def _empty_snapshot(self) -> InspectionSnapshot:
            return InspectionSnapshot(
                status_by_seat={seat.seat_no: "empty" for seat in self.seats},
                behavior_by_seat={seat.seat_no: REAL_STATUS_NOTE["empty"] for seat in self.seats},
                occupied_count=0,
                alert_count=0,
                empty_count=len(self.seats),
                focus_seat=None,
                events=[],
            )

        def _pending_snapshot(self) -> InspectionSnapshot:
            return InspectionSnapshot(
                status_by_seat={seat.seat_no: "normal" for seat in self.seats},
                behavior_by_seat={seat.seat_no: "待检测" for seat in self.seats},
                occupied_count=0,
                alert_count=0,
                empty_count=0,
                focus_seat=None,
                events=[],
            )

        def _configure_window_style(self) -> None:
            app_font = build_qt_font(10, families=self.ui_font_families)
            self.setFont(app_font)
            self.setStyleSheet(
                f"""
                QWidget {{
                    background: {APP_BG};
                    color: {TEXT_PRIMARY};
                    font-family: {self.ui_font_stack};
                }}
                QFrame#panel {{
                    background: {PANEL_BG};
                    border: 1px solid {PANEL_BORDER};
                    border-radius: 16px;
                }}
                QFrame#panelHeader {{
                    background: {PANEL_HEADER_BG};
                    border-top-left-radius: 16px;
                    border-top-right-radius: 16px;
                    border-bottom: 1px solid {PANEL_BORDER};
                }}
                QFrame#metricCard {{
                    background: {PANEL_MUTED_BG};
                    border: 1px solid {PANEL_BORDER};
                    border-radius: 12px;
                }}
                QLabel#panelTitle {{
                    color: {TEXT_PRIMARY};
                    background: transparent;
                }}
                QLabel#subtitle {{
                    color: {TEXT_SECONDARY};
                    background: rgba(19, 24, 47, 0.55);
                    border: 1px solid rgba(94, 181, 255, 0.18);
                    border-radius: 10px;
                    padding: 4px 10px;
                }}
                QLabel#statusChip {{
                    border-radius: 10px;
                    padding: 6px 12px;
                    font-weight: 700;
                }}
                QLabel#statusText {{
                    color: {TEXT_PRIMARY};
                    background: {PANEL_MUTED_BG};
                    border: 1px solid rgba(94, 181, 255, 0.12);
                    border-radius: 10px;
                    padding: 6px 12px;
                }}
                QLabel#errorText {{
                    color: #FFD7D8;
                    background: rgba(255, 94, 98, 0.16);
                    border: 1px solid rgba(255, 94, 98, 0.55);
                    border-radius: 10px;
                    padding: 8px 12px;
                }}
                QLabel#imageSurface {{
                    background: #0F1327;
                    border: none;
                    border-radius: 12px;
                }}
                QPushButton {{
                    background: {PANEL_HEADER_BG};
                    color: {TEXT_PRIMARY};
                    border: 1px solid {PANEL_BORDER};
                    border-radius: 10px;
                    padding: 8px 16px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background: #3B436E;
                }}
                QPushButton#primaryButton {{
                    background: {ACCENT};
                    color: #0F1327;
                    border-color: {ACCENT};
                }}
                QPushButton#primaryButton:hover {{
                    background: #7BC8FF;
                }}
                QPushButton:disabled {{
                    color: #7E86AD;
                    background: #252A46;
                    border-color: #252A46;
                }}
                QLineEdit {{
                    background: #F3F6FF;
                    color: #1A1F33;
                    border: none;
                    border-radius: 8px;
                    padding: 8px 10px;
                }}
                QComboBox {{
                    background: {PANEL_HEADER_BG};
                    color: {TEXT_PRIMARY};
                    border: 1px solid {PANEL_BORDER};
                    border-radius: 8px;
                    padding: 6px 10px;
                }}
                QTableWidget {{
                    background: {PANEL_MUTED_BG};
                    border: none;
                    gridline-color: {PANEL_BORDER};
                    selection-background-color: #355284;
                    selection-color: {TEXT_PRIMARY};
                    alternate-background-color: #202548;
                }}
                QTableWidget::item {{
                    padding: 6px 8px;
                }}
                QHeaderView::section {{
                    background: {PANEL_HEADER_BG};
                    color: {TEXT_PRIMARY};
                    border: none;
                    border-bottom: 1px solid {PANEL_BORDER};
                    padding: 8px;
                    font-weight: 600;
                }}
                QTableCornerButton::section {{
                    background: {PANEL_HEADER_BG};
                    border: none;
                }}
                QSplitter::handle {{
                    background: rgba(94, 181, 255, 0.12);
                    border-radius: 4px;
                }}
                QSplitter::handle:hover {{
                    background: rgba(94, 181, 255, 0.28);
                }}
                """
            )

        def _create_panel(self, title: str, subtitle: str) -> tuple[QFrame, QVBoxLayout]:
            panel = QFrame()
            panel.setObjectName("panel")
            outer = QVBoxLayout(panel)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)

            header = QFrame()
            header.setObjectName("panelHeader")
            header.setFixedHeight(54)
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(16, 8, 16, 8)
            header_layout.setSpacing(10)

            title_label = QLabel(title)
            title_label.setObjectName("panelTitle")
            title_label.setFont(build_qt_font(13, bold=True, families=self.ui_font_families))

            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("subtitle")
            subtitle_label.setFont(build_qt_font(9, families=self.ui_font_families))
            subtitle_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            subtitle_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            subtitle_label.setMaximumWidth(260)

            header_layout.addWidget(title_label)
            header_layout.addStretch(1)
            header_layout.addWidget(subtitle_label)

            body_widget = QWidget()
            body_layout = QVBoxLayout(body_widget)
            body_layout.setContentsMargins(14, 14, 14, 14)
            body_layout.setSpacing(14)

            outer.addWidget(header)
            outer.addWidget(body_widget, 1)
            return panel, body_layout

        def _build_layout(self) -> None:
            central = QWidget()
            self.setCentralWidget(central)
            shell = QVBoxLayout(central)
            shell.setContentsMargins(18, 18, 18, 18)
            shell.setSpacing(12)

            main_panel, main_body = self._create_panel("实时检测画面", "主监控 / 绑定框")
            seat_panel, seat_body = self._create_panel("座位态势图", "30席位状态总览")
            table_panel, table_body = self._create_panel("状态记录", "绑定变化 / 事件记录")
            detail_panel, detail_body = self._create_panel("局部预览", "当前聚焦座位")

            for panel in (main_panel, seat_panel, table_panel, detail_panel):
                panel.setMinimumSize(220, 180)

            self.left_column_splitter = QSplitter(Qt.Orientation.Vertical)
            self.left_column_splitter.setChildrenCollapsible(False)
            self.left_column_splitter.setHandleWidth(10)
            self.left_column_splitter.addWidget(main_panel)
            self.left_column_splitter.addWidget(table_panel)
            self.left_column_splitter.setStretchFactor(0, 6)
            self.left_column_splitter.setStretchFactor(1, 3)

            self.right_column_splitter = QSplitter(Qt.Orientation.Vertical)
            self.right_column_splitter.setChildrenCollapsible(False)
            self.right_column_splitter.setHandleWidth(10)
            self.right_column_splitter.addWidget(seat_panel)
            self.right_column_splitter.addWidget(detail_panel)
            self.right_column_splitter.setStretchFactor(0, 6)
            self.right_column_splitter.setStretchFactor(1, 3)

            self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
            self.workspace_splitter.setChildrenCollapsible(False)
            self.workspace_splitter.setHandleWidth(10)
            self.workspace_splitter.addWidget(self.left_column_splitter)
            self.workspace_splitter.addWidget(self.right_column_splitter)
            self.workspace_splitter.setStretchFactor(0, 3)
            self.workspace_splitter.setStretchFactor(1, 2)
            self.left_column_splitter.setSizes([620, 340])
            self.right_column_splitter.setSizes([620, 340])
            self.workspace_splitter.setSizes([980, 640])
            shell.addWidget(self.workspace_splitter, 1)

            footer = QWidget()
            footer_layout = QVBoxLayout(footer)
            footer_layout.setContentsMargins(0, 2, 0, 0)
            footer_layout.setSpacing(10)
            shell.addWidget(footer)

            metrics_row = QHBoxLayout()
            metrics_row.setSpacing(10)
            self.metric_cards = {
                "occupied": MetricCard("在座人数", STATUS_EMPTY, self.ui_font_families),
                "alerts": MetricCard("预警席位", STATUS_WARNING, self.ui_font_families),
                "empty": MetricCard("空位数", STATUS_EMPTY, self.ui_font_families),
                "rate": MetricCard("异常率", ACCENT, self.ui_font_families),
            }
            for key in ("occupied", "alerts", "empty", "rate"):
                metrics_row.addWidget(self.metric_cards[key], 1)
            main_body.addLayout(metrics_row)

            self.main_image_label = AspectRatioImageLabel((260, 160), scale_mode="fit")
            self.main_image_label.setObjectName("imageSurface")
            main_body.addWidget(self.main_image_label, 1)

            progress_row = QHBoxLayout()
            progress_row.setSpacing(10)
            self.video_progress_slider = QSlider(Qt.Orientation.Horizontal)
            self.video_progress_slider.setRange(0, 1000)
            self.video_progress_slider.setValue(0)
            self.video_progress_slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.video_progress_slider.sliderPressed.connect(self._on_slider_pressed)
            self.video_progress_slider.sliderReleased.connect(self._on_slider_released)
            self.video_progress_slider.valueChanged.connect(self._on_slider_value_changed)
            progress_row.addWidget(self.video_progress_slider, 1)
            self.video_progress_label = QLabel("00:00 / 00:00")
            self.video_progress_label.setStyleSheet(f"color:{TEXT_SECONDARY};")
            progress_row.addWidget(self.video_progress_label)
            main_body.addLayout(progress_row)

            legend_row = QHBoxLayout()
            legend_row.setSpacing(10)
            for text, bg_color, fg_color in [
                ("正常", STATUS_NORMAL, STATUS_NORMAL_TEXT),
                ("空位", STATUS_EMPTY, STATUS_EMPTY_TEXT),
                ("预警", STATUS_WARNING, "#22170F"),
                ("重点", STATUS_CRITICAL, "#2B0F12"),
            ]:
                badge = QLabel(text)
                badge.setStyleSheet(
                    f"background:{bg_color}; color:{fg_color}; border-radius:8px; padding:4px 12px; font-weight:700;"
                )
                legend_row.addWidget(badge)
            legend_row.addStretch(1)
            seat_body.addLayout(legend_row)

            self.seat_map_widget = SeatMapWidget(self.ui_font_families)
            self.seat_map_widget.seat_clicked.connect(self._focus_seat)
            seat_body.addWidget(self.seat_map_widget, 1)

            self.events_table = QTableWidget(0, 5)
            self.events_table.setHorizontalHeaderLabels(
                ["检测时刻", "考场名称", "座位号", "状态等级", "状态说明"]
            )
            self.events_table.verticalHeader().setVisible(False)
            self.events_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.events_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.events_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.events_table.setShowGrid(True)
            self.events_table.setAlternatingRowColors(True)
            self.events_table.setWordWrap(False)
            self.events_table.horizontalHeader().setStretchLastSection(True)
            self.events_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.events_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            self.events_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            self.events_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            self.events_table.itemSelectionChanged.connect(self._on_table_select)
            table_body.addWidget(self.events_table, 1)

            self.table_summary_label = QLabel("事件 0 条")
            self.table_summary_label.setStyleSheet(f"color:{TEXT_SECONDARY};")
            table_body.addWidget(self.table_summary_label)

            self.detail_image_label = AspectRatioImageLabel((220, 160), scale_mode="fit")
            self.detail_image_label.setObjectName("imageSurface")
            detail_body.addWidget(self.detail_image_label, 1)

            self.detail_text_label = QLabel("等待巡检数据")
            self.detail_text_label.setStyleSheet(f"color:{TEXT_SECONDARY};")
            self.detail_text_label.setWordWrap(True)
            detail_body.addWidget(self.detail_text_label)

            controls_row = QHBoxLayout()
            controls_row.setSpacing(10)
            footer_layout.addLayout(controls_row)

            left_controls = QHBoxLayout()
            left_controls.setSpacing(10)
            right_controls = QHBoxLayout()
            right_controls.setSpacing(10)
            controls_row.addLayout(left_controls, 1)
            controls_row.addLayout(right_controls)

            self.play_button = self._make_button("播放", self._resume_playback, primary=True)
            left_controls.addWidget(self.play_button)
            self.pause_button = self._make_button("暂停", self._pause_playback)
            left_controls.addWidget(self.pause_button)
            self.stop_button = self._make_button("停止", self._stop_playback)
            left_controls.addWidget(self.stop_button)
            left_controls.addWidget(self._make_button("复位界面", self._reset_dashboard))
            left_controls.addWidget(self._make_button("清空记录", self._clear_records))
            self.load_button = self._make_button("加载素材", self._load_source)
            left_controls.addWidget(self.load_button)

            speed_label = QLabel("播放速度")
            speed_label.setStyleSheet(f"color:{TEXT_SECONDARY};")
            right_controls.addWidget(speed_label)
            self.speed_combo = QComboBox()
            self.speed_combo.addItems(["0.5x", "0.75x", "1.0x", "1.25x", "1.5x"])
            self.speed_combo.setCurrentText("0.75x")
            self.speed_combo.currentTextChanged.connect(self._set_playback_speed)
            right_controls.addWidget(self.speed_combo)

            locate_label = QLabel("定位座位")
            locate_label.setStyleSheet(f"color:{TEXT_SECONDARY};")
            right_controls.addWidget(locate_label)
            self.search_input = QLineEdit()
            self.search_input.setFixedWidth(84)
            self.search_input.returnPressed.connect(self._focus_seat_from_input)
            right_controls.addWidget(self.search_input)
            right_controls.addWidget(self._make_button("定位", self._focus_seat_from_input))
            right_controls.addWidget(self._make_button("全屏", self._toggle_fullscreen))

            status_row = QHBoxLayout()
            status_row.setSpacing(10)
            footer_layout.addLayout(status_row)

            self.mode_chip_label = QLabel("等待启动")
            self.mode_chip_label.setObjectName("statusChip")
            status_row.addWidget(self.mode_chip_label)

            self.runtime_status_label = QLabel("检测线程尚未启动")
            self.runtime_status_label.setObjectName("statusText")
            status_row.addWidget(self.runtime_status_label, 2)

            self.progress_label = QLabel("等待巡检数据")
            self.progress_label.setObjectName("statusText")
            status_row.addWidget(self.progress_label, 2)

            self.source_label = QLabel("")
            self.source_label.setObjectName("statusText")
            self.source_label.setWordWrap(True)
            footer_layout.addWidget(self.source_label)

            self.error_label = QLabel("")
            self.error_label.setObjectName("errorText")
            self.error_label.setWordWrap(True)
            self.error_label.hide()
            footer_layout.addWidget(self.error_label)
            self._update_playback_buttons()

        def _make_button(self, text: str, callback, primary: bool = False) -> QPushButton:
            button = QPushButton(text)
            if primary:
                button.setObjectName("primaryButton")
            button.clicked.connect(callback)
            return button

        def _format_progress_time(self, frame_idx: int, total_frames: int | None, fps: float) -> str:
            current_text = format_video_time(max(0, frame_idx), fps)
            if total_frames is not None and total_frames > 0:
                total_text = format_video_time(total_frames, fps)
            else:
                total_text = "--:--.--"
            return f"{current_text} / {total_text}"

        def _update_video_progress(self) -> None:
            total_frames = self.current_total_frames
            frame_idx = max(0, int(self.current_frame_idx))
            if total_frames is not None and total_frames > 0:
                self.video_progress_slider.setRange(0, total_frames)
                if not self.slider_is_scrubbing:
                    self.video_progress_slider.setValue(min(frame_idx, total_frames))
            else:
                self.video_progress_slider.setRange(0, 1000)
                if not self.slider_is_scrubbing:
                    self.video_progress_slider.setValue(0)
            self.video_progress_label.setText(
                self._format_progress_time(
                    self.video_progress_slider.value() if self.slider_is_scrubbing else frame_idx,
                    total_frames,
                    self.current_fps,
                )
            )
            self.video_progress_slider.setEnabled(self._can_seek_playback())

        def _can_seek_playback(self) -> bool:
            if self.mode == "mock":
                return self.provider is not None and self.provider.can_seek()
            return self.live_finished and self.provider is not None and self.provider.can_seek()

        def _load_replay_rows(self, csv_path: str | Path | None) -> None:
            self.replay_rows_by_frame = {}
            if not csv_path:
                return
            csv_file = Path(csv_path)
            if not csv_file.exists():
                return
            with csv_file.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    frame_idx = to_int(row.get("frame_idx"))
                    if frame_idx is None:
                        continue
                    self.replay_rows_by_frame.setdefault(frame_idx, []).append(row)

        def _apply_rows_for_frame(self, frame_idx: int) -> None:
            rows = self.replay_rows_by_frame.get(frame_idx, [])
            self.latest_live_rows = list(rows)
            snapshot, self.live_status_map = build_snapshot_from_binding_rows(
                rows=rows,
                seat_numbers=[seat.seat_no for seat in self.seats],
                room_name=self.room_name,
                frame_idx=frame_idx,
                fps=self.current_fps,
                previous_status=None,
            )
            self.snapshot = snapshot

        def _seek_playback(self, frame_idx: int) -> None:
            if self.provider is None or not self.provider.can_seek():
                return
            frame = self.provider.seek(frame_idx)
            self.current_frame = frame
            self.current_clean_frame = frame.copy()
            self.current_frame_idx, self.current_total_frames, self.current_fps = self.provider.progress_state()
            if self.mode == "live" and self.live_finished:
                self._apply_rows_for_frame(self.current_frame_idx)
                self.latest_progress_text = f"回看帧 {self.current_frame_idx}/{self.current_total_frames or '?'}"
            self._refresh_dashboard()

        def _on_slider_pressed(self) -> None:
            self.slider_is_scrubbing = True

        def _on_slider_released(self) -> None:
            if not self.slider_is_scrubbing:
                return
            self.slider_is_scrubbing = False
            if self._can_seek_playback():
                self._seek_playback(self.video_progress_slider.value())
            else:
                self._refresh_dashboard()

        def _on_slider_value_changed(self, value: int) -> None:
            if self.slider_is_scrubbing:
                self.video_progress_label.setText(
                    self._format_progress_time(value, self.current_total_frames, self.current_fps)
                )

        def _effective_provider_interval_ms(self) -> int:
            fps = self.provider.playback_fps() if self.provider is not None else 5.0
            return max(40, min(1500, int(round(1000.0 / max(0.1, fps * self.playback_speed)))))

        def _effective_live_interval_ms(self, fps: float | None = None) -> int:
            base_fps = fps or getattr(self, "latest_live_fps", 25.0) or 25.0
            stride = self.live_config.frame_callback_stride if self.live_config is not None else 1
            return max(
                70,
                min(
                    1500,
                    int(round((1000.0 * max(1, stride)) / max(0.1, base_fps * self.playback_speed))),
                ),
            )

        def _update_poll_timer(self) -> None:
            if not hasattr(self, "timer"):
                return
            if self.mode == "live":
                self.timer.setInterval(60)
            else:
                self.timer.setInterval(self._effective_provider_interval_ms())

        def _update_playback_buttons(self) -> None:
            self.play_button.setEnabled(not self.playing)
            self.pause_button.setEnabled(self.playing)

        def _set_playback_speed(self, text: str) -> None:
            try:
                self.playback_speed = max(0.25, float(text.lower().replace("x", "").strip()))
            except ValueError:
                self.playback_speed = 1.0
            self.last_presented_live_at = 0.0
            self.last_presented_mock_at = 0.0
            self._update_poll_timer()
            self._refresh_dashboard()

        def _resume_playback(self) -> None:
            self.playing = True
            self.stop_mode = False
            self.last_presented_live_at = 0.0
            self.last_presented_mock_at = 0.0
            if self.mode == "live":
                self.latest_progress_text = "已恢复播放，正在按当前速度展示实时结果"
            else:
                self.latest_progress_text = "已恢复播放"
            self._update_playback_buttons()

        def _pause_playback(self) -> None:
            self.playing = False
            self.stop_mode = False
            if self.mode == "live":
                self.latest_progress_text = "已暂停画面，检测仍在后台继续"
            else:
                self.latest_progress_text = "已暂停播放"
            self._update_playback_buttons()
            self._refresh_dashboard()

        def _stop_playback(self) -> None:
            self.playing = False
            self.stop_mode = True
            self.last_presented_live_at = 0.0
            self.last_presented_mock_at = 0.0
            if self.mode == "live":
                self.pending_live_frame_queue.clear()
                self.latest_progress_text = "已停止显示，点击播放继续查看最新实时结果"
            elif self.provider is not None and self.engine is not None:
                self.provider.reset()
                self.engine.reset()
                self.snapshot = self.engine.step()
                self.current_frame = self.provider.read_next()
                self.records.clear()
                self.events_table.setRowCount(0)
                self.latest_progress_text = "已停止并回到起始位置"
            self._update_playback_buttons()
            self._refresh_dashboard()

        def _set_status_chip(self, text: str, bg_color: str, fg_color: str) -> None:
            self.mode_chip_label.setText(text)
            self.mode_chip_label.setStyleSheet(
                f"background:{bg_color}; color:{fg_color}; border-radius:10px; padding:6px 12px; font-weight:700;"
            )

        def _update_status_widgets(self) -> None:
            if self.source_error:
                self._set_status_chip("检测异常", STATUS_CRITICAL, "#2B0F12")
            elif self.mode == "live" and self.live_frame_received:
                self._set_status_chip("实时检测", ACCENT, "#0F1327")
            elif self.mode == "live":
                self._set_status_chip("启动中", STATUS_WARNING, "#2A170B")
            else:
                self._set_status_chip("模拟巡检", STATUS_EMPTY, "#0E2B21")

            self.runtime_status_label.setText(self.runtime_message or "等待巡检线程状态")
            self.progress_label.setText(self.latest_progress_text or "等待巡检数据")
            self.source_label.setText(self._source_text())

            if self.source_error:
                error_text = f"异常: {self.source_error}"
                if self.source_traceback:
                    self.error_label.setToolTip(self.source_traceback)
                self.error_label.setText(error_text)
                self.error_label.show()
            else:
                self.error_label.hide()
                self.error_label.setToolTip("")

        def _source_text(self) -> str:
            if self.live_session is not None:
                text = f"视频源: {self.live_session.description()}"
            elif self.provider is not None:
                text = f"素材源: {self.provider.description()}"
            else:
                text = "素材源: 模拟巡检"
            return text

        def _configure_data_source(self, source: str | Path | None, room_name: str) -> None:
            source_path = Path(source).expanduser().resolve() if source else None
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
                    self.provider = FrameProvider(source)
                    self.pending_live_frame_queue.clear()
                    self.live_session = RealBindingSession(self.live_config)
                    self.live_session.start()
                    if self.provider.error is None:
                        self.current_frame = self.provider.read_next()
                        self.current_clean_frame = self.current_frame.copy()
                        self.current_frame_idx, self.current_total_frames, self.current_fps = self.provider.progress_state()
                    else:
                        self.current_frame = render_placeholder_image(
                            (1920, 1080),
                            "ExamWatchAI",
                            "正在启动真实检测流水线",
                            str(source_path) if source_path else None,
                        )
                        self.current_clean_frame = self.current_frame.copy()
                        self.current_frame_idx = 0
                        self.current_total_frames = None
                        self.current_fps = 25.0
                    self.snapshot = self._pending_snapshot()
                    self.live_frame_received = False
                    self.live_finished = False
                    self.latest_live_rows = []
                    self.latest_live_source_size = self.layout_base_size
                    self.replay_rows_by_frame = {}
                    self.runtime_message = "模型启动中，正在加载视频与权重"
                    self.latest_progress_text = "等待座位布局与首帧回调"
                    self.source_error = None
                    self.source_traceback = None
                    self.load_button.setEnabled(False)
                    self._update_poll_timer()
                    self._update_playback_buttons()
                    return
                if wants_live:
                    self.source_error = "未提供真实检测配置，已回退到 mock 模式。"

            self.mode = "mock"
            self.provider = FrameProvider(source)
            self.engine = MockInspectionEngine(self.seats, room_name=room_name)
            self.pending_live_frame_queue.clear()
            self.source_error = self.provider.error
            self.source_traceback = None
            self.live_finished = False
            self.replay_rows_by_frame = {}
            self.current_frame = self.provider.read_next()
            self.current_clean_frame = self.current_frame.copy()
            self.current_frame_idx, self.current_total_frames, self.current_fps = self.provider.progress_state()
            self.snapshot = self.engine.step()
            self.latest_live_rows = []
            self.latest_live_source_size = self.layout_base_size
            self.runtime_message = "模拟巡检中"
            self.latest_progress_text = "正在使用 mock 数据驱动界面"
            self.load_button.setEnabled(True)
            self._update_poll_timer()
            self._update_playback_buttons()

        def _reset_dashboard(self) -> None:
            self.records.clear()
            self.events_table.setRowCount(0)
            self.selected_seat = None
            self.live_status_map = None
            self.latest_live_rows = []
            self.latest_live_source_size = self.layout_base_size
            self.pending_live_frame_queue.clear()
            self.last_presented_live_at = 0.0
            self.last_presented_mock_at = 0.0
            self.live_finished = False
            self.replay_rows_by_frame = {}
            if self.mode == "mock" and self.provider is not None and self.engine is not None:
                self.provider.reset()
                self.engine.reset()
                self.snapshot = self.engine.step()
                self.current_frame = self.provider.read_next()
                self.current_clean_frame = self.current_frame.copy()
                self.current_frame_idx, self.current_total_frames, self.current_fps = self.provider.progress_state()
            self._update_playback_buttons()
            self._refresh_dashboard()

        def _clear_records(self) -> None:
            self.records.clear()
            self.events_table.setRowCount(0)
            self.table_summary_label.setText("事件 0 条")

        def _load_source(self) -> None:
            if self.mode == "live":
                self.detail_text_label.setText("真实检测模式下，请重新启动并通过 --source 指定视频。")
                return
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择图片或视频素材",
                "",
                "Media (*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
            )
            selected_path = file_path
            if not selected_path:
                selected_path = QFileDialog.getExistingDirectory(self, "选择截图序列目录")
            if not selected_path:
                return
            self.provider = FrameProvider(selected_path)
            self.provider.reset()
            self.source_error = self.provider.error
            self.current_frame = self.provider.read_next()
            self.current_clean_frame = self.current_frame.copy()
            self.current_frame_idx, self.current_total_frames, self.current_fps = self.provider.progress_state()
            self.last_presented_mock_at = 0.0
            self._update_poll_timer()
            self._refresh_dashboard()

        def _focus_seat(self, seat_no: int) -> None:
            if seat_no not in self.seat_lookup:
                return
            self.selected_seat = seat_no
            self.search_input.setText(str(seat_no))
            self._refresh_dashboard()

        def _focus_seat_from_input(self) -> None:
            raw_value = self.search_input.text().strip()
            if not raw_value:
                return
            try:
                seat_no = int(raw_value)
            except ValueError:
                self.detail_text_label.setText(f"无法定位: 输入的座位号 `{raw_value}` 不是数字。")
                return
            if seat_no not in self.seat_lookup:
                self.detail_text_label.setText(f"无法定位: 当前布局中没有座位 {seat_no}。")
                return
            self._focus_seat(seat_no)

        def _toggle_fullscreen(self) -> None:
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                self.showFullScreen()
            else:
                self.showNormal()

        def keyPressEvent(self, event) -> None:
            if event.key() == Qt.Key.Key_Escape and self.fullscreen:
                self.fullscreen = False
                self.showNormal()
                return
            super().keyPressEvent(event)

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
            self.snapshot = self._pending_snapshot()
            self.latest_live_rows = []
            self.latest_live_source_size = (frame_width, frame_height)
            self.latest_progress_text = "座位布局已建立，等待实时绑定结果"

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
                    return Image.fromarray(frame_bgr.copy())
            return render_placeholder_image(
                (1920, 1080),
                "ExamWatchAI",
                "收到的画面格式暂不支持",
                None,
            )

        def _apply_live_frame(self, payload: dict[str, Any]) -> None:
            self.live_frame_received = True
            self.source_traceback = None
            self.source_error = None
            self.current_frame = self._image_from_bgr_frame(payload.get("frame_bgr"))
            raw_frame_bgr = payload.get("frame_bgr_raw")
            if raw_frame_bgr is not None:
                self.current_clean_frame = self._image_from_bgr_frame(raw_frame_bgr)
            else:
                self.current_clean_frame = self.current_frame.copy()
            self.latest_live_source_size = (
                int(payload.get("frame_width", self.current_frame.size[0])),
                int(payload.get("frame_height", self.current_frame.size[1])),
            )
            frame_idx = int(payload.get("frame_idx", 0))
            fps = float(payload.get("fps", 25.0))
            self.latest_live_fps = fps
            self.current_frame_idx = frame_idx
            total_frames = payload.get("total_frames")
            self.current_total_frames = int(total_frames) if total_frames not in (None, "", "?") else None
            self.current_fps = fps
            rows = payload.get("rows") or []
            self.latest_live_rows = list(rows)
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
            finished_now = payload.get("finish") is not None
            if payload.get("layout") is not None:
                self._apply_live_layout(payload["layout"])
                should_refresh = True
            if payload.get("error") is not None:
                error_payload = payload["error"]
                self.source_error = str(error_payload.get("message", "未知错误"))
                self.source_traceback = error_payload.get("traceback")
                self.latest_progress_text = "检测线程启动失败"
                self.detail_text_label.setText(self.source_error)
                should_refresh = True
            incoming_frames = payload.get("frames") or []
            if incoming_frames and self.stop_mode:
                self.pending_live_frame_queue.clear()
                self.pending_live_frame_queue.append(incoming_frames[-1])
            else:
                for frame_payload in incoming_frames:
                    self.pending_live_frame_queue.append(frame_payload)
            now = time.monotonic()
            if (
                self.playing
                and self.pending_live_frame_queue
                and (
                    len(self.pending_live_frame_queue) >= self.live_buffer_target
                    or self.live_finished
                    or finished_now
                )
            ):
                due_interval = self._effective_live_interval_ms()
                if (
                    self.last_presented_live_at <= 0.0
                    or (now - self.last_presented_live_at) * 1000.0 >= due_interval
                ):
                    frame_payload = self.pending_live_frame_queue.popleft()
                    self._apply_live_frame(frame_payload)
                    self.last_presented_live_at = now
            elif self.playing and self.pending_live_frame_queue and not (self.live_finished or finished_now):
                self.latest_progress_text = f"检测缓冲中，待播帧 {len(self.pending_live_frame_queue)}"
            elif incoming_frames and not self.playing:
                self.latest_progress_text = "播放已暂停，实时检测结果正在缓存。"
            elif (
                self.playing
                and not self.live_frame_received
                and self.provider is not None
                and self.provider.error is None
            ):
                due_interval = self._effective_provider_interval_ms()
                if (
                    self.last_presented_mock_at <= 0.0
                    or (now - self.last_presented_mock_at) * 1000.0 >= due_interval
                ):
                    self.current_frame = self.provider.read_next()
                    self.current_clean_frame = self.current_frame.copy()
                    self.current_frame_idx, self.current_total_frames, self.current_fps = self.provider.progress_state()
                    self.last_presented_mock_at = now
                    self.latest_progress_text = self.runtime_message or "检测线程启动中"
                    should_refresh = True
            if payload.get("finish") is not None:
                self.runtime_message = "检测完成"
                self.source_error = None
                self.source_traceback = None
                self.live_finished = True
                finish_payload = payload.get("finish") or {}
                saved_outputs = finish_payload.get("saved_outputs") or {}
                self._load_replay_rows(saved_outputs.get("csv"))
                self.latest_progress_text = "检测完成，可拖动进度条回看结果"
                should_refresh = True
            if should_refresh and not incoming_frames:
                self._refresh_dashboard()

        def _poll_loop(self) -> None:
            if self.mode == "live":
                self._poll_live_session()
                if (
                    self.live_finished
                    and self.playing
                    and not self.pending_live_frame_queue
                    and self.provider is not None
                    and self.provider.can_seek()
                ):
                    now = time.monotonic()
                    due_interval = self._effective_provider_interval_ms()
                    if (
                        self.last_presented_mock_at <= 0.0
                        or (now - self.last_presented_mock_at) * 1000.0 >= due_interval
                    ):
                        self.current_frame = self.provider.read_next()
                        self.current_clean_frame = self.current_frame.copy()
                        self.current_frame_idx, self.current_total_frames, self.current_fps = self.provider.progress_state()
                        self._apply_rows_for_frame(self.current_frame_idx)
                        self.last_presented_mock_at = now
                        self.latest_progress_text = (
                            f"检测结果回放中 | 帧 {self.current_frame_idx}/{self.current_total_frames or '?'}"
                        )
                        self._refresh_dashboard()
            elif self.playing and self.provider is not None and self.engine is not None:
                now = time.monotonic()
                due_interval = self._effective_provider_interval_ms()
                if (
                    self.last_presented_mock_at <= 0.0
                    or (now - self.last_presented_mock_at) * 1000.0 >= due_interval
                ):
                    self.current_frame = self.provider.read_next()
                    self.current_clean_frame = self.current_frame.copy()
                    self.current_frame_idx, self.current_total_frames, self.current_fps = self.provider.progress_state()
                    self.snapshot = self.engine.step()
                    self._append_events(self.snapshot.events)
                    self.last_presented_mock_at = now
                    self._refresh_dashboard()

        def _append_events(self, events: list[AlertRecord]) -> None:
            if not events:
                return
            for event in reversed(events):
                self.records.insert(0, event)
                self.events_table.insertRow(0)
                for col, value in enumerate(
                    [
                        event.timestamp,
                        event.room_name,
                        str(event.seat_no),
                        event.level_label,
                        event.behavior,
                    ]
                ):
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(
                        int(Qt.AlignmentFlag.AlignCenter)
                        if col != 4
                        else int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                    )
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.events_table.setItem(0, col, item)
            self.records = self.records[:200]
            while self.events_table.rowCount() > 200:
                self.events_table.removeRow(self.events_table.rowCount() - 1)

        def _refresh_dashboard(self) -> None:
            occupied_count = self.snapshot.occupied_count
            alert_count = self.snapshot.alert_count
            empty_count = self.snapshot.empty_count
            alert_rate = 0.0 if occupied_count <= 0 else (alert_count / occupied_count) * 100.0

            self.metric_cards["occupied"].set_value(str(occupied_count))
            self.metric_cards["alerts"].set_value(str(alert_count))
            self.metric_cards["empty"].set_value(str(empty_count))
            self.metric_cards["rate"].set_value(f"{alert_rate:.1f}%")
            self.table_summary_label.setText(f"事件 {len(self.records)} 条")
            self._update_status_widgets()
            self._update_video_progress()

            focus_seat = self.selected_seat or self.snapshot.focus_seat
            self.seat_map_widget.set_state(self.seats, self.snapshot, focus_seat)
            self._render_main_view()
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

        def _row_for_seat(self, seat_no: int) -> dict[str, Any] | None:
            confirmed_candidate: dict[str, Any] | None = None
            pending_candidate: dict[str, Any] | None = None
            for row in self.latest_live_rows:
                role = str(row.get("role", "unknown"))
                seat_display = to_int(row.get("seat_no_display"))
                seat_current = to_int(row.get("seat_no_current"))
                if role == "student" and seat_display == seat_no:
                    confirmed_candidate = row
                    break
                if role in {"student", "unknown"} and seat_current == seat_no:
                    pending_candidate = row
            return confirmed_candidate or pending_candidate

        def _row_bbox(self, row: dict[str, Any]) -> tuple[float, float, float, float] | None:
            x1 = row.get("x1")
            y1 = row.get("y1")
            x2 = row.get("x2")
            y2 = row.get("y2")
            values = [x1, y1, x2, y2]
            if any(value is None for value in values):
                return None
            try:
                left, top, right, bottom = [float(value) for value in values]
            except (TypeError, ValueError):
                return None
            if right <= left or bottom <= top:
                return None
            src_w, src_h = self.latest_live_source_size
            dst_w, dst_h = self.current_frame.size
            scale_x = dst_w / max(1.0, float(src_w))
            scale_y = dst_h / max(1.0, float(src_h))
            left *= scale_x
            right *= scale_x
            top *= scale_y
            bottom *= scale_y
            return left, top, right, bottom

        def _crop_focus_region(
            self,
            frame: Image.Image,
            seat_box: tuple[float, float, float, float],
            person_box: tuple[float, float, float, float] | None,
        ) -> Image.Image:
            frame_w, frame_h = frame.size
            if person_box is not None:
                x1, y1, x2, y2 = person_box
                seat_x1, seat_y1, seat_x2, seat_y2 = seat_box
                left = min(x1, seat_x1)
                top = min(y1, seat_y1)
                right = max(x2, seat_x2)
                bottom = max(y2, seat_y2)
                pad_x = max(90.0, (x2 - x1) * 0.85)
                pad_y_top = max(120.0, (y2 - y1) * 0.75)
                pad_y_bottom = max(80.0, (y2 - y1) * 0.35)
                crop_box = (
                    max(0.0, left - pad_x),
                    max(0.0, top - pad_y_top),
                    min(frame_w, right + pad_x),
                    min(frame_h, bottom + pad_y_bottom),
                )
            else:
                x1, y1, x2, y2 = seat_box
                pad_x = max(120.0, (x2 - x1) * 1.3)
                pad_y = max(100.0, (y2 - y1) * 1.8)
                crop_box = (
                    max(0.0, x1 - pad_x),
                    max(0.0, y1 - pad_y),
                    min(frame_w, x2 + pad_x),
                    min(frame_h, y2 + pad_y),
                )
            return frame.crop(tuple(int(value) for value in crop_box)).convert("RGB")

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
            focus_row = self._row_for_seat(focus_seat) if focus_seat is not None else None
            focus_person_box = self._row_bbox(focus_row) if focus_row is not None else None

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

            if focus_person_box is not None and focus_seat is not None:
                draw.rounded_rectangle(focus_person_box, radius=14, outline=(255, 235, 109), width=4)
                self._overlay_tag(
                    draw,
                    (focus_person_box[0], max(18.0, focus_person_box[1] - 42.0)),
                    f"学生 {focus_seat:02d}",
                    (255, 235, 109),
                )

            if self.provider is not None and self.provider.error and self.mode != "live":
                self._overlay_tag(draw, (24.0, 24.0), self.provider.error[:46], (255, 156, 109))
            self.main_image_label.set_pil_image(frame.convert("RGB"))

        def _render_detail_view(self) -> None:
            focus_seat = self.selected_seat or self.snapshot.focus_seat
            if focus_seat is None and self.seats:
                focus_seat = self.seats[0].seat_no
            if focus_seat is None or focus_seat not in self.seat_lookup:
                self.detail_text_label.setText("暂无可聚焦的座位。")
                self.detail_image_label.set_pil_image(
                    render_placeholder_image((960, 540), "ExamWatchAI", "暂无聚焦座位", None)
                )
                return

            seat = self.seat_lookup[focus_seat]
            base_frame = self.current_clean_frame if self.current_clean_frame is not None else self.current_frame
            frame = base_frame.copy().convert("RGBA")
            draw = ImageDraw.Draw(frame, "RGBA")
            seat_box = self._scaled_box(seat, frame.size)
            live_row = self._row_for_seat(focus_seat)
            person_box = self._row_bbox(live_row) if live_row is not None else None
            status = self.snapshot.status_by_seat.get(focus_seat, "normal")
            behavior = self.snapshot.behavior_by_seat.get(focus_seat, STATUS_LABELS[status])
            _, _, outline_rgb = self._status_colors(status)
            draw.rounded_rectangle(seat_box, radius=14, outline=outline_rgb, width=4)
            if person_box is not None:
                draw.rounded_rectangle(person_box, radius=14, outline=(255, 235, 109), width=4)
                self._overlay_tag(
                    draw,
                    (person_box[0], max(18.0, person_box[1] - 42.0)),
                    f"已绑定学生 {focus_seat:02d}",
                    (255, 235, 109),
                )
            crop = self._crop_focus_region(frame, seat_box, person_box)
            self.detail_image_label.set_pil_image(crop)
            detail_parts = [
                f"聚焦座位 {focus_seat:02d}",
                f"状态: {STATUS_LABELS.get(status, '正常')}",
                f"说明: {behavior or '正常'}",
            ]
            if live_row is not None:
                role = str(live_row.get("role", "unknown"))
                role_label = {
                    "student": "学生",
                    "teacher": "教师",
                    "unknown": "待确认目标",
                }.get(role, role)
                detail_parts.append(f"角色: {role_label}")
                track_id = live_row.get("track_id")
                if track_id not in (None, ""):
                    detail_parts.append(f"跟踪ID: {track_id}")
                person_conf = live_row.get("person_conf")
                if person_conf not in (None, ""):
                    try:
                        detail_parts.append(f"置信度: {float(person_conf):.2f}")
                    except (TypeError, ValueError):
                        pass
                if to_bool(live_row.get("is_bound")):
                    detail_parts.append("绑定: 已确认")
                elif to_int(live_row.get("seat_no_current")) == focus_seat:
                    detail_parts.append("绑定: 待确认")
            else:
                detail_parts.append("绑定: 当前未检测到学生")
            detail_parts.append(self.latest_progress_text)
            self.detail_text_label.setText(" | ".join(detail_parts))

        def _on_table_select(self) -> None:
            selected_items = self.events_table.selectedItems()
            if not selected_items:
                return
            seat_item = self.events_table.item(selected_items[0].row(), 2)
            if seat_item is None:
                return
            try:
                seat_no = int(seat_item.text())
            except ValueError:
                return
            self._focus_seat(seat_no)

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            if hasattr(self, "resize_refresh_timer"):
                self.resize_refresh_timer.start(90)


else:
    class DetectionDashboard:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                f"当前环境未检测到 PyQt6，无法启动 PyQt 界面: {QT_IMPORT_ERROR}"
            )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_source = resolve_demo_source(project_root)
    default_layout = project_root / "detect" / "seats.json"
    default_person_weights = project_root / "exam_seat_binding" / "weight" / "yolo11speopel.pt"
    default_desk_weights = project_root / "exam_seat_binding" / "weight" / "yolo11desk.pt"
    default_output_dir = project_root / "exam_seat_binding" / "output"

    parser = argparse.ArgumentParser(description="ExamWatchAI PyQt 检测巡检 UI")
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
    parser.add_argument("--room-name", type=str, default="第一考场", help="界面显示的考场名称")
    parser.add_argument("--refresh-ms", type=int, default=220, help="界面刷新间隔，单位毫秒")
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
    parser.add_argument("--frame-callback-stride", type=int, default=3, help="真实检测回调抽帧间隔")
    parser.add_argument("--frame-callback-max-width", type=int, default=1280, help="回调画面最大宽度")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        live_outputs = parse_outputs(args.pipeline_outputs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not QT_AVAILABLE:
        raise SystemExit(
            "未检测到 PyQt6。请在你的运行环境中先安装 `PyQt6` 再启动当前界面。"
        )

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
        frame_callback_stride=args.frame_callback_stride,
        frame_callback_max_width=args.frame_callback_max_width,
    )

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("ExamWatchAI")
    qt_app.setStyle("Fusion")
    app_font = build_qt_font(10, families=pick_qt_font_families())
    qt_app.setFont(app_font)

    window = DetectionDashboard(
        source=args.source or None,
        layout_path=args.layout or None,
        room_name=args.room_name,
        refresh_ms=args.refresh_ms,
        mode=args.mode,
        live_config=live_config,
    )
    window.show()
    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
