# seat_label_tool.py   
#python detect/seat_label_tool.py --image data/first_frame.png --out detect/seats.json --max-seats 30
import argparse
import json
import os
from pathlib import Path

import cv2


class SeatLabelTool:
    def __init__(self, image_path: str, out_json: str, max_seats: int = 30):
        self.image_path = image_path
        self.out_json = out_json
        self.max_seats = max_seats

        self.img0 = cv2.imread(image_path)
        if self.img0 is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")

        self.img = self.img0.copy()
        self.seats = []  # list of dict: {"id": int, "x1":..., "y1":..., "x2":..., "y2":...}

        self.drawing = False
        self.x1 = self.y1 = self.x2 = self.y2 = 0

        self.win = "SeatLabelTool"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x1, self.y1 = x, y
            self.x2, self.y2 = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.x2, self.y2 = x, y
            self.refresh(preview=True)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.x2, self.y2 = x, y
            self.refresh(preview=True)

    def refresh(self, preview=False):
        self.img = self.img0.copy()

        # draw existing seats
        for s in self.seats:
            cv2.rectangle(self.img, (s["x1"], s["y1"]), (s["x2"], s["y2"]), (0, 255, 0), 2)
            cv2.putText(self.img, f"seat {s['id']}", (s["x1"], max(0, s["y1"] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # draw preview rectangle
        if preview and (self.x1 != self.x2 and self.y1 != self.y2):
            x1, y1, x2, y2 = self._norm_rect(self.x1, self.y1, self.x2, self.y2)
            cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        info = f"Seats: {len(self.seats)}/{self.max_seats} | Keys: n=add  u=undo  s=save  q=quit"
        cv2.putText(self.img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)

    @staticmethod
    def _norm_rect(x1, y1, x2, y2):
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def add_current_rect(self):
        if len(self.seats) >= self.max_seats:
            print("已达到最大 seat 数量")
            return
        if self.x1 == self.x2 or self.y1 == self.y2:
            print("当前没有有效矩形，先用鼠标拖拽画框")
            return

        x1, y1, x2, y2 = self._norm_rect(self.x1, self.y1, self.x2, self.y2)
        seat_id = len(self.seats)
        self.seats.append({"id": seat_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        print(f"Added seat {seat_id}: ({x1},{y1})-({x2},{y2})")

        # reset rect
        self.x1 = self.y1 = self.x2 = self.y2 = 0
        self.refresh(preview=False)

    def undo(self):
        if self.seats:
            removed = self.seats.pop()
            print(f"Undo seat {removed['id']}")
        self.refresh(preview=False)

    def save(self):
        os.makedirs(str(Path(self.out_json).parent), exist_ok=True)
        data = {
            "image": self.image_path,
            "max_seats": self.max_seats,
            "seats": self.seats
        }
        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved: {self.out_json}")

    def run(self):
        self.refresh()
        while True:
            cv2.imshow(self.win, self.img)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('n'):
                self.add_current_rect()
            elif key == ord('u'):
                self.undo()
            elif key == ord('s'):
                self.save()
            elif key == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="用于标注的图片（建议视频第一帧）")
    ap.add_argument("--out", type=str, default="detect/seats.json", help="输出 seats.json 路径")
    ap.add_argument("--max-seats", type=int, default=30, help="seat 数量（默认 30）")
    args = ap.parse_args()

    tool = SeatLabelTool(args.image, args.out, args.max_seats)
    tool.run()


if __name__ == "__main__":
    main()