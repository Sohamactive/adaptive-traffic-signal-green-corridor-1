"""Tkinter dashboard for testing lane-based vehicle detection.

Launch with::

    python -m gui.dashboard

or import and call :func:`launch` from another module.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from detection.lane_scanner import LANE_NAMES, LaneScanner

# ── layout constants ────────────────────────────────────────────────
FRAME_W = 320
FRAME_H = 240
SCAN_INTERVAL_MS = 100  # ms between individual lane scans
HIGHLIGHT_COLOR = "#27ae60"  # green accent for busiest lane
DEFAULT_BG = "#2c3e50"
CARD_BG = "#34495e"
TEXT_FG = "#ecf0f1"
COUNT_FG = "#f39c12"


class LaneCard(tk.Frame):
    """Widget showing one lane's annotated frame and vehicle count."""

    def __init__(self, master: tk.Widget, lane_name: str, **kw):
        super().__init__(master, bg=CARD_BG, bd=2, relief="groove", **kw)
        self.lane_name = lane_name

        # header
        self.header = tk.Label(
            self,
            text=f"Lane {lane_name}",
            font=("Helvetica", 14, "bold"),
            bg=CARD_BG,
            fg=TEXT_FG,
        )
        self.header.pack(pady=(6, 2))

        # video canvas
        self.canvas = tk.Label(self, bg="black", width=FRAME_W, height=FRAME_H)
        self.canvas.pack(padx=6, pady=4)

        # count label
        self.count_var = tk.StringVar(value="Vehicles: –")
        self.count_label = tk.Label(
            self,
            textvariable=self.count_var,
            font=("Helvetica", 16, "bold"),
            bg=CARD_BG,
            fg=COUNT_FG,
        )
        self.count_label.pack(pady=(0, 6))

        self._photo: Optional[ImageTk.PhotoImage] = None

    # ── update helpers ──────────────────────────────────────────────

    def set_frame(self, bgr_img: np.ndarray) -> None:
        """Display a BGR numpy image on the canvas."""
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((FRAME_W, FRAME_H))
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.configure(image=self._photo)

    def set_count(self, count: int) -> None:
        self.count_var.set(f"Vehicles: {count}")

    def set_highlight(self, on: bool) -> None:
        color = HIGHLIGHT_COLOR if on else CARD_BG
        self.configure(bg=color)
        self.header.configure(bg=color)
        self.count_label.configure(bg=color)


class Dashboard(tk.Tk):
    """Main window: 2×2 grid of lane cards + status bar."""

    def __init__(self):
        super().__init__()
        self.title("Adaptive Traffic Signal – Lane Monitor")
        self.configure(bg=DEFAULT_BG)
        self.resizable(False, False)

        self._scanner: Optional[LaneScanner] = None
        self._running = False
        self._scan_index = 0  # which lane we scan next
        self._lock = threading.Lock()

        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        # title bar
        title = tk.Label(
            self,
            text="Lane Vehicle Counter",
            font=("Helvetica", 20, "bold"),
            bg=DEFAULT_BG,
            fg=TEXT_FG,
        )
        title.grid(row=0, column=0, columnspan=2, pady=(10, 4))

        # lane cards in 2×2 grid
        self.cards: Dict[str, LaneCard] = {}
        positions = {"N": (1, 0), "S": (1, 1), "E": (2, 0), "W": (2, 1)}
        for lane, (r, c) in positions.items():
            card = LaneCard(self, lane)
            card.grid(row=r, column=c, padx=8, pady=6)
            self.cards[lane] = card

        # control frame
        ctrl = tk.Frame(self, bg=DEFAULT_BG)
        ctrl.grid(row=3, column=0, columnspan=2, pady=8)

        self.start_btn = ttk.Button(ctrl, text="▶  Start", command=self._on_start)
        self.start_btn.pack(side="left", padx=6)

        self.stop_btn = ttk.Button(ctrl, text="■  Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)

        # status / busiest lane
        self.status_var = tk.StringVar(value="Status: idle")
        status_lbl = tk.Label(
            self,
            textvariable=self.status_var,
            font=("Helvetica", 13),
            bg=DEFAULT_BG,
            fg=TEXT_FG,
        )
        status_lbl.grid(row=4, column=0, columnspan=2, pady=(0, 10))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── scanner lifecycle ───────────────────────────────────────────

    def _on_start(self) -> None:
        if self._running:
            return
        try:
            self._scanner = LaneScanner(camera_index=0)
            self._scanner.start()
        except RuntimeError as exc:
            self.status_var.set(f"Error: {exc}")
            return

        self._running = True
        self._scan_index = 0
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Status: scanning…")
        self._tick()

    def _on_stop(self) -> None:
        self._running = False
        if self._scanner:
            self._scanner.release()
            self._scanner = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Status: stopped")

    def _on_close(self) -> None:
        self._on_stop()
        self.destroy()

    # ── scan loop (runs on the Tk main thread via after()) ──────────

    def _tick(self) -> None:
        if not self._running or self._scanner is None:
            return

        lane = LANE_NAMES[self._scan_index]

        try:
            count, _raw, annotated = self._scanner.scan_lane(lane)
        except RuntimeError as exc:
            self.status_var.set(f"Camera error: {exc}")
            self._on_stop()
            return

        # update UI for the scanned lane
        card = self.cards[lane]
        card.set_frame(annotated)
        card.set_count(count)

        # advance to next lane
        self._scan_index = (self._scan_index + 1) % len(LANE_NAMES)

        # after a full cycle, update busiest highlight
        if self._scan_index == 0:
            busiest_name, busiest_count = self._scanner.busiest_lane()
            for ln, c in self.cards.items():
                c.set_highlight(ln == busiest_name)
            self.status_var.set(
                f"Busiest lane: {busiest_name} ({busiest_count} vehicles)  |  "
                f"Counts: {self._scanner.lane_counts()}"
            )

        # schedule next tick
        self.after(SCAN_INTERVAL_MS, self._tick)


def launch() -> None:
    """Create and run the dashboard (blocking)."""
    app = Dashboard()
    app.mainloop()


if __name__ == "__main__":
    launch()
