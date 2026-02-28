"""Real-time lane-based vehicle counting using a single webcam.

The webcam simulates four intersection lanes (N, S, E, W) by capturing one
frame per lane in round-robin order.  Each frame is fed through
:class:`VehicleCounter` and the per-lane counts are stored so that the GUI or
control layer can query the busiest lane at any time.

Usage from the GUI / control loop::

    from detection.lane_scanner import LaneScanner

    scanner = LaneScanner(camera_index=0)
    scanner.start()

    # on each tick / timer callback:
    snapshot = scanner.scan_all_lanes()
    # snapshot -> {"N": 5, "S": 3, "E": 8, "W": 2}

    busiest = scanner.busiest_lane()
    # busiest -> ("E", 8)

    annotated_frames = scanner.annotated_frames
    # dict of lane_name -> np.ndarray (BGR image with boxes)

    scanner.release()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from detection.vehicalCount import VehicleCounter

# The four simulated lanes in scan order.
LANE_NAMES = ("N", "S", "E", "W")


@dataclass
class LaneScanResult:
    """Snapshot returned by a single full scan cycle."""

    counts: Dict[str, int]
    """Vehicle count per lane name."""

    busiest: Tuple[str, int]
    """(lane_name, count) of the lane with the most vehicles."""

    frames: Dict[str, np.ndarray] = field(repr=False)
    """Raw (un-annotated) camera frame captured for each lane."""

    annotated: Dict[str, np.ndarray] = field(repr=False)
    """Annotated frame (with bounding boxes) for each lane."""


class LaneScanner:
    """Cycle through lanes on a single webcam and count vehicles per lane.

    Parameters
    ----------
    camera_index:
        OpenCV camera index (``0`` = default webcam).
    lane_names:
        Tuple of lane identifiers.  Defaults to ``("N", "S", "E", "W")``.
    model_path:
        Optional path to a custom YOLO weights file; ``None`` uses the
        project default.
    """

    def __init__(
        self,
        camera_index: int = 0,
        lane_names: Tuple[str, ...] = LANE_NAMES,
        model_path=None,
    ):
        self.lane_names = lane_names
        self._counter = VehicleCounter(model_path=model_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._camera_index = camera_index

        # latest results keyed by lane name
        self.counts: Dict[str, int] = {ln: 0 for ln in lane_names}
        self.frames: Dict[str, np.ndarray] = {}
        self.annotated_frames: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the webcam.  Call once before :meth:`scan_all_lanes`."""
        if self._cap is not None and self._cap.isOpened():
            return
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self._camera_index}"
            )

    def release(self) -> None:
        """Release the webcam resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # scanning
    # ------------------------------------------------------------------

    def _grab_frame(self) -> np.ndarray:
        """Read a single frame from the webcam."""
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera not started – call start() first")
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def scan_lane(self, lane_name: str) -> Tuple[int, np.ndarray, np.ndarray]:
        """Capture one frame for *lane_name* and run detection.

        Returns ``(count, raw_frame, annotated_frame)``.
        """
        frame = self._grab_frame()
        count, results = self._counter.count_frame(frame)
        annotated = results.plot()

        # store latest
        self.counts[lane_name] = count
        self.frames[lane_name] = frame
        self.annotated_frames[lane_name] = annotated

        return count, frame, annotated

    def scan_all_lanes(self) -> LaneScanResult:
        """Run one full cycle: capture + detect for every lane, one by one.

        Between lanes the user is expected to reposition the camera (or, for a
        demo, the same view is reused).  In practice you would prompt the GUI
        to show which lane is being scanned.

        Returns a :class:`LaneScanResult` with all counts and frames.
        """
        raw: Dict[str, np.ndarray] = {}
        ann: Dict[str, np.ndarray] = {}

        for lane in self.lane_names:
            count, frame, annotated = self.scan_lane(lane)
            raw[lane] = frame
            ann[lane] = annotated

        busiest = self.busiest_lane()

        return LaneScanResult(
            counts=dict(self.counts),
            busiest=busiest,
            frames=raw,
            annotated=ann,
        )

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def busiest_lane(self) -> Tuple[str, int]:
        """Return ``(lane_name, count)`` of the lane with the highest count.

        If there is a tie the first lane in scan order wins.
        """
        return max(self.counts.items(), key=lambda kv: kv[1])

    def lane_counts(self) -> Dict[str, int]:
        """Return a copy of the current per-lane counts."""
        return dict(self.counts)
