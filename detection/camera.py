"""Camera capture and throttled detection stream utilities."""

from __future__ import annotations

import time
from typing import Generator

import cv2
import numpy as np

from config import CAMERA_BUFFER_SIZE, CAMERA_INDEX, INFERENCE_INTERVAL
from detection.vehicle_detector import VehicleDetector


class CameraStream:
    """Thin wrapper around ``cv2.VideoCapture`` for a single camera source.

    Parameters:
        source: Camera index (int) or video file path (str).
        buffer_size: OpenCV capture buffer size hint.
    """

    def __init__(
        self,
        source: int | str = CAMERA_INDEX,
        buffer_size: int = CAMERA_BUFFER_SIZE,
    ) -> None:
        self._source = source
        self._buffer_size = buffer_size
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open (or re-open) the underlying capture device."""
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self._source)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

    def read(self) -> tuple[bool, np.ndarray]:
        """Read a single frame, delegating to the underlying ``VideoCapture``."""
        if self._cap is None:
            self.open()
        assert self._cap is not None
        return self._cap.read()

    def release(self) -> None:
        """Release the capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def generate_annotated_stream(
    detector: VehicleDetector,
    camera: CameraStream | None = None,
    interval: float = INFERENCE_INTERVAL,
) -> Generator[bytes, None, None]:
    """Yield MJPEG frames with detection overlays at a throttled rate.

    This generator continuously reads from *camera*, but only runs the heavy
    YOLO model every *interval* seconds.  In between, it simply drains the
    camera buffer to keep the feed fresh — costing almost zero CPU.

    Args:
        detector: A ``VehicleDetector`` instance (already loaded).
        camera:   An open ``CameraStream``.  If *None*, one is created with
                  default settings.
        interval: Minimum seconds between successive YOLO inferences.

    Yields:
        Raw bytes of each MJPEG boundary + JPEG payload, ready to be
        streamed via an HTTP ``multipart/x-mixed-replace`` response.
    """
    if camera is None:
        camera = CameraStream()
        camera.open()

    last_inference_time: float = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        current_time = time.time()

        if current_time - last_inference_time >= interval:
            result = detector.detect(frame)
            annotated = detector.draw_vehicle_count(
                result.annotated_frame, result.vehicle_count
            )

            _, buffer = cv2.imencode(".jpg", annotated)
            frame_bytes = buffer.tobytes()

            last_inference_time = current_time

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
