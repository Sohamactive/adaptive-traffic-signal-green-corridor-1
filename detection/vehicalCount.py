from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

# default constants
BASE = Path(__file__).parent
MODEL_PATH = BASE / "traffic_detection_yolov8s.pt"


class VehicleCounter:
    """Wrap a YOLO model for vehicle counting.

    Primary API for real-time use::

        vc = VehicleCounter()
        count, results = vc.count_frame(frame)   # frame from cv2.VideoCapture
        annotated = vc.annotate_frame(frame)      # frame with bounding boxes

    The class keeps the underlying :class:`ultralytics.YOLO` object in
    ``self.model`` so you can access other Ultralytics features if needed.
    """

    def __init__(self, model_path: Union[str, Path] = None):
        if model_path is None:
            model_path = MODEL_PATH
        self.model = YOLO(str(model_path))

    def predict(self, img_path: Union[str, Path]):
        """Run inference on a single image file and return the raw results."""
        return self.model(str(img_path), conf=0.25, iou=0.5)[0]

    def predict_frame(self, frame: np.ndarray):
        """Run inference directly on a numpy image (BGR/uint8).

        The frame is saved to a temporary buffer for the YOLO call because the
        current ultralytics API expects a path or URL.  If a future version
        supports arrays natively this helper can be simplified.
        """
        # ultralytics can accept numpy arrays directly, so we pass frame
        return self.model(frame, conf=0.25, iou=0.5)[0]

    def count_image(self, img_path: Union[str, Path]) -> Tuple[int, object]:
        """Return `(count, results)` for one image file."""
        results = self.predict(img_path)
        count = len(results.boxes) if results.boxes is not None else 0
        return count, results

    def count_frame(self, frame: np.ndarray) -> Tuple[int, object]:
        """Return `(count, results)` for an in‑memory image frame."""
        results = self.predict_frame(frame)
        count = len(results.boxes) if results.boxes is not None else 0
        return count, results

    def annotate_image(self, img_path: Union[str, Path], out_path: Union[str, Path]):
        """Run inference and save an annotated copy to ``out_path``."""
        _, results = self.count_image(img_path)
        annotated = results.plot()
        cv2.imwrite(str(out_path), annotated)

    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run inference on ``frame`` and return an annotated array.

        The returned image has bounding boxes drawn and shares the same shape
        and dtype as the input.
        """
        _, results = self.count_frame(frame)
        annotated = results.plot()  # this is numpy array
        return annotated

    def count_folder(
        self,
        folder: Union[str, Path],
        max_files: int = -1,
    ) -> List[Tuple[Path, int]]:
        """Process all images in a directory, optionally limiting the number.

        Returns a list of ``(path, count)`` tuples.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f"{folder!r} is not a directory")
        files = list(folder.glob("*.*"))
        if max_files > 0:
            files = files[:max_files]
        results = []
        for img in files:
            cnt, _ = self.count_image(img)
            results.append((img, cnt))
        return results

