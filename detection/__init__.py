"""Vehicle detection subsystem.

Exposes the main detector class and supporting utilities so that other
subsystems (GUI, prediction, optimization) can consume detection results
through a clean interface without depending on YOLO or OpenCV directly.
"""

from detection.vehicle_detector import DetectionResult, VehicleDetector

__all__ = ["VehicleDetector", "DetectionResult"]
