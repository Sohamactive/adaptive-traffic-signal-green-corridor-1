"""Detection module for YOLO inference, tracking, and vehicle counting."""

from detection.lane_scanner import LaneScanner, LaneScanResult
from detection.vehicalCount import VehicleCounter

__all__ = ["VehicleCounter", "LaneScanner", "LaneScanResult"]
