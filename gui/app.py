"""Flask routes for the adaptive traffic signal GUI.

All detection / inference logic lives in the ``detection`` package.
This module is a thin controller that handles HTTP requests and
renders templates — nothing more.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT
from detection import VehicleDetector
from detection.camera import CameraStream, generate_annotated_stream


def create_app() -> Flask:
    """Application factory — builds and returns a configured Flask app."""
    app = Flask(__name__)

    # Shared detector instance (loaded once, reused across requests)
    detector = VehicleDetector()

    # ── Routes ───────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        camera = CameraStream()
        try:
            camera.open()
        except RuntimeError:
            camera = None  # generate_annotated_stream will serve placeholder frames
        return Response(
            generate_annotated_stream(detector, camera),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/predict_frame", methods=["POST"])
    def predict_frame():
        """Accept a single JPEG frame from the browser camera and return
        the annotated frame + vehicle count as JSON."""
        file = request.files.get("frame")
        if not file:
            return jsonify({"error": "No frame uploaded"}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        result = detector.detect(img)  # type: ignore[arg-type]
        annotated = detector.draw_vehicle_count(
            result.annotated_frame, result.vehicle_count
        )
        _, buf = cv2.imencode(".jpg", annotated)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return jsonify({"annotated": b64, "vehicle_count": result.vehicle_count})

    # Lane identifiers for the four-way intersection
    LANES = {
        "laneN": "North",
        "laneS": "South",
        "laneE": "East",
        "laneW": "West",
    }

    @app.route("/test", methods=["GET", "POST"])
    def test():
        lane_counts: dict[str, int] = {}
        lane_images: dict[str, str] = {}

        if request.method == "POST":
            for lane_key in LANES:
                file = request.files.get(lane_key)
                if file and file.filename:
                    # Decode the uploaded image
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    # Run detection
                    result = detector.detect(img)  # type: ignore[arg-type]
                    annotated = detector.draw_vehicle_count(
                        result.annotated_frame, result.vehicle_count
                    )
                    lane_counts[lane_key] = result.vehicle_count

                    # Encode to base64 for embedding in the template
                    _, buf = cv2.imencode(".jpg", annotated)
                    lane_images[lane_key] = base64.b64encode(buf.tobytes()).decode("utf-8")

        return render_template(
            "test.html",
            lane_counts=lane_counts,
            lane_images=lane_images,
            lanes=LANES,
        )

    return app


# Allow running directly with `python gui/app.py` for convenience
if __name__ == "__main__":
    create_app().run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
