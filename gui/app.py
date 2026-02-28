"""Flask routes for the adaptive traffic signal GUI.

All detection / inference logic lives in the ``detection`` package.
This module is a thin controller that handles HTTP requests and
renders templates — nothing more.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
from flask import Flask, Response, render_template, request

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
        camera.open()
        return Response(
            generate_annotated_stream(detector, camera),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/test", methods=["GET", "POST"])
    def test():
        result_image: str | None = None
        vehicle_count: int | None = None

        if request.method == "POST":
            file = request.files.get("image")
            if file and file.filename:
                # Decode the uploaded image
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Run detection
                result = detector.detect(img)
                annotated = detector.draw_vehicle_count(
                    result.annotated_frame, result.vehicle_count
                )
                vehicle_count = result.vehicle_count

                # Encode to base64 for embedding in the template
                _, buf = cv2.imencode(".jpg", annotated)
                result_image = base64.b64encode(buf.tobytes()).decode("utf-8")

        return render_template(
            "test.html", result_image=result_image, vehicle_count=vehicle_count
        )

    return app


# Allow running directly with `python gui/app.py` for convenience
if __name__ == "__main__":
    create_app().run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
