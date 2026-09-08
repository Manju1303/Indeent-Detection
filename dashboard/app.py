"""
dashboard/app.py
Flask + SocketIO real-time monitoring dashboard with Live MJPEG Video Stream Feed.
Access at http://localhost:5000 after starting main.py
"""

import os
import sys
import cv2
import time
import logging
import threading
from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "security_dashboard_key"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Shared references (set by main.py)
_event_logger = None
_camera_manager = None
_pipeline = None
_system_monitor = None


def init_dashboard(event_logger, camera_manager, pipeline, system_monitor):
    """Called by main.py to inject shared references."""
    global _event_logger, _camera_manager, _pipeline, _system_monitor
    _event_logger    = event_logger
    _camera_manager  = camera_manager
    _pipeline        = pipeline
    _system_monitor  = system_monitor


# ── Live MJPEG Video Stream Generator ──────────────────────────

def _generate_video_stream(camera_id: int):
    """Generates real-time MJPEG video stream with live annotations."""
    while True:
        try:
            frame = None
            if _pipeline and camera_id in _pipeline.pipelines:
                cam_pipeline = _pipeline.pipelines[camera_id]
                if _camera_manager:
                    ret, raw_frame = _camera_manager.get_frame(camera_id)
                    if ret and raw_frame is not None:
                        result = cam_pipeline.process(raw_frame)
                        if result and result.annotated_frame is not None:
                            frame = result.annotated_frame
                        else:
                            frame = raw_frame
            elif _camera_manager:
                ret, raw_frame = _camera_manager.get_frame(camera_id)
                if ret and raw_frame is not None:
                    frame = raw_frame

            if frame is None:
                # Blank placeholder frame if camera unavailable
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Camera {camera_id} Initializing...", (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            ret_jpg, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_jpg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        except Exception as e:
            logger.debug(f"[Dashboard] Stream error cam {camera_id}: {e}")
            time.sleep(0.1)

        time.sleep(0.03)   # ~30 FPS


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    """Real-time MJPEG video stream route with live detection overlays."""
    return Response(_generate_video_stream(camera_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    if _event_logger is None:
        return jsonify({"error": "not initialized"})
    return jsonify(_event_logger.get_stats())


@app.route("/api/events")
def api_events():
    if _event_logger is None:
        return jsonify([])
    events = _event_logger.get_recent_events(50)
    return jsonify(events)


@app.route("/api/cameras")
def api_cameras():
    if _camera_manager is None:
        return jsonify([])
    return jsonify(_camera_manager.status())


@app.route("/api/performance")
def api_performance():
    if _event_logger is None:
        return jsonify({})
    perf  = _event_logger.get_performance_avg()
    sys_s = _system_monitor.get_stats() if _system_monitor else {}
    return jsonify({**perf, **sys_s})


@app.route("/api/events/<event_type>")
def api_events_filtered(event_type):
    if _event_logger is None:
        return jsonify([])
    return jsonify(_event_logger.get_recent_events(50, event_type=event_type))


# ── Background Emitter ────────────────────────────────────────

def _emit_loop():
    """Pushes live updates & real-time telemetry to connected dashboard clients every 0.3s."""
    while True:
        try:
            if socketio:
                stats  = _event_logger.get_stats() if _event_logger else {}
                events = _event_logger.get_recent_events(10) if _event_logger else []
                cams   = _camera_manager.status() if _camera_manager else []

                # Collect real-time intent predictions across cameras
                telemetry = []
                if _pipeline:
                    for cam_id, pipe in _pipeline.pipelines.items():
                        # Fetch recent telemetry
                        telemetry.append({
                            "camera_id": cam_id,
                            "fps": round(pipe.fps_ctr.fps, 1),
                        })

                socketio.emit("update", {
                    "stats":     stats,
                    "events":    events,
                    "cameras":   cams,
                    "telemetry": telemetry,
                })
        except Exception as e:
            logger.debug(f"[Dashboard] Emit error: {e}")
        time.sleep(0.3)


def run_dashboard(host="0.0.0.0", port=5000):
    """Start dashboard in background thread."""
    t = threading.Thread(target=_emit_loop, daemon=True)
    t.start()
    logger.info(f"[Dashboard] Starting Live Stream Dashboard at http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False, log_output=False)
