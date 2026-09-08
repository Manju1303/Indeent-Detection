"""
dashboard/app.py
Flask + SocketIO real-time monitoring dashboard.
Access at http://localhost:5000 after starting main.py
"""

import os
import sys
import logging
from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO
import threading
import time

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


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


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


# ── Background emitter ────────────────────────────────────────

def _emit_loop():
    """Pushes live updates to connected dashboard clients every 0.5s."""
    while True:
        try:
            if _event_logger and socketio:
                stats  = _event_logger.get_stats()
                events = _event_logger.get_recent_events(10)
                cams   = _camera_manager.status() if _camera_manager else []
                socketio.emit("update", {
                    "stats":   stats,
                    "events":  events,
                    "cameras": cams,
                })
        except Exception as e:
            logger.debug(f"[Dashboard] Emit error: {e}")
        time.sleep(0.5)


def run_dashboard(host="0.0.0.0", port=5000):
    """Start dashboard in background thread."""
    t = threading.Thread(target=_emit_loop, daemon=True)
    t.start()
    logger.info(f"[Dashboard] Starting at http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False, log_output=False)
