"""
main.py
Open-Source Real-Time Theft & Violence Detection System — v2.0
Entry point. Connects all components and runs the main loop.

Usage:
  python main.py                        # uses config/settings.yaml
  python main.py --config my_config.yaml
  python main.py --no-display           # headless (server mode)
  python main.py --source video.mp4     # test on a video file
"""

import os
import sys
import cv2
import time
import yaml
import signal
import logging
import argparse
import threading
from datetime import datetime

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/system.log", mode="a"),
    ]
)
logger = logging.getLogger("main")


# ── Argument parsing ──────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Security Detection System v2.0")
    p.add_argument("--config",     default="config/settings.yaml")
    p.add_argument("--no-display", action="store_true", help="No OpenCV window (headless)")
    p.add_argument("--source",     default=None,   help="Override camera source")
    p.add_argument("--no-alerts",  action="store_true", help="Disable alerts (dev mode)")
    p.add_argument("--no-dashboard", action="store_true")
    return p.parse_args()


# ── Config loader ────────────────────────────────────────────

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        logger.error(f"Config not found: {path}")
        sys.exit(1)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded: {path}")
    return cfg


# ── Main system class ─────────────────────────────────────────

class SecuritySystem:

    def __init__(self, args):
        self.args     = args
        self.cfg      = load_config(args.config)
        self.running  = False

        os.makedirs("logs", exist_ok=True)
        os.makedirs("logs/alert_images", exist_ok=True)

        # Override source if specified via CLI
        if args.source is not None:
            source = args.source
            try:
                source = int(source)
            except ValueError:
                pass
            for cam in self.cfg["cameras"]:
                cam["source"] = source
            logger.info(f"Source overridden: {source}")

        logger.info("=" * 60)
        logger.info("  Security Detection System v2.0")
        logger.info("  100% Free | No Paid APIs | Runs Locally")
        logger.info("=" * 60)

        self._init_components()

    def _init_components(self):
        """Initialize all system components."""
        from core.camera_manager import CameraManager
        from core.pipeline import MultiCameraPipeline
        from alerts.alert_manager import AlertManager
        from utils.logger import EventLogger
        from utils.fps_counter import SystemMonitor

        logger.info("Initializing components...")

        # Camera manager
        self.cameras = CameraManager(self.cfg["cameras"])
        camera_names = {
            cam["id"]: cam["name"]
            for cam in self.cfg["cameras"]
            if cam.get("enabled", True)
        }

        # Processing pipeline
        self.pipeline = MultiCameraPipeline(self.cfg)

        # Event logger (SQLite)
        self.event_logger = EventLogger(self.cfg["logging"])

        # Alert manager
        if not self.args.no_alerts:
            self.alerts = AlertManager(self.cfg["alerts"], camera_names)
        else:
            self.alerts = None
            logger.info("Alerts DISABLED (--no-alerts flag)")

        # System monitor
        self.sys_monitor = SystemMonitor()

        logger.info("All components initialized.")

    def start(self):
        """Start cameras, dashboard, and main loop."""
        self.running = True

        # Start cameras
        self.cameras.start_all()
        time.sleep(1.0)   # warmup

        # Start dashboard
        if not self.args.no_dashboard and self.cfg["dashboard"]["enabled"]:
            self._start_dashboard()

        logger.info("Starting main detection loop. Press Ctrl+C to stop.")
        self._main_loop()

    def stop(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False
        self.cameras.stop_all()
        cv2.destroyAllWindows()
        logger.info("System stopped.")

    # ── Main loop ────────────────────────────────────────────

    def _main_loop(self):
        display = not self.args.no_display
        perf_log_interval = 5.0   # seconds
        last_perf_log     = time.time()

        while self.running:
            # Fetch all frames
            all_frames = self.cameras.get_all_frames()

            if not all_frames:
                time.sleep(0.01)
                continue

            # Process each camera
            all_results = self.pipeline.process_all(all_frames)

            for cam_id, result in all_results.items():
                if result is None:
                    continue

                # Handle violence detection
                if result.violence_detected:
                    self._on_violence(result)

                # Handle theft events
                for evt in result.theft_events:
                    self._on_theft(evt, result)

                # Display window
                if display and result.annotated_frame is not None:
                    win_name = f"Camera {cam_id} — {self.cameras.streams[cam_id].config.name}"
                    cv2.imshow(win_name, result.annotated_frame)

                # Log performance periodically
                if time.time() - last_perf_log >= perf_log_interval:
                    self.event_logger.log_performance(
                        camera_id=cam_id,
                        fps=result.fps,
                        processing_ms=result.processing_ms,
                        num_detections=len(result.detections),
                    )
                    logger.debug(
                        f"[Cam {cam_id}] FPS={result.fps:.1f} | "
                        f"Proc={result.processing_ms:.0f}ms | "
                        f"Dets={len(result.detections)}"
                    )

            if time.time() - last_perf_log >= perf_log_interval:
                last_perf_log = time.time()

            # Keyboard handler
            if display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("'q' pressed — stopping.")
                    break
                elif key == ord("r"):
                    # Reset violence cooldowns
                    for pl in self.pipeline.pipelines.values():
                        pl.violence.reset_cooldown()
                    logger.info("Cooldowns reset.")

    # ── Event handlers ────────────────────────────────────────

    def _on_violence(self, result):
        cam_name = self.cameras.streams[result.camera_id].config.name
        logger.warning(
            f"[VIOLENCE] Camera: {cam_name} | "
            f"Confidence: {result.violence_score:.0%}"
        )

        # Log to SQLite
        self.event_logger.log_event(
            event_type="violence",
            camera_id=result.camera_id,
            camera_name=cam_name,
            confidence=result.violence_score,
            details={"frame": result.frame_number},
            frame=result.annotated_frame,
        )

        # Send alert
        if self.alerts:
            self.alerts.dispatch_violence(
                camera_id=result.camera_id,
                score=result.violence_score,
                frame=result.annotated_frame,
            )

    def _on_theft(self, evt, result):
        cam_name = self.cameras.streams[evt.camera_id].config.name
        logger.warning(
            f"[THEFT-{evt.level.upper()}] Camera: {cam_name} | "
            f"Object: {evt.object_class} | "
            f"Score: {evt.score:.0%} | "
            f"Zone: {evt.zone_name or 'None'}"
        )

        # Log to SQLite
        self.event_logger.log_event(
            event_type=f"theft_{evt.level}",
            camera_id=evt.camera_id,
            camera_name=cam_name,
            confidence=evt.score,
            details={
                "object_class":    evt.object_class,
                "zone":            evt.zone_name,
                "person_track_id": evt.person_track_id,
                "frame":           evt.frame_number,
            },
            frame=evt.frame_snapshot,
        )

        # Send alert
        if self.alerts:
            self.alerts.dispatch_theft(evt, cam_name)

    # ── Dashboard ─────────────────────────────────────────────

    def _start_dashboard(self):
        try:
            from dashboard.app import init_dashboard, run_dashboard
            init_dashboard(
                event_logger=self.event_logger,
                camera_manager=self.cameras,
                pipeline=self.pipeline,
                system_monitor=self.sys_monitor,
            )
            host = self.cfg["dashboard"]["host"]
            port = self.cfg["dashboard"]["port"]
            t = threading.Thread(
                target=run_dashboard,
                kwargs={"host": host, "port": port},
                daemon=True
            )
            t.start()
            logger.info(f"Dashboard running at http://{host}:{port}")
        except Exception as e:
            logger.error(f"Dashboard failed to start: {e}")


# ── Entry point ───────────────────────────────────────────────

def main():
    args   = parse_args()
    system = SecuritySystem(args)

    # Graceful Ctrl+C
    def on_signal(sig, frame):
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        system.start()
    except KeyboardInterrupt:
        system.stop()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        system.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
