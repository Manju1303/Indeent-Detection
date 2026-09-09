"""
core/camera_manager.py
Multi-camera stream manager with reconnection logic and frame buffering.
Supports: webcam, RTSP streams, video files.
"""

import cv2
import threading
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    id: int
    name: str
    source: any          # int (webcam index) or str (RTSP/file path)
    enabled: bool = True
    zones: List[Dict] = field(default_factory=list)


class CameraStream:
    """
    Individual camera stream with threaded reading and auto-reconnect.
    Prevents main pipeline from blocking on slow/dropped cameras.
    """

    RECONNECT_DELAY = 3.0       # seconds between reconnect attempts
    BUFFER_SIZE     = 5         # max frames buffered per camera

    def __init__(self, config: CameraConfig):
        self.config     = config
        self.cap        = None
        self.frame_buf  = deque(maxlen=self.BUFFER_SIZE)
        self.running    = False
        self._thread    = None
        self._lock      = threading.Lock()
        self.fps        = 0.0
        self.frame_count = 0
        self.connected  = False

    # ── Public API ────────────────────────────────────────────

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info(f"[Camera {self.config.id}] '{self.config.name}' stream started.")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self.cap:
            self.cap.release()
        logger.info(f"[Camera {self.config.id}] '{self.config.name}' stopped.")

    def read(self):
        """Return latest frame or None if buffer empty."""
        with self._lock:
            if self.frame_buf:
                return True, self.frame_buf[-1]
            return False, None

    def is_healthy(self) -> bool:
        return self.running and self.connected

    # ── Internal ──────────────────────────────────────────────

    def _connect(self) -> bool:
        if self.cap:
            self.cap.release()
        source = self.config.source

        # For integer webcam sources on Windows, use DirectShow (CAP_DSHOW) for zero-latency live access
        if isinstance(source, int):
            import platform
            if platform.system() == "Windows":
                self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(source)
            else:
                self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            logger.warning(f"[Camera {self.config.id}] Failed to open source: {source}")
            return False

        # Set buffer size to 1 for real-time live detection (zero frame queuing)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.connected = True
        logger.info(f"[Camera {self.config.id}] Connected to live source: {source}")
        return True

    def _read_loop(self):
        fps_start = time.time()
        fps_frames = 0

        while self.running:
            if not self.connected:
                if not self._connect():
                    time.sleep(self.RECONNECT_DELAY)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"[Camera {self.config.id}] Frame read failed. Retrying in {self.RECONNECT_DELAY}s...")
                self.connected = False
                time.sleep(self.RECONNECT_DELAY)
                continue

            with self._lock:
                self.frame_buf.append(frame)

            self.frame_count += 1
            fps_frames += 1

            # Update FPS every second
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                self.fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.time()


class CameraManager:
    """
    Manages all camera streams. Provides unified frame access.
    """

    def __init__(self, camera_configs: List[dict]):
        self.streams: Dict[int, CameraStream] = {}
        for cfg in camera_configs:
            if not cfg.get("enabled", True):
                continue
            cam_cfg = CameraConfig(
                id=cfg["id"],
                name=cfg["name"],
                source=cfg["source"],
                enabled=cfg.get("enabled", True),
                zones=cfg.get("zones", []),
            )
            self.streams[cam_cfg.id] = CameraStream(cam_cfg)

    def start_all(self):
        for stream in self.streams.values():
            stream.start()
        time.sleep(0.5)   # brief warmup
        logger.info(f"[CameraManager] {len(self.streams)} camera(s) started.")

    def stop_all(self):
        for stream in self.streams.values():
            stream.stop()

    def get_frame(self, camera_id: int):
        """Return (success, frame) for a specific camera."""
        if camera_id not in self.streams:
            return False, None
        return self.streams[camera_id].read()

    def get_all_frames(self) -> Dict[int, any]:
        """Return dict of {camera_id: frame} for all healthy cameras."""
        frames = {}
        for cam_id, stream in self.streams.items():
            ret, frame = stream.read()
            if ret and frame is not None:
                frames[cam_id] = frame
        return frames

    def get_zones(self, camera_id: int) -> List[dict]:
        if camera_id in self.streams:
            return self.streams[camera_id].config.zones
        return []

    def get_fps(self, camera_id: int) -> float:
        if camera_id in self.streams:
            return self.streams[camera_id].fps
        return 0.0

    def status(self) -> List[dict]:
        return [
            {
                "id": cam_id,
                "name": s.config.name,
                "healthy": s.is_healthy(),
                "fps": round(s.fps, 1),
                "frames": s.frame_count,
            }
            for cam_id, s in self.streams.items()
        ]
