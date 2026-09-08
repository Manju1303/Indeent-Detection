"""
core/pipeline.py
Main processing pipeline that connects all components per camera.
Handles frame-skip, pose triggering, and result aggregation.
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Dict, List

from core.detector import DetectionEngine
from core.tracker import ByteTracker
from core.violence_detector import ViolenceDetector
from core.theft_detector import TheftDetector
from utils.drawing import FrameDrawer
from utils.fps_counter import FPSCounter

logger = logging.getLogger(__name__)


class PipelineResult:
    """All outputs from a single frame pass."""
    __slots__ = [
        "camera_id", "frame", "annotated_frame", "frame_number",
        "timestamp", "detections", "tracked", "violence_detected",
        "violence_score", "theft_events", "fps", "processing_ms",
    ]

    def __init__(self, camera_id: int):
        self.camera_id         = camera_id
        self.frame             = None
        self.annotated_frame   = None
        self.frame_number      = 0
        self.timestamp         = time.time()
        self.detections        = []
        self.tracked           = []
        self.violence_detected = False
        self.violence_score    = 0.0
        self.theft_events      = []
        self.fps               = 0.0
        self.processing_ms     = 0.0


class CameraPipeline:
    """
    Per-camera processing pipeline.
    Instantiate one per camera stream.
    """

    def __init__(
        self,
        camera_id: int,
        camera_name: str,
        detector: DetectionEngine,
        tracker_config: dict,
        violence_config: dict,
        theft_config: dict,
        zones: list,
        performance_config: dict,
    ):
        self.camera_id   = camera_id
        self.camera_name = camera_name

        self.detector  = detector
        self.tracker   = ByteTracker(tracker_config)
        self.violence  = ViolenceDetector(violence_config)
        self.theft     = TheftDetector(theft_config, zones)
        self.drawer    = FrameDrawer()
        self.fps_ctr   = FPSCounter()

        self.frame_skip    = max(1, performance_config.get("frame_skip", 2))
        self.max_fps       = performance_config.get("max_fps", 30)
        self._frame_count  = 0
        self._last_frame_t = 0.0

    def process(self, frame: np.ndarray) -> Optional[PipelineResult]:
        """
        Process one frame. Returns PipelineResult or None if frame skipped.
        """
        self._frame_count += 1

        # FPS cap
        now = time.time()
        if self.max_fps > 0:
            min_interval = 1.0 / self.max_fps
            if (now - self._last_frame_t) < min_interval:
                return None
        self._last_frame_t = now

        # Frame skip
        if self._frame_count % self.frame_skip != 0:
            return None

        t_start = time.perf_counter()
        result  = PipelineResult(self.camera_id)
        result.frame        = frame
        result.frame_number = self._frame_count

        # ── 1. Object Detection ───────────────────────────────
        detections = self.detector.detect(frame)
        result.detections = detections

        # ── 2. Tracking ───────────────────────────────────────
        tracked = self.tracker.update(detections)
        result.tracked = tracked

        # ── 3. Violence Detection (every frame for LSTM buffer) ──
        violence_detected, violence_score = self.violence.update(frame)
        result.violence_detected = violence_detected
        result.violence_score    = violence_score

        # ── 4. Theft Detection ────────────────────────────────
        persons = self.detector.get_persons(tracked)
        objects = self.detector.get_objects(tracked)
        weapons = self.detector.get_weapons(tracked)

        theft_events = self.theft.update(
            persons=persons,
            objects=objects,
            weapons=weapons,
            camera_id=self.camera_id,
            frame=frame,
        )
        result.theft_events = theft_events

        # ── 5. Draw Annotations ───────────────────────────────
        annotated = self.drawer.draw(
            frame=frame.copy(),
            detections=tracked,
            violence_score=violence_score,
            violence_detected=violence_detected,
            theft_events=theft_events,
            fps=self.fps_ctr.fps,
            camera_name=self.camera_name,
            zones=self.theft.zones,
        )
        result.annotated_frame = annotated

        # ── 6. Metrics ────────────────────────────────────────
        result.fps           = self.fps_ctr.update()
        result.processing_ms = (time.perf_counter() - t_start) * 1000

        return result


class MultiCameraPipeline:
    """
    Manages multiple CameraPipeline instances.
    Provides unified result collection.
    """

    def __init__(self, settings: dict):
        self.settings    = settings
        self.pipelines: Dict[int, CameraPipeline] = {}
        self._build_pipelines()

    def _build_pipelines(self):
        detector_config     = self.settings["detection"]
        tracker_config      = self.settings["tracking"]
        violence_config     = self.settings["violence"]
        theft_config        = self.settings["theft"]
        perf_config         = self.settings["performance"]
        camera_configs      = self.settings["cameras"]

        # Share one detector across all cameras (saves GPU memory)
        shared_detector = DetectionEngine({
            **detector_config,
            "use_fp16": perf_config.get("use_fp16", False),
        })

        for cam_cfg in camera_configs:
            if not cam_cfg.get("enabled", True):
                continue
            cam_id = cam_cfg["id"]
            self.pipelines[cam_id] = CameraPipeline(
                camera_id=cam_id,
                camera_name=cam_cfg.get("name", f"Camera {cam_id}"),
                detector=shared_detector,
                tracker_config=tracker_config,
                violence_config=violence_config,
                theft_config=theft_config,
                zones=cam_cfg.get("zones", []),
                performance_config=perf_config,
            )
            logger.info(f"[Pipeline] Camera {cam_id} '{cam_cfg['name']}' pipeline ready.")

    def process_frame(self, camera_id: int, frame: np.ndarray) -> Optional[PipelineResult]:
        if camera_id not in self.pipelines:
            return None
        return self.pipelines[camera_id].process(frame)

    def process_all(self, frames: Dict[int, np.ndarray]) -> Dict[int, PipelineResult]:
        results = {}
        for cam_id, frame in frames.items():
            result = self.process_frame(cam_id, frame)
            if result is not None:
                results[cam_id] = result
        return results
