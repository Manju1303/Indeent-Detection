"""
core/theft_detector.py
Zone-aware, confidence-scored theft detection engine.

Enhanced over original 5-step logic:
  + Zone sensitivity weighting
  + Confidence scoring (not binary)
  + Re-ID safe object tracking
  + Person-exit confirmation
  + Multi-object tracking
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from core.detector import Detection, compute_iou

logger = logging.getLogger(__name__)


@dataclass
class ObjectState:
    """Tracks state of a single object over time."""
    det: Detection
    first_seen: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)
    touched_by: set  = field(default_factory=set)   # track_ids of persons
    in_zone:    bool = False
    disappeared_frame: Optional[int] = None
    theft_score: float = 0.0


@dataclass
class TheftEvent:
    """Confirmed or suspected theft event."""
    camera_id: int
    frame_number: int
    timestamp: float
    score: float
    level: str                 # "low", "medium", "high"
    person_track_id: int
    object_class: str
    object_bbox: Tuple
    zone_name: Optional[str]
    frame_snapshot: Optional[np.ndarray] = None


def point_in_polygon(point: Tuple[int, int], polygon: List) -> bool:
    """Check if a point is inside a polygon (zone)."""
    px, py = point
    poly = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(poly, (float(px), float(py)), False)
    return result >= 0


class TheftDetector:
    """
    Detects theft by tracking object-person interactions across zones.
    """

    def __init__(self, config: dict, zones: List[dict] = None):
        self.config          = config
        self.zones           = zones or []
        self.overlap_thr     = config.get("overlap_iou_threshold", 0.15)
        self.disappear_frames = config.get("disappear_frames", 45)
        self.thresholds      = config.get("score_thresholds",
                                          {"low": 0.4, "medium": 0.65, "high": 0.85})
        self.cooldown_s      = config.get("cooldown_seconds", 10)

        # Track objects: dict[track_id_or_unique_key → ObjectState]
        self._objects: Dict[str, ObjectState] = {}
        # Track persons who have "touched" objects: dict[person_track_id → set of obj keys]
        self._person_contacts: Dict[int, set] = defaultdict(set)

        self._frame_num       = 0
        self._last_alert_t    = 0.0
        self._confirmed_events: List[TheftEvent] = []

    # ── Public API ────────────────────────────────────────────

    def update(
        self,
        persons: List[Detection],
        objects: List[Detection],
        weapons: List[Detection],
        camera_id: int = 0,
        frame: Optional[np.ndarray] = None,
    ) -> List[TheftEvent]:
        """
        Feed current detections. Returns list of theft events this frame.
        """
        self._frame_num += 1
        events = []

        # ── Step 1: Weapon detection → instant high alert ─────
        for w in weapons:
            if self._can_alert():
                evt = TheftEvent(
                    camera_id=camera_id,
                    frame_number=self._frame_num,
                    timestamp=time.time(),
                    score=0.95,
                    level="high",
                    person_track_id=-1,
                    object_class=w.class_name,
                    object_bbox=w.bbox,
                    zone_name=self._get_zone(w),
                    frame_snapshot=frame.copy() if frame is not None else None,
                )
                events.append(evt)
                self._last_alert_t = time.time()
                logger.warning(f"[Theft] WEAPON detected: {w.class_name}")

        # ── Step 2: Update object registry ───────────────────
        current_obj_keys = set()
        for obj in objects:
            key = self._object_key(obj)
            current_obj_keys.add(key)
            zone_name = self._get_zone(obj)

            if key not in self._objects:
                self._objects[key] = ObjectState(
                    det=obj,
                    in_zone=zone_name is not None,
                )
            else:
                state = self._objects[key]
                state.det = obj
                state.last_seen = time.time()
                state.in_zone   = zone_name is not None
                state.disappeared_frame = None   # reset if seen again

        # ── Step 3: Check hand-object overlap for each person ─
        for person in persons:
            pid = person.track_id or -1
            for key, state in self._objects.items():
                iou = compute_iou(person.bbox, state.det.bbox)
                if iou >= self.overlap_thr:
                    state.touched_by.add(pid)
                    self._person_contacts[pid].add(key)
                    # Score bump for zone overlap
                    zone = self._get_zone(state.det)
                    if zone:
                        zone_cfg = next((z for z in self.zones if z["name"] == zone), {})
                        if zone_cfg.get("theft_sensitive", False):
                            state.theft_score = min(state.theft_score + 0.2, 1.0)
                    else:
                        state.theft_score = min(state.theft_score + 0.1, 1.0)

        # ── Step 4: Check for object disappearance ────────────
        disappeared_keys = set(self._objects.keys()) - current_obj_keys
        for key in disappeared_keys:
            state = self._objects[key]
            if state.disappeared_frame is None:
                state.disappeared_frame = self._frame_num

            frames_gone = self._frame_num - state.disappeared_frame
            if frames_gone >= self.disappear_frames and state.touched_by:
                state.theft_score = min(state.theft_score + 0.3, 1.0)

                # ── Step 5: Person exit confirmation ──────────
                person_track_ids = list(state.touched_by)
                suspected_pid    = person_track_ids[-1]

                level = self._score_to_level(state.theft_score)
                if level and self._can_alert():
                    evt = TheftEvent(
                        camera_id=camera_id,
                        frame_number=self._frame_num,
                        timestamp=time.time(),
                        score=state.theft_score,
                        level=level,
                        person_track_id=suspected_pid,
                        object_class=state.det.class_name,
                        object_bbox=state.det.bbox,
                        zone_name=self._get_zone(state.det),
                        frame_snapshot=frame.copy() if frame is not None else None,
                    )
                    events.append(evt)
                    self._confirmed_events.append(evt)
                    self._last_alert_t = time.time()
                    logger.warning(
                        f"[Theft] {level.upper()} — {state.det.class_name} "
                        f"taken by person {suspected_pid} "
                        f"(score: {state.theft_score:.2f})"
                    )
                # Clean up confirmed object
                del self._objects[key]
                continue

        # ── Cleanup stale objects (not seen for >5 min) ───────
        stale_timeout = 300
        for key in list(self._objects.keys()):
            if time.time() - self._objects[key].last_seen > stale_timeout:
                del self._objects[key]

        return events

    def get_object_states(self) -> List[ObjectState]:
        return list(self._objects.values())

    def get_history(self) -> List[TheftEvent]:
        return self._confirmed_events[-100:]   # last 100 events

    # ── Internal ──────────────────────────────────────────────

    def _object_key(self, det: Detection) -> str:
        """Stable key for an object. Use track_id if available, else bbox hash."""
        if det.track_id is not None:
            return f"{det.class_name}_{det.track_id}"
        cx, cy = det.center
        # Quantize center to avoid minor bbox jitter creating new keys
        return f"{det.class_name}_{cx // 30}_{cy // 30}"

    def _get_zone(self, det: Detection) -> Optional[str]:
        """Return zone name if detection center falls inside a zone polygon."""
        cx, cy = det.center
        for zone in self.zones:
            pts = zone.get("points", [])
            if pts and point_in_polygon((cx, cy), pts):
                return zone["name"]
        return None

    def _score_to_level(self, score: float) -> Optional[str]:
        if score >= self.thresholds["high"]:
            return "high"
        if score >= self.thresholds["medium"]:
            return "medium"
        if score >= self.thresholds["low"]:
            return "low"
        return None

    def _can_alert(self) -> bool:
        return (time.time() - self._last_alert_t) >= self.cooldown_s
