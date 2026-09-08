"""
core/tracker.py
ByteTrack-inspired multi-object tracker with Re-ID grace period.

The Re-ID grace period solves a critical problem from the original design:
when a person briefly disappears behind an object and reappears, ByteTrack
alone would assign a new track_id — breaking theft disappearance logic.
The grace buffer holds lost tracks for N frames before truly dropping them.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

from core.detector import Detection, compute_iou

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a tracked object across frames."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    hits: int = 1
    missed: int = 0
    is_confirmed: bool = False
    history: List[Tuple[int, int]] = field(default_factory=list)  # center points

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def predict(self):
        """Simple linear prediction based on last two centers."""
        if len(self.history) >= 2:
            dx = self.history[-1][0] - self.history[-2][0]
            dy = self.history[-1][1] - self.history[-2][1]
            cx, cy = self.history[-1]
            x1, y1, x2, y2 = self.bbox
            w, h = x2 - x1, y2 - y1
            # Shift bbox by velocity
            nx = cx + dx - w // 2
            ny = cy + dy - h // 2
            self.bbox = (nx, ny, nx + w, ny + h)


class ByteTracker:
    """
    Lightweight ByteTrack-inspired tracker with:
    - IoU-based Hungarian matching
    - Re-ID grace period (hold lost tracks N frames)
    - Track confirmation after min_hits
    """

    def __init__(self, config: dict):
        self.reid_grace   = config.get("reid_grace_frames", 30)
        self.max_miss     = config.get("max_disappeared", 50)
        self.min_hits     = config.get("min_hits", 3)
        self.iou_thr      = config.get("iou_threshold", 0.3)

        self._tracks:     Dict[int, Track] = {}
        self._lost:       Dict[int, Track] = {}   # Re-ID grace buffer
        self._next_id:    int = 1

    # ── Public API ────────────────────────────────────────────

    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Match detections to existing tracks.
        Returns detections with track_id filled in.
        """
        if not detections:
            self._age_all()
            return []

        # 1. Predict new positions for existing tracks
        for track in self._tracks.values():
            track.predict()

        # 2. Build cost matrix (IoU between tracks and detections)
        track_ids   = list(self._tracks.keys())
        track_boxes = [self._tracks[tid].bbox for tid in track_ids]
        det_boxes   = [d.bbox for d in detections]

        if track_boxes and det_boxes:
            cost_matrix = self._iou_cost(track_boxes, det_boxes)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_dets   = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1 - self.iou_thr):
                    tid = track_ids[r]
                    self._update_track(tid, detections[c])
                    detections[c].track_id = tid
                    matched_tracks.add(tid)
                    matched_dets.add(c)

            # 3. Unmatched detections → try Re-ID grace buffer first
            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue
                reid_tid = self._try_reid(det)
                if reid_tid is not None:
                    # Re-matched from grace buffer
                    self._tracks[reid_tid] = self._lost.pop(reid_tid)
                    self._update_track(reid_tid, det)
                    det.track_id = reid_tid
                else:
                    # Truly new track
                    tid = self._new_track(det)
                    det.track_id = tid

            # 4. Unmatched tracks → move to grace buffer or drop
            for tid in track_ids:
                if tid not in matched_tracks:
                    self._tracks[tid].missed += 1
                    if self._tracks[tid].missed > self.reid_grace:
                        self._lost[tid] = self._tracks.pop(tid)
        else:
            # No existing tracks — create new for all detections
            for det in detections:
                tid = self._new_track(det)
                det.track_id = tid

        # 5. Clean up deeply lost tracks
        self._cleanup_lost()

        # 6. Return only confirmed tracks
        return [d for d in detections if d.track_id is not None and
                self._tracks.get(d.track_id, Track(0, (0,0,0,0), "", 0)).is_confirmed]

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id) or self._lost.get(track_id)

    def get_all_tracks(self) -> List[Track]:
        return list(self._tracks.values())

    # ── Internal ──────────────────────────────────────────────

    def _new_track(self, det: Detection) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = Track(
            track_id=tid,
            bbox=det.bbox,
            class_name=det.class_name,
            confidence=det.confidence,
        )
        return tid

    def _update_track(self, tid: int, det: Detection):
        t = self._tracks[tid]
        t.bbox       = det.bbox
        t.confidence = det.confidence
        t.missed     = 0
        t.hits      += 1
        t.history.append(t.center)
        if len(t.history) > 30:
            t.history.pop(0)
        if t.hits >= self.min_hits:
            t.is_confirmed = True

    def _try_reid(self, det: Detection) -> Optional[int]:
        """Check if detection matches any track in grace buffer (lost tracks)."""
        best_tid  = None
        best_iou  = self.iou_thr
        for tid, lost_track in self._lost.items():
            if lost_track.class_name != det.class_name:
                continue
            iou = compute_iou(lost_track.bbox, det.bbox)
            if iou > best_iou:
                best_iou = iou
                best_tid = tid
        return best_tid

    def _age_all(self):
        for tid in list(self._tracks.keys()):
            self._tracks[tid].missed += 1
            if self._tracks[tid].missed > self.reid_grace:
                self._lost[tid] = self._tracks.pop(tid)
        self._cleanup_lost()

    def _cleanup_lost(self):
        for tid in list(self._lost.keys()):
            if self._lost[tid].missed > self.max_miss:
                del self._lost[tid]
                logger.debug(f"[Tracker] Track {tid} permanently dropped.")

    def _iou_cost(self, boxes_a: list, boxes_b: list) -> np.ndarray:
        """Build NxM cost matrix (1 - IoU)."""
        n, m = len(boxes_a), len(boxes_b)
        cost = np.ones((n, m), dtype=np.float32)
        for i, ba in enumerate(boxes_a):
            for j, bb in enumerate(boxes_b):
                cost[i, j] = 1.0 - compute_iou(ba, bb)
        return cost
