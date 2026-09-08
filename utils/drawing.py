"""
utils/drawing.py
Visualization utilities: bounding boxes, human keypoints, body outline, predictive ghost skeleton (+300ms), and Intent HUD overlays.
"""

import cv2
import numpy as np
from typing import List, Optional
from datetime import datetime

from core.pose_estimator import HumanPose, SKELETON_CONNECTIONS
from core.intent_predictor import IntentPrediction

# Color palette (BGR)
COLORS = {
    "person":           (255, 180,  50),   # orange
    "ghost_skeleton":   (255, 240,   0),   # bright cyan / cyan-yellow (+300ms prediction)
    "outline":          (255, 200, 100),   # translucent cyan
    "weapon":           (  0,   0, 255),   # red
    "object":           ( 50, 200, 255),   # yellow
    "violence":         (  0,   0, 220),   # dark red
    "theft_low":        (  0, 165, 255),   # amber
    "theft_medium":     (  0,  80, 255),   # orange-red
    "theft_high":       (  0,   0, 255),   # red
    "zone_safe":        (  0, 200,  80),   # green
    "zone_alert":       (  0,   0, 200),   # red
    "hud_bg":           ( 20,  20,  20),   # dark gray
    "hud_text":         (255, 255, 255),   # white
    "ok":               ( 80, 200,  80),   # green
    "warn":             (  0, 165, 255),   # amber
    "alert":            (  0,   0, 255),   # red
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


class FrameDrawer:
    """Draws all visual overlays, body outlines, skeletons, and +300ms predictive motion vectors on frames."""

    def draw(
        self,
        frame: np.ndarray,
        detections: list,
        poses: Optional[List[HumanPose]] = None,
        predictions: Optional[List[IntentPrediction]] = None,
        violence_score: float = 0.0,
        violence_detected: bool = False,
        theft_events: list = None,
        fps: float = 0.0,
        camera_name: str = "",
        zones: list = None,
    ) -> np.ndarray:
        """
        Full annotation pipeline with intent & movement prediction overlays.
        """
        if zones:
            self._draw_zones(frame, zones)

        for det in detections:
            self._draw_detection(frame, det)

        if poses:
            pred_map = {p.track_id: p for p in (predictions or []) if p.track_id is not None}
            for pose in poses:
                pred = pred_map.get(pose.track_id)
                self._draw_pose_and_intent(frame, pose, pred)

        self._draw_hud(frame, fps, camera_name, violence_score, violence_detected, theft_events or [], predictions or [])

        if violence_detected:
            self._draw_violence_overlay(frame)

        if theft_events:
            self._draw_theft_overlay(frame, theft_events)

        return frame

    # ── Drawing helpers ───────────────────────────────────────

    def _draw_detection(self, frame, det):
        x1, y1, x2, y2 = det.bbox
        label = det.class_name

        if det.is_weapon:
            color = COLORS["weapon"]
        elif label == "person":
            color = COLORS["person"]
        else:
            color = COLORS["object"]

        thickness = 3 if det.is_weapon else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        track_str = f"#{det.track_id}" if det.track_id else ""
        text      = f"{label} {track_str} {det.confidence:.0%}"

        (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if det.is_weapon:
            cx, cy = det.center
            cv2.putText(frame, "[WEAPON]", (cx - 40, cy - 10),
                        FONT_BOLD, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    def _draw_pose_and_intent(self, frame: np.ndarray, pose: HumanPose, pred: Optional[IntentPrediction]):
        """Draw Body Outline, Skeleton Keypoints, Motion Trajectory Arrow, and +300ms Ghost Skeleton."""
        # 1. Body Outline Polygon (Convex Hull)
        if pose.body_outline and len(pose.body_outline) >= 3:
            pts_arr = np.array(pose.body_outline, dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_arr], COLORS["outline"])
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [pts_arr], True, COLORS["person"], 1, cv2.LINE_AA)

        # 2. Current Skeleton Bones
        kps = pose.keypoints
        for j1, j2 in SKELETON_CONNECTIONS:
            if j1 in kps and j2 in kps:
                p1, p2 = kps[j1], kps[j2]
                cv2.line(frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw Keypoint Joint Circles
        for kp_name, (kx, ky) in kps.items():
            cv2.circle(frame, (kx, ky), 4, (0, 165, 255), -1, cv2.LINE_AA)

        # 3. Motion Trajectory Vector & +300ms Ghost Skeleton
        if pred:
            cx, cy = pose.centroid
            fcx, fcy = pred.future_centroid_300ms

            # Trajectory Line & Arrow
            cv2.arrowedLine(frame, (cx, cy), (fcx, fcy), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(frame, f"+300ms", (fcx + 5, fcy), FONT, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # +300ms Predictive Ghost Skeleton (Dashed / Bright Yellow)
            fkps = pred.future_keypoints_300ms
            if fkps:
                for j1, j2 in SKELETON_CONNECTIONS:
                    if j1 in fkps and j2 in fkps:
                        fp1, fp2 = fkps[j1], fkps[j2]
                        cv2.line(frame, fp1, fp2, COLORS["ghost_skeleton"], 1, cv2.LINE_AA)
                for fkp_name, (fkx, fky) in fkps.items():
                    cv2.circle(frame, (fkx, fky), 3, COLORS["ghost_skeleton"], -1, cv2.LINE_AA)

            # Pre-Action Intent Card Badge over head
            x1, y1, x2, y2 = pose.bbox
            badge_text = f"Intent (<300ms): {pred.predicted_intent} ({pred.intent_confidence:.0%})"
            (bw, bh), _ = cv2.getTextSize(badge_text, FONT_BOLD, 0.5, 1)
            bx1 = max(5, cx - bw // 2)
            by1 = max(40, y1 - 25)

            card_color = COLORS["alert"] if pred.action_risk_level == "HIGH_RISK" else COLORS["warn"] if pred.action_risk_level == "WARNING" else COLORS["hud_bg"]
            cv2.rectangle(frame, (bx1 - 4, by1 - bh - 6), (bx1 + bw + 4, by1 + 4), card_color, -1)
            cv2.rectangle(frame, (bx1 - 4, by1 - bh - 6), (bx1 + bw + 4, by1 + 4), (0, 255, 255), 1)
            cv2.putText(frame, badge_text, (bx1, by1), FONT_BOLD, 0.5, COLORS["hud_text"], 1, cv2.LINE_AA)

    def _draw_zones(self, frame, zones: list):
        for zone in zones:
            pts = zone.get("points", [])
            if not pts:
                continue
            pts_arr = np.array(pts, dtype=np.int32)
            sensitive = zone.get("theft_sensitive", False)
            color = COLORS["zone_alert"] if sensitive else COLORS["zone_safe"]

            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_arr], color)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            cv2.polylines(frame, [pts_arr], True, color, 2)
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            cv2.putText(frame, zone["name"], (cx - 30, cy),
                        FONT, 0.55, color, 1, cv2.LINE_AA)

    def _draw_hud(self, frame, fps, camera_name, violence_score,
                  violence_detected, theft_events, predictions: List[IntentPrediction]):
        h, w = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)

        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"[CAM] {camera_name} | PRE-ACTION AI (<300ms)", (8, 24),
                    FONT_BOLD, 0.55, COLORS["hud_text"], 1, cv2.LINE_AA)
        cv2.putText(frame, ts, (w - 80, 24),
                    FONT, 0.55, COLORS["hud_text"], 1, cv2.LINE_AA)

        fps_color = COLORS["ok"] if fps >= 15 else COLORS["warn"] if fps >= 8 else COLORS["alert"]
        cv2.putText(frame, f"FPS:{fps:.0f}", (w - 160, 24),
                    FONT, 0.55, fps_color, 1, cv2.LINE_AA)

        # Bottom status bar
        cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)

        if predictions:
            latest_pred = predictions[-1]
            intent_str = f"PREACTION: {latest_pred.predicted_intent.upper()}"
            cv2.putText(frame, intent_str, (10, h - 12),
                        FONT_BOLD, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        vbar_w  = int(100 * violence_score)
        vbar_color = COLORS["alert"] if violence_detected else COLORS["warn"] if violence_score > 0.4 else COLORS["ok"]
        cv2.rectangle(frame, (w - 180, h - 28), (w - 80, h - 8), (60, 60, 60), -1)
        cv2.rectangle(frame, (w - 180, h - 28), (w - 180 + vbar_w, h - 8), vbar_color, -1)
        cv2.putText(frame, f"Threat:{violence_score:.0%}", (w - 175, h - 12),
                    FONT, 0.42, COLORS["hud_text"], 1, cv2.LINE_AA)

    def _draw_violence_overlay(self, frame):
        h, w = frame.shape[:2]
        border = 6
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 220), border)
        cv2.putText(frame, "[ALERT] HIGH THREAT DETECTED", (w // 2 - 140, 70),
                    FONT_BOLD, 0.9, (0, 0, 220), 2, cv2.LINE_AA)

    def _draw_theft_overlay(self, frame, events):
        h, w = frame.shape[:2]
        max_evt = max(events, key=lambda e: e.score)
        if max_evt.level == "high":
            color = COLORS["alert"]
            cv2.rectangle(frame, (0, 0), (w, h), color, 4)
            cv2.putText(frame, "[ALERT] INTENT THREAT DETECTED", (w // 2 - 130, 100),
                        FONT_BOLD, 0.9, color, 2, cv2.LINE_AA)
