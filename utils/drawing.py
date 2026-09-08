"""
utils/drawing.py
Visualization utilities: bounding boxes, zones, HUD overlays, threat indicators.
"""

import cv2
import numpy as np
from typing import List, Optional
from datetime import datetime


# Color palette (BGR)
COLORS = {
    "person":   (255, 180,  50),   # orange
    "weapon":   (  0,   0, 255),   # red
    "object":   ( 50, 200, 255),   # yellow
    "violence": (  0,   0, 220),   # dark red
    "theft_low":    (  0, 165, 255),   # amber
    "theft_medium": (  0,  80, 255),   # orange-red
    "theft_high":   (  0,   0, 255),   # red
    "zone_safe":    (  0, 200,  80),   # green
    "zone_alert":   (  0,   0, 200),   # red
    "hud_bg":   (  0,   0,   0),   # black
    "hud_text": (255, 255, 255),   # white
    "ok":       ( 80, 200,  80),   # green
    "warn":     (  0, 165, 255),   # amber
    "alert":    (  0,   0, 255),   # red
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


class FrameDrawer:
    """Draws all visual overlays on frames."""

    def draw(
        self,
        frame: np.ndarray,
        detections: list,
        violence_score: float = 0.0,
        violence_detected: bool = False,
        theft_events: list = None,
        fps: float = 0.0,
        camera_name: str = "",
        zones: list = None,
    ) -> np.ndarray:
        """
        Full annotation pipeline. Returns annotated frame.
        """
        if zones:
            self._draw_zones(frame, zones)

        for det in detections:
            self._draw_detection(frame, det)

        self._draw_hud(frame, fps, camera_name, violence_score, violence_detected, theft_events or [])

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

        # Bounding box
        thickness = 3 if det.is_weapon else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Track ID
        track_str = f"#{det.track_id}" if det.track_id else ""
        text      = f"{label} {track_str} {det.confidence:.0%}"

        # Label background
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Weapon: big warning icon
        if det.is_weapon:
            cx, cy = det.center
            cv2.putText(frame, "[WEAPON]", (cx - 40, cy - 10),
                        FONT_BOLD, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    def _draw_zones(self, frame, zones: list):
        for zone in zones:
            pts = zone.get("points", [])
            if not pts:
                continue
            pts_arr = np.array(pts, dtype=np.int32)
            sensitive = zone.get("theft_sensitive", False)
            color = COLORS["zone_alert"] if sensitive else COLORS["zone_safe"]

            # Semi-transparent fill
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts_arr], color)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Border
            cv2.polylines(frame, [pts_arr], True, color, 2)

            # Zone name
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            cv2.putText(frame, zone["name"], (cx - 30, cy),
                        FONT, 0.55, color, 1, cv2.LINE_AA)

    def _draw_hud(self, frame, fps, camera_name, violence_score,
                  violence_detected, theft_events):
        h, w = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)

        # Camera name + timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"[CAM] {camera_name}", (8, 24),
                    FONT_BOLD, 0.6, COLORS["hud_text"], 1, cv2.LINE_AA)
        cv2.putText(frame, ts, (w - 80, 24),
                    FONT, 0.55, COLORS["hud_text"], 1, cv2.LINE_AA)

        # FPS
        fps_color = COLORS["ok"] if fps >= 15 else COLORS["warn"] if fps >= 8 else COLORS["alert"]
        cv2.putText(frame, f"FPS:{fps:.0f}", (w - 160, 24),
                    FONT, 0.55, fps_color, 1, cv2.LINE_AA)

        # Bottom status bar
        cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), -1)

        # Violence meter
        vbar_w  = int(120 * violence_score)
        vbar_color = COLORS["alert"] if violence_detected else COLORS["warn"] if violence_score > 0.4 else COLORS["ok"]
        cv2.rectangle(frame, (5, h - 28), (125, h - 8), (60, 60, 60), -1)
        cv2.rectangle(frame, (5, h - 28), (5 + vbar_w, h - 8), vbar_color, -1)
        cv2.putText(frame, f"Violence:{violence_score:.0%}", (8, h - 12),
                    FONT, 0.42, COLORS["hud_text"], 1, cv2.LINE_AA)

        # Theft indicator
        if theft_events:
            max_event = max(theft_events, key=lambda e: e.score)
            level_color = {
                "low": COLORS["warn"],
                "medium": (0, 80, 255),
                "high": COLORS["alert"],
            }.get(max_event.level, COLORS["warn"])
            cv2.putText(frame, f"THEFT: {max_event.level.upper()}",
                        (140, h - 12), FONT_BOLD, 0.5, level_color, 1, cv2.LINE_AA)

    def _draw_violence_overlay(self, frame):
        """Flash red border when violence detected."""
        h, w = frame.shape[:2]
        border = 6
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 220), border)
        cv2.putText(frame, "[ALERT] VIOLENCE DETECTED", (w // 2 - 140, 70),
                    FONT_BOLD, 0.9, (0, 0, 220), 2, cv2.LINE_AA)

    def _draw_theft_overlay(self, frame, events):
        h, w = frame.shape[:2]
        max_evt = max(events, key=lambda e: e.score)
        if max_evt.level == "high":
            color = COLORS["alert"]
            cv2.rectangle(frame, (0, 0), (w, h), color, 4)
            cv2.putText(frame, "[ALERT] THEFT DETECTED", (w // 2 - 130, 100),
                        FONT_BOLD, 0.9, color, 2, cv2.LINE_AA)
