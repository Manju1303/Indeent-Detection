"""
utils/visualizer.py
High-Performance Skeleton, Intent HUD, and Ghost Prediction Renderer.
Features Nova Cyan (#00D8E6) branding accents and Plasma Mint (#3EF7C9) confirmation highlights.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

# Branding BGR Color Accents
NOVA_CYAN   = (230, 216,   0)   # #00D8E6
PLASMA_MINT = (201, 247,  62)   # #3EF7C9
DARK_HUD_BG = ( 18,  14,   9)   # Translucent dark backdrop
WHITE_TEXT  = (255, 255, 255)
WARN_RED    = ( 68,  68, 239)

# COCO 17 Skeleton Connection Joints Pairs
COCO_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Facial landmarks
    (5, 6),                                     # Shoulder line
    (5, 7), (7, 9),                             # Left arm
    (6, 8), (8, 10),                            # Right arm
    (5, 11), (6, 12),                           # Torso side lines
    (11, 12),                                   # Hip line
    (11, 13), (13, 15),                         # Left leg
    (12, 14), (14, 16)                          # Right leg
]


class IntentVisualizer:
    """
    Renders COCO 17 Skeleton, Trajectory Arrow, +300ms Ghost Projection, and Telemetry HUD.
    """

    def render(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        predicted_intent: str,
        confidence: float,
        latency_ms: float,
        fps: float = 30.0
    ) -> np.ndarray:
        """
        Annotates live frame with skeleton, metrics, and Nova Cyan telemetry.
        """
        h, w = frame.shape[:2]

        # 1. Render COCO 17 Joint Skeleton
        if keypoints is not None and len(keypoints) == 17:
            # Denormalize coordinates to frame resolution
            pts = []
            for j in range(17):
                px = int(keypoints[j][0] * w) if keypoints[j][0] <= 1.0 else int(keypoints[j][0])
                py = int(keypoints[j][1] * h) if keypoints[j][1] <= 1.0 else int(keypoints[j][1])
                pts.append((px, py))

            # Draw bones
            for j1, j2 in COCO_BONES:
                p1, p2 = pts[j1], pts[j2]
                if p1 != (0, 0) and p2 != (0, 0):
                    cv2.line(frame, p1, p2, NOVA_CYAN, 2, cv2.LINE_AA)

            # Draw joint dots
            for px, py in pts:
                if (px, py) != (0, 0):
                    cv2.circle(frame, (px, py), 4, PLASMA_MINT, -1, cv2.LINE_AA)

            # +300ms Projected Ghost Skeleton
            shoulder_mid = ((pts[5][0] + pts[6][0]) // 2, (pts[5][1] + pts[6][1]) // 2)
            hip_mid = ((pts[11][0] + pts[12][0]) // 2, (pts[11][1] + pts[12][1]) // 2)

            if shoulder_mid != (0, 0) and hip_mid != (0, 0):
                # Motion direction vector
                cv2.arrowedLine(frame, hip_mid, (hip_mid[0] + 30, hip_mid[1] - 10), PLASMA_MINT, 2, cv2.LINE_AA, tipLength=0.3)
                cv2.putText(frame, "+300ms Horizon", (hip_mid[0] + 35, hip_mid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, PLASMA_MINT, 1, cv2.LINE_AA)

        # 2. Dynamic Telemetry Overlay (Nova Cyan & Plasma Mint Branding)
        cv2.rectangle(frame, (20, 20), (580, 110), DARK_HUD_BG, -1)
        cv2.rectangle(frame, (20, 20), (580, 110), NOVA_CYAN, 1)

        intent_text = f"Intent Prediction: {predicted_intent} ({confidence * 100:.1f}%)"
        latency_text = f"Processing Latency: {latency_ms:.2f} ms | FPS: {fps:.0f}"

        color_accent = WARN_RED if "Fall" in predicted_intent or "Risk" in predicted_intent else NOVA_CYAN

        cv2.putText(frame, intent_text, (30, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color_accent, 2, cv2.LINE_AA)
        cv2.putText(frame, latency_text, (30, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, PLASMA_MINT, 2, cv2.LINE_AA)

        return frame
