"""
core/intent_predictor.py
Sub-300ms Human Intent & Pre-Action Movement Predictor.
Uses Human Torso Dynamics (Torso Center Kinematics, Spine Pitch/Roll Velocity) as the PRIMARY FEATURE.

Features:
  - Sub-300ms Torso Trajectory & Orientation Extrapolation (dt = 0.300s)
  - Torso Angular Pitch/Roll & Translational Velocity Vector Engine
  - Predictive "+300ms Projected Torso & Ghost Skeleton"
  - Pre-Action Intent Classifier ("About to Reach", "About to Lunge", "About to Bend Torso", "About to Twist/Turn", "About to Rise")
  - Ultra-fast real-time performance (< 1ms per frame)
"""

import time
import math
import numpy as np
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from core.pose_estimator import HumanPose, TorsoState

logger = logging.getLogger(__name__)


@dataclass
class IntentPrediction:
    """Pre-action prediction result driven by Human Torso Dynamics."""
    track_id: Optional[int]
    current_posture: str
    predicted_intent: str                   # E.g. "About to Reach Out", "About to Lunge", "About to Bend Torso", "Stable"
    intent_confidence: float
    prediction_window_ms: int = 300
    torso_velocity: Tuple[float, float] = (0.0, 0.0)         # (vx, vy) px/sec of Torso Center
    torso_pitch_velocity_deg_s: float = 0.0                  # Torso forward/backward pitch rate (deg/sec)
    torso_roll_velocity_deg_s: float = 0.0                   # Torso side tilt roll rate (deg/sec)
    future_torso_center_300ms: Tuple[int, int] = (0, 0)
    future_torso_polygon_300ms: List[Tuple[int, int]] = field(default_factory=list)
    future_keypoints_300ms: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    trajectory_vector: Tuple[int, int] = (0, 0)
    action_risk_level: str = "NORMAL"                       # "NORMAL", "CAUTION", "WARNING", "HIGH_RISK"
    latency_ms: float = 0.0


class IntentPredictor:
    """
    Sub-300ms Human Movement & Pre-Action Predictor powered by Torso Kinematics.
    """

    PREDICTION_WINDOW_S = 0.300    # 300 milliseconds
    MAX_HISTORY_LEN     = 15       # Rolling frame history

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.prediction_window_ms = self.config.get("prediction_window_ms", 300)
        self.window_s = self.prediction_window_ms / 1000.0
        self._history: Dict[int, deque] = {}

    # ── Public API ────────────────────────────────────────────

    def predict_intent(self, pose: HumanPose, timestamp: Optional[float] = None) -> IntentPrediction:
        """
        Analyze Torso Dynamics & Keypoint sequence to predict next movement (<300ms).
        """
        t_start = time.perf_counter()
        now = timestamp or time.time()
        tid = pose.track_id or 0

        if tid not in self._history:
            self._history[tid] = deque(maxlen=self.MAX_HISTORY_LEN)

        self._history[tid].append((now, pose))
        history = self._history[tid]

        # 1. Compute Torso Core Kinematics (Primary Feature)
        vx, vy, ax, ay, pitch_rate, roll_rate = self._compute_torso_kinematics(history)

        # 2. Extrapolate Torso Center & Torso Polygon at t + 300ms
        tcx, tcy = pose.torso.center
        future_tcx = int(tcx + vx * self.window_s + 0.5 * ax * (self.window_s ** 2))
        future_tcy = int(tcy + vy * self.window_s + 0.5 * ay * (self.window_s ** 2))

        future_torso_poly = [
            (int(pt[0] + vx * self.window_s), int(pt[1] + vy * self.window_s))
            for pt in pose.torso.polygon
        ]

        # 3. Extrapolate Ghost Keypoints at t + 300ms
        future_kps = self._predict_future_keypoints(history, self.window_s)

        # 4. Classify Pre-Action Intent based on Torso Motion + Limbs
        intent_label, conf, risk = self._classify_torso_intent(pose, vx, vy, ax, ay, pitch_rate, roll_rate, future_kps)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return IntentPrediction(
            track_id=pose.track_id,
            current_posture=pose.posture,
            predicted_intent=intent_label,
            intent_confidence=conf,
            prediction_window_ms=self.prediction_window_ms,
            torso_velocity=(round(vx, 1), round(vy, 1)),
            torso_pitch_velocity_deg_s=round(pitch_rate, 1),
            torso_roll_velocity_deg_s=round(roll_rate, 1),
            future_torso_center_300ms=(future_tcx, future_tcy),
            future_torso_polygon_300ms=future_torso_poly,
            future_keypoints_300ms=future_kps,
            trajectory_vector=(future_tcx - tcx, future_tcy - tcy),
            action_risk_level=risk,
            latency_ms=round(elapsed_ms, 2),
        )

    # ── Torso Kinematic Computations ──────────────────────────

    def _compute_torso_kinematics(self, history: deque) -> Tuple[float, float, float, float, float, float]:
        """Compute Torso Center Translation Velocity, Acceleration, and Torso Pitch/Roll Angular Velocities."""
        if len(history) < 2:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        t_curr, pose_curr = history[-1]
        t_prev, pose_prev = history[-2]
        dt1 = max(0.001, t_curr - t_prev)

        # Torso Center translation
        tc_curr = pose_curr.torso.center
        tc_prev = pose_prev.torso.center

        vx = (tc_curr[0] - tc_prev[0]) / dt1
        vy = (tc_curr[1] - tc_prev[1]) / dt1

        # Torso Pitch & Roll Angular Velocity (deg/sec)
        pitch_rate = (pose_curr.torso.pitch_angle_deg - pose_prev.torso.pitch_angle_deg) / dt1
        roll_rate = (pose_curr.torso.roll_angle_deg - pose_prev.torso.roll_angle_deg) / dt1

        if len(history) >= 3:
            t_prev2, pose_prev2 = history[-3]
            dt2 = max(0.001, t_prev - t_prev2)
            tc_prev2 = pose_prev2.torso.center

            vx_prev = (tc_prev[0] - tc_prev2[0]) / dt2
            vy_prev = (tc_prev[1] - tc_prev2[1]) / dt2

            ax = (vx - vx_prev) / dt1
            ay = (vy - vy_prev) / dt1
        else:
            ax, ay = 0.0, 0.0

        return vx, vy, ax, ay, pitch_rate, roll_rate

    def _predict_future_keypoints(self, history: deque, dt: float) -> Dict[str, Tuple[int, int]]:
        if len(history) < 2:
            return {k: v for k, v in history[-1][1].keypoints.items()}

        t_curr, pose_curr = history[-1]
        t_prev, pose_prev = history[-2]
        dt_hist = max(0.001, t_curr - t_prev)

        future_kps = {}
        curr_kps = pose_curr.keypoints
        prev_kps = pose_prev.keypoints

        for name, p_curr in curr_kps.items():
            if name in prev_kps:
                p_prev = prev_kps[name]
                kvx = (p_curr[0] - p_prev[0]) / dt_hist
                kvy = (p_curr[1] - p_prev[1]) / dt_hist
                future_kps[name] = (int(p_curr[0] + kvx * dt), int(p_curr[1] + kvy * dt))
            else:
                future_kps[name] = p_curr

        return future_kps

    def _classify_torso_intent(
        self,
        pose: HumanPose,
        vx: float,
        vy: float,
        ax: float,
        ay: float,
        pitch_rate: float,
        roll_rate: float,
        future_kps: Dict[str, Tuple[int, int]]
    ) -> Tuple[str, float, str]:
        """Classify Intent based primarily on Torso Pitch/Roll rates, Torso Acceleration, and Limbs."""
        torso_speed = math.hypot(vx, vy)
        torso_accel = math.hypot(ax, ay)

        # 1. Torso Pitch Forward Rate (Rapid torso lean forward)
        if pitch_rate > 35.0 or (pose.torso.pitch_angle_deg > 25 and vy > 120.0):
            if torso_accel > 350.0:
                return "About to Lunge / Forward Charge", 0.94, "HIGH_RISK"
            return "About to Bend Torso Forward", 0.89, "CAUTION"

        # 2. Torso Pitch Backward Rate (Rising / Standing Up)
        if pitch_rate < -30.0 and pose.posture in ("Sitting", "Torso Bending Forward", "Lying Down"):
            return "About to Stand Up Right", 0.91, "NORMAL"

        # 3. Torso Roll Rate (Side dodge / lateral lean)
        if abs(roll_rate) > 30.0 or abs(pose.torso.roll_angle_deg) > 20:
            return "About to Lean / Dodge Sideways", 0.87, "CAUTION"

        # 4. Rapid Arm Extension relative to Torso
        if pose.arm_extension_left > 0.85 or pose.arm_extension_right > 0.85:
            if torso_accel > 300.0:
                return "About to Strike / Reach Fast", 0.92, "HIGH_RISK"
            return "About to Reach Out", 0.88, "WARNING"

        # 5. Torso Lateral Translation (Turning)
        if vx < -180.0:
            return "About to Turn Left", 0.85, "NORMAL"
        if vx > 180.0:
            return "About to Turn Right", 0.85, "NORMAL"

        if torso_speed > 250.0:
            return "Fast Torso Translation / Running", 0.88, "WARNING"

        return "Torso Stable / Static Stance", 0.95, "NORMAL"
