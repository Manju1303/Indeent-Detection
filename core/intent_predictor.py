"""
core/intent_predictor.py
Sub-300ms Human Intent & Pre-Action Movement Predictor.

Features:
  - Sub-300ms Kinematic Trajectory Extrapolation (dt = 0.300s)
  - Keypoint Velocity & Acceleration Vector Engine
  - Predictive "Ghost Skeleton" Generation (+300ms in future)
  - Pre-Action Intent Classifier ("About to Reach", "About to Lunge", "About to Bend", "About to Turn", "About to Rise")
  - Ultra-fast real-time performance (< 2ms per frame)
"""

import time
import math
import numpy as np
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from core.pose_estimator import HumanPose

logger = logging.getLogger(__name__)


@dataclass
class IntentPrediction:
    """Pre-action prediction result for one human."""
    track_id: Optional[int]
    current_posture: str
    predicted_intent: str                   # E.g. "About to Reach Out", "About to Lunge", "About to Bend", "Stable"
    intent_confidence: float
    prediction_window_ms: int = 300
    centroid_velocity: Tuple[float, float] = (0.0, 0.0)    # (vx, vy) in px/sec
    centroid_acceleration: Tuple[float, float] = (0.0, 0.0)
    future_centroid_300ms: Tuple[int, int] = (0, 0)
    future_keypoints_300ms: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    trajectory_vector: Tuple[int, int] = (0, 0)             # (dx, dy) movement displacement over 300ms
    action_risk_level: str = "NORMAL"                       # "NORMAL", "CAUTION", "WARNING", "HIGH_RISK"
    latency_ms: float = 0.0


class IntentPredictor:
    """
    Predicts human next-movements and pre-action intent under 300ms.
    Maintains timestamped trajectory histories per tracked individual.
    """

    PREDICTION_WINDOW_S = 0.300    # 300 milliseconds
    MAX_HISTORY_LEN     = 15       # Rolling frame history

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.prediction_window_ms = self.config.get("prediction_window_ms", 300)
        self.window_s = self.prediction_window_ms / 1000.0

        # Memory per tracked human: track_id -> deque of (timestamp, HumanPose)
        self._history: Dict[int, deque] = {}

    # ── Public API ────────────────────────────────────────────

    def predict_intent(self, pose: HumanPose, timestamp: Optional[float] = None) -> IntentPrediction:
        """
        Analyze current pose & historical motion sequence to predict next movement (<300ms).
        """
        t_start = time.perf_counter()
        now = timestamp or time.time()
        tid = pose.track_id or 0

        if tid not in self._history:
            self._history[tid] = deque(maxlen=self.MAX_HISTORY_LEN)

        self._history[tid].append((now, pose))
        history = self._history[tid]

        # 1. Compute velocity & acceleration vectors
        vx, vy, ax, ay = self._compute_kinematics(history)

        # 2. Extrapolate future centroid at t + 300ms
        cx, cy = pose.centroid
        future_cx = int(cx + vx * self.window_s + 0.5 * ax * (self.window_s ** 2))
        future_cy = int(cy + vy * self.window_s + 0.5 * ay * (self.window_s ** 2))

        # 3. Extrapolate future keypoints ("Ghost Skeleton") at t + 300ms
        future_kps = self._predict_future_keypoints(history, self.window_s)

        # 4. Classify Pre-Action Intent & Action Risk
        intent_label, conf, risk = self._classify_intent(pose, vx, vy, ax, ay, future_kps)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return IntentPrediction(
            track_id=pose.track_id,
            current_posture=pose.posture,
            predicted_intent=intent_label,
            intent_confidence=conf,
            prediction_window_ms=self.prediction_window_ms,
            centroid_velocity=(round(vx, 1), round(vy, 1)),
            centroid_acceleration=(round(ax, 1), round(ay, 1)),
            future_centroid_300ms=(future_cx, future_cy),
            future_keypoints_300ms=future_kps,
            trajectory_vector=(future_cx - cx, future_cy - cy),
            action_risk_level=risk,
            latency_ms=round(elapsed_ms, 2),
        )

    # ── Kinematic Math & Intent Inference ─────────────────────

    def _compute_kinematics(self, history: deque) -> Tuple[float, float, float, float]:
        """Compute centroid velocity (vx, vy) and acceleration (ax, ay) from history."""
        if len(history) < 2:
            return 0.0, 0.0, 0.0, 0.0

        t_curr, pose_curr = history[-1]
        t_prev, pose_prev = history[-2]
        dt1 = max(0.001, t_curr - t_prev)

        c_curr = pose_curr.centroid
        c_prev = pose_prev.centroid

        vx = (c_curr[0] - c_prev[0]) / dt1
        vy = (c_curr[1] - c_prev[1]) / dt1

        if len(history) >= 3:
            t_prev2, pose_prev2 = history[-3]
            dt2 = max(0.001, t_prev - t_prev2)
            c_prev2 = pose_prev2.centroid

            vx_prev = (c_prev[0] - c_prev2[0]) / dt2
            vy_prev = (c_prev[1] - c_prev2[1]) / dt2

            ax = (vx - vx_prev) / dt1
            ay = (vy - vy_prev) / dt1
        else:
            ax, ay = 0.0, 0.0

        return vx, vy, ax, ay

    def _predict_future_keypoints(self, history: deque, dt: float) -> Dict[str, Tuple[int, int]]:
        """Project each keypoint position dt seconds into the future using kinematic velocity extrapolation."""
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

                # Predict future position: p(t + dt) = p(t) + v * dt
                f_x = int(p_curr[0] + kvx * dt)
                f_y = int(p_curr[1] + kvy * dt)
                future_kps[name] = (f_x, f_y)
            else:
                future_kps[name] = p_curr

        return future_kps

    def _classify_intent(
        self,
        pose: HumanPose,
        vx: float,
        vy: float,
        ax: float,
        ay: float,
        future_kps: Dict[str, Tuple[int, int]]
    ) -> Tuple[str, float, str]:
        """Classify pre-action movement intent and security/safety risk level."""
        speed = math.hypot(vx, vy)
        accel_mag = math.hypot(ax, ay)

        # Wrist expansion velocity check
        wrist_l_vel = 0.0
        wrist_r_vel = 0.0
        if "LEFT_WRIST" in pose.keypoints and "LEFT_WRIST" in future_kps:
            wrist_l_vel = math.hypot(future_kps["LEFT_WRIST"][0] - pose.keypoints["LEFT_WRIST"][0],
                                     future_kps["LEFT_WRIST"][1] - pose.keypoints["LEFT_WRIST"][1]) / self.window_s
        if "RIGHT_WRIST" in pose.keypoints and "RIGHT_WRIST" in future_kps:
            wrist_r_vel = math.hypot(future_kps["RIGHT_WRIST"][0] - pose.keypoints["RIGHT_WRIST"][0],
                                     future_kps["RIGHT_WRIST"][1] - pose.keypoints["RIGHT_WRIST"][1]) / self.window_s

        max_wrist_speed = max(wrist_l_vel, wrist_r_vel)

        # Rules Engine for Pre-Action Intent Detection (< 300ms)
        if max_wrist_speed > 350.0 or pose.arm_extension_left > 0.85 or pose.arm_extension_right > 0.85:
            if accel_mag > 400.0 or speed > 300.0:
                return "About to Lunge / Strike", 0.92, "HIGH_RISK"
            return "About to Reach Out / Grab", 0.89, "WARNING"

        if vy > 180.0 or (pose.posture == "Standing Upright" and pose.torso_angle_deg > 25):
            return "About to Bend / Pick Up", 0.86, "CAUTION"

        if vy < -180.0 and pose.posture in ("Sitting", "Bending Forward", "Lying Down"):
            return "About to Stand Up", 0.88, "NORMAL"

        if vx < -200.0:
            return "About to Turn Left", 0.84, "NORMAL"

        if vx > 200.0:
            return "About to Turn Right", 0.84, "NORMAL"

        if speed > 250.0:
            return "Fast Approach / Running", 0.87, "WARNING"

        return "Stable / Static Stance", 0.95, "NORMAL"
