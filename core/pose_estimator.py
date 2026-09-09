"""
core/pose_estimator.py
Real-time Human Pose, Keypoint & Torso Core Analysis Engine.
The Torso (Shoulders, Hips, Spine Vector, Pitch/Roll/Yaw) is the PRIMARY feature.
"""

import cv2
import numpy as np
import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# Keypoint Index Mapping (Standard 33 MediaPipe / COCO Keypoint format)
KEYPOINT_MAP = {
    "NOSE": 0, "LEFT_EYE": 2, "RIGHT_EYE": 5, "LEFT_EAR": 7, "RIGHT_EAR": 8,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
}

# Skeleton Bone Connections (Joint pairs for skeleton rendering)
SKELETON_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
]


@dataclass
class TorsoState:
    """Detailed anatomical and kinematic metrics of the Human Torso (Primary Feature)."""
    center: Tuple[int, int]                    # Midpoint of torso (between shoulders & hips)
    polygon: List[Tuple[int, int]]             # [LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_HIP, LEFT_HIP]
    spine_neck_pt: Tuple[int, int]             # Midpoint between shoulders
    spine_pelvis_pt: Tuple[int, int]           # Midpoint between hips
    spine_length_px: float                     # Distance from neck to pelvis
    pitch_angle_deg: float                     # Forward/backward inclination from vertical (0° = upright)
    roll_angle_deg: float                      # Side tilt angle from horizontal (0° = level shoulders)
    yaw_facing_ratio: float                    # Body facing aspect ratio (-1.0 = facing left, 0 = frontal, +1.0 = facing right)
    shoulder_width_px: float
    hip_width_px: float


@dataclass
class HumanPose:
    """Represents full pose, keypoints, torso core metrics, and body outline for one human."""
    track_id: Optional[int]
    bbox: Tuple[int, int, int, int]           # (x1, y1, x2, y2)
    keypoints: Dict[str, Tuple[int, int]]     # {"NOSE": (x,y), ...}
    keypoint_confidences: Dict[str, float]
    torso: TorsoState                         # PRIMARY FEATURE: Complete Torso Telemetry
    posture: str                              # "Standing Upright", "Bending Forward", "Reaching", "Lunging", "Sitting", "Running"
    posture_confidence: float
    arm_extension_left: float                 # 0.0 (retracted) to 1.0 (fully extended)
    arm_extension_right: float
    body_outline: List[Tuple[int, int]]       # Convex hull polygon points
    centroid: Tuple[int, int]


class PoseEstimator:
    """
    Real-time Human Pose & Primary Torso Core Estimator.
    Analyzes Torso Quadrilateral, Spine Vector, Pitch/Roll/Yaw, and Body Contour.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.min_detection_confidence = self.config.get("min_detection_confidence", 0.5)
        self._mp_pose = None
        self._init_mediapipe()

    # ── Public API ────────────────────────────────────────────

    def estimate_pose(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int], track_id: Optional[int] = None) -> HumanPose:
        """
        Extract keypoints, Primary Torso Core Metrics, posture classification, and body outline.
        """
        x1, y1, x2, y2 = person_bbox
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 1. Extract Keypoints
        keypoints, keypoint_confs = self._extract_keypoints(frame, person_bbox)

        # 2. Extract Primary Feature: TORSO CORE STATE
        torso = self._extract_torso_core(keypoints, (cx, cy), w, h)

        # 3. Compute Arm Extensions
        arm_ext_l, arm_ext_r = self._compute_arm_extensions(keypoints, torso.spine_length_px)

        # 4. Classify Posture based on Torso Pitch/Roll & Joint Angles
        posture, posture_conf = self._classify_posture(w, h, torso, arm_ext_l, arm_ext_r, keypoints)

        # 5. Generate Body Outline (Convex Hull around keypoints)
        outline = self._generate_body_outline(keypoints, person_bbox)

        return HumanPose(
            track_id=track_id,
            bbox=person_bbox,
            keypoints=keypoints,
            keypoint_confidences=keypoint_confs,
            torso=torso,
            posture=posture,
            posture_confidence=posture_conf,
            arm_extension_left=arm_ext_l,
            arm_extension_right=arm_ext_r,
            body_outline=outline,
            centroid=(cx, cy),
        )

    # ── Torso Core Extraction (Primary Feature) ───────────────

    def _extract_torso_core(self, kps: Dict[str, Tuple[int, int]], centroid: Tuple[int, int], w: int, h: int) -> TorsoState:
        """Extract Torso Quadrilateral, Spine Vector, Pitch, Roll, and Yaw Orientation."""
        ls = kps.get("LEFT_SHOULDER", (centroid[0] - int(w * 0.25), int(centroid[1] - h * 0.2)))
        rs = kps.get("RIGHT_SHOULDER", (centroid[0] + int(w * 0.25), int(centroid[1] - h * 0.2)))
        lh = kps.get("LEFT_HIP", (centroid[0] - int(w * 0.18), int(centroid[1] + h * 0.25)))
        rh = kps.get("RIGHT_HIP", (centroid[0] + int(w * 0.18), int(centroid[1] + h * 0.25)))

        # Spine Vector (Neck to Pelvis)
        neck_x, neck_y = (ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2
        pelvis_x, pelvis_y = (lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2
        torso_cx, torso_cy = (neck_x + pelvis_x) // 2, (neck_y + pelvis_y) // 2

        spine_length = max(1.0, math.hypot(pelvis_x - neck_x, pelvis_y - neck_y))

        # Torso Pitch (Forward/backward inclination angle from vertical)
        dx_spine = neck_x - pelvis_x
        dy_spine = pelvis_y - neck_y
        pitch_deg = math.degrees(math.atan2(abs(dx_spine), max(1, dy_spine)))

        # Torso Roll (Side tilt angle of shoulders from horizontal)
        dx_shoulder = rs[0] - ls[0]
        dy_shoulder = rs[1] - ls[1]
        roll_deg = math.degrees(math.atan2(dy_shoulder, max(1, abs(dx_shoulder))))

        # Torso Yaw Facing Ratio (Asymmetry between shoulder width & hip width relative to bounding box)
        shoulder_width = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
        hip_width = math.hypot(rh[0] - lh[0], rh[1] - lh[1])
        facing_ratio = (shoulder_width - w * 0.5) / max(1.0, w * 0.5)
        facing_ratio = max(-1.0, min(1.0, facing_ratio))

        torso_poly = [ls, rs, rh, lh]

        return TorsoState(
            center=(torso_cx, torso_cy),
            polygon=torso_poly,
            spine_neck_pt=(neck_x, neck_y),
            spine_pelvis_pt=(pelvis_x, pelvis_y),
            spine_length_px=round(spine_length, 1),
            pitch_angle_deg=round(pitch_deg, 1),
            roll_angle_deg=round(roll_deg, 1),
            yaw_facing_ratio=round(facing_ratio, 2),
            shoulder_width_px=round(shoulder_width, 1),
            hip_width_px=round(hip_width, 1),
        )

    # ── Internal Helpers ──────────────────────────────────────

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=0.5,
            )
            logger.info("[PoseEstimator] MediaPipe Pose solution loaded.")
        except Exception as e:
            logger.info(f"[PoseEstimator] MediaPipe Pose unavailable ({e}). Using geometric skeleton fallback.")

    def _extract_keypoints(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, float]]:
        """Extract keypoints using MediaPipe if available, or geometric fallback."""
        x1, y1, x2, y2 = bbox
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        keypoints = {}
        confs = {}

        if self._mp_pose is not None:
            try:
                # Square-pad bounding box to maintain 1:1 aspect ratio for MediaPipe
                max_dim = max(w, h)
                cx_box, cy_box = (x1 + x2) // 2, (y1 + y2) // 2
                half_dim = int(max_dim * 0.6)

                rx1, ry1 = max(0, cx_box - half_dim), max(0, cy_box - half_dim)
                rx2, ry2 = min(frame.shape[1], cx_box + half_dim), min(frame.shape[0], cy_box + half_dim)
                crop = frame[ry1:ry2, rx1:rx2]

                if crop.size > 0:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    results = self._mp_pose.process(rgb)
                    if results.pose_landmarks:
                        cw, ch = crop.shape[1], crop.shape[0]
                        mp_kps = results.pose_landmarks.landmark
                        for name, idx in KEYPOINT_MAP.items():
                            if idx < len(mp_kps):
                                lm = mp_kps[idx]
                                keypoints[name] = (int(rx1 + lm.x * cw), int(ry1 + lm.y * ch))
                                confs[name] = float(lm.visibility)
                        if len(keypoints) >= 8:
                            return keypoints, confs
            except Exception as e:
                logger.debug(f"[PoseEstimator] MediaPipe fallback: {e}")

        # Geometric Skeleton Fallback
        cx = (x1 + x2) // 2
        keypoints["NOSE"] = (cx, int(y1 + h * 0.15))
        keypoints["LEFT_EYE"] = (int(cx - w * 0.08), int(y1 + h * 0.12))
        keypoints["RIGHT_EYE"] = (int(cx + w * 0.08), int(y1 + h * 0.12))
        keypoints["LEFT_EAR"] = (int(cx - w * 0.15), int(y1 + h * 0.14))
        keypoints["RIGHT_EAR"] = (int(cx + w * 0.15), int(y1 + h * 0.14))

        keypoints["LEFT_SHOULDER"] = (int(cx - w * 0.25), int(y1 + h * 0.25))
        keypoints["RIGHT_SHOULDER"] = (int(cx + w * 0.25), int(y1 + h * 0.25))

        keypoints["LEFT_ELBOW"] = (int(cx - w * 0.32), int(y1 + h * 0.42))
        keypoints["RIGHT_ELBOW"] = (int(cx + w * 0.32), int(y1 + h * 0.42))

        keypoints["LEFT_WRIST"] = (int(cx - w * 0.35), int(y1 + h * 0.58))
        keypoints["RIGHT_WRIST"] = (int(cx + w * 0.35), int(y1 + h * 0.58))

        keypoints["LEFT_HIP"] = (int(cx - w * 0.18), int(y1 + h * 0.55))
        keypoints["RIGHT_HIP"] = (int(cx + w * 0.18), int(y1 + h * 0.55))

        keypoints["LEFT_KNEE"] = (int(cx - w * 0.20), int(y1 + h * 0.76))
        keypoints["RIGHT_KNEE"] = (int(cx + w * 0.20), int(y1 + h * 0.76))

        keypoints["LEFT_ANKLE"] = (int(cx - w * 0.22), int(y1 + h * 0.95))
        keypoints["RIGHT_ANKLE"] = (int(cx + w * 0.22), int(y1 + h * 0.95))

        for name in keypoints:
            confs[name] = 0.85

        return keypoints, confs

    def _compute_arm_extensions(self, kps: Dict[str, Tuple[int, int]], spine_len: float) -> Tuple[float, float]:
        ext_l = 0.5
        if "LEFT_SHOULDER" in kps and "LEFT_WRIST" in kps:
            arm_l = math.hypot(kps["LEFT_WRIST"][0] - kps["LEFT_SHOULDER"][0], kps["LEFT_WRIST"][1] - kps["LEFT_SHOULDER"][1])
            ext_l = min(1.0, arm_l / max(1.0, spine_len * 1.2))

        ext_r = 0.5
        if "RIGHT_SHOULDER" in kps and "RIGHT_WRIST" in kps:
            arm_r = math.hypot(kps["RIGHT_WRIST"][0] - kps["RIGHT_SHOULDER"][0], kps["RIGHT_WRIST"][1] - kps["RIGHT_SHOULDER"][1])
            ext_r = min(1.0, arm_r / max(1.0, spine_len * 1.2))

        return round(ext_l, 2), round(ext_r, 2)

    def _classify_posture(self, w: int, h: int, torso: TorsoState, arm_l: float, arm_r: float, kps: Dict[str, Tuple[int, int]]) -> Tuple[str, float]:
        aspect_ratio = w / float(h)

        if aspect_ratio > 1.2:
            return "Lying Down", 0.90
        elif aspect_ratio > 0.85:
            if torso.pitch_angle_deg > 35:
                return "Torso Bending Forward", 0.88
            return "Sitting", 0.82
        else:
            if torso.pitch_angle_deg > 30:
                return "Torso Bending Forward", 0.86
            elif abs(torso.roll_angle_deg) > 20:
                return "Torso Leaning Sideways", 0.87
            elif arm_l > 0.85 or arm_r > 0.85:
                return "Reaching Out", 0.92
            elif "LEFT_ANKLE" in kps and "RIGHT_ANKLE" in kps:
                if abs(kps["LEFT_ANKLE"][0] - kps["RIGHT_ANKLE"][0]) > w * 0.6:
                    return "Lunging Stance", 0.88
            return "Standing Upright", 0.95

    def _generate_body_outline(self, kps: Dict[str, Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        pts = np.array(list(kps.values()), dtype=np.int32)
        if len(pts) >= 3:
            hull = cv2.convexHull(pts)
            return [tuple(p[0]) for p in hull]

        x1, y1, x2, y2 = bbox
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


class RealTimePoseEstimator:
    """
    ONNX & Geometric 17-Joint COCO Frontend Pose Estimator.
    Binds to CUDA/CPU Execution Providers and outputs normalized 17x2 joint matrix.
    """

    COCO_17_KEYS = [
        "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.session = None
        self.fallback_estimator = PoseEstimator()
        if model_path:
            self._init_onnx_session(model_path)

    def _init_onnx_session(self, model_path: str):
        try:
            import os
            if os.path.exists(model_path):
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(model_path, opts, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                logger.info(f"[RealTimePoseEstimator] Bound ONNX session: {model_path}")
            else:
                logger.info(f"[RealTimePoseEstimator] ONNX model '{model_path}' not found. Using fallback estimator.")
        except Exception as e:
            logger.info(f"[RealTimePoseEstimator] ONNX load notice ({e}). Using fallback estimator.")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (192, 256))
        img = resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def extract_keypoints(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Extract keypoints returning normalized shape (17, 2).
        """
        h, w = frame.shape[:2]
        if bbox is None:
            bbox = (int(w * 0.25), int(h * 0.1), int(w * 0.75), int(h * 0.9))

        if self.session is not None:
            try:
                input_tensor = self.preprocess(frame)
                outputs = self.session.run(None, {self.input_name: input_tensor})
                raw_kps = outputs[0]
                if isinstance(raw_kps, np.ndarray) and raw_kps.shape[-2:] == (17, 2):
                    return raw_kps.reshape(17, 2)
            except Exception as e:
                logger.debug(f"[RealTimePoseEstimator] ONNX inference error: {e}")

        # Fallback using geometric pose estimator
        pose = self.fallback_estimator.estimate_pose(frame, bbox, track_id=1)
        matrix = np.zeros((17, 2), dtype=np.float32)
        for idx, key in enumerate(self.COCO_17_KEYS):
            if key in pose.keypoints:
                pt = pose.keypoints[key]
                matrix[idx] = [pt[0] / float(w), pt[1] / float(h)]
            else:
                matrix[idx] = [0.5, 0.5]
        return matrix
