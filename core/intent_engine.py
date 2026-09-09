"""
core/intent_engine.py
Sub-300ms Spatio-Temporal Intent Engine (ST-GCN / GRU Sequence Classifier).
Maintains a 15-frame history buffer (~500ms) to predict next human movement intent (<300ms horizon).
"""

import os
import time
import logging
import numpy as np
from collections import deque
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class Sub300msIntentEngine:
    """
    Spatio-Temporal Sequence Memory Intent Engine.
    Evaluates keypoint sequence tensors of shape (1, Time, Joints, Channels) -> (1, 15, 17, 2).
    """

    CLASSES = [
        "Standing Still",
        "Walking",
        "Reaching Forward",
        "Bending Down",
        "Sudden Fall / Risk"
    ]

    def __init__(self, model_path: str = "models/intent_stgcn.onnx", window_size: int = 15, num_joints: int = 17):
        self.window_size = window_size
        self.num_joints = num_joints
        self.history_buffer = deque(maxlen=window_size)
        self.session = None
        self.input_name = None

        self._init_onnx_session(model_path)

    def _init_onnx_session(self, model_path: str):
        try:
            if os.path.exists(model_path):
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(model_path, opts, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                logger.info(f"[Sub300msIntentEngine] Bound ONNX model: {model_path}")
            else:
                logger.info(f"[Sub300msIntentEngine] ONNX model '{model_path}' not found. Using kinematic sequence classifier.")
        except Exception as e:
            logger.info(f"[Sub300msIntentEngine] ONNX load notice ({e}). Using kinematic sequence classifier.")

    def update_buffer(self, keypoints: np.ndarray):
        """
        keypoints shape expected: (17, 2) normalized coordinates [0.0 - 1.0].
        """
        if isinstance(keypoints, np.ndarray) and keypoints.shape == (self.num_joints, 2):
            self.history_buffer.append(keypoints)
        else:
            # Reshape or default
            kps = np.zeros((self.num_joints, 2), dtype=np.float32)
            self.history_buffer.append(kps)

    def is_ready(self) -> bool:
        return len(self.history_buffer) == self.window_size

    def predict_intent(self) -> Tuple[str, float]:
        """
        Predict human action intent over the spatio-temporal memory window.
        """
        if not self.is_ready():
            progress = len(self.history_buffer) / float(self.window_size)
            return "Populating Buffer...", round(progress, 2)

        sequence_data = np.array(self.history_buffer, dtype=np.float32)   # (15, 17, 2)
        input_tensor = np.expand_dims(sequence_data, axis=0)                # (1, 15, 17, 2)

        # 1. Run ONNX Inference if session loaded
        if self.session is not None:
            try:
                outputs = self.session.run(None, {self.input_name: input_tensor})
                probs = outputs[0]
                if probs.ndim == 2:
                    probs = probs[0]
                best_idx = int(np.argmax(probs))
                conf = float(probs[best_idx])
                return self.CLASSES[min(best_idx, len(self.CLASSES) - 1)], round(conf, 2)
            except Exception as e:
                logger.debug(f"[Sub300msIntentEngine] ONNX inference error: {e}")

        # 2. Kinematic Spatio-Temporal Sequence Fallback Classifier
        return self._classify_kinematic_sequence(sequence_data)

    def _classify_kinematic_sequence(self, sequence: np.ndarray) -> Tuple[str, float]:
        """
        Analyzes 15-frame sequence (15, 17, 2) for rapid motion, reaching, bending, or falling.
        """
        # COCO Indices: 5: L_SHOULDER, 6: R_SHOULDER, 9: L_WRIST, 10: R_WRIST, 11: L_HIP, 12: R_HIP
        curr = sequence[-1]
        prev = sequence[0]

        # Calculate translation velocity of hip/shoulder center over sequence
        hip_center_curr = (curr[11] + curr[12]) * 0.5
        hip_center_prev = (prev[11] + prev[12]) * 0.5
        dy = hip_center_curr[1] - hip_center_prev[1]
        dx = hip_center_curr[0] - hip_center_prev[0]
        speed = np.hypot(dx, dy)

        # Arm extension vector relative to shoulders
        shoulder_center = (curr[5] + curr[6]) * 0.5
        wrist_center = (curr[9] + curr[10]) * 0.5
        arm_extension = np.hypot(wrist_center[0] - shoulder_center[0], wrist_center[1] - shoulder_center[1])

        # Pitch tilt (neck/shoulder center to hip center)
        dy_spine = hip_center_curr[1] - shoulder_center[1]
        dx_spine = abs(shoulder_center[0] - hip_center_curr[0])

        if dy > 0.15:   # Rapid downward drop
            return "Sudden Fall / Risk", 0.94
        elif dy_spine < 0.10:   # Torso leaning far forward
            return "Bending Down", 0.89
        elif arm_extension > 0.30:   # Extended arms
            return "Reaching Forward", 0.92
        elif speed > 0.08:   # Translational movement
            return "Walking", 0.86
        else:
            return "Standing Still", 0.95
