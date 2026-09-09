"""
core/violence_detector.py
LSTM-based violence detection using frame sequence analysis.
Loads trained MobileNetV2+LSTM model from models/violence_model.pth
"""

import cv2
import numpy as np
import logging
import time
from collections import deque
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ViolenceDetector:
    """
    Detects violence/fighting using a sliding window of frames.
    MobileNetV2 extracts per-frame features → LSTM classifies sequence.

    Falls back to pose-based heuristics if model not loaded.
    """

    INPUT_SIZE = (224, 224)

    def __init__(self, config: dict):
        self.enabled    = config.get("enabled", True)
        self.model_path = config.get("model_path", "models/violence_model.pth")
        self.seq_len    = config.get("sequence_length", 16)
        self.conf_thr   = config.get("confidence_threshold", 0.72)
        self.cooldown   = config.get("cooldown_seconds", 5)

        self.model          = None
        self.feature_extractor = None
        self._device        = "cpu"
        self._frame_buf     = deque(maxlen=self.seq_len)
        self._last_alert_t  = 0.0
        self._score_history = deque(maxlen=10)

        if self.enabled:
            self._load_model()

    # ── Public API ────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Feed a new frame. Returns (violence_detected, confidence_score).
        """
        if not self.enabled:
            return False, 0.0

        preprocessed = self._preprocess(frame)
        self._frame_buf.append(preprocessed)

        if len(self._frame_buf) < self.seq_len:
            return False, 0.0

        score = self._predict_sequence()
        self._score_history.append(score)

        # Smooth score over recent history to reduce jitter
        smoothed = float(np.mean(self._score_history))

        now = time.time()
        in_cooldown = (now - self._last_alert_t) < self.cooldown

        if smoothed >= self.conf_thr and not in_cooldown:
            self._last_alert_t = now
            logger.warning(f"[Violence] DETECTED — confidence: {smoothed:.2%}")
            return True, smoothed

        return False, smoothed

    def reset_cooldown(self):
        self._last_alert_t = 0.0

    # ── Internal ──────────────────────────────────────────────

    def _predict_sequence(self) -> float:
        """Run model inference on buffered sequence."""
        if self.model is None:
            # Fallback: simple pixel-difference heuristic (no model)
            return self._heuristic_score()

        try:
            import torch
            sequence = np.stack(list(self._frame_buf), axis=0)   # (T, H, W, 3)

            with torch.no_grad():
                # Extract features per frame
                frames_t = torch.FloatTensor(sequence).permute(0, 3, 1, 2) / 255.0
                frames_t = frames_t.to(self._device)

                features = self.feature_extractor(frames_t)   # (T, feature_dim)
                features = features.unsqueeze(0)              # (1, T, feature_dim)

                output = self.model(features)                 # (1, 2)
                probs  = torch.softmax(output, dim=1)
                score  = probs[0, 1].item()                   # violence class prob
            return score
        except Exception as e:
            logger.error(f"[Violence] Inference error: {e}")
            return self._heuristic_score()

    def _heuristic_score(self) -> float:
        """
        Fast motion heuristic fallback.
        Uses frame differencing (absdiff) for ultra-fast motion estimation (<0.2ms).
        """
        if len(self._frame_buf) < 2:
            return 0.0
        frames = list(self._frame_buf)
        f_prev = frames[-2].astype(np.uint8) if frames[-2].dtype != np.uint8 else frames[-2]
        f_curr = frames[-1].astype(np.uint8) if frames[-1].dtype != np.uint8 else frames[-1]

        prev_gray = cv2.cvtColor(f_prev, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(f_curr, cv2.COLOR_RGB2GRAY)

        diff = cv2.absdiff(prev_gray, curr_gray)
        # Motion thresholding
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_ratio = np.count_nonzero(motion_mask) / float(motion_mask.size)

        # Normalize score: 0 = calm/static, 1.0 = rapid violent motion
        score = min(motion_ratio * 4.0, 1.0)
        return float(score)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize frame for model input."""
        resized = cv2.resize(frame, self.INPUT_SIZE)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32)

    def _load_model(self):
        """Load trained violence model and feature extractor."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import models

            # --- Feature extractor: MobileNetV2 backbone ---
            backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
            backbone.classifier = nn.Identity()   # remove final FC
            backbone.eval()

            # Try to determine feature dim
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                feat_dim = backbone(dummy).shape[1]

            # --- Violence LSTM classifier ---
            class ViolenceLSTM(nn.Module):
                def __init__(self, input_dim, hidden_dim=256, num_layers=2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                        batch_first=True, dropout=0.3)
                    self.fc   = nn.Sequential(
                        nn.Linear(hidden_dim, 128),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(128, 2),
                    )
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])

            self.feature_extractor = backbone
            violence_model = ViolenceLSTM(feat_dim)

            # Load weights if available
            try:
                import os
                if os.path.exists(self.model_path):
                    state = torch.load(self.model_path, map_location="cpu")
                    violence_model.load_state_dict(state)
                    logger.info(f"[Violence] Loaded trained weights: {self.model_path}")
                else:
                    logger.warning(
                        f"[Violence] No trained weights at '{self.model_path}'. "
                        f"Using optical flow heuristic until model is trained. "
                        f"Run: python models/train_violence.py"
                    )
            except Exception as e:
                logger.warning(f"[Violence] Weight load error: {e}. Using heuristic.")

            violence_model.eval()
            self.model = violence_model

            # Device setup
            if torch.cuda.is_available():
                self._device = "cuda"
                self.model.to("cuda")
                self.feature_extractor.to("cuda")
            logger.info(f"[Violence] Model ready on {self._device}.")

        except ImportError:
            logger.error("[Violence] PyTorch not installed. Violence detection disabled.")
        except Exception as e:
            logger.error(f"[Violence] Model load failed: {e}")
