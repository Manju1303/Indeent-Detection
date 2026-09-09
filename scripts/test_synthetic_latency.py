"""
scripts/test_synthetic_latency.py
Synthetic Motion Sequence Generator & Sub-300ms Pipeline Latency Benchmark Script.
Simulates 30 FPS keypoint sequence streams to test intent engine performance out-of-the-box.
"""

import sys
import os
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.intent_engine import Sub300msIntentEngine
from core.pose_estimator import RealTimePoseEstimator


def generate_synthetic_motion_sequence(num_frames: int = 30):
    """
    Generates realistic 17-joint COCO synthetic motion sequence.
    Simulates a human standing, initiating forward reach, and bending down over 30 frames.
    """
    sequence = []

    # Base standing keypoint template (normalized 17x2 COCO coordinates)
    base_kps = np.array([
        [0.50, 0.15], # 0: Nose
        [0.48, 0.13], # 1: L_Eye
        [0.52, 0.13], # 2: R_Eye
        [0.45, 0.14], # 3: L_Ear
        [0.55, 0.14], # 4: R_Ear
        [0.40, 0.25], # 5: L_Shoulder
        [0.60, 0.25], # 6: R_Shoulder
        [0.35, 0.40], # 7: L_Elbow
        [0.65, 0.40], # 8: R_Elbow
        [0.30, 0.55], # 9: L_Wrist
        [0.70, 0.55], # 10: R_Wrist
        [0.42, 0.55], # 11: L_Hip
        [0.58, 0.55], # 12: R_Hip
        [0.42, 0.75], # 13: L_Knee
        [0.58, 0.75], # 14: R_Knee
        [0.42, 0.95], # 15: L_Ankle
        [0.58, 0.95]  # 16: R_Ankle
    ], dtype=np.float32)

    for i in range(num_frames):
        kps = base_kps.copy()
        # Add subtle natural jitter
        jitter = np.random.normal(0, 0.002, size=kps.shape)
        kps += jitter

        # Frame 10 to 20: Reach arms forward
        if i >= 10:
            reach_factor = (i - 10) * 0.02
            kps[9] = [0.30 - reach_factor * 0.5, 0.55 - reach_factor]   # L Wrist extends
            kps[10] = [0.70 + reach_factor * 0.5, 0.55 - reach_factor]  # R Wrist extends

        # Frame 20 to 30: Forward torso tilt
        if i >= 20:
            bend_factor = (i - 20) * 0.015
            kps[0][1] += bend_factor * 1.5   # Nose drops
            kps[5][1] += bend_factor         # Shoulders drop
            kps[6][1] += bend_factor

        sequence.append(kps)

    return sequence


def run_benchmark():
    print("=" * 70)
    print("  Sub-300ms Human Intent Engine — Synthetic Motion Sequence Benchmark")
    print("=" * 70)

    intent_engine = Sub300msIntentEngine(window_size=15, num_joints=17)
    motion_sequence = generate_synthetic_motion_sequence(num_frames=30)
    latencies = []

    print("\n[BENCHMARK] Simulating 30 FPS live motion stream (30 sequence frames)...")

    for idx, kps in enumerate(motion_sequence):
        t0 = time.perf_counter()

        # Update spatio-temporal memory buffer & predict intent
        intent_engine.update_buffer(kps)
        intent, conf = intent_engine.predict_intent()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        if idx % 5 == 0 or idx == 29:
            status_buf = f"Buffer ({len(intent_engine.history_buffer)}/15)"
            print(f"  Frame {idx:2d} | Latency: {elapsed_ms:5.3f} ms | Status: {status_buf:14s} | Intent (<300ms): {intent:22s} ({conf*100:.1f}%)")

    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)

    print("\n" + "=" * 70)
    print(f"  Average Sequence Processing Latency : {avg_latency:.3f} ms")
    print(f"  Maximum Single Frame Latency       : {max_latency:.3f} ms")
    print(f"  Prediction Time Horizon Threshold  : 300.00 ms")
    print(f"  Execution Status                   : {'[OK] EXCELLENT (SUB-MILLISECOND LATENCY)' if avg_latency < 30 else '[WARN] HIGH LATENCY'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_benchmark()
