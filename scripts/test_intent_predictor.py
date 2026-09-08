"""
scripts/test_intent_predictor.py
Benchmark and verification script for Sub-300ms Human Intent & Pre-Action Prediction Engine.
"""

import sys
import os
import time
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.pose_estimator import PoseEstimator
from core.intent_predictor import IntentPredictor


def run_benchmark():
    print("=" * 60)
    print("  Human Intent & Pre-Action Prediction Benchmark (<300ms)")
    print("=" * 60)

    pose_est = PoseEstimator()
    intent_pred = IntentPredictor({"prediction_window_ms": 300})

    # Create dummy 1080p frame
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Simulated motion sequence of a standing person starting to reach / lunge forward
    start_bbox = [800, 300, 1100, 900]
    latencies = []

    print("\nSimulating real-time video stream (30 frames)...")
    for i in range(30):
        # Shift bounding box forward to simulate motion
        bbox = (start_bbox[0] + i * 8, start_bbox[1], start_bbox[2] + i * 8, start_bbox[3])

        t0 = time.perf_counter()
        pose = pose_est.estimate_pose(dummy_frame, bbox, track_id=1)
        pred = intent_pred.predict_intent(pose)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        if i % 5 == 0 or i == 29:
            print(f"  Frame {i:2d} | Latency: {elapsed_ms:4.2f}ms | Posture: {pose.posture:16s} | Intent (<300ms): {pred.predicted_intent:25s} | Risk: {pred.action_risk_level}")

    avg_ms = np.mean(latencies)
    max_ms = np.max(latencies)

    print("\n" + "=" * 60)
    print(f"  Average Frame Latency : {avg_ms:.2f} ms")
    print(f"  Maximum Frame Latency : {max_ms:.2f} ms")
    print(f"  Prediction Window     : 300 ms")
    print(f"  Real-Time Status      : {'[OK] EXCELLENT (WELL BELOW 300ms REQUIREMENT)' if avg_ms < 30 else '[WARN] SLOW'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_benchmark()
