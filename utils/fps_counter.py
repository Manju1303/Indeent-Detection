"""
utils/fps_counter.py
Rolling FPS counter and performance monitor.
"""

import time
from collections import deque
import psutil
import os


class FPSCounter:
    """Computes rolling FPS over a window of recent frames."""

    def __init__(self, window: int = 30):
        self._timestamps = deque(maxlen=window)
        self.fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self.fps = (len(self._timestamps) - 1) / elapsed
        return self.fps


class SystemMonitor:
    """Monitors CPU, RAM, and GPU usage."""

    def __init__(self):
        self._process = psutil.Process(os.getpid())

    def get_stats(self) -> dict:
        cpu  = psutil.cpu_percent(interval=None)
        ram  = psutil.virtual_memory()
        proc = self._process.memory_info().rss / (1024 ** 2)   # MB

        stats = {
            "cpu_percent":    cpu,
            "ram_total_gb":   round(ram.total / (1024 ** 3), 1),
            "ram_used_gb":    round(ram.used  / (1024 ** 3), 1),
            "ram_percent":    ram.percent,
            "process_ram_mb": round(proc, 1),
        }

        # Try GPU stats (optional)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                stats["gpu_util"]    = int(parts[0])
                stats["gpu_mem_mb"]  = int(parts[1])
                stats["gpu_total_mb"] = int(parts[2])
        except Exception:
            pass

        return stats
