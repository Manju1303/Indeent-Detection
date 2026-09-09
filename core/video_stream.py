"""
core/video_stream.py
Multi-Threaded Low-Latency Frame Capture Engine.
Decouples camera ingestion from processing loop to prevent queue lag and frame drops.
"""

import cv2
import time
import threading
import platform
import logging

logger = logging.getLogger(__name__)


class LiveVideoStream:
    """
    Multi-Threaded Video Reader.
    Captures live frames asynchronously in a background thread.
    """

    def __init__(self, src=0, width=1280, height=720):
        self.src = src
        self.width = width
        self.height = height

        # DirectShow on Windows for zero-latency camera capture
        if isinstance(src, int) and platform.system() == "Windows":
            self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.stream.isOpened():
                self.stream = cv2.VideoCapture(src)
        else:
            self.stream = cv2.VideoCapture(src)

        if self.stream.isOpened():
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Eliminates internal queue lag

        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        logger.info(f"[LiveVideoStream] Started capture thread for source: {self.src}")
        return self

    def update(self):
        while self.started:
            if not self.stream.isOpened():
                time.sleep(0.05)
                continue

            grabbed, frame = self.stream.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                time.sleep(0.01)

            time.sleep(0.005)   # Yield CPU to prevent thrashing

    def read(self):
        with self.read_lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.started = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.stream.isOpened():
            self.stream.release()
        logger.info("[LiveVideoStream] Stopped video stream.")
