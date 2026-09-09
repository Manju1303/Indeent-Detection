"""
core/detector.py
YOLOv8-based object and weapon detection engine.
Handles model loading, inference, class filtering, and result formatting.
"""

import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Classes that trigger weapon/threat flag immediately
WEAPON_CLASSES = {"knife", "scissors", "gun", "pistol", "rifle"}

# COCO class names (YOLOv8 pretrained)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    is_weapon: bool = False
    track_id: Optional[int] = None    # filled by tracker

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def iou(self, other: "Detection") -> float:
        return compute_iou(self.bbox, other.bbox)


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DetectionEngine:
    """
    Wraps YOLOv8 for object detection.
    Falls back to graceful degradation if model not found.
    """

    def __init__(self, config: dict):
        self.config     = config
        self.model      = None
        self.device     = self._resolve_device(config.get("device", "auto"))
        self.conf_thr   = config.get("confidence_threshold", 0.45)
        self.iou_thr    = config.get("iou_threshold", 0.45)
        self.img_size   = config.get("image_size", 640)
        self.use_fp16   = config.get("use_fp16", False)
        self.classes_of_interest = config.get("classes_of_interest", [])
        self._class_ids_of_interest = []

        self._load_model(config.get("model", "yolov8s.pt"))

    # ── Public ────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a frame. Returns list of Detection objects."""
        if self.model is None:
            return []

        t0 = time.perf_counter()

        predict_kwargs = {
            "source": frame,
            "conf": self.conf_thr,
            "iou": self.iou_thr,
            "imgsz": self.img_size,
            "device": self.device,
            "verbose": False,
            "classes": self._class_ids_of_interest if self._class_ids_of_interest else None,
        }
        if self.use_fp16 and self.device != "cpu":
            predict_kwargs["half"] = True

        results = self.model.predict(**predict_kwargs)

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                conf     = float(box.conf[0])
                cls_name = self.model.names.get(cls_id, str(cls_id))
                xyxy     = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    is_weapon=cls_name.lower() in WEAPON_CLASSES,
                ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Detection: {len(detections)} objects in {elapsed_ms:.1f}ms")
        return detections

    def filter_by_class(self, detections: List[Detection], class_names: List[str]) -> List[Detection]:
        names_set = set(n.lower() for n in class_names)
        return [d for d in detections if d.class_name.lower() in names_set]

    def get_persons(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.class_name == "person"]

    def get_weapons(self, detections: List[Detection]) -> List[Detection]:
        return [d for d in detections if d.is_weapon]

    def get_objects(self, detections: List[Detection]) -> List[Detection]:
        """Return non-person, non-weapon detections (potential theft targets)."""
        return [d for d in detections if d.class_name != "person" and not d.is_weapon]

    # ── Internal ──────────────────────────────────────────────

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            # Build class ID filter list
            if self.classes_of_interest:
                name_to_id = {v.lower(): k for k, v in self.model.names.items()}
                self._class_ids_of_interest = [
                    name_to_id[c.lower()]
                    for c in self.classes_of_interest
                    if c.lower() in name_to_id
                ]
            logger.info(f"[Detector] YOLOv8 loaded: {model_path} on {self.device}")
        except ImportError:
            logger.error("[Detector] ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            logger.error(f"[Detector] Failed to load model '{model_path}': {e}")

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("[Detector] CUDA GPU detected.")
                return "cuda"
            if torch.backends.mps.is_available():
                logger.info("[Detector] Apple MPS detected.")
                return "mps"
        except ImportError:
            pass
        logger.info("[Detector] Using CPU.")
        return "cpu"
