"""
scripts/download_models.py
Downloads all required pretrained models.
All models are free and open-source.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_yolov8():
    """Download YOLOv8s weights (auto-downloaded by ultralytics on first use)."""
    logger.info("Downloading YOLOv8s model...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")   # auto-downloads to ~/.config/Ultralytics/
        logger.info("✓ YOLOv8s downloaded successfully.")
        return True
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return False
    except Exception as e:
        logger.error(f"YOLOv8 download failed: {e}")
        return False


def check_mediapipe():
    """Verify MediaPipe is available."""
    try:
        import mediapipe as mp
        logger.info("✓ MediaPipe available.")
        return True
    except ImportError:
        logger.warning("MediaPipe not installed. Run: pip install mediapipe")
        return False


def check_pytorch():
    """Verify PyTorch installation and check for GPU."""
    try:
        import torch
        gpu = torch.cuda.is_available()
        logger.info(f"✓ PyTorch {torch.__version__} | GPU: {'YES (' + torch.cuda.get_device_name(0) + ')' if gpu else 'No (CPU only)'}")
        return True
    except ImportError:
        logger.error("PyTorch not installed. See: https://pytorch.org/get-started/locally/")
        return False


def check_opencv():
    """Verify OpenCV."""
    try:
        import cv2
        logger.info(f"✓ OpenCV {cv2.__version__}")
        return True
    except ImportError:
        logger.error("OpenCV not installed. Run: pip install opencv-python")
        return False


def create_directories():
    """Create necessary project directories."""
    dirs = [
        "models", "logs", "logs/alert_images",
        "data/violence_dataset/train/fight",
        "data/violence_dataset/train/non-fight",
        "data/violence_dataset/val/fight",
        "data/violence_dataset/val/non-fight",
        "alerts", "config",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("✓ Project directories created.")


def main():
    print("\n" + "="*55)
    print("  Security Detection System — Setup & Model Download")
    print("="*55 + "\n")

    create_directories()
    ok_cv   = check_opencv()
    ok_pt   = check_pytorch()
    ok_mp   = check_mediapipe()
    ok_yolo = download_yolov8()

    print("\n" + "="*55)
    print("  Setup Summary")
    print("="*55)
    print(f"  OpenCV:     {'✓' if ok_cv else '✗ MISSING'}")
    print(f"  PyTorch:    {'✓' if ok_pt else '✗ MISSING'}")
    print(f"  MediaPipe:  {'✓' if ok_mp else '✗ MISSING'}")
    print(f"  YOLOv8s:    {'✓' if ok_yolo else '✗ MISSING'}")

    if all([ok_cv, ok_pt, ok_yolo]):
        print("\n  ✅ Ready to run! Start with: python main.py")
    else:
        print("\n  ⚠️  Fix missing dependencies above, then re-run.")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
