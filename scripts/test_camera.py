"""
scripts/test_camera.py
Quick test utility to verify cameras are working before starting the main system.
Usage:
  python scripts/test_camera.py                  # test camera 0
  python scripts/test_camera.py --source 1       # test camera index 1
  python scripts/test_camera.py --source rtsp://... --name "Store Cam"
"""

import cv2
import argparse
import time
import sys


def test_camera(source, name="Camera"):
    print(f"\nTesting: {name} (source={source})")
    print("Press 'q' to quit, 's' to save a screenshot\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open: {source}")
        return False

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame_count = 0
    start_time  = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        frame_count += 1
        elapsed = time.time() - start_time
        fps     = frame_count / elapsed if elapsed > 0 else 0

        # Overlay stats
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{name}", (10, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}  |  {w}x{h}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, "Press Q to quit | S to screenshot", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow(f"Camera Test — {name}", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  📸 Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Camera OK — captured {frame_count} frames at avg {fps:.1f} FPS")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=0,
                   help="Camera source: 0 (webcam), RTSP URL, or video file path")
    p.add_argument("--name",   default="Test Camera")
    args = p.parse_args()

    source = args.source
    try:
        source = int(source)   # numeric index
    except (ValueError, TypeError):
        pass   # keep as string (RTSP or path)

    success = test_camera(source, args.name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
