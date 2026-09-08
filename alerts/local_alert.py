"""
alerts/local_alert.py
Local alert channel: console log + optional sound file playback.
No internet required. Works completely offline.
"""

import logging
import os
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class LocalChannel:
    """Console logging and optional audio alert."""

    def __init__(self, config: dict):
        self.sound_file      = config.get("sound_file", "alerts/alert.wav")
        self.log_to_console  = config.get("log_to_console", True)
        self._sound_available = self._check_sound()

    def send(self, alert):
        if self.log_to_console:
            ts = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
            print(f"\n{'='*60}")
            print(f"  🚨 SECURITY ALERT [{ts}]")
            print(f"  Type       : {alert.event_type.upper()}")
            print(f"  Camera     : {alert.camera_name}")
            print(f"  Confidence : {alert.confidence:.0%}")
            print(f"  Message    : {alert.message}")
            print(f"{'='*60}\n")

        if self._sound_available:
            t = threading.Thread(target=self._play_sound, daemon=True)
            t.start()

    def _play_sound(self):
        try:
            from playsound import playsound
            playsound(self.sound_file, block=False)
        except Exception:
            try:
                import sys
                if sys.platform == "win32":
                    import winsound
                    winsound.PlaySound(self.sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    print("\a\a\a")   # terminal bell fallback
            except Exception:
                print("\a\a\a")   # terminal bell fallback

    def _check_sound(self) -> bool:
        if not os.path.exists(self.sound_file):
            return False
        try:
            import playsound
            return True
        except ImportError:
            import sys
            return sys.platform == "win32"
