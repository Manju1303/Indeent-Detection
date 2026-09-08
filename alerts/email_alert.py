"""
alerts/email_alert.py
Free email alert channel via Python's built-in smtplib.
Works with Gmail, Outlook, or any SMTP server.

Gmail setup:
  1. Enable 2FA on your Google account
  2. Generate an App Password: myaccount.google.com → Security → App Passwords
  3. Use that App Password in settings.yaml (not your regular password)
"""

import cv2
import smtplib
import logging
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

logger = logging.getLogger(__name__)


class EmailChannel:
    """Sends alert emails with optional embedded image."""

    def __init__(self, config: dict):
        self.smtp_server     = config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port       = config.get("smtp_port", 587)
        self.sender_email    = config["sender_email"]
        self.sender_password = config["sender_password"]
        self.recipient       = config["recipient_email"]
        self.send_image      = config.get("send_image", True)

    def send(self, alert):
        try:
            msg = MIMEMultipart()
            msg["From"]    = self.sender_email
            msg["To"]      = self.recipient
            msg["Subject"] = f"🚨 Security Alert: {alert.event_type.upper()} — {alert.camera_name}"

            body = f"""
<html><body>
<h2>Security Alert</h2>
<p><b>Type:</b> {alert.event_type.upper()}</p>
<p><b>Camera:</b> {alert.camera_name}</p>
<p><b>Time:</b> {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</p>
<p><b>Confidence:</b> {alert.confidence:.0%}</p>
<p><b>Message:</b><br>{alert.message}</p>
</body></html>
            """
            msg.attach(MIMEText(body, "html"))

            # Attach frame snapshot
            if self.send_image and alert.frame_snapshot is not None:
                success, buf = cv2.imencode(".jpg", alert.frame_snapshot,
                                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    img = MIMEImage(buf.tobytes(), name="alert_frame.jpg")
                    msg.attach(img)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient, msg.as_string())

            logger.info(f"[Email] Alert sent to {self.recipient}")

        except smtplib.SMTPException as e:
            logger.error(f"[Email] SMTP error: {e}")
        except Exception as e:
            logger.error(f"[Email] Unexpected error: {e}")


# ─────────────────────────────────────────────────────────────────────────────


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
        self.sound_file       = config.get("sound_file", "alerts/alert.wav")
        self.log_to_console   = config.get("log_to_console", True)
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
            # Fallback: system beep
            print("\a\a\a")

    def _check_sound(self) -> bool:
        if not os.path.exists(self.sound_file):
            logger.debug(f"[Local] Sound file not found: {self.sound_file}. Using beep fallback.")
            return False
        try:
            import playsound
            return True
        except ImportError:
            return False
