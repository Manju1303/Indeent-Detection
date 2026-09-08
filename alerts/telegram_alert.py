"""
alerts/telegram_alert.py
Free Telegram Bot API alert channel.
No paid subscription required — uses the free Telegram Bot API.

Setup:
  1. Message @BotFather on Telegram → /newbot → get TOKEN
  2. Add bot to a group or message it directly → get CHAT_ID
  3. Fill in config/settings.yaml
"""

import cv2
import logging
import requests
import tempfile
import os
import time
from io import BytesIO

logger = logging.getLogger(__name__)


class TelegramChannel:
    """Sends alerts via Telegram Bot API (completely free)."""

    BASE_URL = "https://api.telegram.org/bot{token}/{method}"
    TIMEOUT  = 10   # seconds

    def __init__(self, config: dict):
        self.token       = config["bot_token"]
        self.chat_id     = config["chat_id"]
        self.send_image  = config.get("send_image", True)
        self._last_sent  = 0.0

    def send(self, alert):
        """Send alert message and optional image to Telegram."""
        # Send text message
        self._send_message(alert.message)

        # Send image if available
        if self.send_image and alert.frame_snapshot is not None:
            self._send_photo(alert.frame_snapshot, caption=alert.event_type.upper())

    def _send_message(self, text: str):
        url = self.BASE_URL.format(token=self.token, method="sendMessage")
        try:
            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }, timeout=self.TIMEOUT)
            if not resp.ok:
                logger.error(f"[Telegram] sendMessage failed: {resp.text}")
        except requests.RequestException as e:
            logger.error(f"[Telegram] Network error: {e}")

    def _send_photo(self, frame, caption: str = ""):
        """Encode frame as JPEG and send to Telegram."""
        try:
            success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                return
            url = self.BASE_URL.format(token=self.token, method="sendPhoto")
            resp = requests.post(url,
                data={"chat_id": self.chat_id, "caption": caption},
                files={"photo": ("alert.jpg", buf.tobytes(), "image/jpeg")},
                timeout=self.TIMEOUT,
            )
            if not resp.ok:
                logger.error(f"[Telegram] sendPhoto failed: {resp.text}")
        except Exception as e:
            logger.error(f"[Telegram] Photo send error: {e}")

    def test_connection(self) -> bool:
        """Test if bot token and chat_id are valid."""
        try:
            url = self.BASE_URL.format(token=self.token, method="getMe")
            resp = requests.get(url, timeout=5)
            if resp.ok:
                bot_name = resp.json()["result"]["username"]
                logger.info(f"[Telegram] Connected as @{bot_name}")
                return True
        except Exception:
            pass
        return False
