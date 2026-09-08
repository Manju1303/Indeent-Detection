"""
alerts/alert_manager.py
Centralized alert dispatcher with:
- Global cooldown to prevent spam
- Per-event-type cooldowns
- Multi-channel dispatch (Telegram, Email, Local)
- Alert history tracking
"""

import logging
import time
import threading
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Standardized alert payload."""
    event_type: str          # "violence", "theft_low", "theft_medium", "theft_high", "weapon"
    camera_id: int
    camera_name: str
    message: str
    confidence: float
    timestamp: float
    frame_snapshot: Optional[object] = None   # numpy array
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AlertManager:
    """
    Receives alert objects and dispatches to configured channels.
    Handles throttling so the same event doesn't spam every second.
    """

    def __init__(self, config: dict, camera_names: dict = None):
        self.config        = config
        self.camera_names  = camera_names or {}
        self.min_conf      = config.get("min_confidence", 0.65)
        self.global_cd     = config.get("global_cooldown_seconds", 15)

        self._channels     = []
        self._history: List[Alert] = []
        self._last_alert:  dict = {}   # event_type → last timestamp
        self._lock         = threading.Lock()

        self._init_channels()

    # ── Public API ────────────────────────────────────────────

    def dispatch(self, alert: Alert):
        """Send alert to all configured channels if not throttled."""
        if alert.confidence < self.min_conf:
            logger.debug(f"[Alerts] Suppressed — confidence too low: {alert.confidence:.2%}")
            return

        with self._lock:
            now = time.time()
            last = self._last_alert.get(alert.event_type, 0)
            if (now - last) < self.global_cd:
                logger.debug(f"[Alerts] Throttled '{alert.event_type}' (cooldown active)")
                return
            self._last_alert[alert.event_type] = now

        self._history.append(alert)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        logger.info(f"[Alerts] Dispatching: {alert.event_type} | cam={alert.camera_id} | conf={alert.confidence:.2%}")

        # Dispatch in parallel threads so one slow channel doesn't block others
        threads = []
        for channel in self._channels:
            t = threading.Thread(target=self._send_safe, args=(channel, alert), daemon=True)
            t.start()
            threads.append(t)

    def dispatch_violence(self, camera_id: int, score: float, frame=None):
        cam_name = self.camera_names.get(camera_id, f"Camera {camera_id}")
        alert = Alert(
            event_type="violence",
            camera_id=camera_id,
            camera_name=cam_name,
            message=f"⚠️ VIOLENCE DETECTED on {cam_name} (confidence: {score:.0%})",
            confidence=score,
            timestamp=time.time(),
            frame_snapshot=frame,
        )
        self.dispatch(alert)

    def dispatch_theft(self, event, camera_name: str):
        level_emoji = {"low": "🔶", "medium": "🔴", "high": "🚨"}.get(event.level, "⚠️")
        alert = Alert(
            event_type=f"theft_{event.level}",
            camera_id=event.camera_id,
            camera_name=camera_name,
            message=(
                f"{level_emoji} THEFT {event.level.upper()} on {camera_name}\n"
                f"Object: {event.object_class}\n"
                f"Zone: {event.zone_name or 'Unknown'}\n"
                f"Confidence: {event.score:.0%}"
            ),
            confidence=event.score,
            timestamp=event.timestamp,
            frame_snapshot=event.frame_snapshot,
            metadata={"person_track_id": event.person_track_id},
        )
        self.dispatch(alert)

    def dispatch_weapon(self, camera_id: int, weapon_class: str, score: float, frame=None):
        cam_name = self.camera_names.get(camera_id, f"Camera {camera_id}")
        alert = Alert(
            event_type="weapon",
            camera_id=camera_id,
            camera_name=cam_name,
            message=f"🔫 WEAPON DETECTED on {cam_name}: {weapon_class} (conf: {score:.0%})",
            confidence=score,
            timestamp=time.time(),
            frame_snapshot=frame,
        )
        self.dispatch(alert)

    def get_recent(self, n: int = 50) -> List[Alert]:
        return self._history[-n:]

    # ── Internal ──────────────────────────────────────────────

    def _init_channels(self):
        cfg = self.config

        if cfg.get("telegram", {}).get("enabled", False):
            try:
                from alerts.telegram_alert import TelegramChannel
                self._channels.append(TelegramChannel(cfg["telegram"]))
                logger.info("[Alerts] Telegram channel enabled.")
            except Exception as e:
                logger.error(f"[Alerts] Telegram init failed: {e}")

        if cfg.get("email", {}).get("enabled", False):
            try:
                from alerts.email_alert import EmailChannel
                self._channels.append(EmailChannel(cfg["email"]))
                logger.info("[Alerts] Email channel enabled.")
            except Exception as e:
                logger.error(f"[Alerts] Email init failed: {e}")

        if cfg.get("local", {}).get("enabled", True):
            try:
                from alerts.local_alert import LocalChannel
                self._channels.append(LocalChannel(cfg["local"]))
                logger.info("[Alerts] Local alert channel enabled.")
            except Exception as e:
                logger.error(f"[Alerts] Local alert init failed: {e}")

        if not self._channels:
            logger.warning("[Alerts] No alert channels configured!")

    def _send_safe(self, channel, alert: Alert):
        try:
            channel.send(alert)
        except Exception as e:
            logger.error(f"[Alerts] Channel {type(channel).__name__} failed: {e}")
