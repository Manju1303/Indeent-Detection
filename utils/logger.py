"""
utils/logger.py
SQLite-backed event logger for all detections and alerts.
Provides full audit trail, performance metrics, and dashboard data.
"""

import sqlite3
import logging
import os
import time
import cv2
import json
from datetime import datetime
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class EventLogger:
    """
    Writes all events to SQLite. No external database required.
    Thread-safe via connection-per-call pattern.
    """

    def __init__(self, config: dict):
        self.db_path         = config.get("db_path", "logs/events.db")
        self.save_images     = config.get("save_alert_images", True)
        self.images_dir      = config.get("alert_images_dir", "logs/alert_images")
        self.max_images      = config.get("max_stored_images", 500)

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if self.save_images:
            os.makedirs(self.images_dir, exist_ok=True)

        self._init_db()
        logger.info(f"[Logger] SQLite database: {self.db_path}")

    # ── Public API ────────────────────────────────────────────

    def log_event(
        self,
        event_type: str,
        camera_id: int,
        camera_name: str,
        confidence: float,
        details: dict,
        frame=None,
    ):
        """Log a detection or alert event."""
        image_path = None
        if frame is not None and self.save_images:
            image_path = self._save_image(frame, event_type, camera_id)

        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO events (timestamp, event_type, camera_id, camera_name,
                                        confidence, details_json, image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    time.time(),
                    event_type,
                    camera_id,
                    camera_name,
                    confidence,
                    json.dumps(details),
                    image_path,
                ))
        except sqlite3.DatabaseError as e:
            if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                self._repair_corrupt_db()

    def log_performance(self, camera_id: int, fps: float, processing_ms: float,
                        num_detections: int):
        """Log per-frame performance metrics."""
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO performance (timestamp, camera_id, fps, processing_ms, num_detections)
                    VALUES (?, ?, ?, ?, ?)
                """, (time.time(), camera_id, fps, processing_ms, num_detections))
        except sqlite3.DatabaseError as e:
            if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                self._repair_corrupt_db()

    def get_recent_events(self, n: int = 100, event_type: str = None) -> List[dict]:
        """Fetch recent events for dashboard."""
        try:
            with self._connect() as conn:
                if event_type:
                    rows = conn.execute("""
                        SELECT * FROM events WHERE event_type = ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (event_type, n)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT * FROM events ORDER BY timestamp DESC LIMIT ?
                    """, (n,)).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except sqlite3.DatabaseError:
            self._repair_corrupt_db()
            return []

    def get_stats(self) -> dict:
        """Summary statistics for dashboard."""
        try:
            with self._connect() as conn:
                total      = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                violence   = conn.execute("SELECT COUNT(*) FROM events WHERE event_type='violence'").fetchone()[0]
                theft_high = conn.execute("SELECT COUNT(*) FROM events WHERE event_type='theft_high'").fetchone()[0]
                theft_med  = conn.execute("SELECT COUNT(*) FROM events WHERE event_type='theft_medium'").fetchone()[0]
                theft_low  = conn.execute("SELECT COUNT(*) FROM events WHERE event_type='theft_low'").fetchone()[0]
                weapons    = conn.execute("SELECT COUNT(*) FROM events WHERE event_type='weapon'").fetchone()[0]

                today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
                today       = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE timestamp >= ?", (today_start,)
                ).fetchone()[0]

            return {
                "total":      total,
                "today":      today,
                "violence":   violence,
                "theft_high": theft_high,
                "theft_med":  theft_med,
                "theft_low":  theft_low,
                "weapons":    weapons,
            }
        except sqlite3.DatabaseError:
            self._repair_corrupt_db()
            return {"total": 0, "today": 0, "violence": 0, "theft_high": 0, "theft_med": 0, "theft_low": 0, "weapons": 0}

    def get_performance_avg(self, camera_id: int = None, last_n: int = 100) -> dict:
        """Average performance metrics."""
        try:
            with self._connect() as conn:
                if camera_id is not None:
                    rows = conn.execute("""
                        SELECT fps, processing_ms, num_detections FROM performance
                        WHERE camera_id = ? ORDER BY timestamp DESC LIMIT ?
                    """, (camera_id, last_n)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT fps, processing_ms, num_detections FROM performance
                        ORDER BY timestamp DESC LIMIT ?
                    """, (last_n,)).fetchall()

            if not rows:
                return {"fps": 0, "processing_ms": 0, "num_detections": 0}
            fps   = sum(r[0] for r in rows) / len(rows)
            ms    = sum(r[1] for r in rows) / len(rows)
            dets  = sum(r[2] for r in rows) / len(rows)
            return {"fps": round(fps, 1), "processing_ms": round(ms, 1), "num_detections": round(dets, 1)}
        except sqlite3.DatabaseError:
            self._repair_corrupt_db()
            return {"fps": 0, "processing_ms": 0, "num_detections": 0}

    # ── Internal ──────────────────────────────────────────────

    def _init_db(self):
        try:
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=5000;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp    REAL    NOT NULL,
                        event_type   TEXT    NOT NULL,
                        camera_id    INTEGER NOT NULL,
                        camera_name  TEXT,
                        confidence   REAL,
                        details_json TEXT,
                        image_path   TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance (
                        id             INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp      REAL,
                        camera_id      INTEGER,
                        fps            REAL,
                        processing_ms  REAL,
                        num_detections INTEGER
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)")
        except sqlite3.DatabaseError as e:
            if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                self._repair_corrupt_db()

    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.DatabaseError as e:
            if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                logger.warning(f"[Logger] Corrupted database detected at '{self.db_path}'. Recreating database...")
                self._repair_corrupt_db()
                conn = sqlite3.connect(self.db_path, timeout=10)
                conn.execute("PRAGMA busy_timeout=5000;")
                conn.row_factory = sqlite3.Row
                return conn
            raise

    def _repair_corrupt_db(self):
        try:
            if os.path.exists(self.db_path):
                bak_path = f"{self.db_path}.corrupt_{int(time.time())}"
                try:
                    os.rename(self.db_path, bak_path)
                except Exception:
                    os.remove(self.db_path)
                logger.info(f"[Logger] Corrupt database backup created at {bak_path}")
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    REAL    NOT NULL,
                    event_type   TEXT    NOT NULL,
                    camera_id    INTEGER NOT NULL,
                    camera_name  TEXT,
                    confidence   REAL,
                    details_json TEXT,
                    image_path   TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp      REAL,
                    camera_id      INTEGER,
                    fps            REAL,
                    processing_ms  REAL,
                    num_detections INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)")
            conn.close()
        except Exception as err:
            logger.error(f"[Logger] Failed to repair database: {err}")

    def _save_image(self, frame, event_type: str, camera_id: int) -> Optional[str]:
        try:
            ts   = int(time.time())
            name = f"{event_type}_cam{camera_id}_{ts}.jpg"
            path = os.path.join(self.images_dir, name)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self._prune_images()
            return path
        except Exception as e:
            logger.error(f"[Logger] Image save error: {e}")
            return None

    def _prune_images(self):
        """Keep only the most recent N images."""
        try:
            files = sorted(
                [f for f in os.listdir(self.images_dir) if f.endswith(".jpg")],
                key=lambda f: os.path.getmtime(os.path.join(self.images_dir, f))
            )
            while len(files) > self.max_images:
                os.remove(os.path.join(self.images_dir, files.pop(0)))
        except Exception:
            pass

    def _row_to_dict(self, row) -> dict:
        d = dict(row)
        if d.get("details_json"):
            try:
                d["details"] = json.loads(d["details_json"])
            except Exception:
                d["details"] = {}
        d["datetime"] = datetime.fromtimestamp(d["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        return d
