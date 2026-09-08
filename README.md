# 🔒 Open-Source Real-Time Theft & Violence Detection System

**100% Free | No Paid APIs | Runs Locally | Production Ready**

---

## Enhanced Architecture (v2.0)

```
CCTV / IP Camera / Video File
         │
         ▼
  ┌──────────────┐
  │  Frame Queue  │  ← Multiprocessing buffer
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────┐
  │                    Detection Engine                   │
  │  YOLOv8s → ByteTrack → Re-ID Grace → Pose (cond.)   │
  └──────┬───────────────────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
Violence    Theft
 LSTM      Rule Engine
  Model    (Overlap +
           Disappear +
            Zone)
    │         │
    └────┬────┘
         ▼
  ┌──────────────┐
  │ Event Logger │ → SQLite (with FPS, confidence, frame)
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────┐
  │          Alert Throttler             │
  │  (prevents duplicate/spam alerts)   │
  └──────┬───────────────────────────────┘
         │
   ┌─────┴──────┐
   ▼            ▼
Telegram     Local
  Bot        Alarm
             Email
         │
         ▼
  ┌──────────────┐
  │   Dashboard   │ ← Flask + SQLite live view
  └──────────────┘
```

---

## Enhancements Over Original Design

| Feature | Original | Enhanced v2.0 |
|---|---|---|
| Tracking | ByteTrack | ByteTrack + Re-ID grace period |
| Theft logic | 5-step basic | Zone-aware + confidence scoring |
| Alert system | Telegram only | Telegram + Email + Local buzzer |
| False alarms | Threshold only | Throttler + event cooldown |
| Logging | None | Full SQLite audit trail |
| Dashboard | Optional Flask | Built-in live Flask dashboard |
| Multi-camera | Not mentioned | Full multi-stream support |
| Config | Hardcoded | YAML-based config system |
| Violence dataset | RWF-2000 noted | Training script included |

---

## Installation

```bash
# 1. Clone and setup
cd theft_violence_detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download pretrained YOLOv8s
python scripts/download_models.py

# 4. Configure your cameras and settings
nano config/settings.yaml

# 5. Run the system
python main.py

# 6. Open dashboard
http://localhost:5000
```

---

## Directory Structure

```
theft_violence_detection/
├── main.py                     # Entry point
├── requirements.txt
├── config/
│   └── settings.yaml           # All configuration
├── core/
│   ├── camera_manager.py       # Multi-camera stream handler
│   ├── detector.py             # YOLOv8 detection engine
│   ├── tracker.py              # ByteTrack + Re-ID grace
│   ├── violence_detector.py    # LSTM violence classifier
│   ├── theft_detector.py       # Zone-aware theft logic
│   └── pipeline.py             # Full pipeline orchestrator
├── models/
│   ├── violence_model.py       # MobileNetV2+LSTM architecture
│   └── train_violence.py       # Training script
├── alerts/
│   ├── alert_manager.py        # Throttler + dispatcher
│   ├── telegram_alert.py       # Telegram bot
│   ├── email_alert.py          # SMTP email
│   └── local_alert.py          # Buzzer / sound
├── dashboard/
│   ├── app.py                  # Flask dashboard
│   └── templates/
│       └── index.html          # Live monitoring UI
├── utils/
│   ├── logger.py               # SQLite event logger
│   ├── drawing.py              # Visualization overlays
│   └── fps_counter.py          # Performance monitor
└── scripts/
    ├── download_models.py      # Auto model downloader
    └── test_camera.py          # Camera test utility
```

---

## Performance Targets

| Metric | CPU Only | With GPU |
|---|---|---|
| FPS | 12–18 | 30–60 |
| Weapon Detection | 94–97% | 94–97% |
| Violence Detection | 82–90% | 85–92% |
| Theft Detection | 78–88% | 80–90% |
| Alert Delay | < 1.5s | < 0.8s |

---

## License

This project is open source under MIT License.
Model weights: YOLOv8 (AGPL-3.0), MediaPipe (Apache 2.0), PyTorch (BSD)

> ⚠️ For commercial deployment, replace YOLOv8 with RT-DETR or purchase Ultralytics Enterprise License.
