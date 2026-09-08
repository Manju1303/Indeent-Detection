# Real-Time Human Torso & Pre-Action Movement Prediction System (<300ms)

An open-source AI computer vision system powered by **Human Torso Dynamics** (Torso Quadrilateral, Spine Vector, Torso Pitch/Roll/Yaw) as its **PRIMARY FEATURE** for real-time human posture extraction and **Sub-300ms Pre-Action Movement Prediction** ("Intent / Indent Detection").

---

## Primary Feature & Capabilities

- 👕 **Human Torso Core Analysis (Primary Feature)**:
  - Real-time **Torso Quadrilateral Polygon** (`Shoulders -> Hips`).
  - **Spine Vector Line** (Neck Midpoint to Pelvis Midpoint).
  - **Torso Pitch Angle** (Forward/backward inclination) & **Torso Roll Angle** (Side tilt).
  - **Torso Yaw Orientation** (Facing direction).
- ⚡ **Sub-300ms Motion Extrapolation**: Kinematic torso velocity & acceleration engine predicts future body position at $t + 300\text{ms}$ (latency **< 0.1 ms per frame**).
- 👻 **+300ms Torso Ghost Projection**: Renders projected future torso polygon, ghost skeleton, and trajectory arrows on live video.
- 🧠 **Pre-Action Intent Classifier**: Predicts immediate next torso actions (*"About to Bend Torso Forward"*, *"About to Lunge / Charge"*, *"About to Stand Up"*, *"About to Turn"*).
- 🌐 **Live Web Dashboard**: Flask + SocketIO real-time monitoring interface at `http://localhost:5000`.

---

## Benchmark Metrics

| Metric | Performance |
|---|---|
| **Primary Feature** | **Human Torso & Spine Core** |
| **Average Prediction Latency** | **0.06 ms** |
| **Maximum Frame Latency** | **0.17 ms** |
| **Target Prediction Horizon** | **300 ms** |
| **Frame Rate** | **Up to 60 FPS** |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Benchmark Test
```bash
python scripts/test_intent_predictor.py
```

### 3. Start System
```bash
python main.py
```

Open `http://localhost:5000` in your web browser to view live predictions.
