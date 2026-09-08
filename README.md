# Real-Time Human Intent & Pre-Action Movement Prediction System (<300ms)

An open-source AI computer vision system for real-time **Human Position, Posture, Body Outline Extraction**, and **Sub-300ms Pre-Action Movement Prediction** ("Intent / Indent Detection").

---

## Key Features

- 🎯 **33-Point Body Keypoints & Outline Tracking**: Real-time extraction of skeleton joints and body contour polygon (convex hull).
- ⚡ **Sub-300ms Motion Extrapolation**: Kinematic velocity & acceleration vector engine predicts future body keypoint positions at $t + 300\text{ms}$ (latency **< 0.1 ms per frame**).
- 👻 **+300ms Predictive Ghost Skeleton Overlay**: Renders projected future human pose and trajectory arrows directly on the live video stream.
- 🧠 **Pre-Action Intent Classifier**: Predicts immediate next actions (*"About to Reach Out"*, *"About to Lunge / Strike"*, *"About to Bend / Pick Up"*, *"About to Turn"*, *"About to Rise"*).
- 🌐 **Live Web Dashboard**: Flask + SocketIO real-time monitoring interface at `http://localhost:5000`.

---

## Benchmark Metrics

| Metric | Performance |
|---|---|
| **Average Prediction Latency** | **0.05 ms** |
| **Maximum Frame Latency** | **0.14 ms** |
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
