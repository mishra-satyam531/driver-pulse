# Driver Pulse
> **Real-time telematics analytics engine for Uber drivers — stress detection, earnings forecasting, and empathetic AI coaching.**

## Demo & Deployment
- **Demo Video:** [https://drive.google.com/file/d/1D5KNUXIQk0hExsqL1zQyBdQxZHp8uy93/view?usp=sharing]
- **Live Application:** [https://driver-pulse-uber-hackathon.streamlit.app/]

**Note to Judges:**  
All backend outputs are pre-generated and committed to the repo. The dashboard loads instantly with no pipeline setup required. To regenerate outputs from scratch, see the [Backend Pipelines](#backend-pipelines) section.

---

# Overview

This repository contains the **Driver Pulse system** built for the Uber Driver Pulse Hackathon.

The project combines:

- Driver stress detection from telemetry
- Real-time earnings velocity analysis
- Machine learning goal prediction
- GenAI-generated driver insights
- A Streamlit dashboard to visualize everything

The system analyzes **accelerometer data, audio intensity, and earnings logs** to identify stressful driving moments and determine whether a driver is on track to meet their earnings goal.

The **frontend dashboard displays these insights in a glanceable format** so drivers (and judges) can quickly understand their performance.

---

# System Architecture

The architecture follows a **decoupled backend + lightweight frontend design**.

## Core Components

### 1. Safety & Stress Engine
Analyzes **vehicle motion and cabin audio levels** to detect stressful events.

Key signals:
- Horizontal jerk
- Sudden braking
- Loud cabin noise

Outputs structured logs of flagged stress events.

---

### 2. Earnings Velocity Engine
Calculates **real-time earnings pace (₹/hr)** during a shift.

Handles edge cases such as:

- Cold start periods
- Shift end detection
- Low-data velocity noise

Outputs driver earnings velocity metrics.

---

### 3. ML Goal Predictor
A **Random Forest model** predicts whether a driver will reach their daily earnings goal.

Inputs include:
- Driver rating
- Experience level
- Earnings velocity
- Remaining shift time

Outputs:
- Forecast status (`ahead`, `on_track`, `at_risk`)
- ML confidence score

---

### 4. LLM Insights Engine
Uses **GenAI (Llama 3 via Groq)** to convert raw system outputs into **driver-friendly insights**.

Example:

Instead of:

Jerk > 4.0 detected


Driver sees:

> “We noticed a sudden stop earlier. Take it easy — smooth driving keeps trips comfortable.”

The LLM layer **does not decide events**, it only **translates them into supportive feedback**.

---

### 5. Driver Dashboard
A **Streamlit web application** that visualizes:

- Stress events
- Trip summaries
- Earnings velocity
- Goal predictions
- AI insights

The frontend **does not run ML models** — it simply reads **structured outputs from the backend**.

---

# Project Structure

```
driver-pulse/
├── app/
│   └── driver_pulse_app.py        # Streamlit dashboard (5 tabs)
├── backend/
│   ├── stress_model.py            # Physics-based stress detection engine
│   ├── earnings_velocity.py       # Real-time earnings velocity calculator
│   ├── goal_predictor.py          # Random Forest ML goal predictor
│   ├── driver_insights.py         # Groq LLM safety insight generator
│   ├── earnings_insights.py       # Groq LLM earnings coaching generator
│   └── api.py                     # FastAPI wrapper exposing backend endpoints
├── data/
│   ├── sensor_data/               # Raw accelerometer + audio CSVs
│   ├── processed_outputs/         # All backend JSON/CSV outputs
│   ├── earnings/                  # Earnings velocity logs + driver goals
│   └── drivers/                   # Driver metadata (name, city, rating)
├── utils/
│   └── seed_stress_data.py        # Generates synthetic telematics data
├── architecture.md                # System architecture diagram (Mermaid)
├── design_doc.md                  # Full design document
├── progress_log.md                # Development decisions log
├── requirements.txt
└── README.md
```

---

# Backend Pipelines

> **Note:** Pre-generated outputs are already committed. Run these only if you want to regenerate from scratch.

## 0. Seed Sensor Data

```bash
python utils/seed_stress_data.py
```

Generates `data/sensor_data/accelerometer_data.csv` and `audio_intensity_data.csv`.

---

## 1. Stress Detection

```bash
python -m backend.stress_model
```

Outputs:
```
data/processed_outputs/flagged_moments.json
data/processed_outputs/flagged_moments.csv
```

---

## 2. Earnings Velocity

```bash
python -m backend.earnings_velocity
```

Outputs:
```
data/processed_outputs/earnings_velocity_output.json
data/processed_outputs/trip_summaries.csv
```

---

## 3. ML Goal Predictor

```bash
python -m backend.goal_predictor
```

Trains a Random Forest classifier (300 trees, 5-fold stratified CV), saves model artifacts, and outputs:
```
data/processed_outputs/goal_predictions_output.json
data/processed_outputs/models/goal_model.pkl
data/processed_outputs/models/goal_label_encoder.pkl
```

---

## 4. Safety LLM Insights

```bash
python -m backend.driver_insights
```

Calls Groq `llama-3.1-8b-instant` for medium/high severity stress events. Output:
```
data/processed_outputs/trip_insights_final.json
```

---

## 5. Earnings Coaching Insights

```bash
python -m backend.earnings_insights
```

Generates personalized earnings coaching per driver. Output:
```
data/processed_outputs/earnings_insights_final.json
```


---

# Running the Dashboard

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Start the app:**
```bash
streamlit run app/driver_pulse_app.py
```

**Open in browser:**
```
http://localhost:8501
```

The dashboard has 5 tabs:
| Tab | Description |
|-----|-------------|
| Trip Summary | Fleet-level stress event overview + raw data export |
| Flagged Moments | Per-driver incident reports with GPS map + LLM insights + TTS |
| Earnings Velocity | Real-time earnings pace, goal gauge, and end-of-shift forecast |
| Test API | Live interactive model testing with JSON output |
| System Architecture | Engine documentation, formulas, and design rationale |


---

# Trade-offs & Assumptions

### Precomputed Outputs
The dashboard reads **pre-generated CSV/JSON logs** instead of running models in real time.

This ensures:
- Faster UI
- Simpler deployment
- Reliable hackathon demos

---

### Rule-Based Stress Detection
Stress detection uses **deterministic physics rules** rather than deep learning for:
- Explainability
- Debuggability
- Traceable thresholds

---

### Synthetic Dataset
Mock telematics data simulates scenarios such as:
- Harsh braking
- Loud cabin arguments
- False positives (potholes, sirens)

This helps demonstrate edge-case handling.

---

# Key Design Decisions

### Privacy-First Audio Processing
Raw audio **never leaves the device**. The edge node extracts only a rolling decibel mean — a single anonymous number. The cloud receives `85.2 dB`, not a conversation recording.

### Offline Resilience
Sensor batches are queued locally if connectivity drops. The backend uses `pd.merge_asof` with absolute Unix timestamps so delayed or out-of-order events still produce accurate stress insights.

### Explainable Outputs
Every flagged event includes `signal_type`, `raw_value`, `threshold`, and `explanation` — full traceability from sensor to decision. No black-box flags.

### LLM as Translator, Not Decider
The Groq LLM layer never classifies events — it only translates structured sensor output into empathetic driver language. Rule-based logic stays in Python; the model handles tone.

### Battery-Aware Architecture
Heavy compute (Random Forest, LLM) runs in the cloud. The phone sends small batched payloads every 30 seconds and renders a read-only JSON snapshot — no continuous WebSocket drain. 
