# Driver Pulse

## Demo & Deployment
- **Demo Video:** [Insert Unlisted YouTube or Google Drive URL]
- **Live Application:** [Insert Streamlit / Cloud deployment URL]

**Optional Judge Login**
- Username: judge@uber.com  
- Password: hackathon2026

**Note to Judges:**  
The application may take **30–60 seconds to start** on free-tier hosting.

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


backend/
stress_model.py
earnings_velocity.py
goal_predictor.py
driver_insights.py
earnings_insights.py

utils/
seed_stress_data.py

data/
sensor_data/
processed_outputs/
earnings/
drivers/

app/
driver_pulse_app.py

README.md
design_doc.md
progress_log.md
requirements.txt


---

# Backend Pipelines

## Stress Detection

Run:


python backend/stress_model.py


Output:


data/processed_outputs/flagged_moments.json
data/processed_outputs/stress_analysis_output.json


---

## Earnings Velocity

Run:


python backend/earnings_velocity.py


Output:


data/processed_outputs/earnings_velocity_output.json
data/processed_outputs/trip_summaries.csv


---

## ML Goal Predictor

Run:


python backend/goal_predictor.py


This script automatically:

- Trains a Random Forest model
- Saves `goal_model.pkl`
- Saves `goal_label_encoder.pkl`

Outputs:


goal_predictions_output.json


---

## Safety LLM Insights

Run:


python backend/driver_insights.py


Output:


trip_insights_final.json


---

## Earnings Coaching Insights

Run:


python backend/earnings_insights.py


Output:


earnings_insights_final.json


---

# Running the Dashboard

Start the frontend:


streamlit run app/driver_pulse_app.py


Open in browser:


http://localhost:8501


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

# Next Steps

### Frontend
- Improve dashboard design
- Add additional charts
- Improve mobile responsiveness

### ML Team
- Tune thresholds in stress detection
- Improve goal prediction accuracy
- Add new behavioral features

### Demo Preparation
Record a **2–3 minute walkthrough video**:

0:00–0:30 – Architecture overview  
0:30–2:00 – Live dashboard walkthrough  
2:00–3:00 – Explain backend outputs and models