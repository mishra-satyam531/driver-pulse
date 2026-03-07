# Driver Pulse: Backend Architecture

This repository contains the backend data processing pipelines for the Driver Pulse system. The architecture is designed as a set of decoupled Python microservices that process raw telemetry and earnings logs, outputting structured data for the Frontend App to consume.

## System Architecture

The system utilizes a **decoupled, feature-driven architecture** to allow independent development and deployment across the stack:

1. **Safety & Stress Engine:** Uses deterministic physics and audio thresholds to flag critical driver stress events.
2. **Earnings & Forecasting Engine:** Uses a hybrid approach (real-time math rules + Machine Learning) to predict if a driver will hit their daily earning target.
3. **LLM Insights Engine:** Translates raw safety warnings into supportive, personalized feedback for drivers using GenAI.
4. **Frontend App (Driver Dashboard):** A standalone application that simply reads from the `processed_outputs` folder to render traffic lights, stress warnings, and confidence bars.

This separation of concerns ensures the frontend does not need to compute ML or physics—it requires only API-like JSON files.

---

## Core Backend Modules

### 1. `backend/earnings_velocity.py` (Rule-Based Pace Engine)
Transforms raw shift logs into a real-time pace signal (₹/hr). 
* **The "Why":** Instead of showing drivers their earnings only at the end of the day, a running velocity is calculated. It handles critical edge cases like "cold starts" (suppressing unreliable velocity in the first 15 mins) and shift-end cutoffs.
* **Output:** `earnings_velocity_output.json` & `trip_summaries.csv`

### 2. `backend/goal_predictor.py` (ML Forecasting Layer)
Adds a Machine Learning layer on top of the rule-based velocity measurements.
* **The "Why":** Driver goal achievement is non-linear. A rule might indicate a driver is "on track", but a lightweight Random Forest model learns from historical data (experience, rating, time left) to attach a **probabilistic confidence score**.
* **Output:** `goal_predictions_output.json` (enriched velocity log)
* **Note on Models:** This script automatically trains the ML model (`goal_model.pkl`) and label encoder (`goal_label_encoder.pkl`) and saves them to `data/processed_outputs/models/`. **These `.pkl` files are generated on the fly** and are rebuilt every time the script runs on new data.

### 3. `backend/stress_model.py` (Safety & Stress Engine)
Analyzes telemetry (Jerk) and Audio levels to detect stress.
* **The "Why":** Uses physics heuristics (e.g., Horizontal Jerk > 4.0, Audio > 85dB) to flag absolute critical conflicts. ML is purposefully *not* used here because physics limits are deterministic.
* **Output:** `flagged_moments.json` & `stress_analysis_output.json`

### 4. `backend/driver_insights.py` (LLM Context Layer - Safety)
Connects to an LLM (Llama 3 via Groq) to enrich the flagged safety events with human-readable, supportive feedback.
* **The "Why":** Raw telemetry (e.g., "Jerk > 4.0") is stressful for a driver to read. Instead of robotic errors, the GenAI agent scales down the aggression and prints a supportive 1-2 sentence message (e.g., "We noticed a sudden stop, Alex. Safety first!").
* **Output:** `trip_insights_final.json`

### 5. `backend/earnings_insights.py` (LLM Context Layer - Earnings)
Connects to the same LLM to provide motivational financial coaching based on the ML predictions.
* **The "Why":** Rather than telling a driver "Your velocity ratio is 0.8 / At Risk", the LLM translates the prediction into a gentle, purely positive nudge (e.g., "You're slightly behind your pace, Alex. Try moving toward downtown for higher demand.").
* **Output:** `earnings_insights_final.json`

---

## 🚀 How to Run

Because the system is fully modular, the engines can be run independently in any order. The output files will be written directly to `data/processed_outputs/` for the frontend to consume.

Run from the project root:

```bash
# 1. Run the Safety/Stress Pipeline
python backend/stress_model.py

# 2. Run the Earnings Velocity Engine
python backend/earnings_velocity.py

# 3. Train and Run the ML Goal Predictor
python backend/goal_predictor.py

# 4. Generate AI Notifications from Safety Events
python backend/driver_insights.py

# 5. Generate AI Financial Coaching from Earnings Predictions
python backend/earnings_insights.py
```

## 📊 Data Contracts (Outputs)
All finalized data for the Frontend App is dropped into `/data/processed_outputs/`. 
The frontend expects these specific structures:
- `stress_analysis_output.json`: Array of events with `is_stress_event` flags and exact timestamps.
- `trip_summaries.csv`: One summary row per driver per shift date.
- `goal_predictions_output.json`: Rich event log containing `forecast_status` (explainable rule label) and `ml_confidence` (0.0 to 1.0 score for the UI).
- `trip_insights_final.json`: Generative AI contextual notifications derived from safety flags.
- `earnings_insights_final.json`: Generative AI motivational coaching derived from ML goal predictions.
