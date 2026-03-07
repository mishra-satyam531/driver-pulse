Driver Pulse: Team [Your Team Name]
- Demo Video: [Insert Unlisted YouTube or open Google Drive URL here]
- Live Application: [Insert Streamlit / cloud deployment URL here]
- (Optional) Judge Login Credentials: Username: judge@uber.com \| Password: hackathon2026
- Note to Judges: [Insert spin-up warning here, e.g., "The app may take 30–60 seconds to start on free-tier hosting."]

---

## Overview

This repository contains the **Driver Pulse engine and dashboard** for the Uber Driver Pulse hackathon.
Your teammates' model ingests raw accelerometer and audio intensity data, applies heuristics to detect stressful driving moments, and generates transparent logs of flagged events and earnings velocity.
This web app surfaces those insights in a **driver-facing, glanceable dashboard** that lets judges trace every UI element back to a structured output record.

The project is organized into three main layers:
- **Data & Models** (`backend/`, `utils/`, `data/`): Scripts to seed mock telematics, detect stress moments, and generate natural-language insights.
- **Processed Outputs** (`data/processed_outputs/`, `data/earnings/`): CSV/JSON logs of flagged moments and earnings velocity.
- **Driver Dashboard** (`app/driver_pulse_app.py`): Streamlit-based web UI that visualizes stress flags and earnings goals for each driver.

Judges should be able to understand what your system does within a few minutes of using the dashboard or watching the demo video.

---

## Live Deployment

Once deployed, a judge should be able to:
1. Open the **Live Application** link above.
2. Select a driver and trip.
3. See **flagged stress moments** on a timeline and in a structured table.
4. Inspect **earnings velocity** versus goal and whether the driver is ahead, on track, or at risk.

Recommended hosting options (per hackathon brief):
- **Streamlit Community Cloud / Hugging Face Spaces** for this Python dashboard.
- Alternatively: any public cloud host that can run `streamlit run app/driver_pulse_app.py`.

If your deployment may sleep on free tiers, add a short note at the top of this README (for example, "The Render backend may take ~60 seconds to wake up").

---

## Local Setup Instructions

### Prerequisites

- Python 3.10+ installed.
- `pip` package manager.
- (Optional) A virtual environment tool such as `venv` or `conda`.

### 1. Clone the Repository

```bash
git clone <your-repo-url>.git
cd driver-pulse
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS / Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate or Refresh Mock Data (Optional)

If you want to regenerate the mock telematics data locally:

```bash
python utils/seed_stress_data.py
python backend/stress_model.py
python backend/driver_insights.py  # Requires OPENAI_API_KEY / GROQ key if you want fresh LLM insights
```

This pipeline will:
- Produce synthetic accelerometer and audio streams.
- Run the stress detection rules to create `data/processed_outputs/flagged_moments.csv` and `.json`.
- Use the LLM helper to add empathetic one-line insights per medium/high event in `trip_insights_final.json`.

### 5. Run the Driver Pulse Dashboard Locally

```bash
streamlit run app/driver_pulse_app.py
```

Then open the URL shown in your terminal (by default, `http://localhost:8501`).

---

## App Walkthrough

The Streamlit app organizes the experience into four tabs:

- **Trip Overview**  
  High-level table of trips with:
  - First/last timestamp.
  - Number of flagged events.
  - Count of high-severity flags.
  - Maximum combined stress score.

- **Flagged Moments**  
  For a selected driver and trip, this tab shows:
  - A bar chart of **combined stress score** over time.
  - A structured table with:
    - `timestamp`, `flag_type`, `severity`, `motion_score`, `audio_score`, `combined_score`.
    - `explanation` and `context` (extracted from model output).
    - (If available) `llm_insight` – a one-line, empathetic, driver-facing explanation.
  Severity cells are color-coded (green / amber / red) to pass the "glanceable" test.

- **Earnings & Goals**  
  Uses `data/earnings/earnings_velocity_log.csv` and `data/earnings/driver_goals.csv` to show:
  - Current cumulative earnings and earnings velocity.
  - Target velocity to hit the driver’s goal by the end of the shift.
  - `velocity_delta` and `forecast_status` (ahead / on_track / at_risk).
  - A time-series chart of cumulative earnings and velocity vs target.

- **How this works**  
  A concise explanation of how the engine fuses motion, audio, and earnings data into driver-facing insights.

---

## Trade-offs & Assumptions

- **Precomputed pipeline over real-time streaming**  
  For hackathon simplicity and reliability, the dashboard reads from **precomputed CSV/JSON outputs** produced by the stress detection scripts.
  This avoids running heavy models on every UI interaction and keeps the system easy to demo and deploy.

- **Single-trip mock telematics**  
  The provided synthetic telematics focuses on a single `mock_trip_001` with carefully constructed edge cases (harsh braking, cabin arguments, false positives).
  This lets us demonstrate nuanced behavior (like vetoing potholes and sirens) without requiring hours of raw data.

- **Rule-based stress detection**  
  The core engine uses engineered features (jerk, rolling audio windows) and threshold rules instead of a black-box deep learning model.
  This is intentional for **explainability**: every flag can be traced back to clear thresholds and metrics.

- **LLM insights as a thin layer**  
  The optional LLM layer in `backend/driver_insights.py` never decides whether an event is stressed or not.
  It only translates already-flagged events into short, empathetic sentences for the driver.

Add any additional assumptions you and your teammates made about the data (for example, units, sampling rates, or decisions to ignore certain edge cases).

---

## Project Structure

```text
backend/
  stress_model.py        # Ingests sensor CSVs, computes features, and exports flagged_moments.csv/json
  driver_insights.py     # Adds short LLM-generated insights per stress event

utils/
  seed_stress_data.py    # Generates a synthetic telematics dataset for local testing

data/
  sensor_data/           # Raw accelerometer and audio intensity CSVs
  processed_outputs/     # Model outputs (flagged_moments.csv/json, trip_insights_final.json)
  earnings/              # Earnings velocity and goal CSVs
  drivers/               # Driver metadata (names, IDs, etc.)

app/
  driver_pulse_app.py    # Streamlit dashboard used for the live web demo

README.md                # This file (engineering handoff & submission map)
design_doc.md            # Detailed product & algorithm design
progress_log.md          # Development history log for judges
requirements.txt         # Python dependencies (including Streamlit)
```

---

## Next Steps for the Team

- **You / Web dev**  
  - Wire in any additional outputs from the ML team (e.g., per-trip stress scores, new event types).
  - Polish the visual design and labels to match the story in your demo video.

- **ML teammates**  
  - Fine-tune thresholds and features in `backend/stress_model.py`.
  - Extend earnings velocity logic or add new goal types if desired.

- **For demo & submission**  
  - Record a 2–3 minute walkthrough following the hackathon video script:
    - 0:00–0:30 – Architecture and approach.
    - 0:30–2:00 – Live UI walkthrough (flagged events + earnings).
    - 2:00–3:00 – Under the hood (point directly to the CSV/JSON logs).
  - Paste the video and deployment URLs into the header of this README before submission.

