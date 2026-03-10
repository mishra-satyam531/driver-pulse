## Driver Pulse – Progress Log

This log captures the major development steps and design pivots for the Driver Pulse prototype.

---

### 2026-03-04 – Repository Initialization & Data Exploration

- Imported the official hackathon datasets and example outputs into a dedicated `driver-pulse` repository.
- Inspected `sensor_data`, `earnings`, `trips`, and `processed_outputs` folders to understand available signals and reference schemas.
- Identified the two core user-facing stories:
  - Stressful moments during trips (from accelerometer + audio).
  - Earnings velocity vs daily goals.

---

### 2026-03-04 – Stress Detection Rules & Mock Data

- Implemented `utils/seed_stress_data.py` to generate a synthetic, hour-long telematics stream with curated edge cases:
  - True harsh braking events.
  - Pothole-like vertical jolts.
  - Siren-style short-lived audio spikes.
  - Longer cabin arguments with high sustained noise.
- Built `backend/stress_model.py` to:
  - Load accelerometer and audio CSVs.
  - Compute engineered features (horizontal/vertical jerk, 15s rolling audio).
  - Fuse streams by `trip_id` and timestamp.
  - Apply explainable rules for `HARSH_MOTION`, `SUSTAINED_NOISE`, and `CRITICAL_CONFLICT`.
- Aggregated consecutive stress seconds into event blocks and exported:
  - `data/processed_outputs/flagged_moments.json`
  - `data/processed_outputs/flagged_moments.csv`

---

### 2026-03-05 – Event Scoring & Absorption Logic

- Added scoring logic to derive:
  - `motion_score`, `audio_score`, and `combined_score` in \[0, 1\].
  - `severity` buckets (low / medium / high).
- Implemented **event absorption**: conflict moments now absorb lesser flags within ±15 seconds to avoid duplicate alerts.
- Ensured each event record includes:
  - `flag_id`, `driver_id`, `trip_id`, timestamps, duration, GPS snapshot.
  - A human-readable `explanation` and `context` string.

---

### 2026-03-05 – LLM-Based Trip Insights Layer

- Created `backend/driver_insights.py` to add empathetic, one-line narratives for medium/high severity events.
- Integrated with an OpenAI-compatible LLM endpoint (Groq) using a strict system prompt:
  - No blame, no driving advice.
  - At most 1–2 sentences.
  - Only mention audio/motion when the underlying data justifies it.
- Wrote results to `data/processed_outputs/trip_insights_final.json` with an `llm_insight` field per event.

---

### 2026-03-06 – Earnings Velocity & Goals Wiring

- Loaded and sanity-checked `data/earnings/earnings_velocity_log.csv` and `data/earnings/driver_goals.csv`.
- Confirmed that each row encodes:
  - Cumulative earnings.
  - Current vs target hourly velocity.
  - `velocity_delta` and `forecast_status` (ahead / on_track / at_risk).
- Documented how these logs will be joined on `driver_id` and surfaced in the UI.

---

### 2026-03-07 – Driver Pulse Dashboard (Streamlit)

- Added `app/driver_pulse_app.py`, a Streamlit UI that sits on top of the existing processed outputs.
- Implemented four main tabs:
  - **Trip Overview:** aggregates flagged events by driver and trip, with counts and max stress scores.
  - **Flagged Moments:** driver/trip filters, bar chart of combined stress score over time, and a table with severity-aware styling and optional `llm_insight` column.
  - **Earnings & Goals:** driver selector, KPI metrics (current earnings, velocity vs target), line chart of earnings and velocities over time, and a detailed table.
  - **How this works:** short narrative tying together raw signals, heuristics, and UI decisions.
- Cached CSV/JSON loads with `st.cache_data` so repeated interactions stay responsive.

---

### 2026-03-07 – Documentation & Submission Prep

- Created `README.md` following the hackathon formatting requirements:
  - Top-level section for **Demo Video** and **Live Application** URLs.
  - Setup instructions, trade-offs, and app walkthrough.
- Wrote this `progress_log.md` to expose how the solution evolved over time.
- Authored `design_doc.md` summarizing product vision, stress detection and earnings algorithms, and the MVP cut line.

---

### Potential Future Iterations (Post-Hackathon)

- Add per-trip summary cards (overall stress score, calm vs tense segments, top moments).
- Let drivers provide feedback on events (useful / not useful) and collect those labels for future model tuning.
- Experiment with lightweight ML on top of the engineered features while preserving explainability.

