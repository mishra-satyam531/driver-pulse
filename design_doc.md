## Driver Pulse – Design Document

### 1. Product Vision

**Persona:** Alex, a typical Uber driver who juggles navigation, rider interactions, and earnings goals during long shifts.

**Problem:** Today, Alex only sees a final earnings number at the end of the shift. She does not know:
- When a moment during the trip was unusually stressful or risky.
- Whether she is pacing ahead of or behind her daily earnings goal in the middle of a shift.
- How raw telematics data (accelerometer, audio intensity) translates into meaningful, actionable feedback.

**Solution:** Driver Pulse converts messy sensor logs into **glanceable, driver-friendly summaries**:
- Surfaces a small set of **flagged moments** that combine motion and cabin audio signals.
- Shows **earnings velocity** vs goal so Alex can see if she is ahead, on track, or at risk.
- Provides short, empathetic insights describing what happened without judging the driver.

The UI is designed to be **explainable to another engineer** and **understandable within seconds** to a non-technical judge.

---

### 2. Stress Detection Algorithm

#### 2.1 Ingestion & Cleaning

Source files:
- `data/sensor_data/accelerometer_data.csv`
- `data/sensor_data/audio_intensity_data.csv`

Key steps (see `backend/stress_model.py`):
- Parse timestamps as timezone-aware datetimes and floor to the nearest second.
- Drop rows with invalid timestamps.
- Sort each stream by `trip_id` and `timestamp` for consistent windowing.

#### 2.2 Feature Engineering

**Motion features**
- Compute **horizontal magnitude** from X/Y axes:
  - \( \text{horizontal\_magnitude} = \sqrt{accel\_x^2 + accel\_y^2} \)
- Compute **Horizontal_Jerk** as the per-second difference within a trip:
  - \( \text{Horizontal\_Jerk}_t = \text{horizontal\_magnitude}_t - \text{horizontal\_magnitude}_{t-1} \)
- Detrend gravity from Z-axis (`accel_z_adj = accel_z - 9.8`) and compute **Vertical_Jerk** as the absolute diff.

**Audio features**
- Clip raw audio levels into a realistic band `[30, 120] dB`.
- Compute a **15-second rolling mean** of clipped audio per trip to capture sustained cabin noise.

#### 2.3 Sensor Fusion

We perform an as-of merge between accelerometer and audio streams:
- Group audio by `trip_id` and merge into the motion stream using `merge_asof` on `timestamp` with a ±60s tolerance.
- This produces a **fused dataframe** where each motion sample is enriched with nearby audio context (`Audio_Rolling_15s`, `audio_class`).

#### 2.4 Rule-Based Stress Flags

We use simple, explainable rules instead of opaque ML:
- **Harsh motion:** `Horizontal_Jerk > 4.0` and `Vertical_Jerk < 2.0` (filters out potholes / vertical jolts).
- **Sustained noise:** `Audio_Rolling_15s > 85 dB` and `audio_class == "argument"`.
- **Critical conflict:** both harsh motion and sustained noise at once.

Each second is labeled using these rules; rows with `Stress_Flag` are kept as raw stress samples.

#### 2.5 Event Aggregation & Scoring

Sensor samples can fire for multiple consecutive seconds. To avoid overwhelming the driver:
- We group consecutive seconds with the same `Stress_Flag`, same `trip_id`, and <1s gap into **event blocks**.
- For each block we compute:
  - Start timestamp and elapsed time.
  - Duration in seconds.
  - Max horizontal/vertical jerk and max rolling audio within the block.
  - A categorical **flag_type**:
    - `harsh_braking`, `audio_spike`, or `conflict_moment`.
  - A synthetic **context** string (`"Motion: harsh_braking | Audio: audio_spike"`).

We then derive scores:
- **motion_score** scales `Horizontal_Jerk` into \[0, 1\] above the 4.0 m/s² threshold.
- **audio_score** scales `Audio_Rolling_15s` into \[0, 1\] above the 85 dB threshold.
- **combined_score** is the max of the two.
- **severity** buckets:
  - `high` if combined_score ≥ 0.7,
  - `medium` if ≥ 0.4,
  - otherwise `low`.

#### 2.6 Event Absorption Logic

To avoid double-counting during chaotic segments:
- For every `conflict_moment`, we remove any `harsh_braking` or `audio_spike` events that fall within ±15 seconds of that conflict.
- This ensures the driver only sees a **single, richer event** instead of a burst of overlapping flags.

The final result is exported as:
- `data/processed_outputs/flagged_moments.csv`
- `data/processed_outputs/flagged_moments.json`

Each row is a single, human-explainable event that the UI can point to.

---

### 3. Earnings Velocity Algorithm

Source files:
- `data/earnings/earnings_velocity_log.csv`
- `data/earnings/driver_goals.csv`

For each driver:
- `earnings_velocity_log.csv` tracks:
  - `cumulative_earnings`
  - `elapsed_hours`
  - `current_velocity` (earnings per hour)
  - `target_velocity`
  - `velocity_delta = current_velocity - target_velocity`
  - `forecast_status` (ahead / on_track / at_risk)

- `driver_goals.csv` defines:
  - Goal earnings, goal hours, shift start/end.
  - Current earnings and hours at the time of logging.
  - `earnings_velocity` and `goal_completion_forecast`.

The dashboard:
- Displays the most recent row per driver as a **snapshot** of how they are pacing.
- Plots cumulative earnings and velocities over time to show trajectory.
- Surfaces `velocity_delta` and `forecast_status` in simple language (e.g., "ahead of pace by ₹80/hr").

Edge cases:
- If logs for a driver are missing or sparse, the UI degrades gracefully by showing "—" instead of numbers and an info message.

---

### 4. Earnings Velocity + Stress Fusion in the UI

We deliberately **decouple** stress detection and earnings forecasting in the engine, but present them together in the dashboard:
- The **Flagged Moments** tab answers: *"What were the stressful or risky moments on this trip?"*
- The **Earnings & Goals** tab answers: *"Given how my shift is going, am I on pace to hit my goal?"*

This separation keeps each metric explainable, while still allowing a judge to:
- Filter to a driver, pick a trip, and inspect moments of stress.
- Then switch to the earnings tab to see whether those moments tend to happen when the driver is behind/under pressure.

---

### 5. Execution Strategy & MVP Cut Line

**MVP scope (what we built):**
- Single, well-instrumented mock trip with curated edge cases.
- Rule-based stress detection with transparent thresholds and engineered features.
- Earnings velocity viewed per driver, based on provided logs.
- Streamlit dashboard with:
  - Trip overview.
  - Flagged moment drill-down.
  - Earnings vs goal visualization.
  - Plain-language explanation tab.

**Cut line (what we intentionally did NOT build):**
- Real-time mobile SDK or on-device processing.
- Per-driver personalization of thresholds or models.
- Full-fledged multi-day history, streaks, or gamification.
- Complex ML models (e.g., RNNs / transformers) over time-series; our focus is explainability.

**Rollout path (if this were a real product):**
1. **Closed beta** with a few drivers using a post-trip review screen that summarizes their day.
2. **Collect feedback** on false positives/negatives and refine thresholds, window sizes, and absorption logic.
3. **Gradual integration** into in-app notifications or real-time prompts if drivers find value and trust the signals.
4. **Longer-term**: experiment with lightweight ML classifiers on top of the engineered features, but keep human-explainable labels in the loop.

---

### 6. Risks and Mitigations

- **False positives due to noisy data**  
  - Mitigation: use vertical jerk to filter out potholes; aggregate into events; absorb minor events near a conflict moment.

- **Driver trust and perception**  
  - Mitigation: use empathetic, non-judgmental LLM phrasing; never assign blame; keep explanations short and factual.

- **Performance & reliability in hackathon environment**  
  - Mitigation: precompute outputs as CSV/JSON; the UI performs only light filtering and plotting.

---

### 7. How to Extend This Design

If the team has more time after the hackathon:
- Add **per-trip summary cards** (overall stress score, calm vs tense segments, top 3 moments).
- Incorporate **driver feedback** flags to label events as "useful" or "noise" and feed that back into threshold tuning.
- Add **privacy-aware audio features** (e.g., local processing on device, no raw audio upload) while preserving the same surface-level metrics.

