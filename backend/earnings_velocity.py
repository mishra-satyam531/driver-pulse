"""
earnings_velocity.py
Provides real-time earnings velocity features.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
BASE_DIR     = Path(__file__).resolve().parents[1]
DATA_DIR     = BASE_DIR / "data"
OUTPUT_DIR   = DATA_DIR / "processed_outputs"
OUTPUT_PATH  = OUTPUT_DIR / "earnings_velocity_output.json"
SUMMARY_PATH = OUTPUT_DIR / "trip_summaries.csv"

# Thresholds
COLD_START_HOURS = 0.25   # suppress forecast for first 15 min as it is too unreliable
AHEAD_BUFFER     = 1.05   # require 5% buffer to call "ahead"

# Data loading
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    goals   = pd.read_csv(DATA_DIR / "earnings" / "driver_goals.csv")
    vel_log = pd.read_csv(DATA_DIR / "earnings" / "earnings_velocity_log.csv")
    drivers = pd.read_csv(DATA_DIR / "drivers"  / "drivers.csv")
    
    # DATA CLEANING: Earnings - Remove negative earnings and unrealistic values
    if 'cumulative_earnings' in vel_log.columns:
        vel_log['cumulative_earnings'] = vel_log['cumulative_earnings'].clip(lower=0)
        print("DATA CLEANING: Earnings clipped to remove negative values")
    
    # DATA CLEANING: Goals - Ensure target earnings are positive and realistic
    if 'target_earnings' in goals.columns:
        goals['target_earnings'] = goals['target_earnings'].clip(lower=50, upper=10000)
        print("DATA CLEANING: Target earnings clipped to [50, 10000] range for realism")
    
    # DATA CLEANING: Time - Remove invalid elapsed hours
    if 'elapsed_hours' in vel_log.columns:
        vel_log['elapsed_hours'] = vel_log['elapsed_hours'].clip(lower=0, upper=24)
        print("DATA CLEANING: Elapsed hours clipped to [0, 24] range")
    
    return goals, vel_log, drivers

# Timestamp parsing
def parse_timestamp(ts_str: str, date_str: str) -> pd.Timestamp:
    ts_str = str(ts_str).strip()
    if " " in ts_str or "T" in ts_str:
        return pd.to_datetime(ts_str, errors="coerce")
    return pd.to_datetime(f"{date_str} {ts_str}", errors="coerce")

# Velocity computations
def compute_current_velocity(earned: float, elapsed_h: float) -> float | None:
    if elapsed_h < COLD_START_HOURS:
        return None
    return round(earned / elapsed_h, 2) if elapsed_h > 0 else 0.0

def compute_target_velocity(target: float, earned: float, remaining_h: float) -> float | None:
    # None when shift is over should be treated as "missed".
    needed = target - earned
    if needed <= 0:
        return 0.0
    if remaining_h <= 0:
        return None
    return round(needed / remaining_h, 2)

def forecast_status(
    current_v: float | None,
    target: float,
    earned: float,
    remaining_h: float,
    elapsed_h: float,
) -> str:
    """
    Rule-based forecast of goal attainment.
    Priority order:
      1. earned >= target -> "achieved"
      2. elapsed < 15 min or velocity unknown -> "insufficient_data"
      3. shift over, goal not met -> "at_risk"
      4. projected >= target * 1.05 -> "ahead"
      5. projected >= target -> "on_track"
      6. otherwise -> "at_risk"
    """
    if earned >= target:
        return "achieved"
    if elapsed_h < COLD_START_HOURS or current_v is None:
        return "insufficient_data"
    if remaining_h <= 0:
        return "at_risk"

    projected = earned + current_v * remaining_h
    if projected >= target * AHEAD_BUFFER:
        return "ahead"
    if projected >= target:
        return "on_track"
    return "at_risk"

# Batch processing
def compute_velocity_metrics(
    goals: pd.DataFrame,
    vel_log: pd.DataFrame,
    drivers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich every velocity-log row with computed metrics and a forecast label.
    """
    # One goal row per driver (first occurrence wins)
    primary_goals = goals.drop_duplicates(subset="driver_id", keep="first")

    merged = (
        vel_log
        .merge(primary_goals[["driver_id", "target_earnings", "shift_end_time"]], on="driver_id", how="left")
        .merge(drivers[["driver_id", "city", "experience_months", "rating"]], on="driver_id", how="left")
    )

    records = []
    for _, row in merged.iterrows():
        date_str   = str(row.get("date", "2024-02-06"))
        current_ts = parse_timestamp(row["timestamp"], date_str)
        shift_end  = pd.to_datetime(f"{date_str} {row.get('shift_end_time', '20:00:00')}", errors="coerce")

        if pd.isnull(current_ts) or pd.isnull(shift_end):
            continue

        elapsed_h   = float(row.get("elapsed_hours", 0))
        earned      = float(row.get("cumulative_earnings", 0))
        target      = float(row.get("target_earnings", 1400))
        trips       = int(row.get("trips_completed", 0))
        remaining_h = max((shift_end - current_ts).total_seconds() / 3600, 0.0)

        current_v = compute_current_velocity(earned, elapsed_h)
        target_v  = compute_target_velocity(target, earned, remaining_h)
        delta     = round(current_v - target_v, 2) if (current_v is not None and target_v is not None) else None
        projected = round(earned + current_v * remaining_h, 2) if current_v is not None else None
        status    = forecast_status(current_v, target, earned, remaining_h, elapsed_h)

        records.append({
            # identifiers
            "log_id":                   row.get("log_id", ""),
            "driver_id":                row.get("driver_id", ""),
            "city":                     row.get("city", ""),
            # time
            "timestamp":                current_ts.isoformat(),
            "elapsed_hours":            round(elapsed_h, 3),
            "remaining_hours":          round(remaining_h, 3),
            # earnings progress
            "cumulative_earnings":      earned,
            "target_earnings":          target,
            "pct_to_goal":              round(earned / target * 100, 1) if target > 0 else 0.0,
            "trips_completed":          trips,
            "trips_per_hour":           round(trips / elapsed_h, 2) if elapsed_h > 0 else 0.0,
            # velocity metrics
            "current_velocity":         round(current_v, 2) if current_v is not None else None,
            "target_velocity":          round(target_v, 2) if target_v is not None else None,
            "velocity_delta":           round(delta, 2) if delta is not None else None,
            "projected_final_earnings": round(projected, 2) if projected is not None else None,
            # forecast
            "forecast_status":          status,
            # output schema 
            "signal_type":              "EARNINGS",
            "raw_value":                current_v,
            "event_label":              status.upper(),
            "threshold":                target_v,
        })

    return pd.DataFrame(records)

# Trip summaries
def build_trip_summaries(velocity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the velocity log into one summary row per driver per date.
    Produces trip_summaries.csv.
    """
    if velocity_df.empty:
        return pd.DataFrame()

    df = velocity_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    def _summarise(grp: pd.DataFrame) -> dict:
        last = grp.sort_values("timestamp").iloc[-1]
        return {
            "driver_id":                  last["driver_id"],
            "city":                       last.get("city", ""),
            "date":                       str(last["date"]),
            "total_elapsed_hours":        round(float(last["elapsed_hours"]), 2),
            "total_earnings":             float(last["cumulative_earnings"]),
            "target_earnings":            float(last["target_earnings"]),
            "pct_to_goal":                float(last["pct_to_goal"]),
            "total_trips":                int(last["trips_completed"]),
            "avg_earnings_per_hour":      round(float(last["current_velocity"]), 2) if last["current_velocity"] is not None else 0.0,
            "projected_final_earnings":   round(float(last["projected_final_earnings"]), 2) if last["projected_final_earnings"] is not None else 0.0,
            "final_forecast":             last["forecast_status"],
        }

    return pd.DataFrame(
        df.groupby(["driver_id", "date"], group_keys=False).apply(_summarise).tolist()
    )

# Export
def export_velocity_output(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        OUTPUT_PATH.write_text("[]", encoding="utf-8")
        return
    OUTPUT_PATH.write_text(
        json.dumps(df.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  Velocity log -> {OUTPUT_PATH}")

def export_trip_summaries(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        SUMMARY_PATH.write_text("", encoding="utf-8")
        return
    df.to_csv(SUMMARY_PATH, index=False)
    print(f"  Trip summaries -> {SUMMARY_PATH}")

# Entrypoint
def run_earnings_velocity_model() -> pd.DataFrame:
    """
    Load data, compute velocity metrics, export outputs, return velocity DataFrame
    (used downstream by goal_predictor.py).
    """
    goals, vel_log, drivers = load_data()
    df = compute_velocity_metrics(goals, vel_log, drivers)
    export_velocity_output(df)
    summaries = build_trip_summaries(df)
    export_trip_summaries(summaries)

    if not df.empty:
        print(f"  Earnings velocity: {len(df)} events processed")
        print(f"  Forecast distribution:\n{df['forecast_status'].value_counts().to_string()}")

    return df

if __name__ == "__main__":
    run_earnings_velocity_model()
