import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ACCEL_PATH = DATA_DIR / "sensor_data" / "accelerometer_data.csv"
AUDIO_PATH = DATA_DIR / "sensor_data" / "audio_intensity_data.csv"
OUTPUT_DIR = DATA_DIR / "processed_outputs"
OUTPUT_PATH = OUTPUT_DIR / "flagged_moments.json"


def load_sensor_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load sensor data from CSV files with error handling."""
    # Check if files exist
    if not ACCEL_PATH.exists() or not AUDIO_PATH.exists():
        print("Error: Please ensure sensor CSV files exist in the data/sensor_data/ directory.")
        print(f"Expected files:")
        print(f"  - {ACCEL_PATH}")
        print(f"  - {AUDIO_PATH}")
        raise FileNotFoundError("Sensor data files not found")
    
    try:
        accel_df = pd.read_csv(ACCEL_PATH)
        audio_df = pd.read_csv(AUDIO_PATH)
    except Exception as e:
        print(f"Error reading sensor CSV files: {e}")
        raise

    # DATA CLEANING: Accelerometer - Clip to physically realistic range [-20, 20] m/s²
    accel_cols = ['accel_x', 'accel_y', 'accel_z']
    for col in accel_cols:
        if col in accel_df.columns:
            accel_df[col] = accel_df[col].clip(lower=-20, upper=20)
    print("DATA CLEANING: Accelerometer data clipped to [-20, 20] m/s² to remove physical anomalies.")
    
    # DATA CLEANING: Accelerometer - Interpolate missing values
    accel_df[accel_cols] = accel_df.groupby('trip_id')[accel_cols].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    print("DATA CLEANING: Accelerometer data interpolated to fill missing values.")

    # Explicit timestamp conversion for both dataframes
    accel_df["timestamp"] = pd.to_datetime(accel_df["timestamp"], utc=True, errors="coerce")
    audio_df["timestamp"] = pd.to_datetime(audio_df["timestamp"], utc=True, errors="coerce")
    
    # Create exact-match merge columns BEFORE any formatting/rounding
    accel_df['merge_time'] = accel_df['timestamp'].dt.round('s')
    audio_df['merge_time'] = audio_df['timestamp'].dt.round('s')

    accel_df = accel_df.dropna(subset=["timestamp"])
    audio_df = audio_df.dropna(subset=["timestamp"])

    accel_df = accel_df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
    audio_df = audio_df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

    return accel_df, audio_df


def compute_motion_metrics(accel_df: pd.DataFrame) -> pd.DataFrame:
    df = accel_df.copy()

    df["horizontal_magnitude"] = np.hypot(df["accel_x"], df["accel_y"])
    df["Horizontal_Jerk"] = (
        df.groupby("trip_id", group_keys=False)["horizontal_magnitude"].diff()
    )

    df["accel_z_adj"] = df["accel_z"] - 9.8
    df["Vertical_Jerk"] = (
        df.groupby("trip_id", group_keys=False)["accel_z_adj"].diff().abs()
    )

    return df


def compute_audio_metrics(audio_df: pd.DataFrame) -> pd.DataFrame:
    df = audio_df.copy()

    # DATA CLEANING: Audio - Clip to [30, 120] dB to remove noise floor and sensor outliers
    df["audio_level_clipped"] = df["audio_level"].clip(lower=30, upper=120)
    print("DATA CLEANING: Audio levels clipped to [30, 120] dB range to remove noise floor and sensor outliers")

    # Use groupby + time-based rolling on a MultiIndex, then align by position.
    rolled = (
        df.set_index("timestamp")
        .groupby("trip_id")["audio_level_clipped"]
        .rolling("15s")
        .mean()
    )
    df["Audio_Rolling_15s"] = rolled.values

    return df


def fuse_sensors(accel_df: pd.DataFrame, audio_df: pd.DataFrame) -> pd.DataFrame:
    # 1. Standardize trip_ids so 'TRIP001' and 'trip_101' don't break the merge
    accel_df['trip_id'] = accel_df['trip_id'].astype(str).str.lower().str.replace('trip0', 'trip_')
    audio_df['trip_id'] = audio_df['trip_id'].astype(str).str.lower().str.replace('trip0', 'trip_')

    # 2. Select only needed audio columns
    audio_merge_cols = ["trip_id", "merge_time", "Audio_Rolling_15s", "audio_class"]
    audio_subset = audio_df[[c for c in audio_merge_cols if c in audio_df.columns]].copy()

    # 3. Sort by time (STRICT REQUIREMENT for merge_asof)
    accel_df = accel_df.sort_values('merge_time')
    audio_subset = audio_subset.sort_values('merge_time')

    # 4. PROFESSIONAL SENSOR FUSION: Snap the closest audio reading to the accel reading
    fused = pd.merge_asof(
        accel_df,
        audio_subset,
        on='merge_time',
        by='trip_id',
        direction='backward' # Uses the most recent audio reading for that exact second
    )
    
    # 5. Clean up temporary columns
    fused = fused.drop(columns=['merge_time'])
    
    # 6. Forward-fill any gaps so the audio score persists across the whole 30-second window
    fused['Audio_Rolling_15s'] = fused.groupby('trip_id')['Audio_Rolling_15s'].ffill().fillna(50.0)
    fused['audio_class'] = fused.groupby('trip_id')['audio_class'].ffill().fillna('normal')
    
    return fused


def apply_stress_rules(fused_df: pd.DataFrame) -> pd.DataFrame:
    df = fused_df.copy()

    harsh_motion = (df["Horizontal_Jerk"] > 4.0) & (df["Vertical_Jerk"] < 2.0)
    sustained_noise = (df["Audio_Rolling_15s"] > 85) & (df["audio_class"] == "argument")
    critical_conflict = harsh_motion & sustained_noise

    conditions = [critical_conflict, harsh_motion, sustained_noise]
    choices = ["CRITICAL_CONFLICT", "HARSH_MOTION", "SUSTAINED_NOISE"]

    df["Stress_Flag"] = np.select(conditions, choices, default=None)

    flagged = df[df["Stress_Flag"].notna()].copy()
    if flagged.empty:
        return flagged

    keep_cols = [
        "trip_id",
        "driver_id",
        "timestamp",
        "elapsed_sec",
        "speed_kmh",
        "gps_lat",
        "gps_lon",
        "Horizontal_Jerk",
        "Vertical_Jerk",
        "Audio_Rolling_15s",
        "audio_class",
        "Stress_Flag",
    ]
    existing_cols = [c for c in keep_cols if c in flagged.columns]
    return flagged[existing_cols]


def export_flagged(flagged_df: pd.DataFrame) -> pd.DataFrame:
    """Export flagged moments to JSON and CSV files with Uber compliance schema.
    
    Args:
        flagged_df: DataFrame with flagged stress events
        
    Returns:
        pd.DataFrame: Final processed DataFrame with all compliance columns
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if flagged_df.empty:
        OUTPUT_PATH.write_text("[]", encoding="utf-8")
        return

    df = flagged_df.copy()

    # Normalize timestamp to timezone-aware datetime before formatting.
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    df = df.dropna(subset=["timestamp", "Stress_Flag"]).copy()
    df = df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

    # Collapse consecutive rows with the same Stress_Flag into event blocks.
    prev_trip = df["trip_id"].shift()
    prev_flag = df["Stress_Flag"].shift()
    prev_ts = df["timestamp"].shift()

    trip_change = df["trip_id"].ne(prev_trip)
    flag_change = df["Stress_Flag"].ne(prev_flag)
    time_gap = (df["timestamp"] - prev_ts).gt(pd.Timedelta(seconds=1))

    event_start = trip_change | flag_change | time_gap
    df["event_block_id"] = event_start.cumsum()

    def _duration_seconds(ts: pd.Series) -> int:
        t0 = ts.iloc[0]
        t1 = ts.iloc[-1]
        return int((t1 - t0).total_seconds()) + 1

    aggregated = (
        df.groupby(["trip_id", "driver_id", "Stress_Flag", "event_block_id"], as_index=False)
        .agg(
            driver_id=("driver_id", "first"),
            timestamp=("timestamp", "first"),
            elapsed_sec=("elapsed_sec", "first"),
            speed_kmh=("speed_kmh", "first"),
            gps_lat=("gps_lat", "first"),
            gps_lon=("gps_lon", "first"),
            Horizontal_Jerk=("Horizontal_Jerk", "max"),
            Vertical_Jerk=("Vertical_Jerk", "max"),
            Audio_Rolling_15s=("Audio_Rolling_15s", "max"),
            audio_class=("audio_class", "first"),
            duration_seconds=("timestamp", _duration_seconds),
        )
        .drop(columns=["event_block_id"])
    )

    aggregated = aggregated.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

    flag_type_map = {
        "CRITICAL_CONFLICT": "conflict_moment",
        "HARSH_MOTION": "harsh_braking",
        "SUSTAINED_NOISE": "audio_spike",
    }

    aggregated["flag_type"] = aggregated["Stress_Flag"].map(flag_type_map).fillna(
        "unknown"
    )
    aggregated = aggregated.drop(columns=["Stress_Flag"])
    
    # Event absorption filter: remove lesser flags within 15 seconds of conflict moments
    conflict_indices = aggregated[aggregated["flag_type"] == "conflict_moment"].index
    rows_to_drop = []
    
    for conflict_idx in conflict_indices:
        conflict_time = aggregated.loc[conflict_idx, "elapsed_sec"]
        current_trip = aggregated.loc[conflict_idx, "trip_id"]
        time_window_start = conflict_time - 15
        time_window_end = conflict_time + 15
        
        # Find lesser flags within 15 seconds of this conflict moment (same trip only)
        lesser_flags = aggregated[
            (aggregated.index != conflict_idx) &
            (aggregated["trip_id"] == current_trip) &
            (aggregated["flag_type"].isin(["audio_spike", "harsh_braking"])) &
            (aggregated["elapsed_sec"] >= time_window_start) &
            (aggregated["elapsed_sec"] <= time_window_end)
        ]
        
        rows_to_drop.extend(lesser_flags.index.tolist())
    
    # Drop the identified rows and reset index
    aggregated = aggregated.drop(rows_to_drop).reset_index(drop=True)
    
    aggregated["flag_id"] = [
        f"FLAG{i:03d}" for i in range(1, len(aggregated) + 1)
    ]

    motion_type = pd.Series(
        np.where(
            aggregated["flag_type"].isin(["harsh_braking", "conflict_moment"]),
            "harsh_braking",
            "none",
        ),
        dtype=str,
    )
    audio_type = pd.Series(
        np.where(
            aggregated["flag_type"].isin(["audio_spike", "conflict_moment"]),
            "audio_spike",
            "none",
        ),
        dtype=str,
    )
    aggregated["context"] = "Motion: " + motion_type + " | Audio: " + audio_type

    motion_score = np.clip((aggregated["Horizontal_Jerk"].astype(float) - 4.0) / 4.0, 0.0, 1.0).round(2)
    audio_score = np.clip((aggregated["Audio_Rolling_15s"].astype(float) - 85.0) / 15.0, 0.0, 1.0).round(2)
    aggregated["motion_score"] = motion_score
    aggregated["audio_score"] = audio_score
    aggregated["combined_score"] = np.maximum(motion_score, audio_score).round(2)

    aggregated["severity"] = np.select(
        [aggregated["combined_score"] >= 0.7, aggregated["combined_score"] >= 0.4],
        ["high", "medium"],
        default="low",
    )

    hj = aggregated["Horizontal_Jerk"].astype(float).round(1)
    ar = aggregated["Audio_Rolling_15s"].fillna(0).astype(float).round(0).astype(int)
    aggregated["explanation"] = np.where(
        aggregated["flag_type"].eq("conflict_moment"),
        "Combined signal: Harsh braking (" + hj.astype(str) + " m/s^2) + sustained high audio (" + ar.astype(str) + " dB)",
        np.where(
            aggregated["flag_type"].eq("harsh_braking"),
            "Harsh braking detected (" + hj.astype(str) + " m/s^2) with audio level (" + ar.astype(str) + " dB)",
            "Sustained high audio detected (" + ar.astype(str) + " dB) during " + aggregated["audio_class"].astype(str),
        ),
    )

    aggregated["timestamp"] = aggregated["timestamp"].dt.tz_convert("UTC").dt.strftime(
        "%d-%m-%Y %H:%M"
    )

    for col in [
        "motion_score",
        "audio_score",
        "combined_score",
        "Horizontal_Jerk",
        "Vertical_Jerk",
        "Audio_Rolling_15s",
        "speed_kmh",
        "gps_lat",
        "gps_lon",
    ]:
        if col in aggregated.columns:
            aggregated[col] = aggregated[col].astype(float).round(2)

    # Add Uber's required schema columns
    signal_type_map = {
        "harsh_braking": "ACCELEROMETER",
        "audio_spike": "AUDIO",
        "conflict_moment": "COMBINED"
    }
    aggregated["signal_type"] = aggregated["flag_type"].map(signal_type_map).fillna("UNKNOWN")
    
    # Create raw_value column based on signal type
    def create_raw_value(row):
        if row["signal_type"] == "ACCELEROMETER":
            return f"{row['Horizontal_Jerk']} m/s^2"
        elif row["signal_type"] == "AUDIO":
            return f"{row['Audio_Rolling_15s']} dB"
        elif row["signal_type"] == "COMBINED":
            return f"{row['Horizontal_Jerk']} m/s^2, {row['Audio_Rolling_15s']} dB"
        else:
            return "UNKNOWN"
    
    aggregated["raw_value"] = aggregated.apply(create_raw_value, axis=1)
    
    # Create threshold column based on signal type
    threshold_map = {
        "ACCELEROMETER": "4.0 m/s^2",
        "AUDIO": "85 dB",
        "COMBINED": "4.0 m/s^2, 85 dB"
    }
    aggregated["threshold"] = aggregated["signal_type"].map(threshold_map).fillna("UNKNOWN")
    
    # Create event_label column (uppercase version of flag_type)
    aggregated["event_label"] = aggregated["flag_type"].str.upper()

    aggregated = aggregated[
        [
            "flag_id",
            "driver_id",
            "trip_id",
            "timestamp",
            "flag_type",
            "severity",
            "motion_score",
            "audio_score",
            "combined_score",
            "explanation",
            "context",
            "elapsed_sec",
            "duration_seconds",
            "speed_kmh",
            "gps_lat",
            "gps_lon",
            "Horizontal_Jerk",
            "Vertical_Jerk",
            "Audio_Rolling_15s",
            "audio_class",
            "signal_type",
            "raw_value",
            "threshold",
            "event_label",
        ]
    ]

    records = aggregated.to_dict(orient="records")
    OUTPUT_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    
    # Also export as CSV
    csv_path = OUTPUT_DIR / "flagged_moments.csv"
    aggregated.to_csv(csv_path, index=False)
    
    # Return the final DataFrame for API use
    return aggregated


def run_stress_moment_model() -> pd.DataFrame:
    """Main execution function: load sensor data, process stress events, and export results.
    
    This function assumes sensor CSV files already exist in data/sensor_data/ directory.
    To generate initial data, run seed_stress_data.py separately.
    
    Returns:
        pd.DataFrame: Processed and aggregated stress events DataFrame
    """
    accel_df, audio_df = load_sensor_data()
    accel_metrics = compute_motion_metrics(accel_df)
    audio_metrics = compute_audio_metrics(audio_df)
    fused = fuse_sensors(accel_metrics, audio_metrics)
    flagged = apply_stress_rules(fused)
    
    # Export to files (keeping compliance logic intact) and get final DataFrame
    final_df = export_flagged(flagged)
    
    return final_df


if __name__ == "__main__":
    run_stress_moment_model()
