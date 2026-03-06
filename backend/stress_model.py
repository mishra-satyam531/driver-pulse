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
    accel_df = pd.read_csv(ACCEL_PATH)
    audio_df = pd.read_csv(AUDIO_PATH)

    accel_df["timestamp"] = pd.to_datetime(
        accel_df["timestamp"], utc=True, errors="coerce"
    ).dt.floor("s")
    audio_df["timestamp"] = pd.to_datetime(
        audio_df["timestamp"], utc=True, errors="coerce"
    ).dt.floor("s")

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

    df["audio_level_clipped"] = df["audio_level"].clip(lower=30, upper=120)

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
    accel_sorted = accel_df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
    audio_sorted = audio_df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

    audio_cols = [
        "trip_id",
        "timestamp",
        "Audio_Rolling_15s",
        "audio_class",
    ]
    audio_subset = audio_sorted[audio_cols]

    fused = pd.merge_asof(
        accel_sorted,
        audio_subset,
        on="timestamp",
        by="trip_id",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=60),
    )
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


def export_flagged(flagged_df: pd.DataFrame) -> None:
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
        df.groupby(["trip_id", "Stress_Flag", "event_block_id"], as_index=False)
        .agg(
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

    aggregated["timestamp"] = aggregated["timestamp"].dt.tz_convert("UTC").dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    records = aggregated.to_dict(orient="records")
    OUTPUT_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")


def run_stress_moment_model() -> None:
    accel_df, audio_df = load_sensor_data()
    accel_metrics = compute_motion_metrics(accel_df)
    audio_metrics = compute_audio_metrics(audio_df)
    fused = fuse_sensors(accel_metrics, audio_metrics)
    flagged = apply_stress_rules(fused)
    export_flagged(flagged)


if __name__ == "__main__":
    run_stress_moment_model()

