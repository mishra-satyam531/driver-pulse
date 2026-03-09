from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "sensor_data"


def generate_mock_telematics_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)

    n_seconds = 3600
    elapsed_seconds = np.arange(n_seconds, dtype=int)

    start_ts = pd.Timestamp("2025-01-01T00:00:00Z")
    timestamps = start_ts + pd.to_timedelta(elapsed_seconds, unit="s")

    trip_id = f"trip_{rng.integers(1000, 9999)}"
    trip_ids = np.full(n_seconds, trip_id, dtype=object)
    
    driver_id = f"DRV{rng.integers(100, 999)}"
    driver_ids = np.full(n_seconds, driver_id, dtype=object)

    base_lat = 37.7749
    base_lon = -122.4194
    lat_drift = rng.normal(0.0, 1e-5, size=n_seconds).cumsum()
    lon_drift = rng.normal(0.0, 1e-5, size=n_seconds).cumsum()
    gps_lat = base_lat + lat_drift
    gps_lon = base_lon + lon_drift

    speed_kmh = 50.0 + rng.normal(0.0, 3.0, size=n_seconds)
    speed_kmh = np.clip(speed_kmh, 40.0, 60.0)

    accel_x = rng.normal(0.0, 0.2, size=n_seconds)
    accel_y = rng.normal(0.0, 0.2, size=n_seconds)
    accel_z = rng.normal(9.8, 0.2, size=n_seconds)

    audio_level = rng.normal(65.0, 5.0, size=n_seconds)
    audio_class = rng.choice(
        ["normal", "quiet"], size=n_seconds, p=[0.7, 0.3]
    ).astype(object)

    # Event 1 (True Harsh Brake): at second 500 for 2 seconds
    idx_event1 = (elapsed_seconds >= 500) & (elapsed_seconds < 502)
    accel_y[idx_event1] = 5.0

    # Event 2 (Pothole Veto Test): at second 1000 for 1 second
    idx_event2 = elapsed_seconds == 1000
    accel_y[idx_event2] = 4.5
    accel_z[idx_event2] = 14.0

    # Event 3 (Siren False Alarm Test): 1500..1520
    idx_event3 = (elapsed_seconds >= 1500) & (elapsed_seconds <= 1520)
    audio_level[idx_event3] = 100.0
    audio_class[idx_event3] = "loud"

    # Event 4 (True Argument): 2000..2025
    idx_event4 = (elapsed_seconds >= 2000) & (elapsed_seconds <= 2025)
    audio_level[idx_event4] = 95.0
    audio_class[idx_event4] = "argument"

    # Event 5 (Total Chaos): 3000..3039 (40 seconds)
    idx_event5 = (elapsed_seconds >= 3000) & (elapsed_seconds < 3040)
    audio_level[idx_event5] = 98.0
    audio_class[idx_event5] = "argument"

    # Event 6 (Test Case: Extreme Brake, Quiet Cabin): at second 3500
    idx_event6 = elapsed_seconds == 3500
    accel_y[idx_event6] = 6.0  # Creates a massive 6.0 m/s^2 horizontal jerk
    audio_level[idx_event6] = 65.0  # Normal quiet cabin noise
    audio_class[idx_event6] = "normal"

    # Midpoint harsh brake inside Total Chaos: exactly at 3020
    idx_event5_brake = elapsed_seconds == 3020
    accel_y[idx_event5_brake] = 5.5
    accel_z[idx_event5_brake] = 9.8

    accel_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "trip_id": trip_ids,
            "driver_id": driver_ids,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_sec": elapsed_seconds,
            "accel_x": accel_x,
            "accel_y": accel_y,
            "accel_z": accel_z,
            "speed_kmh": speed_kmh,
            "gps_lat": gps_lat,
            "gps_lon": gps_lon,
        }
    )

    audio_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "trip_id": trip_ids,
            "audio_level": audio_level,
            "audio_class": audio_class,
        }
    )

    return accel_df, audio_df


def export_csvs(accel_df: pd.DataFrame, audio_df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    accel_path = DATA_DIR / "accelerometer_data.csv"
    audio_path = DATA_DIR / "audio_intensity_data.csv"

    accel_df.to_csv(accel_path, index=False)
    audio_df.to_csv(audio_path, index=False)


def main() -> None:
    accel_df, audio_df = generate_mock_telematics_data()
    export_csvs(accel_df, audio_df)


if __name__ == "__main__":
    main()

