from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed_outputs"

FLAGGED_PATH = PROC_DIR / "flagged_moments.json"
EARNINGS_PATH = PROC_DIR / "earnings_velocity_output.json"
OUT_CSV = PROC_DIR / "uber_compliance_log.csv"

# Threshold constants (align with stress_model.py)
JERK_THRESHOLD = 4.0
AUDIO_THRESHOLD = 85.0


def _read_json_array(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def build_from_flagged(flag_events: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for ev in flag_events:
        ts = ev.get("timestamp")
        flag_type = str(ev.get("flag_type", "")).lower()
        jerk = ev.get("Horizontal_Jerk")
        aud = ev.get("Audio_Rolling_15s")

        if flag_type == "harsh_braking":
            rows.append({
                "timestamp": ts,
                "signal_type": "ACCELEROMETER",
                "raw_value": jerk,
                "threshold": JERK_THRESHOLD,
                "event_label": "HARSH_BRAKING",
            })
        elif flag_type == "audio_spike":
            rows.append({
                "timestamp": ts,
                "signal_type": "AUDIO",
                "raw_value": aud,
                "threshold": AUDIO_THRESHOLD,
                "event_label": "NOISE_SPIKE",
            })
        elif flag_type == "conflict_moment":
            # Emit two lines to make the evidence explicit
            rows.append({
                "timestamp": ts,
                "signal_type": "ACCELEROMETER",
                "raw_value": jerk,
                "threshold": JERK_THRESHOLD,
                "event_label": "HARSH_BRAKING",
            })
            rows.append({
                "timestamp": ts,
                "signal_type": "AUDIO",
                "raw_value": aud,
                "threshold": AUDIO_THRESHOLD,
                "event_label": "NOISE_SPIKE",
            })
    return rows


def build_from_earnings(earn_events: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for ev in earn_events:
        ts = ev.get("timestamp")
        sig = str(ev.get("signal_type", "EARNINGS")).upper()
        raw = ev.get("raw_value")
        thr = ev.get("threshold")
        label = str(ev.get("event_label", ev.get("forecast_status", ""))).upper()
        rows.append({
            "timestamp": ts,
            "signal_type": sig,
            "raw_value": raw,
            "threshold": thr,
            "event_label": label,
        })
    return rows


def generate_uber_compliance_log() -> Path:
    flag_events = _read_json_array(FLAGGED_PATH)
    earn_events = _read_json_array(EARNINGS_PATH)

    rows = build_from_flagged(flag_events) + build_from_earnings(earn_events)
    df = pd.DataFrame(rows, columns=["timestamp", "signal_type", "raw_value", "threshold", "event_label"])
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Compliance log -> {OUT_CSV}  ({len(df)} rows)")
    return OUT_CSV


if __name__ == "__main__":
    generate_uber_compliance_log()
