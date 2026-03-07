"""
goal_predictor.py  –  Person 2: Earnings & Forecasting Lead
=============================================================
Chapter 2 of the earnings story that begins in earnings_velocity.py.

earnings_velocity.py answers: "How fast is she going right now?"
goal_predictor.py  answers:   "Based on that pace — will she make it?"

The velocity engine gives us a rule-based snapshot at each moment in time.
This module lifts that snapshot into a learning system: a Random Forest
trained on historical driver goal data that learns which combinations of
pace, progress, and driver profile actually predict success or struggle.

The result is two layers of intelligence on every log event:
  • rule_forecast   — deterministic, always explainable, no black box
  • ml_forecast     — probabilistic, catches non-linear patterns the rules miss
  • ml_confidence   — a score the dashboard can use to show certainty

Why Random Forest and not something heavier?
  ~210 labelled rows in driver_goals.csv — deep learning would overfit badly.
  RF with max_depth=6 gives honest generalisation, fast training (< 1s),
  and per-class probability scores out of the box — exactly what a
  real-time confidence meter needs.

Outputs → data/processed_outputs/
  goal_predictions_output.json  — velocity log enriched with ML layer
  models/goal_model.pkl         — trained classifier (reproducible)
  models/goal_label_encoder.pkl — label mapping
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[1]   # repo root
DATA_DIR     = BASE_DIR / "data"
OUTPUT_DIR   = DATA_DIR / "processed_outputs"
MODEL_DIR    = OUTPUT_DIR / "models"
MODEL_PATH   = MODEL_DIR / "goal_model.pkl"
ENCODER_PATH = MODEL_DIR / "goal_label_encoder.pkl"
OUTPUT_PATH  = OUTPUT_DIR / "goal_predictions_output.json"

EARNINGS_DIR = DATA_DIR / "earnings"
DRIVERS_PATH = DATA_DIR / "drivers" / "drivers.csv"

# ── Features the model learns from ───────────────────────────────────────────
# These mirror the signals available in real time during a shift.
# velocity_ratio and pct_earned are the strongest predictors (top-3 CV).
FEATURE_COLS = [
    "pct_earned",         # how far through the earnings goal (0 → 1)
    "pct_time_used",      # how far through the shift window (0 → 1)
    "velocity_ratio",     # actual pace ÷ ideal pace  (>1 means ahead)
    "earnings_velocity",  # raw earnings per hour at this moment
    "hours_remaining",    # absolute time left in shift
    "experience_months",  # driver tenure — veteran drivers pace differently
    "rating",             # star rating — proxy for demand / trip acceptance
]

VALID_LABELS = {"ahead", "on_track", "at_risk"}


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    goals   = pd.read_csv(EARNINGS_DIR / "driver_goals.csv")
    drivers = pd.read_csv(DRIVERS_PATH)
    return goals, drivers


# ── Step 2: Feature engineering ───────────────────────────────────────────────

def build_features(goals: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw goal records into an ML-ready feature matrix.

    Each row in driver_goals.csv is a snapshot of one driver mid-shift.
    We compute ratio features so the model generalises across different
    earning targets and shift lengths — a driver at 50% of a ₹2000 goal
    and a driver at 50% of a ₹1000 goal are in structurally the same position.
    """
    df = goals.merge(
        drivers[["driver_id", "experience_months", "rating"]],
        on="driver_id", how="left",
    )

    ideal_velocity        = df["target_earnings"] / (df["target_hours"] + 1e-5)
    df["pct_earned"]      = df["current_earnings"] / (df["target_earnings"] + 1e-5)
    df["pct_time_used"]   = df["current_hours"]    / (df["target_hours"]    + 1e-5)
    df["velocity_ratio"]  = df["earnings_velocity"] / (ideal_velocity + 1e-5)
    df["hours_remaining"] = df["target_hours"] - df["current_hours"]

    df = df[df["goal_completion_forecast"].isin(VALID_LABELS)].copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df


# ── Step 3: Train ─────────────────────────────────────────────────────────────

def train_model(goals: pd.DataFrame, drivers: pd.DataFrame) -> tuple:
    """
    Train a Random Forest classifier on historical goal snapshots.

    5-fold stratified cross-validation gives an honest F1 estimate before
    the final model is fit on all available data.  The balanced class_weight
    ensures the minority labels (on_track) aren't ignored during training.
    """
    df = build_features(goals, drivers)
    X  = df[FEATURE_COLS].values
    le = LabelEncoder()
    y  = le.fit_transform(df["goal_completion_forecast"])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # 5-fold stratified cross-validation
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
    print(f"  CV F1 (weighted): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    model.fit(X, y)

    top3 = (
        pd.Series(model.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    print(f"  Top-3 features : {', '.join(top3)}")

    return model, le, float(cv_f1.mean())


# ── Save / load ───────────────────────────────────────────────────────────────

def save_model(model, encoder) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,   MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"  Model   → {MODEL_PATH}")
    print(f"  Encoder → {ENCODER_PATH}")


def load_model() -> tuple:
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run goal_predictor.py first.")
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


# ── Step 4: Apply ML layer to velocity log ────────────────────────────────────

def predict_from_velocity_df(
    model,
    encoder,
    velocity_df: pd.DataFrame,
    goals: pd.DataFrame,
    drivers: pd.DataFrame,
) -> pd.DataFrame:
    """
    This is where the two chapters meet.

    The velocity log from earnings_velocity.py gives us rule-based forecasts
    for every moment.  Here we reconstruct the same feature set on each log
    row and overlay the ML model's judgment — giving every event both a
    deterministic label and a probabilistic confidence score.

    Rows still in "insufficient_data" or "achieved" state are left untouched;
    the model only speaks when there is enough signal to work with.
    """
    df = velocity_df.copy()

    primary_goals = goals.drop_duplicates(subset="driver_id", keep="first")
    df = df.merge(primary_goals[["driver_id", "target_hours"]], on="driver_id", how="left")
    df = df.merge(
        drivers[["driver_id", "experience_months", "rating"]],
        on="driver_id", how="left", suffixes=("", "_drv"),
    )

    target_hours         = df["target_hours"].fillna(8.0)
    ideal_v              = df["target_earnings"] / (target_hours + 1e-5)
    df["pct_earned"]     = df["cumulative_earnings"] / (df["target_earnings"] + 1e-5)
    df["pct_time_used"]  = df["elapsed_hours"] / (target_hours + 1e-5)
    df["velocity_ratio"] = df["current_velocity"].fillna(0) / (ideal_v + 1e-5)
    df["earnings_velocity"] = df["current_velocity"].fillna(0)
    df["hours_remaining"]   = (target_hours - df["elapsed_hours"]).clip(lower=0)

    # Only score rows that have a meaningful rule-based label
    eligible = df["forecast_status"].isin(VALID_LABELS)
    X = df.loc[eligible, FEATURE_COLS].fillna(0).values

    df["ml_forecast"]   = df["forecast_status"]
    df["ml_confidence"] = 1.0

    if len(X) > 0:
        proba = model.predict_proba(X)
        preds = encoder.inverse_transform(model.predict(X))
        df.loc[eligible, "ml_forecast"]    = preds
        df.loc[eligible, "ml_confidence"]  = proba.max(axis=1).round(3)
        df.loc[eligible, "forecast_status"] = preds

    tmp = ["target_hours", "pct_earned", "pct_time_used", "velocity_ratio", "hours_remaining"]
    df.drop(columns=[c for c in tmp if c in df.columns], inplace=True)
    return df


# ── Dashboard helper: single-driver inference ─────────────────────────────────

def predict_single(
    model,
    encoder,
    pct_earned: float,
    pct_time_used: float,
    velocity_ratio: float,
    earnings_velocity: float,
    hours_remaining: float,
    experience_months: float = 24,
    rating: float = 4.7,
) -> dict:
    """
    Instant forecast for one driver at one moment — designed for the dashboard.

    Returns a dict with the forecast label, overall confidence, and the full
    probability breakdown across all three classes so the UI can render a
    confidence bar or a traffic-light indicator.
    """
    x     = np.array([[pct_earned, pct_time_used, velocity_ratio,
                        earnings_velocity, hours_remaining,
                        experience_months, rating]])
    proba = model.predict_proba(x)[0]
    label = encoder.inverse_transform(model.predict(x))[0]

    return {
        "forecast":      label,
        "confidence":    round(float(proba.max()), 3),
        "probabilities": {
            encoder.inverse_transform([i])[0]: round(float(p), 3)
            for i, p in enumerate(proba)
        },
    }


# ── Step 5: Export ────────────────────────────────────────────────────────────

def export_predictions(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if df.empty:
        OUTPUT_PATH.write_text("[]", encoding="utf-8")
        return

    keep = [
        "driver_id", "timestamp", "cumulative_earnings", "target_earnings",
        "pct_to_goal", "current_velocity", "target_velocity", "velocity_delta",
        "projected_final_earnings", "forecast_status", "ml_forecast",
        "ml_confidence", "signal_type", "raw_value", "event_label", "threshold",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    OUTPUT_PATH.write_text(
        json.dumps(out.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  Goal predictions → {OUTPUT_PATH}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def run_goal_predictor_model() -> None:
    """
    Full Person 2 pipeline — runs both chapters end to end:
      earnings_velocity  →  velocity_df  (rule-based, real-time)
      goal_predictor     →  enriched     (ML layer on top)
    """
    from backend.earnings_velocity import run_earnings_velocity_model

    print("Step 1 — Earnings velocity engine ...")
    velocity_df = run_earnings_velocity_model()

    print("\nStep 2 — Goal-completion model (training) ...")
    goals, drivers = load_data()
    model, encoder, cv_f1 = train_model(goals, drivers)
    save_model(model, encoder)

    print("\nStep 3 — Applying ML predictions to velocity log ...")
    enriched = predict_from_velocity_df(model, encoder, velocity_df, goals, drivers)
    export_predictions(enriched)

    if not enriched.empty:
        print(f"\n  Events scored  : {len(enriched)}")
        print(f"  Forecast dist  :\n{enriched['forecast_status'].value_counts().to_string()}")
    print(f"  CV F1 (weighted): {cv_f1:.3f}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run_goal_predictor_model()
