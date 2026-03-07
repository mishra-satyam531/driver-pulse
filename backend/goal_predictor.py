"""
goal_predictor.py  –  Person 2: Earnings & Forecasting Lead
=============================================================
Trains a lightweight Random Forest classifier on driver_goals.csv to predict
whether a driver is "ahead", "on_track", or "at_risk" of hitting their
earnings goal.

The ML model is intentionally simple and lightweight:
  • ~210 training rows  → Random Forest (no deep-learning overhead)
  • 5-fold stratified CV  → gives honest F1 estimate on small data
  • balanced class_weight  → handles the natural class skew gracefully
  • Probability output  → dashboard can show a confidence meter

Outputs (→ data/processed_outputs/):
  goal_predictions_output.json   – enriched velocity log with ML forecasts
  goal_model.pkl / goal_label_encoder.pkl  → model artefacts (gitignored)
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
BASE_DIR      = Path(__file__).resolve().parents[1]          # repo root
DATA_DIR      = BASE_DIR / "data"
OUTPUT_DIR    = DATA_DIR / "processed_outputs"
MODEL_DIR     = OUTPUT_DIR / "models"
MODEL_PATH    = MODEL_DIR / "goal_model.pkl"
ENCODER_PATH  = MODEL_DIR / "goal_label_encoder.pkl"
OUTPUT_PATH   = OUTPUT_DIR / "goal_predictions_output.json"

EARNINGS_DIR  = DATA_DIR / "earnings"
DRIVERS_PATH  = DATA_DIR / "drivers" / "drivers.csv"

# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "pct_earned",         # current_earnings / target_earnings
    "pct_time_used",      # current_hours / target_hours
    "velocity_ratio",     # earnings_velocity / ideal_velocity
    "earnings_velocity",  # raw ₹/hr from goals CSV
    "hours_remaining",    # target_hours − current_hours
    "experience_months",  # driver tenure
    "rating",             # driver star rating
]

VALID_LABELS = {"ahead", "on_track", "at_risk"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    goals   = pd.read_csv(EARNINGS_DIR / "driver_goals.csv")
    drivers = pd.read_csv(DRIVERS_PATH)
    return goals, drivers


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(goals: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """
    Build the ML feature matrix from driver_goals.csv.

    Features derived:
      pct_earned      = current_earnings / target_earnings   (how close to goal)
      pct_time_used   = current_hours / target_hours          (how far through shift)
      velocity_ratio  = actual velocity / ideal velocity      (> 1 → ahead of pace)
      hours_remaining = target_hours − current_hours
      earnings_velocity, experience_months, rating            (raw / from profile)

    Target: goal_completion_forecast  (ahead / on_track / at_risk)
    """
    df = goals.merge(
        drivers[["driver_id", "experience_months", "rating"]],
        on="driver_id", how="left",
    )

    ideal_velocity       = df["target_earnings"] / (df["target_hours"] + 1e-5)
    df["pct_earned"]     = df["current_earnings"] / (df["target_earnings"] + 1e-5)
    df["pct_time_used"]  = df["current_hours"]    / (df["target_hours"]    + 1e-5)
    df["velocity_ratio"] = df["earnings_velocity"] / (ideal_velocity + 1e-5)
    df["hours_remaining"] = df["target_hours"] - df["current_hours"]

    # Only rows with a labelled forecast
    df = df[df["goal_completion_forecast"].isin(VALID_LABELS)].copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(goals: pd.DataFrame, drivers: pd.DataFrame) -> tuple:
    """
    Train a Random Forest on driver_goals.csv.

    Design rationale:
      • Random Forest handles non-linear interactions (velocity × time-of-shift)
      • Probability scores → dashboard confidence meter
      • Small dataset (~210 rows): light model, deep trees avoided (max_depth=6)
      • balanced class_weight counters natural label imbalance
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

    # 5-fold stratified cross-validation (honest estimate on small data)
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
    print(f"  CV F1 (weighted): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    model.fit(X, y)

    feat_imp = (
        pd.Series(model.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
    )
    print(f"  Top-3 features : {', '.join(feat_imp.head(3).index.tolist())}")

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
        raise FileNotFoundError(
            "Trained model not found.  Run goal_predictor.py first."
        )
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


# ── Inference on velocity log ─────────────────────────────────────────────────

def predict_from_velocity_df(
    model,
    encoder,
    velocity_df: pd.DataFrame,
    goals: pd.DataFrame,
    drivers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the trained model to the velocity-log output from earnings_velocity.py.
    Adds 'ml_forecast' and 'ml_confidence' columns.
    The ml_forecast overrides 'forecast_status' for eligible rows.
    """
    df = velocity_df.copy()

    # Attach goal/shift info
    primary_goals = goals.drop_duplicates(subset="driver_id", keep="first")
    df = df.merge(
        primary_goals[["driver_id", "target_hours"]],
        on="driver_id", how="left",
    )
    df = df.merge(
        drivers[["driver_id", "experience_months", "rating"]],
        on="driver_id", how="left",
        suffixes=("", "_drv"),
    )

    target_hours     = df.get("target_hours", pd.Series([8.0] * len(df))).fillna(8.0)
    ideal_v          = df["target_earnings"] / (target_hours + 1e-5)

    df["pct_earned"]        = df["cumulative_earnings"] / (df["target_earnings"] + 1e-5)
    df["pct_time_used"]     = df["elapsed_hours"]       / (target_hours + 1e-5)
    df["velocity_ratio"]    = df["current_velocity"].fillna(0) / (ideal_v + 1e-5)
    df["earnings_velocity"] = df["current_velocity"].fillna(0)
    df["hours_remaining"]   = (target_hours - df["elapsed_hours"]).clip(lower=0)

    # Only run the model on rows that already have a rule-based label
    eligible_mask = df["forecast_status"].isin(VALID_LABELS)
    X = df.loc[eligible_mask, FEATURE_COLS].fillna(0).values

    df["ml_forecast"]   = df["forecast_status"]
    df["ml_confidence"] = 1.0

    if len(X) > 0:
        proba = model.predict_proba(X)
        preds = encoder.inverse_transform(model.predict(X))
        df.loc[eligible_mask, "ml_forecast"]    = preds
        df.loc[eligible_mask, "ml_confidence"]  = proba.max(axis=1).round(3)
        df.loc[eligible_mask, "forecast_status"] = preds

    # Clean up temporary feature columns
    tmp_cols = ["target_hours", "pct_earned", "pct_time_used",
                "velocity_ratio", "hours_remaining"]
    df.drop(columns=[c for c in tmp_cols if c in df.columns], inplace=True)

    return df


# ── Single-driver inference (for dashboard) ───────────────────────────────────

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
    Predict goal completion for a single driver at a moment in time.
    Called by the dashboard to power per-driver real-time forecasts.

    Returns:
        {
          "forecast": "ahead" | "on_track" | "at_risk",
          "confidence": 0.0–1.0,
          "probabilities": {label: score, ...}
        }
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


# ── Export ────────────────────────────────────────────────────────────────────

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
    Full pipeline:
      1. Run the velocity engine  →  velocity_df
      2. Train the ML model on driver_goals.csv
      3. Apply ML forecasts on top of the rule-based velocity output
      4. Export enriched predictions
    """
    # Import here (same package) to avoid circular deps at module level
    from backend.earnings_velocity import run_earnings_velocity_model

    print("Running earnings velocity engine ...")
    velocity_df = run_earnings_velocity_model()

    print("\nTraining goal-completion model ...")
    goals, drivers = load_data()
    model, encoder, cv_f1 = train_model(goals, drivers)
    save_model(model, encoder)

    print("\nApplying ML predictions ...")
    enriched = predict_from_velocity_df(model, encoder, velocity_df, goals, drivers)
    export_predictions(enriched)

    if not enriched.empty:
        print(f"\n  Goal forecasts : {len(enriched)} events")
        print(f"  Forecast dist  :\n{enriched['forecast_status'].value_counts().to_string()}")
    print(f"  ML CV F1       : {cv_f1:.3f}")


if __name__ == "__main__":
    # Allow running from repo root:  python -m backend.goal_predictor
    # or from backend/:              python goal_predictor.py
    if __package__ is None:
        # Running as a plain script — patch the import path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from backend.earnings_velocity import run_earnings_velocity_model  # noqa: F401
    run_goal_predictor_model()
