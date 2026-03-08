"""
goal_predictor.py
ML forecasting model for driver goal achievement.
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

# Paths
BASE_DIR     = Path(__file__).resolve().parents[1]
DATA_DIR     = BASE_DIR / "data"
OUTPUT_DIR   = DATA_DIR / "processed_outputs"
MODEL_DIR    = OUTPUT_DIR / "models"
MODEL_PATH   = MODEL_DIR / "goal_model.pkl"
ENCODER_PATH = MODEL_DIR / "goal_label_encoder.pkl"
OUTPUT_PATH  = OUTPUT_DIR / "goal_predictions_output.json"

EARNINGS_DIR = DATA_DIR / "earnings"
DRIVERS_PATH = DATA_DIR / "drivers" / "drivers.csv"

# Features mapped to corresponding signal attributes
FEATURE_COLS = [
    "pct_earned",         # progress toward goal (0 to 1)
    "pct_time_used",      # fraction of shift elapsed (0 to 1)
    "velocity_ratio",     # actual pace / ideal pace (>1 = ahead)
    "earnings_velocity",  # raw earnings/hr
    "hours_remaining",    # time left in hours
    "experience_months",  # driver tenure
    "rating",             # driver star rating
]

VALID_LABELS = {"ahead", "on_track", "at_risk"}

# Data loading
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    goals   = pd.read_csv(EARNINGS_DIR / "driver_goals.csv")
    drivers = pd.read_csv(DRIVERS_PATH)
    return goals, drivers

# Feature engineering
def build_features(goals: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    df = goals.merge(
        drivers[["driver_id", "experience_months", "rating"]],
        on="driver_id", how="left",
    )

    ideal_velocity        = df["target_earnings"] / (df["target_hours"] + 1e-5)
    df["pct_earned"]      = df["current_earnings"] / (df["target_earnings"] + 1e-5)
    df["pct_time_used"]   = df["current_hours"]    / (df["target_hours"]    + 1e-5)
    df["velocity_ratio"]  = df["earnings_velocity"] / (ideal_velocity + 1e-5)
    df["hours_remaining"] = df["target_hours"] - df["current_hours"]

    # Drop rows without a labelled outcome
    df = df[df["goal_completion_forecast"].isin(VALID_LABELS)].copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df

# Training
def train_model(goals: pd.DataFrame, drivers: pd.DataFrame) -> tuple:
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

    # 5-fold stratified CV
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
    print(f"  CV F1 (weighted): {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")

    model.fit(X, y)

    top3 = (
        pd.Series(model.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    print(f"  Top-3 features : {', '.join(top3)}")

    return model, le, float(cv_f1.mean())

# Save and load
def save_model(model, encoder) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,   MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"  Model   -> {MODEL_PATH}")
    print(f"  Encoder -> {ENCODER_PATH}")

def load_model() -> tuple:
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run goal_predictor.py first.")
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)

# Inference on velocity log
def predict_from_velocity_df(
    model,
    encoder,
    velocity_df: pd.DataFrame,
    goals: pd.DataFrame,
    drivers: pd.DataFrame,
) -> pd.DataFrame:
    df = velocity_df.copy()

    primary_goals = goals.drop_duplicates(subset="driver_id", keep="first")
    df = df.merge(primary_goals[["driver_id", "target_hours"]], on="driver_id", how="left")
    df = df.merge(
        drivers[["driver_id", "experience_months", "rating"]],
        on="driver_id", how="left", suffixes=("", "_drv"),
    )

    target_hours          = df["target_hours"].fillna(8.0)
    ideal_v               = df["target_earnings"] / (target_hours + 1e-5)
    df["pct_earned"]      = df["cumulative_earnings"] / (df["target_earnings"] + 1e-5)
    df["pct_time_used"]   = df["elapsed_hours"] / (target_hours + 1e-5)
    df["velocity_ratio"]  = df["current_velocity"].fillna(0) / (ideal_v + 1e-5)
    df["earnings_velocity"]  = df["current_velocity"].fillna(0)
    df["hours_remaining"] = (target_hours - df["elapsed_hours"]).clip(lower=0)

    # Only score eligible rows
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

# Dashboard: single-driver inference
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

# Export
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
    for col in ["current_velocity", "target_velocity", "velocity_delta", "projected_final_earnings"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: round(float(x), 2) if pd.notna(x) and x is not None else x)
    OUTPUT_PATH.write_text(
        json.dumps(out.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  Goal predictions -> {OUTPUT_PATH}")

# Entrypoint
def run_goal_predictor_model() -> None:
    from backend.earnings_velocity import run_earnings_velocity_model

    print("Step 1 - Earnings velocity engine ...")
    velocity_df = run_earnings_velocity_model()

    print("\nStep 2 - Training goal-completion model ...")
    goals, drivers = load_data()
    model, encoder, cv_f1 = train_model(goals, drivers)
    save_model(model, encoder)

    print("\nStep 3 - Applying ML predictions ...")
    enriched = predict_from_velocity_df(model, encoder, velocity_df, goals, drivers)
    export_predictions(enriched)

    if not enriched.empty:
        print(f"\n  Events scored   : {len(enriched)}")
        print(f"  Forecast dist   :\n{enriched['forecast_status'].value_counts().to_string()}")
    print(f"  CV F1 (weighted): {cv_f1:.3f}")

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run_goal_predictor_model()
