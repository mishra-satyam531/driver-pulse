"""
main.py  –  Driver Pulse: Full Pipeline Orchestrator
=====================================================
Run from the repo root:

    python main.py

This wires together all three modules:
  1. Stress / safety engine   (Person 1) – backend/stress_model.py
  2. Earnings velocity engine (Person 2) – backend/earnings_velocity.py
  3. Goal completion ML model (Person 2) – backend/goal_predictor.py

All outputs land in  data/processed_outputs/
"""

import sys
from pathlib import Path

# Ensure repo root is on the path so relative imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.stress_model       import run_stress_moment_model
from backend.earnings_velocity  import run_earnings_velocity_model
from backend.goal_predictor     import (
    load_data,
    train_model,
    save_model,
    predict_from_velocity_df,
    export_predictions,
)


def main() -> None:
    print("=" * 60)
    print("  Driver Pulse  –  Full Pipeline")
    print("=" * 60)

    # ── Step 1: Stress & Safety Engine (Person 1) ────────────────
    print("\n[1/3]  Stress & safety detection ...")
    run_stress_moment_model()
    print("       ✓ flagged_moments.json  written")

    # ── Step 2: Earnings Velocity Engine (Person 2) ───────────────
    print("\n[2/3]  Earnings velocity calculation ...")
    velocity_df = run_earnings_velocity_model()
    print("       ✓ earnings_velocity_output.json + trip_summaries.csv  written")

    # ── Step 3: Goal Completion ML Model (Person 2) ───────────────
    print("\n[3/3]  Goal-completion ML model ...")
    goals, drivers = load_data()
    model, encoder, cv_f1 = train_model(goals, drivers)
    save_model(model, encoder)

    enriched = predict_from_velocity_df(model, encoder, velocity_df, goals, drivers)
    export_predictions(enriched)
    print("       ✓ goal_predictions_output.json  written")
    print(f"       ✓ CV F1 (weighted) = {cv_f1:.3f}")

    print("\n" + "=" * 60)
    print("  Pipeline complete!  All outputs → data/processed_outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
