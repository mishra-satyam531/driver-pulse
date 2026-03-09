import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "earnings"

def randomize_earnings_data():
    rng = np.random.default_rng()
    
    # 1. Randomize driver_goals.csv
    goals_path = DATA_DIR / "driver_goals.csv"
    if goals_path.exists():
        goals = pd.read_csv(goals_path)
        # Assign random progression between 10% and 110% of targeted earnings
        goals["current_earnings"] = goals["target_earnings"] * rng.uniform(0.1, 1.1, len(goals))
        goals["current_earnings"] = goals["current_earnings"].round(0)
        
        # Assign random elapsed shift hours
        goals["current_hours"] = goals["target_hours"] * rng.uniform(0.1, 0.9, len(goals))
        goals["current_hours"] = goals["current_hours"].round(1)
        
        # Update naive derived columns
        goals["earnings_velocity"] = (goals["current_earnings"] / goals["current_hours"]).round(2)
        
        goals.to_csv(goals_path, index=False)
        print(f"Randomized driver_goals.csv ({len(goals)} rows)")
        
    # 2. Randomize earnings_velocity_log.csv
    log_path = DATA_DIR / "earnings_velocity_log.csv"
    if log_path.exists():
        vel_log = pd.read_csv(log_path)
        # Randomize elapsed hours 
        vel_log["elapsed_hours"] = rng.uniform(0.5, 8.0, len(vel_log)).round(2)
        
        # Randomize earnings driven by a randomized average speed (rupees per hour)
        simulated_hourly_rate = rng.uniform(50, 400, len(vel_log))
        vel_log["cumulative_earnings"] = (vel_log["elapsed_hours"] * simulated_hourly_rate).round(0)
        
        # Randomize trips completed
        vel_log["trips_completed"] = (vel_log["elapsed_hours"] * rng.uniform(1, 3, len(vel_log))).astype(int)
        
        vel_log.to_csv(log_path, index=False)
        print(f"Randomized earnings_velocity_log.csv ({len(vel_log)} rows)")

if __name__ == "__main__":
    randomize_earnings_data()
