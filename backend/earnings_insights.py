import json
import os
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from openai import OpenAI

def _require_env(name: str) -> str:
    v = os.getenv(name, "")
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v

API_KEY = _require_env("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed_outputs" / "goal_predictions_output.json"
DRIVERS_PATH = BASE_DIR / "data" / "drivers" / "drivers.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed_outputs" / "earnings_insights_final.json"

def load_drivers_data() -> Dict[str, str]:
    if not DRIVERS_PATH.exists():
        return {}
    try:
        drivers_df = pd.read_csv(DRIVERS_PATH)
        drivers_df['first_name'] = drivers_df['name'].str.split().str[0]
        return dict(zip(drivers_df['driver_id'], drivers_df['first_name']))
    except Exception:
        return {}

def load_predictions() -> List[Dict[str, Any]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}. Run goal_predictor.py first.")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_earnings_insight(event: dict, driver_name: str, client: OpenAI) -> str:
    system_prompt = f"""You are 'Drive Pulse', an encouraging, stress-free financial assistant for gig workers.
Translate their current earnings progress into brief, motivating coaching.

STRICT RULES:
1. Keep it to 1-2 sentences. Address the driver as '{driver_name}'.
2. DO NOT stress the driver or sound like a strict boss.
3. If forecast is 'ahead' or 'achieved', celebrate their pacing.
4. If forecast is 'at_risk', offer a gentle, optimistic nudge.
5. Use the provided context (percentage to goal, velocity). Avoid dumping raw numbers without context."""

    pct = event.get('pct_to_goal', 0)
    status = event.get('forecast_status', 'unknown')
    target = event.get('target_earnings', 0)
    current = event.get('cumulative_earnings', 0)
    
    user_prompt = f"Driver has earned ₹{current} out of ₹{target} ({pct}%). The ML predicts they are currently: {status}."
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating LLM insight: {e}")
        return ""

def process_earnings_insights(events: List[Dict[str, Any]], drivers_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    processed_events = []
    
    # To avoid API rate limits, we'll only generate insights for the LAST recorded event
    # per driver (their current/final state), rather than thousands of historical logs.
    
    # Find latest event per driver
    latest_events = {}
    for event in events:
        did = event.get('driver_id')
        if did:
            latest_events[did] = event
            
    # Process a defined batch size of drivers (e.g. for a live dashboard updates stream)
    batch_size = 5
    latest_events = dict(list(latest_events.items())[:batch_size])
    print(f"Generating personalized earnings coaching for {len(latest_events)} drivers...")
    
    for driver_id, event in latest_events.items():
        driver_name = drivers_mapping.get(driver_id, 'Alex')
        
        # Only generate a message if we have a valid forecast
        if event.get('forecast_status') in ['ahead', 'on_track', 'at_risk', 'achieved']:
            insight = generate_earnings_insight(event, driver_name, client)
            event['earnings_llm_insight'] = insight
            processed_events.append(event)
            
    return processed_events

def save_insights(events: List[Dict[str, Any]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    print(f"Earnings insights saved to: {OUTPUT_PATH}")

def main():
    print("Loading drivers data...")
    drivers_mapping = load_drivers_data()
    
    print("Loading goal predictions...")
    events = load_predictions()
    
    processed_events = process_earnings_insights(events, drivers_mapping)
    save_insights(processed_events)

if __name__ == "__main__":
    main()
