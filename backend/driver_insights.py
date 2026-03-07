import json
from pathlib import Path
from typing import Dict, List, Any
import os

import pandas as pd
from openai import OpenAI

# Configuration - easily update these with your API details
API_KEY = os.getenv("OPENAI_API_KEY", "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")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
# File paths
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed_outputs" / "flagged_moments.json"
DRIVERS_PATH = BASE_DIR / "data" / "drivers" / "drivers.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed_outputs" / "trip_insights_final.json"

# System prompt for the LLM
SYSTEM_PROMPT = """You are 'Drive Pulse', an empathetic, non-judgmental safety assistant for Uber drivers. Translate raw telematics into brief, supportive insights.

STRICT RULES:
1. DO NOT judge or assign blame. DO NOT give driving advice.
2. Keep it to 1-2 sentences. Address the driver as 'Alex'.
3. AUDIO RULE: ONLY mention noise or stressful environments if the input includes 'high audio', 'argument', or 'audio spike'. If audio is normal/quiet, stay silent about it.
4. MOTION RULE: ONLY mention braking or jarring movements if motion_score >= 0.4 or the input includes 'harsh braking'.
5. PURE FOCUS: If only one sensor (Motion or Audio) triggers a 'medium/high' flag, focus ONLY on that sensor. Do not add 'filler' about the other sensor being normal."""


def load_drivers_data() -> Dict[str, str]:
    """Load drivers data and return a mapping of driver_id to first_name."""
    if not DRIVERS_PATH.exists():
        print(f"Warning: Drivers file not found at {DRIVERS_PATH}")
        print("Using default driver name 'Alex' for all events.")
        return {}
    
    try:
        drivers_df = pd.read_csv(DRIVERS_PATH)
        # Extract first name from full name (split on space and take first part)
        drivers_df['first_name'] = drivers_df['name'].str.split().str[0]
        return dict(zip(drivers_df['driver_id'], drivers_df['first_name']))
    except Exception as e:
        print(f"Error loading drivers data: {e}")
        print("Using default driver name 'Alex' for all events.")
        return {}


def load_flagged_moments() -> List[Dict[str, Any]]:
    """Load the flagged moments JSON data."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_llm_insight(explanation: str, timestamp: str, driver_name: str, client: OpenAI) -> str:
    """Generate LLM insight for a given event with personalized driver name."""
    
    # Update system prompt to use the actual driver name
    personalized_system_prompt = f"""You are 'Drive Pulse', an empathetic, non-judgmental safety assistant for Uber drivers. Translate raw telematics into brief, supportive insights.

STRICT RULES:
1. DO NOT judge or assign blame. DO NOT give driving advice.
2. Keep it to 1-2 sentences. Address the driver as '{driver_name}'.
3. AUDIO RULE: ONLY mention noise or stressful environments if the input includes 'high audio', 'argument', or 'audio spike'. If audio is normal/quiet, stay silent about it.
4. MOTION RULE: ONLY mention braking or jarring movements if motion_score >= 0.4 or the input includes 'harsh braking'.
5. PURE FOCUS: If only one sensor (Motion or Audio) triggers a 'medium/high' flag, focus ONLY on that sensor. Do not add 'filler' about the other sensor being normal."""
    
    user_prompt = f"Input Data: \"{explanation} at {timestamp}.\""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": personalized_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating LLM insight: {e}")
        return "Unable to generate insight at this time."


def process_events_with_llm(events: List[Dict[str, Any]], drivers_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """Process events and add LLM insights for medium/high severity events."""
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    
    processed_events = []
    
    for event in events:
        # Check if severity is medium or high
        severity = event.get('severity', '').lower()
        
        if severity in ['medium', 'high']:
            explanation = event.get('explanation', '')
            timestamp = event.get('timestamp', '')
            driver_id = event.get('driver_id', '')
            
            # Get driver name, fallback to 'Alex' if not found
            driver_name = drivers_mapping.get(driver_id, 'Alex')
            
            # Generate LLM insight with personalized driver name
            llm_insight = generate_llm_insight(explanation, timestamp, driver_name, client)
            event['llm_insight'] = llm_insight
        else:
            # For low severity events, add empty insight or skip
            event['llm_insight'] = ""
        
        processed_events.append(event)
    
    return processed_events


def save_trip_insights(events: List[Dict[str, Any]]) -> None:
    """Save the processed events with LLM insights to the output file."""
    
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    
    print(f"Trip insights saved to: {OUTPUT_PATH}")


def main():
    """Main function to process flagged moments and generate insights."""
    
    try:
        # Load the drivers data
        print("Loading drivers data...")
        drivers_mapping = load_drivers_data()
        print(f"Loaded {len(drivers_mapping)} driver mappings")
        
        # Load the flagged moments
        print("Loading flagged moments...")
        events = load_flagged_moments()
        print(f"Loaded {len(events)} events")
        
        # Process events with LLM
        print("Generating LLM insights for medium/high severity events...")
        processed_events = process_events_with_llm(events, drivers_mapping)
        
        # Save the results
        print("Saving trip insights...")
        save_trip_insights(processed_events)
        
        # Print summary
        insights_generated = sum(1 for event in processed_events if event.get('llm_insight'))
        print(f"Generated {insights_generated} LLM insights out of {len(events)} total events")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
