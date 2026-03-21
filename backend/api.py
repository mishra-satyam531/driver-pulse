"""
backend/api.py
FastAPI application wrapping stress_model.py and earnings_velocity.py for dynamic API endpoints.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import existing processing functions
from stress_model import run_stress_moment_model
from earnings_velocity import run_earnings_velocity_model

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Global cache variables
_stress_events_cache: List[Dict[str, Any]] = []
_earnings_status_cache: List[Dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to compute and cache data on startup."""
    global _stress_events_cache, _earnings_status_cache
    
    print("Computing and caching stress events...")
    try:
        stress_df = run_stress_moment_model()
        if not stress_df.empty:
            # Clean up NaN values for JSON serialization
            records = stress_df.to_dict(orient="records")
            cleaned_records = []
            for record in records:
                cleaned_record = {}
                for key, value in record.items():
                    if pd.isna(value):
                        cleaned_record[key] = None
                    elif isinstance(value, float) and value.is_integer():
                        cleaned_record[key] = int(value)
                    else:
                        cleaned_record[key] = value
                cleaned_records.append(cleaned_record)
            _stress_events_cache = cleaned_records
        else:
            _stress_events_cache = []
        print(f"Cached {len(_stress_events_cache)} stress events")
    except Exception as e:
        print(f"Error caching stress events: {e}")
        _stress_events_cache = []
    
    print("Computing and caching earnings status...")
    try:
        earnings_df = run_earnings_velocity_model()
        if not earnings_df.empty:
            # Clean up NaN values for JSON serialization
            records = earnings_df.to_dict(orient="records")
            cleaned_records = []
            for record in records:
                cleaned_record = {}
                for key, value in record.items():
                    if pd.isna(value):
                        cleaned_record[key] = None
                    elif isinstance(value, float) and value.is_integer():
                        cleaned_record[key] = int(value)
                    else:
                        cleaned_record[key] = value
                cleaned_records.append(cleaned_record)
            _earnings_status_cache = cleaned_records
        else:
            _earnings_status_cache = []
        print(f"Cached {len(_earnings_status_cache)} earnings records")
    except Exception as e:
        print(f"Error caching earnings status: {e}")
        _earnings_status_cache = []
    
    yield
    
    # Cleanup on shutdown (if needed)
    print("Shutting down API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Driver Pulse API",
    description="Uber hackathon backend API for stress events and earnings velocity",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"message": "Driver Pulse API is running", "version": "1.0.0"}


@app.get("/api/stress_events")
def get_stress_events() -> List[Dict[str, Any]]:
    """
    Get stress events from the stress model processing pipeline.
    
    Returns:
        List of dictionaries containing flagged stress moments with all compliance columns
    """
    try:
        return _stress_events_cache
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stress events: {str(e)}")


@app.get("/api/earnings_status")
def get_earnings_status() -> List[Dict[str, Any]]:
    """
    Get earnings status from the earnings velocity processing pipeline.
    
    Returns:
        List of dictionaries containing earnings velocity metrics and forecasts
    """
    try:
        return _earnings_status_cache
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get earnings status: {str(e)}")


@app.get("/api/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "api": "Driver Pulse API"}


@app.get("/api/trip_insights")
def get_trip_insights() -> List[Dict[str, Any]]:
    """
    Get trip insights with LLM analysis.
    
    Returns:
        List of dictionaries containing trip insights and driver analysis
    """
    try:
        trip_insights_path = DATA_DIR / "processed_outputs" / "trip_insights_final.json"
        if not trip_insights_path.exists():
            return []
        
        with open(trip_insights_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data if isinstance(data, list) else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trip insights: {str(e)}")


@app.get("/api/driver_goals")
def get_driver_goals() -> List[Dict[str, Any]]:
    """
    Get driver goals and targets.
    
    Returns:
        List of dictionaries containing driver goals information
    """
    try:
        driver_goals_path = DATA_DIR / "earnings" / "driver_goals.csv"
        if not driver_goals_path.exists():
            return []
        
        df = pd.read_csv(driver_goals_path)
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get driver goals: {str(e)}")


@app.get("/api/drivers")
def get_drivers() -> List[Dict[str, Any]]:
    """
    Get driver information and profiles.
    
    Returns:
        List of dictionaries containing driver information
    """
    try:
        drivers_path = DATA_DIR / "drivers" / "drivers.csv"
        if not drivers_path.exists():
            return []
        
        df = pd.read_csv(drivers_path)
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get drivers: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
