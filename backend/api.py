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
from backend.stress_model import run_stress_moment_model
from backend.earnings_velocity import run_earnings_velocity_model

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


@app.post("/api/refresh")
def refresh_data():
    """
    Re-run the data pipeline and update the in-memory caches.
    
    Returns:
        Success message with updated cache status
    """
    global _stress_events_cache, _earnings_status_cache
    
    print("Refreshing stress events cache...")
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
        print(f"Refreshed {len(_stress_events_cache)} stress events")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh stress events: {str(e)}")
    
    print("Refreshing earnings status cache...")
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
        print(f"Refreshed {len(_earnings_status_cache)} earnings records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh earnings status: {str(e)}")
    
    return {
        "status": "success", 
        "message": "Data pipeline re-run and cache updated",
        "stress_events_count": len(_stress_events_cache),
        "earnings_records_count": len(_earnings_status_cache)
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
