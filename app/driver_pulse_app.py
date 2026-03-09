import pandas as pd
import streamlit as st
import os
import tempfile
import json
from pathlib import Path
from gtts import gTTS
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from deep_translator import GoogleTranslator

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.driver_insights import API_KEY, BASE_URL

DATA_DIR = BASE_DIR / "data"

FLAGGED_MOMENTS_CSV = BASE_DIR / "processed_outputs" / "flagged_moments.csv"
TRIP_INSIGHTS_JSON = BASE_DIR / "processed_outputs" / "trip_insights_final.json"
EARNINGS_VELOCITY_CSV = DATA_DIR / "earnings" / "earnings_velocity_log.csv"
DRIVER_GOALS_CSV = DATA_DIR / "earnings" / "driver_goals.csv"
DRIVERS_CSV = DATA_DIR / "drivers" / "drivers.csv"


@st.cache_data
def load_flagged_moments() -> pd.DataFrame:
    if not FLAGGED_MOMENTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(FLAGGED_MOMENTS_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_data
def load_trip_insights() -> pd.DataFrame:
    if not TRIP_INSIGHTS_JSON.exists():
        return pd.DataFrame()
    records = json.loads(TRIP_INSIGHTS_JSON.read_text(encoding="utf-8"))
    return pd.DataFrame(records)


@st.cache_data
def load_earnings_velocity() -> pd.DataFrame:
    if not EARNINGS_VELOCITY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(EARNINGS_VELOCITY_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_data
def load_driver_goals() -> pd.DataFrame:
    if not DRIVER_GOALS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(DRIVER_GOALS_CSV)
    return df


@st.cache_data
def load_drivers() -> pd.DataFrame:
    if not DRIVERS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(DRIVERS_CSV)


def style_severity(value: str) -> str:
    color_map = {
        "high": "#ff4b4b",
        "medium": "#ffa600",
        "low": "#2ecc71",
    }
    color = color_map.get(str(value).lower(), "#cccccc")
    return f"background-color: {color}; color: white; font-weight: 600;"

@st.cache_data
def get_text(text: str, target_lang_name: str) -> str:
    """Uses Google Translate for UI elements, cached for performance."""
    if target_lang_name == "English":
        return text
    
    indian_langs = {
        "Hindi (हिन्दी)": "hi", "Bengali (বাংলা)": "bn", "Marathi (मराठी)": "mr",
        "Telugu (తెలుగు)": "te", "Tamil (தமிழ்)": "ta", "Gujarati (ગુજરાતી)": "gu",
        "Kannada (ಕನ್ನಡ)": "kn", "Malayalam (മലയാളം)": "ml"
    }
    target_code = indian_langs.get(target_lang_name, "en")
    try:
        translated = GoogleTranslator(source='auto', target=target_code).translate(text)
        return translated if translated else text
    except:
        return text

def translate_text(text: str, target_lang_name: str) -> str:
    """Uses Google Translate for dynamic insights."""
    lang_map = {
        "Hindi (हिन्दी)": "hi", "Bengali (বাংলা)": "bn", "Marathi (मराठी)": "mr",
        "Telugu (తెలుగు)": "te", "Tamil (தமிழ்)": "ta", "Gujarati (ગુજરાતી)": "gu",
        "Kannada (ಕನ್ನಡ)": "kn", "Malayalam (മലയാളம்)": "ml"
    }
    target_code = lang_map.get(target_lang_name, "en")
    try:
        return GoogleTranslator(source='auto', target=target_code).translate(text)
    except:
        return text

def render_trip_overview(flagged_df: pd.DataFrame) -> None:
    lang_name = st.session_state.get("selected_lang_name", "English")
    st.subheader(get_text("Trip Summary", lang_name))

    if flagged_df.empty:
        st.info("No flagged moments available yet.")
        return

    drivers = sorted(flagged_df["driver_id"].dropna().unique().tolist())
    selected_driver = st.selectbox(f"{get_text('Driver', lang_name)} Filter", ["All"] + drivers, key="overview_driver_filter")
    
    view_df = flagged_df if selected_driver == "All" else flagged_df[flagged_df["driver_id"] == selected_driver]

    trip_summary = (
        view_df.groupby(["driver_id", "trip_id"], as_index=False)
        .agg(
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            flags_count=("flag_id", "count"),
            high_flags=("severity", lambda s: (s.str.lower() == "high").sum()),
            max_combined_score=("combined_score", "max"),
        )
        .sort_values("first_timestamp")
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(get_text("Total Trips", lang_name), len(trip_summary))
    col2.metric(get_text("Total Stress Events", lang_name), trip_summary["flags_count"].sum())
    col3.metric(get_text("Critical (High) Events", lang_name), trip_summary["high_flags"].sum())

    st.write("---")
    
    st.write(f"#### {get_text('Trip', lang_name)} Analysis")
    fig1 = px.bar(
        trip_summary, 
        x="trip_id", 
        y="flags_count", 
        color="flags_count", 
        color_continuous_scale=["#3b82f6", "#1e3a8a"],
        height=600,
        title="Volume of Stress Events by Trip"
    )
    fig1.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.scatter(
        trip_summary, 
        x="trip_id", 
        y="max_combined_score", 
        size="flags_count", 
        color="max_combined_score", 
        color_continuous_scale=["#3b82f6", "#1e3a8a"],
        size_max=40,
        height=600,
        title="Max Severity Score vs Event Frequency"
    )
    fig2.update_layout(margin=dict(t=40, b=40, l=40, r=40))
    st.plotly_chart(fig2, use_container_width=True)

    st.write("---")
    st.write("#### Mathematical Formulas for Stress Engine")
    st.latex(r"Motion\_Score = \max\left(0, \min\left(1, \frac{Jerk - 4.0}{4.0}\right)\right)")
    st.latex(r"Audio\_Score = \max\left(0, \min\left(1, \frac{AudioLevel - 85.0}{15.0}\right)\right)")
    st.latex(r"Combined\_Score = \max(Motion\_Score, Audio\_Score)")

    st.write("#### Detailed Trip Log (Raw CSV Export)")
    st.dataframe(view_df.head(100), use_container_width=True, hide_index=True)


def render_flagged_moments(flagged_df: pd.DataFrame, insights_df: pd.DataFrame) -> None:
    lang_name = st.session_state.get("selected_lang_name", "English")
    lang_code = st.session_state.get("selected_lang_code", "en")
    st.subheader(get_text("Incident Reports", lang_name))

    if flagged_df.empty:
        st.info("No flagged events found.")
        return

    drivers = sorted(flagged_df["driver_id"].dropna().unique().tolist())
    selected_driver = st.selectbox(get_text("Driver", lang_name), drivers, index=0 if drivers else None, key="flagged_driver_select")

    trips = sorted(flagged_df[flagged_df["driver_id"] == selected_driver]["trip_id"].dropna().unique().tolist())
    selected_trip = st.selectbox(get_text("Trip", lang_name), trips, index=0 if trips else None, key="flagged_trip_select")

    df_trip = flagged_df[(flagged_df["driver_id"] == selected_driver) & (flagged_df["trip_id"] == selected_trip)].copy()

    if df_trip.empty:
        st.warning("No flags for this selection.")
        return

    st.metric("Total Events", len(df_trip))

    st.write(f"#### {get_text('Event Intensity Timeline', lang_name)}")
    fig = px.bar(
        df_trip, x="timestamp", y=["motion_score", "audio_score"], 
        barmode="group", height=400,
        color_discrete_map={"motion_score": "#1e3a8a", "audio_score": "#93c5fd"}
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"**{get_text('Incident Map Location', lang_name)}**", unsafe_allow_html=True)
    if "gps_lat" in df_trip.columns and "gps_lon" in df_trip.columns:
        import folium
        from streamlit_folium import st_folium
        
        map_df = df_trip[["gps_lat", "gps_lon", "flag_type", "severity", "timestamp"]].dropna().copy()
        
        if not map_df.empty:
            center_lat = map_df["gps_lat"].mean()
            center_lon = map_df["gps_lon"].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")
            
            color_map = {"low": "lightblue", "medium": "blue", "high": "darkblue"}
            
            for idx, row in map_df.iterrows():
                # Extract localized strings for hover tags
                f_type = get_text(row['flag_type'].replace("_", " ").title(), lang_name)
                sev = get_text(row['severity'].upper(), lang_name)
                
                # Leaflet Marker
                folium.Marker(
                    location=[row["gps_lat"], row["gps_lon"]],
                    popup=f"<b>{sev}</b><br>{f_type}<br>{row['timestamp']}",
                    tooltip=f"{f_type} ({sev})",
                    icon=folium.Icon(color=color_map.get(row["severity"], "blue"), icon="info-sign")
                ).add_to(m)
                
            st_folium(m, use_container_width=True, height=450, returned_objects=[])
        else:
            st.info("No GPS data available for these incidents.")
    else:
        st.info("No GPS data available for these incidents.")
    
    st.write("---")

    if not insights_df.empty and "llm_insight" in insights_df.columns:
        merged = df_trip.merge(insights_df[["flag_id", "llm_insight"]], on="flag_id", how="left")
    else:
        merged = df_trip

    st.write(f"#### {get_text('Trip', lang_name)} Incident Stream")
    for idx, row in merged.iterrows():
        with st.container(border=True):
            cols = st.columns([1, 6, 1])
            sev = row["severity"].upper()
            sc = "#1e3a8a" if sev == "HIGH" else ("#3b82f6" if sev == "MEDIUM" else "#93c5fd")
            text_color = "white" if sev in ["HIGH", "MEDIUM"] else "#1e3a8a"
            
            cols[0].markdown(f"<div style='background:{sc}; color:{text_color}; padding:4px; border-radius:6px; text-align:center; font-weight:700;'>{sev}</div>", unsafe_allow_html=True)
            cols[1].markdown(f"**{row['flag_type'].replace('_', ' ').title()}** | {row['timestamp'].strftime('%H:%M:%S')}")
            
            insight = row.get("llm_insight", "Processing...")
            st.info(f"Insight: {insight}")
            
            with cols[2]:
                if st.button("🔊", key=f"btn_audio_{row['flag_id']}", help=f"Listen to Insight in {lang_name}"):
                    with st.spinner("Preparing Audio..."):
                        translated = translate_text(insight, lang_name)
                        speak_text(translated, lang_code)


def render_earnings_view(velocity_df: pd.DataFrame, goals_df: pd.DataFrame, drivers_df: pd.DataFrame) -> None:
    lang_name = st.session_state.get("selected_lang_name", "English")
    
    if velocity_df.empty or goals_df.empty:
        st.info("No earnings data available.")
        return

    common_drivers = sorted(set(velocity_df["driver_id"].unique()).intersection(goals_df["driver_id"].unique()))
    selected_driver = st.selectbox(f"{get_text('Driver', lang_name)} Filter", common_drivers, key="earnings_driver_select")

    vel_driver = velocity_df[velocity_df["driver_id"] == selected_driver].copy()
    goals_driver = goals_df[goals_df["driver_id"] == selected_driver].sort_values("date")

    goals_driver = goals_driver.sort_values("date")
    current_goal = goals_driver.iloc[-1] if not goals_driver.empty else None

    if not vel_driver.empty:
        latest = vel_driver.sort_values("timestamp").iloc[-1]
        curr, target, delta = latest["cumulative_earnings"], latest["target_velocity"], latest["velocity_delta"]
        status = latest["forecast_status"].upper()
    else:
        curr = target = delta = 0
        status = "N/A"

    st.write(f"### {get_text('Current Earnings', lang_name)}: ₹{curr:,.2f}")
    st.write(f"**{get_text('Target Hourly', lang_name)}:** ₹{target:.1f}/hr")
    st.write("---")

    c1, c2, c3 = st.columns(3)
    c1.metric(get_text("Goal Target", lang_name), f"₹{current_goal['target_earnings'] if current_goal is not None else 0:,.0f}")
    c2.metric(get_text("Velocity Delta", lang_name), f"₹{delta:.1f}")
    c3.metric(get_text("Forecast Status", lang_name), status)

    st.write("---")
    st.write("#### Velocity Formulas")
    st.latex(r"Current\_Velocity = \frac{Cumulative\_Earnings}{Elapsed\_Hours}")
    st.latex(r"Target\_Velocity = \frac{Target\_Earnings - Cumulative\_Earnings}{Remaining\_Hours}")
    st.latex(r"Velocity\_Delta = Current\_Velocity - Target\_Velocity")
    
    st.write(f"#### {get_text('Detailed Earnings Table (Raw Outputs)', lang_name)}")
    st.dataframe(vel_driver.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)


def render_how_it_works() -> None:
    lang_name = st.session_state.get("selected_lang_name", "English")
    st.subheader(get_text("How Driver Pulse Works", lang_name))
    st.markdown(get_text("Driver Pulse combines motion and audio telematics signals with earnings data to surface **glanceable, explainable insights** for Uber drivers.", lang_name))
    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, height=260):
            st.markdown(f"<h4 style='color: #2563eb;'>{get_text('Stress Detection Engine', lang_name)}</h4>", unsafe_allow_html=True)
            st.markdown(f"""
            - {get_text('Ingests sparse **accelerometer** and **audio intensity** streams.', lang_name)}
            - {get_text('Computes engineered features (e.g., horizontal jerk, rolling noise windows).', lang_name)}
            - {get_text('Applies strict rule-based logic to flag *harsh motion*, *sustained cabin noise*, and *critical conflicts*.', lang_name)}
            """)
        
        with st.container(border=True, height=260):
            st.markdown(f"<h4 style='color: #2563eb;'>{get_text('Earnings Velocity Model', lang_name)}</h4>", unsafe_allow_html=True)
            st.markdown(f"""
            - {get_text('Tracks cumulative earnings versus daily personalized goals.', lang_name)}
            - {get_text('Computes real-time *current pace (₹/hr)* versus *target velocity needed*.', lang_name)}
            - {get_text('Uses an **ML Model** (Random Forest) to safely forecast if a driver is *ahead*, *on_track*, or *at_risk*.', lang_name)}
            """)
            
    with col2:
        with st.container(border=True, height=260):
            st.markdown(f"<h4 style='color: #2563eb;'>{get_text('Flag Aggregation & Scoring', lang_name)}</h4>", unsafe_allow_html=True)
            st.markdown(f"""
            - {get_text('Collapses chaotic bursts of raw telematics samples into **single intuitive events**.', lang_name)}
            - {get_text('Computes standard **0-1 scale severity scores** for Motion and Audio intensity.', lang_name)}
            - {get_text('Generates strict, transparent explanations mapping exactly to the triggered sensors.', lang_name)}
            """)
            
        with st.container(border=True, height=260):
            st.markdown(f"<h4 style='color: #2563eb;'>{get_text('Drive Pulse LLM Insights', lang_name)}</h4>", unsafe_allow_html=True)
            st.markdown(f"""
            - {get_text('Integrates a fast, localized **LLM** for human-readable driver support.', lang_name)}
            - {get_text('Logic filters ensure only actionable *Medium* or *High* severity events consume API tokens.', lang_name)}
            - {get_text('Binds strict empathy and non-judgmental guardrails, along with Real-time **Google Translate & Text-to-Speech**.', lang_name)}
            """)


def render_test_api() -> None:
    lang_name = st.session_state.get("selected_lang_name", "English")
    st.subheader(get_text("Interactive Model Testing", lang_name))
    st.markdown(get_text("Use this section to generate random inputs and test how the backend models behave exactly like an API.", lang_name))
    
    test_type = st.radio(get_text("Select Model to Test:", lang_name), [get_text("Earnings Velocity & Goal Predictor", lang_name), get_text("Stress & Driver Insights", lang_name)])
    if st.button(get_text("Generate Random Values", lang_name)):
        import random
        st.session_state["rand_earned"] = round(random.uniform(50, 1500), 2)
        st.session_state["rand_target"] = round(random.uniform(1000, 2000), 2)
        st.session_state["rand_elapsed"] = round(random.uniform(1.0, 7.0), 2)
        st.session_state["rand_remaining"] = round(random.uniform(1.0, 7.0), 2)
        
        st.session_state["rand_jerk"] = round(random.uniform(1.0, 8.0), 2)
        st.session_state["rand_audio"] = round(random.uniform(60.0, 110.0), 2)
        st.session_state["rand_audio_class"] = random.choice(["normal", "quiet", "loud", "argument"])
        st.session_state["did_generate"] = True

    if not st.session_state.get("did_generate", False):
        st.info(get_text("Click 'Generate Random Values' above to populate the inputs.", lang_name))
        return

    st.write(f"### {get_text('Simulated Input Values', lang_name)}")
    if test_type == get_text("Earnings Velocity & Goal Predictor", lang_name):
        cols = st.columns(4)
        earned = cols[0].number_input(get_text("Earned (₹)", lang_name), value=st.session_state["rand_earned"])
        target = cols[1].number_input(get_text("Target (₹)", lang_name), value=st.session_state["rand_target"])
        elapsed_h = cols[2].number_input(get_text("Elapsed Hours", lang_name), value=st.session_state["rand_elapsed"])
        rem_h = cols[3].number_input(get_text("Remaining Hours", lang_name), value=st.session_state["rand_remaining"])
        
        st.write("---")
        st.write(f"### {get_text('Output', lang_name)}")
        import sys
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        from backend.earnings_velocity import compute_current_velocity, compute_target_velocity, forecast_status
        from backend.goal_predictor import load_model, predict_single
        
        current_v = compute_current_velocity(earned, elapsed_h)
        target_v = compute_target_velocity(target, earned, rem_h)
        rule_status = forecast_status(current_v, target, earned, rem_h, elapsed_h)
        
        st.json({
            "module": "earnings_velocity",
            "current_velocity_per_hr": current_v,
            "target_velocity_needed": target_v,
            "rule_based_status": rule_status
        })
        
        try:
            model, encoder = load_model()
            pct_earned = earned / max(target, 0.01)
            target_hours = elapsed_h + rem_h
            pct_time_used = elapsed_h / max(target_hours, 0.01)
            ideal_v = target / max(target_hours, 0.01)
            velocity_ratio = (current_v or 0) / max(ideal_v, 0.01)
            
            ml_pred = predict_single(
                model, encoder,
                pct_earned=pct_earned,
                pct_time_used=pct_time_used,
                velocity_ratio=velocity_ratio,
                earnings_velocity=current_v or 0,
                hours_remaining=rem_h,
                experience_months=24,
                rating=4.8
            )
            st.json({
                "module": "goal_predictor_ml",
                "ml_forecast": ml_pred["forecast"],
                "confidence": ml_pred["confidence"],
                "class_probabilities": ml_pred["probabilities"]
            })
        except Exception as e:
            st.warning(f"{get_text('ML Model not trained or failed to load. Try running the backend pipeline first. Error:', lang_name)} {str(e)}")
            
    else:
        cols = st.columns(3)
        jerk = cols[0].number_input(get_text("Horizontal Jerk (m/s²)", lang_name), value=st.session_state["rand_jerk"])
        audio_level = cols[1].number_input(get_text("Audio Level (dB)", lang_name), value=st.session_state["rand_audio"])
        audio_class = cols[2].selectbox(get_text("Audio Class", lang_name), ["normal", "quiet", "loud", "argument"], index=["normal", "quiet", "loud", "argument"].index(st.session_state["rand_audio_class"]))

        st.write("---")
        st.write(f"### {get_text('Output', lang_name)}")
        import sys
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        import numpy as np
        
        harsh_motion = (jerk > 4.0)
        sustained_noise = (audio_level > 85) and (audio_class == "argument")
        critical_conflict = harsh_motion and sustained_noise

        flag = None
        flag_type = "none"
        if critical_conflict:
            flag = "CRITICAL_CONFLICT"
            flag_type = "conflict_moment"
        elif harsh_motion:
            flag = "HARSH_MOTION"
            flag_type = "harsh_braking"
        elif sustained_noise:
            flag = "SUSTAINED_NOISE"
            flag_type = "audio_spike"
            
        motion_score = round(float(np.clip((jerk - 4.0) / 4.0, 0.0, 1.0)), 2)
        audio_score = round(float(np.clip((audio_level - 85.0) / 15.0, 0.0, 1.0)), 2)
        combined_score = max(motion_score, audio_score)
        
        severity = "low"
        if combined_score >= 0.7: severity = "high"
        elif combined_score >= 0.4: severity = "medium"

        explanation = "Normal Context"
        if flag_type == "conflict_moment":
             explanation = f"Combined signal: Harsh braking ({jerk} m/s^2) + sustained high audio ({int(audio_level)} dB)"
        elif flag_type == "harsh_braking":
             explanation = f"Harsh braking detected ({jerk} m/s^2) with audio level ({int(audio_level)} dB)"
        elif flag_type == "audio_spike":
             explanation = f"Sustained high audio detected ({int(audio_level)} dB) during {audio_class}"
             
        res = {
            "module": "stress_model",
            "stress_flag": flag,
            "flag_type": flag_type,
            "severity": severity,
            "scores": {
               "motion": motion_score,
               "audio": audio_score,
               "combined": combined_score
            },
            "explanation": explanation
        }
        st.json(res)
        
        if flag and severity in ["medium", "high"]:
            from backend.driver_insights import generate_llm_insight, API_KEY, BASE_URL
            from openai import OpenAI
            client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            try:
                insight = generate_llm_insight(explanation, "2024-02-06 14:30:00", "Alex (Mock)", client)
                st.write("#### LLM Driver Insight (API Output)")
                st.success(f"Drive Pulse App says: '{insight}'")
                
                lang_name = st.session_state.get("selected_lang_name", "English")
                lang_code = st.session_state.get("selected_lang_code", "en")
                if st.button("🔊", key="voice_test_api", help=f"Listen in {lang_name}"):
                    with st.spinner(f"Translating to {lang_name}..."):
                        translated = translate_text(insight, lang_name)
                        st.write(f"*{lang_name}: {translated}*")
                        speak_text(translated, lang_code)
            except Exception as e:
                st.error("Could not fetch LLM insight. " + str(e))
        else:
            st.info("severity is low, no LLM insight triggered.")




def speak_text(text: str, lang_code: str):
    """Generates and plays audio for the given text and language."""
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3", autoplay=True)
    except Exception as e:
        st.error(f"Voice error: {e}")




def main() -> None:
    st.set_page_config(
        page_title="Driver Pulse",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(f"""
        <style>
        [data-testid="stSidebar"] {{
            border-right: 1px solid #e5e7eb;
        }}
        /* Completely neutralise any fake markdown links so they can't be clicked or seen */
        .stMarkdown a {{
            pointer-events: none !important;
            text-decoration: none !important;
            color: inherit !important;
            cursor: default !important;
        }}
        .header-anchor {{
            display: none !important;
        }}
        /* Make radio buttons larger and span the full width */
        div[role='radiogroup'] > label {{
            padding: 12px 15px;
            font-size: 1.15rem !important;
            font-weight: 500;
            cursor: pointer;
            width: 100%;
            border-radius: 6px;
            margin-bottom: 5px;
            background-color: #f3f4f6;
        }}
        div[role='radiogroup'] > label:hover {{
            background-color: #e5e7eb;
        }}
        div[role='radiogroup'] span {{
            font-size: 1.15rem !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='font-size: 4rem; color: #111827; padding-bottom: 20px;'>Driver Pulse <span style='font-size: 1.5rem; color: #3b82f6; vertical-align: middle;'>Analytics Engine</span></h1>", unsafe_allow_html=True)
    
    indian_langs = {
        "English": "en", "Hindi (हिन्दी)": "hi", "Bengali (বাংলা)": "bn", 
        "Marathi (मराठी)": "mr", "Telugu (తెలుగు)": "te", "Tamil (தமிழ்)": "ta",
        "Gujarati (ગુજરાતી)": "gu", "Kannada (ಕನ್ನಡ)": "kn", "Malayalam (മലയാളം)": "ml"
    }
    
    st.sidebar.markdown(f"### {get_text('Language', st.session_state.get('selected_lang_name', 'English'))}")
    default_lang = st.session_state.get("selected_lang_name", "English")
    selected_lang_name = st.sidebar.selectbox("Preferred Language", list(indian_langs.keys()), 
                                              index=list(indian_langs.keys()).index(default_lang), label_visibility="collapsed")
    st.session_state["selected_lang_name"] = selected_lang_name
    st.session_state["selected_lang_code"] = indian_langs[selected_lang_name]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {get_text('Navigation', selected_lang_name)}")

    flagged_df = load_flagged_moments()
    insights_df = load_trip_insights()
    velocity_df = load_earnings_velocity()
    goals_df = load_driver_goals()
    drivers_df = load_drivers()

    tab_names = [
        get_text("Trip Summary", selected_lang_name),
        get_text("Flagged Moments", selected_lang_name),
        get_text("Earnings Velocity", selected_lang_name),
        get_text("Test API", selected_lang_name),
        get_text("System Architecture", selected_lang_name)
    ]
    
    tab_keys = ["trip_summary", "flagged_moments", "earnings_velocity", "test_api", "system_architecture"]
    
    if "tab" in st.query_params:
        try:
            default_tab_idx = tab_keys.index(st.query_params["tab"])
        except ValueError:
            default_tab_idx = 0
    else:
        default_tab_idx = st.session_state.get("active_tab_idx", 0)

    active_tab = st.sidebar.radio("Menu", tab_names, index=default_tab_idx, label_visibility="collapsed")
    
    current_idx = tab_names.index(active_tab)
    st.session_state["active_tab_idx"] = current_idx
    st.query_params["tab"] = tab_keys[current_idx]

    if active_tab == tab_names[0]: 
        render_trip_overview(flagged_df)
    elif active_tab == tab_names[1]: 
        render_flagged_moments(flagged_df, insights_df)
    elif active_tab == tab_names[2]: 
        render_earnings_view(velocity_df, goals_df, drivers_df)
    elif active_tab == tab_names[3]: 
        render_test_api()
    elif active_tab == tab_names[4]: 
        render_how_it_works()

if __name__ == "__main__":
    main()

