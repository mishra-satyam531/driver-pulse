import pandas as pd
import numpy as np
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

FLAGGED_MOMENTS_CSV = DATA_DIR / "processed_outputs" / "flagged_moments.csv"
TRIP_INSIGHTS_JSON = DATA_DIR / "processed_outputs" / "trip_insights_final.json"
EARNINGS_VELOCITY_CSV = DATA_DIR / "earnings" / "earnings_velocity_log.csv"
DRIVER_GOALS_CSV = DATA_DIR / "earnings" / "driver_goals.csv"
DRIVERS_CSV = DATA_DIR / "drivers" / "drivers.csv"


@st.cache_data(ttl=2)
def load_flagged_moments() -> pd.DataFrame:
    if not FLAGGED_MOMENTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(FLAGGED_MOMENTS_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    if "combined_score" in df.columns:
        df["combined_score"] = df["combined_score"].fillna(0)
    if "motion_score" in df.columns:
        df["motion_score"] = df["motion_score"].fillna(0)
    if "audio_score" in df.columns:
        df["audio_score"] = df["audio_score"].fillna(0)
    return df


@st.cache_data(ttl=2)
def load_trip_insights() -> pd.DataFrame:
    if not TRIP_INSIGHTS_JSON.exists():
        return pd.DataFrame()
    records = json.loads(TRIP_INSIGHTS_JSON.read_text(encoding="utf-8"))
    return pd.DataFrame(records)


@st.cache_data(ttl=2)
def load_earnings_velocity() -> pd.DataFrame:
    if not EARNINGS_VELOCITY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(EARNINGS_VELOCITY_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    
    new_rows = []
    for driver in df['driver_id'].dropna().unique()[:5]: 
        base_time = pd.Timestamp("2024-02-06 07:00:00")
        for i in range(15):
            t = base_time + pd.Timedelta(minutes=30*i)
            earned = 100 + (30 * i) + np.random.randint(-10, 20)
            elapsed = 0.5 * (i + 1)
            cur_v = earned / elapsed
            new_rows.append({
                "log_id": f"VEL_GEN_{driver}_{i}",
                "driver_id": driver,
                "date": "2024-02-06",
                "timestamp": t,
                "cumulative_earnings": earned,
                "elapsed_hours": elapsed,
                "current_velocity": cur_v,
                "target_velocity": 175.0,
                "velocity_delta": cur_v - 175.0,
                "trips_completed": i + 1,
                "forecast_status": "ahead" if cur_v > 175 else "at_risk"
            })
    
    if new_rows:
        gen_df = pd.DataFrame(new_rows)
        df = df[~df['log_id'].astype(str).str.startswith("VEL_GEN_")]
        df = pd.concat([df, gen_df], ignore_index=True)
        
    df = df.sort_values("timestamp")
    df["timestamp"] = df["timestamp"].dt.strftime("%H:%M")
    return df


@st.cache_data(ttl=2)
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
        "high": "#dc2626", # Red
        "medium": "#f59e0b", # Orange
        "low": "#fcd34d", # Yellow
    }
    color = color_map.get(str(value).lower(), "#e5e7eb")
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
        view_df.sort_values("timestamp"), 
        x="timestamp", 
        y="combined_score", 
        color="trip_id",
        height=450,
        title="Individual Stress Event Intensity",
        labels={"combined_score": "Combined Score", "timestamp": "Time"},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig1.update_layout(margin=dict(t=40, b=40, l=40, r=40), font=dict(family="Inter, sans-serif"), legend_title_text="Trip ID")
    st.plotly_chart(fig1, use_container_width=True)
    
    fig2 = px.scatter(
        view_df.sort_values("timestamp"), 
        x="timestamp", 
        y="combined_score", 
        size="combined_score",
        color="severity",
        height=450,
        title="Incident Severity Timeline",
        labels={"combined_score": "Score", "timestamp": "Time"},
        template="plotly_white",
        color_discrete_map={"low": "#fcd34d", "medium": "#f59e0b", "high": "#dc2626"}
    )
    fig2.update_layout(margin=dict(t=40, b=40, l=40, r=40), font=dict(family="Inter, sans-serif"), legend_title_text="Severity")
    st.plotly_chart(fig2, use_container_width=True)



    st.write("#### Detailed Trip Log (Raw CSV Export)")
    st.dataframe(view_df.head(100), use_container_width=True, hide_index=True)


def render_flagged_moments(flagged_df: pd.DataFrame, insights_df: pd.DataFrame) -> None:
    lang_name = st.session_state.get("selected_lang_name", "English")
    lang_code = st.session_state.get("selected_lang_code", "en")
    st.subheader(get_text("Incident Reports", lang_name))

    if flagged_df.empty:
        st.info("No flagged events found.")
        return

    original_drivers = sorted(flagged_df["driver_id"].dropna().unique().tolist())
    
    # Map IDs to actual driver names for a richer UI
    drivers_df = load_drivers()
    driver_mapping = {}
    for d in original_drivers:
        name = d
        if not drivers_df.empty:
            row = drivers_df[drivers_df["driver_id"] == d]
            if not row.empty:
                name = f"{row['name'].iloc[0]} ({d})"
        driver_mapping[d] = name
        
    drivers_display = [driver_mapping[d] for d in original_drivers]
    selected_display = st.selectbox(get_text("Driver", lang_name), drivers_display, index=0 if drivers_display else None, key="flagged_driver_select")
    
    selected_driver = None
    for d, disp in driver_mapping.items():
        if disp == selected_display:
            selected_driver = d
            break

    trips = sorted(flagged_df[flagged_df["driver_id"] == selected_driver]["trip_id"].dropna().unique().tolist())
    selected_trip = st.selectbox(get_text("Trip", lang_name), trips, index=0 if trips else None, key="flagged_trip_select")

    df_trip = flagged_df[(flagged_df["driver_id"] == selected_driver) & (flagged_df["trip_id"] == selected_trip)].copy()

    if df_trip.empty:
        st.warning("No flags for this selection.")
        return

    st.metric("Total Events", len(df_trip))

    fig = px.bar(
        df_trip, x="timestamp", y=["motion_score", "audio_score"], 
        barmode="group", height=400,
        color_discrete_map={"motion_score": "#f59e0b", "audio_score": "#fcd34d"}
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="", font=dict(family="Inter, sans-serif"))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"**{get_text('Incident Map Location', lang_name)}**", unsafe_allow_html=True)
    if "gps_lat" in flagged_df.columns and "gps_lon" in flagged_df.columns:
        import folium
        from streamlit_folium import st_folium
        
        all_driver_flags = flagged_df[flagged_df["driver_id"] == selected_driver].dropna(subset=["gps_lat", "gps_lon"]).copy()
        
        if not all_driver_flags.empty:
            center_lat = all_driver_flags["gps_lat"].mean()
            center_lon = all_driver_flags["gps_lon"].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="CartoDB positron")
            
            # Map severity to hex colors
            color_map = {"low": "#fcd34d", "medium": "#f59e0b", "high": "#dc2626"}
            
            for idx, row in all_driver_flags.iterrows():
                f_type = get_text(row['flag_type'].replace("_", " ").title(), lang_name)
                sev = get_text(row['severity'].upper(), lang_name)
                dot_color = color_map.get(row["severity"], "#9ca3af")
                
                # Add deterministic jitter to separate overlapping markers
                jitter_lat = row["gps_lat"] + (0.00015 * (idx % 4 - 1.5))
                jitter_lon = row["gps_lon"] + (0.00015 * (idx % 2 - 0.5))
                
                is_selected_trip = (row["trip_id"] == selected_trip)
                gmaps_url = f"https://www.google.com/maps?q={row['gps_lat']},{row['gps_lon']}"
                
                popup_html = f"""
                <div style="font-family: 'Inter', sans-serif; font-size: 14px; min-width: 150px;">
                    <div style="font-weight: 700; margin-bottom: 2px;">{sev}: {f_type}</div>
                    <div style="font-size: 11px; color: #6b7280; margin-bottom: 8px;">Trip: {row['trip_id']}</div>
                    <a href="{gmaps_url}" target="_blank" style="display: block; background: #1e3a8a; color: white; padding: 8px 12px; border-radius: 4px; text-decoration: none; text-align: center; font-size: 12px; font-weight: 600;">
                        View on Maps
                    </a>
                </div>
                """
                
                folium.CircleMarker(
                    location=[jitter_lat, jitter_lon],
                    radius=12 if is_selected_trip else 8,
                    popup=folium.Popup(popup_html, max_width=300),
                    color=dot_color,
                    fill=True,
                    fill_color=dot_color,
                    fill_opacity=0.9 if is_selected_trip else 0.4,
                    weight=3 if is_selected_trip else 1
                ).add_to(m)
                
            # Add path line between points
            if len(all_driver_flags) > 1:
                path_points = all_driver_flags[["gps_lat", "gps_lon"]].values.tolist()
                folium.PolyLine(path_points, color="#1e3a8a", weight=2, opacity=0.6).add_to(m)

            st_folium(m, use_container_width=True, height=500, returned_objects=[], key="incident_report_map")
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
            sc = "#dc2626" if sev == "HIGH" else ("#f59e0b" if sev == "MEDIUM" else "#fcd34d")
            text_color = "white" if sev == "HIGH" else "#111827"
            
            cols[0].markdown(f"<div style='background:{sc}; color:{text_color}; padding:4px; border-radius:6px; text-align:center; font-weight:700;'>{sev}</div>", unsafe_allow_html=True)
            cols[1].markdown(f"**{row['flag_type'].replace('_', ' ').title()}** | {row['timestamp'].strftime('%H:%M:%S')}")
            cols[1].markdown(f"<span style='font-size: 0.8rem; color: #6b7280;'>Location: {row['gps_lat']:.4f}, {row['gps_lon']:.4f}</span>", unsafe_allow_html=True)
            
            # Confidence Badge & Feature Contributions
            conf_score = row["combined_score"]
            conf_label = "High Confidence" if conf_score > 0.8 else ("Medium Confidence" if conf_score > 0.4 else "Low Confidence")
            conf_color = "#10b981" if conf_score > 0.8 else ("#f59e0b" if conf_score > 0.4 else "#6b7280")
            
            cols[1].markdown(f"<span style='background:{conf_color}; color:white; padding:2px 6px; border-radius:4px; font-size:10px; font-weight:700; vertical-align:middle; margin-left:10px;'>{conf_label}</span>", unsafe_allow_html=True)
            
            # Feature Contributions
            with st.expander(get_text("Explainability (Feature Contributions)", lang_name)):
                expl_col1, expl_col2 = st.columns(2)
                expl_col1.metric("Motion Impact", f"{int(row['motion_score']*100)}%")
                expl_col2.metric("Audio Impact", f"{int(row['audio_score']*100)}%")
                st.progress(row["motion_score"], text="Motion Weight")
                st.progress(row["audio_score"], text="Audio Weight")

            insight = row.get("llm_insight", "Processing...")
            st.info(f"Insight: {insight}")
            
            with cols[2]:
                if st.button("Listen", key=f"btn_audio_{row['flag_id']}", help=f"Listen to Insight in {lang_name}"):
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
    current_goal = goals_driver.iloc[-1] if not goals_driver.empty else None

    if not vel_driver.empty:
        latest = vel_driver.sort_values("timestamp").iloc[-1]
        curr = float(latest.get("cumulative_earnings", 0))
        target_v = float(latest.get("target_velocity", 0))
        delta = float(latest.get("velocity_delta", 0))
        status = latest.get("forecast_status", "N/A").upper()
        trips = int(latest.get("trips_completed", 0))
        curr_v = float(latest.get("current_velocity", 0))
        elapsed = float(latest.get("elapsed_hours", 0))
    else:
        curr = target_v = delta = curr_v = trips = elapsed = 0
        status = "N/A"

    target_e = float(current_goal['target_earnings']) if current_goal is not None else 0
    target_hours = float(current_goal['target_hours']) if current_goal is not None and 'target_hours' in current_goal else 8.0
    
    remaining = max(target_hours - elapsed, 0)
    projected = curr + (curr_v * remaining) if curr_v else curr
    
    goal_progress_pct = min(int((curr / target_e) * 100), 100) if target_e > 0 else 0

    st.markdown("<style>.metric-card { border-radius: 8px; padding: 16px; background-color: #ffffff; border: 1px solid #e5e7eb; color: #111827; display: flex; flex-direction: column; justify-content: space-between; height: 100%; }</style>", unsafe_allow_html=True)
    
    st.write("")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.8rem; font-weight: 700; margin-bottom: 8px; text-transform: uppercase;">{get_text('TOTAL EARNINGS TODAY', lang_name)}</div>
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 8px;">₹{curr:,.2f}</div>
            <div style="font-size: 0.85rem; color: #6b7280;">{trips} {get_text('trips completed', lang_name)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.8rem; font-weight: 700; margin-bottom: 8px; text-transform: uppercase;">{get_text('CURRENT VELOCITY', lang_name)}</div>
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 8px;">₹{curr_v:,.2f}/hr</div>
            <div style="font-size: 0.85rem; color: #6b7280;">{get_text('Target:', lang_name)} ₹{target_v:,.2f}/hr</div>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.8rem; font-weight: 700; margin-bottom: 8px; text-transform: uppercase;">{get_text('GOAL PROGRESS', lang_name)}</div>
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 8px;">{goal_progress_pct}%</div>
            <div style="font-size: 0.85rem; color: #6b7280;">₹{curr:,.0f} / ₹{target_e:,.0f}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.8rem; font-weight: 700; margin-bottom: 8px; text-transform: uppercase;">{get_text('PROJECTED FINAL', lang_name)}</div>
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 8px;">₹{projected:,.0f}</div>
            <div style="font-size: 0.85rem; color: #6b7280;">{get_text('End of shift forecast', lang_name)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    
    fc_col1, fc_col2 = st.columns([1.5, 1])
    with st.sidebar:
        st.write("---")
        st.subheader(get_text("Set Daily Goals", lang_name))
        new_target_e = st.number_input(get_text("Target Earnings (₹)", lang_name), value=target_e, step=100.0)
        new_target_h = st.number_input(get_text("Target Hours", lang_name), value=target_hours, step=0.5)
        if new_target_e != target_e or new_target_h != target_hours:
            target_e = new_target_e
            target_hours = new_target_h
            # Recalculate based on new goals
            remaining = max(target_hours - elapsed, 0)
            projected = curr + (curr_v * remaining) if curr_v else curr
            goal_progress_pct = min(int((curr / target_e) * 100), 100) if target_e > 0 else 0

    with fc_col1:
        st.markdown(f"### {get_text('Goal Achievement Forecast', lang_name)}")
        message = "You're on track to meet your goal." if status == "ON_TRACK" else ("You are ahead of your goal." if status == "AHEAD" else "You might fall short of your goal.")
        
        st.markdown(f"""
        <div style="border: 1px solid #e5e7eb; border-radius: 12px; padding: 24px; background: white; margin-top: 16px;">
            <div style="display: flex; align-items: center; color: #111827; font-weight: 600; font-size: 1.1rem;">
                {get_text(message, lang_name)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with fc_col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=goal_progress_pct,
            number={'suffix': "%"},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': get_text("Goal Progress", lang_name), 'font': {'size': 18, 'family': 'Inter, sans-serif'}},
            delta={'reference': 100, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#dc2626"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#4b5563"},
                'bar': {'color': "#1e3a8a"},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#e5e7eb",
                'steps': [
                    {'range': [0, 50], 'color': '#f9fafb'},
                    {'range': [50, 80], 'color': '#f3f4f6'},
                    {'range': [80, 100], 'color': '#e5e7eb'}
                ]
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.markdown(f"### {get_text('Earnings Over Time', lang_name)}")
    
    if not vel_driver.empty:
        vel_driver = vel_driver.sort_values("timestamp")
        line_fig = px.area(vel_driver, x="timestamp", y="cumulative_earnings", 
                           title="Cumulative Earnings vs Final Target", 
                           labels={"cumulative_earnings": "Earnings (₹)", "timestamp": "Time"},
                           template="plotly_white",
                           color_discrete_sequence=["#1e3a8a"])
        line_fig.data[0].name = "Earnings"
        line_fig.data[0].showlegend = True
        line_fig.add_scatter(x=vel_driver["timestamp"], y=[target_e]*len(vel_driver), mode="lines", name="Final Target", line=dict(color="#10b981", dash="dash"))
        line_fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), font=dict(family="Inter, sans-serif"))
        st.plotly_chart(line_fig, use_container_width=True)
        
        vel_driver["prorated_target"] = (target_e / target_hours) * vel_driver["elapsed_hours"]
        line_fig2 = go.Figure()
        line_fig2.add_trace(go.Scatter(x=vel_driver["timestamp"], y=vel_driver["cumulative_earnings"], mode="lines+markers", name="Earned by Time", line=dict(color="#1e3a8a")))
        line_fig2.add_trace(go.Scatter(x=vel_driver["timestamp"], y=vel_driver["prorated_target"], mode="lines+markers", name="Target by Time", line=dict(color="#10b981", dash="dash")))
        line_fig2.update_layout(title="Earnings Pace vs Target Pace", template="plotly_white", height=450, margin=dict(l=20, r=20, t=40, b=20), font=dict(family="Inter, sans-serif"))
        st.plotly_chart(line_fig2, use_container_width=True)

    st.write("---")
    st.markdown(f"### {get_text('Hourly Breakdown', lang_name)}")
    st.dataframe(vel_driver.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)


def render_how_it_works() -> None:
    lang_name = st.session_state.get("selected_lang_name", "English")
    st.markdown(get_text("Driver Pulse integrates telematics signals with real-time analytics to surface proactive, data-driven security for Uber partners.", lang_name))
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
            - {get_text('Integrates a localized analysis engine for driver support.', lang_name)}
            - {get_text('Logic filters ensure only actionable events consume backend resources.', lang_name)}
            - {get_text('Built-in real-time translation and human-voice audio feedback.', lang_name)}
            """)

    st.write("---")
    st.subheader(get_text("Under The Hood: Architecture & Math", lang_name))
    tab_math, tab_arch, tab_viz = st.tabs([
        get_text("Engine Mathematical Formulas", lang_name), 
        get_text("System Constraints & Privacy", lang_name),
        get_text("Visual Strategy & Flow", lang_name)
    ])
    
    with tab_math:
        st.write(f"##### {get_text('Stress Engine (Physics & Audio)', lang_name)}")
        st.latex(r"Motion\_Score = \max\left(0, \min\left(1, \frac{Jerk - 4.0}{4.0}\right)\right)")
        st.latex(r"Audio\_Score = \max\left(0, \min\left(1, \frac{AudioLevel - 85.0}{15.0}\right)\right)")
        st.latex(r"Combined\_Score = \max(Motion\_Score, Audio\_Score)")
        st.write("")
        st.write(f"##### {get_text('Earnings Velocity Planner', lang_name)}")
        st.latex(r"Current\_Velocity = \frac{Cumulative\_Earnings}{Elapsed\_Hours}")
        st.latex(r"Target\_Velocity = \frac{Target\_Earnings - Cumulative\_Earnings}{Remaining\_Hours}")
        st.latex(r"Velocity\_Delta = Current\_Velocity - Target\_Velocity")

    with tab_arch:
        st.write(f"##### {get_text('Network Resilience (Decoupled Offline Queues)', lang_name)}")
        st.markdown(f"{get_text('**Constraint:** Cars lose cellular connectivity in garages, tunnels, and rural borders. <br>**Solution:** The client caches & batches resampled telematics locally. If the API drops, arrays wait. When reconnected, the Backend Engine uses absolute Unix timestamps (`pd.merge_asof`) so entirely out-of-order, delayed events still successfully generate contextually accurate stress insights.', lang_name)}", unsafe_allow_html=True)
        
        st.write(f"##### {get_text('Ethical Audio Surveillance (Physical Minimization)', lang_name)}")
        st.markdown(f"{get_text('**Constraint:** You cannot natively record a passenger conversation. Period. <br>**Solution:** The Edge Node pushes the microphone through a hard DSP filter natively on the device. **The cloud never touches raw audio**. The extractor locally calculates a rolling mean in decibels (`dB`), creating an anonymous 1D numerical array. The Cloud receives `85.2 dB` practically disabling spying.', lang_name)}", unsafe_allow_html=True)

        st.write(f"##### {get_text('Battery Survival (Offloaded Generation)', lang_name)}")
        st.markdown(f"{get_text('**Constraint:** High-frequency continuous polling destroys Uber phones instantly. <br>**Solution:** Rather than streaming continuous WebSockets, the device pings batched chunks. Expensive processes (Random Forest Predictors, LLM Generation) trigger remotely on the backend. The dashboard simply acts as a thin Read-only UI over cached outputs.', lang_name)}", unsafe_allow_html=True)

    with tab_viz:
        import base64
        
        st.write(f"##### {get_text('System Architecture (Data Flow)', lang_name)}")
        chart1 = """
        flowchart TD
            subgraph Edge [The Driver's Device]
                S1[IMU / Accelerometer]
                S2[Microphone / Audio]
                Filter[DSP Filter \n 1D Envelope]
                UI[Streamlit Dashboard]
            end

            subgraph Cloud [Cloud Intelligence]
                Lake[(Time-Series Lake)]
                StressCore{Stress Logic Engine}
                VeloCore{Earnings Predictor}
                LLM[GenAI Insight LLM]
                DB[(Insights DB)]
            end

            S1 & S2 --> Filter
            Filter -->|Encrypted Batches| Lake
            Lake --> StressCore & VeloCore
            StressCore & VeloCore --> DB
            DB -->|High Severity| LLM
            LLM -->|Empathetic Scripts| DB
            DB -->|Fetch| UI
        """
        b64_1 = base64.b64encode(chart1.encode('ascii')).decode('ascii')
        st.image(f"https://mermaid.ink/img/{b64_1}", use_container_width=True)
        
        st.write("---")
        st.write(f"##### {get_text('Stress Detection Pipeline', lang_name)}")
        chart2 = """
        flowchart LR
            A[Raw Telemetry] --> B[Feature Engineering]
            B --> C[Sensor Fusion]
            C --> D[Rule Engine]
            D --> E[Event Aggregator]
            E --> F[Flagged Output]
            
            subgraph Details
                B1[Horizontal Jerk] --> B
                B2[Rolling Audio] --> B
                D1[Jerk > 4.0] --> D
                D2[Audio > 85dB] --> D
            end
        """
        b64_2 = base64.b64encode(chart2.encode('ascii')).decode('ascii')
        st.image(f"https://mermaid.ink/img/{b64_2}", use_container_width=True)


def render_test_api() -> None:
    lang_name = st.session_state.get("selected_lang_name", "English")
    st.subheader(get_text("Interactive Model Testing", lang_name))
    
    test_mode_options = [get_text("Single Event Prediction", lang_name), get_text("Batch Processing (CSV)", lang_name)]
    test_mode = st.radio(get_text("Select Testing Mode:", lang_name), test_mode_options, horizontal=True, key="active_test_mode")
    
    st.write("---")
    
    if test_mode == test_mode_options[0]:
        # Single Event
        st.markdown(get_text("Enter individual values to test model edge cases.", lang_name))
        test_type = st.radio(get_text("Select Model to Test:", lang_name), [get_text("Earnings Velocity & Goal Predictor", lang_name), get_text("Stress & Driver Insights", lang_name)], key="single_test_type")
        
        if st.button(get_text("Generate Random Values", lang_name), key="btn_gen_rand"):
            import random
            st.session_state["rand_earned"] = round(random.uniform(50, 1500), 2)
            st.session_state["rand_target"] = round(random.uniform(1000, 2000), 2)
            st.session_state["rand_elapsed"] = round(random.uniform(1.0, 7.0), 2)
            st.session_state["rand_remaining"] = round(random.uniform(1.0, 7.0), 2)
            
            st.session_state["rand_jerk"] = round(random.uniform(1.0, 8.0), 2)
            st.session_state["rand_audio"] = round(random.uniform(60.0, 110.0), 2)
            st.session_state["rand_audio_class"] = random.choice(["normal", "quiet", "loud", "argument"])
            st.session_state["did_generate"] = True

        if st.session_state.get("did_generate", False):
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
                    st.warning(f"{get_text('ML Model not trained or failed to load. Error:', lang_name)} {str(e)}")
                    
            else:
                cols = st.columns(3)
                jerk = cols[0].number_input(get_text("Horizontal Jerk (m/s²)", lang_name), value=st.session_state["rand_jerk"])
                audio_level = cols[1].number_input(get_text("Audio Level (dB)", lang_name), value=st.session_state["rand_audio"])
                audio_class = cols[2].selectbox(get_text("Audio Class", lang_name), ["normal", "quiet", "loud", "argument"], index=["normal", "quiet", "loud", "argument"].index(st.session_state["rand_audio_class"]))

                st.write("---")
                st.write(f"### {get_text('Output', lang_name)}")
                
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
                    "scores": {"motion": motion_score, "audio": audio_score, "combined": combined_score},
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
                        if st.button(get_text("Listen to Insight", lang_name), key="voice_test_api"):
                            translated = translate_text(insight, lang_name)
                            speak_text(translated, lang_code)
                    except Exception as e:
                        st.error("Could not fetch LLM insight. " + str(e))
                else:
                    st.info("Severity is low, no LLM insight triggered.")
        else:
            st.info(get_text("Click 'Generate Random Values' above to populate the inputs.", lang_name))

    else:
        # Batch Processing
        st.markdown(f"#### {get_text('Step 1: Input Analysis Data', lang_name)}")
        col_inp1, col_sep, col_inp2 = st.columns([10, 1, 10])
        
        with col_inp1:
            st.write(f"**{get_text('Option A: Simulated Demo', lang_name)}**")
            demo_size = st.select_slider(get_text("Inference Window (Rows):", lang_name), options=[5, 10, 15, 30, 50, 100], value=15)
            if st.button(get_text("Search & Generate Analysis", lang_name)):
                try:
                    acc_path = os.path.join(BASE_DIR, "data", "sensor_data", "accelerometer_data.csv")
                    aud_path = os.path.join(BASE_DIR, "data", "sensor_data", "audio_intensity_data.csv")
                    if os.path.exists(acc_path) and os.path.exists(aud_path):
                        df_acc = pd.read_csv(acc_path)
                        df_aud = pd.read_csv(aud_path)
                        df_acc['timestamp'] = pd.to_datetime(df_acc['timestamp'], errors='coerce')
                        df_aud['timestamp'] = pd.to_datetime(df_aud['timestamp'], errors='coerce')
                        
                        full_df = pd.merge_asof(
                            df_acc.sort_values('timestamp'), 
                            df_aud.sort_values('timestamp'), 
                            on='timestamp', direction='nearest', tolerance=pd.Timedelta(seconds=30),
                            suffixes=('', '_aud')
                        )
                        # Official Sync: Instead of manual calculation (find flags), use the official flagged moments
                        flags_df = load_flagged_moments()
                        
                        if not flags_df.empty:
                            # Ensure both are tz-naive for subtraction
                            first_flag_time = pd.to_datetime(flags_df.iloc[0]['timestamp'], dayfirst=True)
                            if first_flag_time.tzinfo is not None:
                                first_flag_time = first_flag_time.tz_localize(None)
                            
                            # Ensure full_df timestamp is also naive
                            raw_ts = full_df['timestamp'].dt.tz_localize(None) if full_df['timestamp'].dt.tz is not None else full_df['timestamp']
                            
                            # Find row index closest to this official flag
                            timedeltas = (raw_ts - first_flag_time).abs()
                            start_idx = timedeltas.idxmin()
                        else:
                            start_idx = 0
                        
                        # Apply pre-incident context shift
                        start_idx = max(0, start_idx - 3)
                        
                        st.session_state["demo_batch_df"] = full_df.iloc[start_idx : start_idx + demo_size].copy()
                    else:
                        st.error("Sample data not found.")
                except Exception as e:
                    st.exception(e)

        with col_sep:
            # Vertical line trick
            st.markdown("<div style='border-left: 1px solid #ddd; height: 180px; margin: auto; width: 1px;'></div>", unsafe_allow_html=True)

        with col_inp2:
            st.write(f"**{get_text('Option B: Custom CSV Upload', lang_name)}**")
            uploaded_files = st.file_uploader(get_text("Upload Sensor Data", lang_name), type=["csv"], key="batch_upload_v2", accept_multiple_files=True)
            if uploaded_files:
                try:
                    dfs = []
                    for f in uploaded_files:
                         df_f = pd.read_csv(f)
                         if 'timestamp' in df_f.columns:
                             df_f['timestamp'] = pd.to_datetime(df_f['timestamp'], errors='coerce')
                         dfs.append(df_f)
                    if len(dfs) > 1:
                        final_df = dfs[0]
                        for i in range(1, len(dfs)):
                            if 'timestamp' in final_df.columns and 'timestamp' in dfs[i].columns:
                                cols_to_use = dfs[i].columns.difference(final_df.columns).tolist() + ['timestamp']
                                final_df = pd.merge_asof(final_df.sort_values('timestamp'), dfs[i][cols_to_use].sort_values('timestamp'), on='timestamp', direction='nearest', tolerance=pd.Timedelta(seconds=30))
                            else:
                                final_df = pd.concat([final_df, dfs[i]], axis=1)
                        st.session_state["demo_batch_df"] = final_df
                        st.success(get_text("Custom files merged successfully.", lang_name))
                    else:
                        st.session_state["demo_batch_df"] = dfs[0]
                except Exception as e:
                    st.error(f"Error: {e}")

        # The Output Section – Always at the bottom
        batch_df = st.session_state.get("demo_batch_df", None)
        if batch_df is not None:
            st.write("---")
            st.markdown(f"### {get_text('Batch Inference Output', lang_name)}")
            
            try:
                has_accel = all(c in batch_df.columns for c in ['accel_x', 'accel_y'])
                has_audio = 'audio_level' in batch_df.columns
                
                if has_accel or has_audio:
                    with st.spinner(get_text("Analyzing telemetry...", lang_name)):
                        # Import production logic directly from backend
                        import sys
                        if str(BASE_DIR) not in sys.path:
                            sys.path.insert(0, str(BASE_DIR))
                        from backend.stress_model import compute_motion_metrics, compute_audio_metrics, fuse_sensors, apply_stress_rules

                        # Safety: Ensure required columns and formatting for the backend model
                        if 'trip_id' not in batch_df.columns: batch_df['trip_id'] = 'test_trip'
                        if 'driver_id' not in batch_df.columns: batch_df['driver_id'] = 'test_driver'
                        if 'audio_class' not in batch_df.columns: batch_df['audio_class'] = 'quiet'

                        # 1. Compute Base Metrics (Production Pipeline)
                        processed_acc = compute_motion_metrics(batch_df) if has_accel else batch_df
                        processed_aud = compute_audio_metrics(batch_df) if has_audio else batch_df
                        
                        # Use production names for fuser
                        if 'Audio_Rolling_15s' not in processed_aud.columns and 'Audio_Rolling' in processed_aud.columns:
                            processed_aud['Audio_Rolling_15s'] = processed_aud['Audio_Rolling']
                        
                        # Ensure backend-compatible structure
                        if not has_accel:
                            processed_acc['Horizontal_Jerk'] = 0.0
                            processed_acc['Vertical_Jerk'] = 0.0
                        if 'Audio_Rolling_15s' not in processed_aud.columns:
                            processed_aud['Audio_Rolling_15s'] = 0.0
                        
                        # 2. Fuse & Apply Official Rules
                        # fuse_sensors expects specific structure
                        fused_df = fuse_sensors(processed_acc, processed_aud)
                        
                        # Crucial: the model checks audio_class
                        if 'audio_class' not in fused_df.columns:
                            fused_df['audio_class'] = 'quiet'
                        
                        # 3. Apply stress rules from backend
                        flagged_results = apply_stress_rules(fused_df)
                        
                        # 3. Calculate scores for the visualization (Timeline)
                        # We calculate these on the fused_df so we can plot the whole timeline
                        fused_df["motion_score"] = np.clip((fused_df["Horizontal_Jerk"] - 4.0) / 4.0, 0.0, 1.0)
                        # Mask bumps
                        fused_df.loc[fused_df["Vertical_Jerk"] > 2.0, "motion_score"] = 0.0
                        
                        fused_df["audio_score"] = np.clip((fused_df["Audio_Rolling_15s"] - 85.0) / 15.0, 0.0, 1.0)
                        fused_df["combined_score"] = np.maximum(fused_df["motion_score"], fused_df["audio_score"])
                        
                        # 4. Display Metrics
                        m1, m2 = st.columns(2)
                        m1.metric(get_text("Analytical Datapoints", lang_name), len(fused_df))
                        m2.metric(get_text("Flagged Moments", lang_name), len(flagged_results), delta=len(flagged_results) if len(flagged_results)>0 else None)
                        
                        st.write(f"#### {get_text('Risk Heatmap Timeline', lang_name)}")
                        st.line_chart(fused_df[['motion_score', 'audio_score', 'combined_score']])
                        
                        if not flagged_results.empty:
                            st.write(f"#### {get_text('Detected High-Risk Moments', lang_name)}")
                            # Enrich flagged_results with scores for the dataframe display
                            flagged_results["combined_score"] = np.maximum(
                                np.clip((flagged_results["Horizontal_Jerk"] - 4.0) / 4.0, 0, 1),
                                np.clip((flagged_results["Audio_Rolling_15s"] - 85.0) / 15.0, 0, 1)
                            )
                            st.dataframe(flagged_results.sort_values('combined_score', ascending=False).head(15), use_container_width=True)
                        else:
                            st.info(get_text("No high-risk moments detected in this specific window.", lang_name))
                else:
                    st.error("Data missing required sensor columns.")
            except Exception as e:
                import traceback
                st.error(f"Inference Error: {str(e)}")
                st.code(traceback.format_exc())

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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif !important;
        }}
        [data-testid="stSidebar"] {{
            border-right: 1px solid #e5e7eb;
        }}
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
    
    active_tab = st.sidebar.radio("Menu", tab_names, key="main_navigation", label_visibility="collapsed")
    
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

