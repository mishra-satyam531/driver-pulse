import json
from pathlib import Path
import pandas as pd
import streamlit as st
import pydeck as pdk
from plotly import graph_objects as go
import streamlit.components.v1 as components
from backend.compliance_log import generate_uber_compliance_log

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

FLAGGED_MOMENTS_CSV = DATA_DIR / "processed_outputs" / "flagged_moments.csv"
TRIP_INSIGHTS_JSON = DATA_DIR / "processed_outputs" / "trip_insights_final.json"
EARNINGS_VELOCITY_CSV = DATA_DIR / "earnings" / "earnings_velocity_log.csv"
DRIVER_GOALS_CSV = DATA_DIR / "earnings" / "driver_goals.csv"
DRIVERS_CSV = DATA_DIR / "drivers" / "drivers.csv"
COMPLIANCE_LOG_CSV = DATA_DIR / "processed_outputs" / "uber_compliance_log.csv"


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


@st.cache_data
def load_compliance_log() -> pd.DataFrame:
    if not COMPLIANCE_LOG_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPLIANCE_LOG_CSV)


def style_severity(value: str) -> str:
    color_map = {
        "high": "#ff4b4b",
        "medium": "#ffa600",
        "low": "#2ecc71",
    }
    color = color_map.get(str(value).lower(), "#cccccc")
    return f"background-color: {color}; color: white; font-weight: 600;"


def render_trip_overview(flagged_df: pd.DataFrame, velocity_df: pd.DataFrame) -> None:
    st.subheader("Live Driving")
    latest = None
    if not velocity_df.empty:
        latest = velocity_df.sort_values("timestamp").iloc[-1]
    current_earnings = latest["cumulative_earnings"] if latest is not None else None
    current_velocity = latest["current_velocity"] if latest is not None else None
    target_velocity = latest["target_velocity"] if latest is not None else None
    velocity_delta = latest["velocity_delta"] if latest is not None else None
    critical_flags = 0
    if not flagged_df.empty:
        if "severity" in flagged_df.columns:
            critical_flags = int((flagged_df["severity"].str.lower() == "high").sum())
        elif "flag_type" in flagged_df.columns:
            critical_flags = int((flagged_df["flag_type"] == "conflict_moment").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Earnings", f"₹{current_earnings:.2f}" if current_earnings is not None else "—")
    if current_velocity is not None:
        c2.metric("Earnings Velocity", f"₹{current_velocity:.2f}/hr", delta=f"{velocity_delta:+.2f}" if velocity_delta is not None else None)
    else:
        c2.metric("Earnings Velocity", "—")
    c3.metric("Safety Status", f"{critical_flags} Critical Alerts")
    if flagged_df.empty or not {"gps_lat", "gps_lon"}.issubset(flagged_df.columns):
        return
    df = flagged_df.dropna(subset=["gps_lat", "gps_lon"]).copy()
    if df.empty:
        return
    df = df.sort_values("timestamp")
    def _color(sev, ftype):
        s = (str(sev).lower() if pd.notna(sev) else "")
        if s == "high" or ftype == "conflict_moment":
            return [255, 75, 75, 180]
        if s == "medium":
            return [255, 166, 0, 180]
        return [46, 204, 113, 180]
    df["color"] = [ _color(sev, ft) for sev, ft in zip(df.get("severity"), df.get("flag_type")) ]
    df_points = df.rename(columns={"gps_lat": "lat", "gps_lon": "lon"})
    path_data = df_points[["lat", "lon"]].to_dict(orient="records")
    viewport = pdk.ViewState(latitude=float(df_points["lat"].iloc[0]), longitude=float(df_points["lon"].iloc[0]), zoom=12)
    layers = [
        pdk.Layer(
            "PathLayer",
            data=[{"path": path_data}],
            get_path="path",
            get_color=[100, 100, 255],
            width_min_pixels=2,
            opacity=0.4,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=df_points,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=25,
            pickable=True,
        ),
    ]
    st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/streets-v11", initial_view_state=viewport, layers=layers), use_container_width=True)


def render_flagged_moments(flagged_df: pd.DataFrame, insights_df: pd.DataFrame) -> None:
    st.subheader("Flagged Moments")

    if flagged_df.empty:
        st.info("No flagged events found. Once your model exports `flagged_moments.csv`, they will appear here.")
        return

    drivers = sorted(flagged_df["driver_id"].dropna().unique().tolist())
    selected_driver = st.selectbox("Driver", drivers, index=0 if drivers else None)

    trips = sorted(
        flagged_df[flagged_df["driver_id"] == selected_driver]["trip_id"]
        .dropna()
        .unique()
        .tolist()
    )
    selected_trip = st.selectbox("Trip", trips, index=0 if trips else None)

    df_trip = flagged_df[
        (flagged_df["driver_id"] == selected_driver)
        & (flagged_df["trip_id"] == selected_trip)
    ].copy()

    if df_trip.empty:
        st.warning("No flags for this selection.")
        return

    st.metric(
        "Flagged Events",
        value=len(df_trip),
        help="Total number of stress-related events detected for this trip.",
    )

    chart_cols = [c for c in ["Horizontal_Jerk", "Audio_Rolling_15s"] if c in df_trip.columns]
    if chart_cols:
        st.line_chart(df_trip.set_index("timestamp")[chart_cols], height=250)

    if not insights_df.empty and "llm_insight" in insights_df.columns:
        merged = df_trip.merge(
            insights_df[["flag_id", "llm_insight"]],
            on="flag_id",
            how="left",
        )
    else:
        merged = df_trip

    def tts_button(key: str, text: str) -> None:
        html = f"""
        <button id="btn-{key}">Speak</button>
        <script>
        const b=document.getElementById("btn-{key}");
        if(b) {{
          b.onclick=()=>{{
            const u=new SpeechSynthesisUtterance({json.dumps(text)});
            u.lang="en-US";
            window.speechSynthesis.speak(u);
          }};
        }}
        </script>
        """
        components.html(html, height=35)

    for _, row in merged.sort_values("timestamp").iterrows():
        sev = str(row.get("severity", "")).lower()
        title = row.get("flag_type", "")
        when = row.get("timestamp", "")
        score = row.get("combined_score", None)
        msg = f"{title} at {when} • Score {score:.2f}" if score is not None else f"{title} at {when}"
        if sev == "high":
            st.error(msg)
        elif sev == "medium":
            st.warning(msg)
        else:
            st.info(msg)
        ai = row.get("llm_insight", "")
        if isinstance(ai, str) and ai.strip():
            st.info(ai)
            tts_button(str(row.get("flag_id", "")), ai)


def render_earnings_view(velocity_df: pd.DataFrame, goals_df: pd.DataFrame, drivers_df: pd.DataFrame, flagged_df: pd.DataFrame) -> None:
    st.subheader("Earnings & Goals")

    if velocity_df.empty or goals_df.empty:
        st.info("Earnings datasets not found. Place CSVs in `data/earnings/` to enable this view.")
        return

    common_drivers = sorted(
        set(velocity_df["driver_id"].unique()).intersection(goals_df["driver_id"].unique())
    )
    if not common_drivers:
        st.warning("No overlapping drivers between velocity and goals tables.")
        return

    driver_labels = []
    id_to_label = {}
    for d in common_drivers:
        name = d
        if not drivers_df.empty:
            row = drivers_df[drivers_df["driver_id"] == d]
            if not row.empty:
                name = row["name"].iloc[0]
        label = f"{name} ({d})"
        driver_labels.append(label)
        id_to_label[d] = label

    selected_label = st.selectbox("Driver", driver_labels)
    selected_driver = None
    for did, label in id_to_label.items():
        if label == selected_label:
            selected_driver = did
            break

    vel_driver = velocity_df[velocity_df["driver_id"] == selected_driver].copy()
    goals_driver = goals_df[goals_df["driver_id"] == selected_driver].copy()

    goals_driver = goals_driver.sort_values("date")
    current_goal = goals_driver.iloc[-1] if not goals_driver.empty else None

    if not vel_driver.empty:
        latest = vel_driver.sort_values("timestamp").iloc[-1]
        current_earnings = latest["cumulative_earnings"]
        target_velocity = latest["target_velocity"]
        current_velocity = latest["current_velocity"]
        velocity_delta = latest["velocity_delta"]
        forecast_status = latest["forecast_status"]
    else:
        current_earnings = target_velocity = current_velocity = velocity_delta = None
        forecast_status = "unknown"

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Earnings", f"{current_earnings:.0f}" if current_earnings is not None else "—")
    col2.metric("Current Velocity", f"{current_velocity:.2f}" if current_velocity is not None else "—")
    col3.metric(
        "Velocity vs Target",
        f"{velocity_delta:.2f}" if velocity_delta is not None else "—",
        help="Positive means ahead of goal pace; negative means falling behind.",
    )

    if current_goal is not None:
        target_amt = current_goal.get("target_earnings", None)
        if target_amt is not None and current_earnings is not None:
            pct = min(max(float(current_earnings) / float(target_amt), 0.0), 1.0)
            st.progress(pct, text=f"₹{current_earnings:.0f} of ₹{float(target_amt):.0f}")

    if not vel_driver.empty:
        chart_df = vel_driver.sort_values("timestamp").set_index("timestamp")
        st.line_chart(
            chart_df[["cumulative_earnings", "current_velocity", "target_velocity"]],
            height=300,
        )

        hours = chart_df["elapsed_hours"].iloc[-1] if "elapsed_hours" in chart_df.columns else None
        safe_score = None
        if not flagged_df.empty:
            df_flags = flagged_df.copy()
            if not drivers_df.empty:
                pass
            highs = int((df_flags["severity"].str.lower() == "high").sum()) if "severity" in df_flags.columns else 0
            meds = int((df_flags["severity"].str.lower() == "medium").sum()) if "severity" in df_flags.columns else 0
            safe_score = max(50, int(100 - highs * 10 - meds * 5))
        summary = "Shift Wrapped: "
        parts = []
        if hours is not None:
            parts.append(f"{float(hours):.1f} hours")
        if current_earnings is not None:
            parts.append(f"₹{float(current_earnings):.0f}")
        if safe_score is not None:
            parts.append(f"{safe_score}% safe driving score")
        st.success("You've driven " + ", ".join(parts) + " today!" if parts else "Shift summary unavailable")

        ratio = None
        if current_velocity is not None and target_velocity is not None and target_velocity not in (0, None):
            ratio = float(current_velocity) / float(target_velocity)
        if ratio is not None:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=ratio, gauge={"axis": {"range": [0, 2]}, "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 1}}, number={"valueformat": ".2f"}, domain={"x": [0, 1], "y": [0, 1]}))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


def render_how_it_works() -> None:
    st.subheader("How It Works")
    st.markdown("We process raw physical thresholds—4.0 m/s² jerk and 85 dB sustained audio—combine them with earnings velocity, and pass summaries to an LLM for empathetic coaching. Backends export structured files that this dashboard renders.")
    if st.button("Generate or Refresh Compliance Log"):
        generate_uber_compliance_log()
        load_compliance_log.clear()
    comp = load_compliance_log()
    comp = load_compliance_log()
    if comp.empty:
        st.info("Compliance log not found. Place uber_compliance_log.csv in data/processed_outputs.")
    else:
        st.dataframe(comp, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Driver Pulse Dashboard",
        page_icon="🚗",
        layout="wide",
    )

    st.title("Driver Pulse – Trip & Earnings Dashboard")
    st.caption(
        "Inspect flagged stress moments and earnings velocity for simulated Uber driver sessions."
    )

    flagged_df = load_flagged_moments()
    insights_df = load_trip_insights()
    velocity_df = load_earnings_velocity()
    goals_df = load_driver_goals()
    drivers_df = load_drivers()
    compliance_df = load_compliance_log()

    tab_overview, tab_flags, tab_earnings, tab_how = st.tabs(
        ["Live Driving", "Flagged Moments", "Earnings & Goals", "How It Works"]
    )

    with tab_overview:
        render_trip_overview(flagged_df, velocity_df)
    with tab_flags:
        render_flagged_moments(flagged_df, insights_df)
    with tab_earnings:
        render_earnings_view(velocity_df, goals_df, drivers_df, flagged_df)
    with tab_how:
        render_how_it_works()


if __name__ == "__main__":
    main()

