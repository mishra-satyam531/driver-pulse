import json
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

FLAGGED_MOMENTS_CSV = DATA_DIR / "processed_outputs" / "flagged_moments.csv"
TRIP_INSIGHTS_JSON = DATA_DIR / "processed_outputs" / "trip_insights_final.json"
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


def render_trip_overview(flagged_df: pd.DataFrame) -> None:
    st.subheader("Trip Overview")

    if flagged_df.empty:
        st.info("No flagged moments available yet. Run the stress model pipeline to generate data.")
        return

    trip_summary = (
        flagged_df.groupby(["driver_id", "trip_id"], as_index=False)
        .agg(
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            flags_count=("flag_id", "count"),
            high_flags=("severity", lambda s: (s.str.lower() == "high").sum()),
            max_combined_score=("combined_score", "max"),
        )
        .sort_values("first_timestamp")
    )

    st.dataframe(
        trip_summary,
        use_container_width=True,
        hide_index=True,
    )


def render_flagged_moments(flagged_df: pd.DataFrame, insights_df: pd.DataFrame) -> None:
    st.subheader("Flagged Stress Moments")

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

    st.bar_chart(
        df_trip.set_index("timestamp")[["combined_score"]],
        height=250,
    )

    if not insights_df.empty and "llm_insight" in insights_df.columns:
        merged = df_trip.merge(
            insights_df[["flag_id", "llm_insight"]],
            on="flag_id",
            how="left",
        )
    else:
        merged = df_trip

    table_cols = [
        "timestamp",
        "flag_type",
        "severity",
        "motion_score",
        "audio_score",
        "combined_score",
        "explanation",
        "context",
    ]
    if "llm_insight" in merged.columns:
        table_cols.append("llm_insight")

    display_df = merged[table_cols].copy()
    st.dataframe(
        display_df.style.applymap(style_severity, subset=["severity"]),
        use_container_width=True,
        hide_index=True,
    )


def render_earnings_view(velocity_df: pd.DataFrame, goals_df: pd.DataFrame, drivers_df: pd.DataFrame) -> None:
    st.subheader("Earnings Velocity")

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
    col2.metric("Current Velocity", f"{current_velocity:.1f}" if current_velocity is not None else "—")
    col3.metric(
        "Velocity vs Target",
        f"{velocity_delta:.1f}" if velocity_delta is not None else "—",
        help="Positive means ahead of goal pace; negative means falling behind.",
    )

    if current_goal is not None:
        st.caption(
            f"Goal: earn {current_goal['target_earnings']} by {current_goal['shift_end_time']} "
            f"(status: {current_goal['status']}, forecast: {current_goal['goal_completion_forecast']})."
        )

    if not vel_driver.empty:
        chart_df = vel_driver.sort_values("timestamp").set_index("timestamp")
        st.line_chart(
            chart_df[["cumulative_earnings", "current_velocity", "target_velocity"]],
            height=300,
        )

        st.dataframe(
            vel_driver[
                [
                    "timestamp",
                    "cumulative_earnings",
                    "current_velocity",
                    "target_velocity",
                    "velocity_delta",
                    "forecast_status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_how_it_works() -> None:
    st.subheader("How Driver Pulse Works")
    st.markdown(
        """
Driver Pulse combines motion and audio telematics signals with earnings data to surface **glanceable, explainable insights** for drivers.

- **Stress detection engine** ingests accelerometer and audio intensity streams, computes engineered features (jerk, rolling noise windows), and applies rules to flag harsh motion, sustained cabin noise, and combined conflict moments.
- **Flag aggregation & scoring** collapses bursts of raw samples into single events with severity, motion/audio scores, and short explanations.
- **Earnings velocity model** tracks cumulative earnings versus daily goals, estimating whether the driver is ahead, on track, or at risk of missing their target.
- **This dashboard** sits on top of the processed outputs, letting a judge click into a trip, inspect stress flags, and see how earnings pace evolves across a shift.
        """
    )


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

    tab_overview, tab_flags, tab_earnings, tab_how = st.tabs(
        ["Trip Overview", "Flagged Moments", "Earnings & Goals", "How this works"]
    )

    with tab_overview:
        render_trip_overview(flagged_df)
    with tab_flags:
        render_flagged_moments(flagged_df, insights_df)
    with tab_earnings:
        render_earnings_view(velocity_df, goals_df, drivers_df)
    with tab_how:
        render_how_it_works()


if __name__ == "__main__":
    main()

