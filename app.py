import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ============================================================
# KOLKATA WEATHER AI - MULTI-PAGE STREAMLIT DASHBOARD
# Expected files:
# dataset/kolkata_weather.csv
# dataset/processed_weather.csv
# models/best_weather_model.pkl
# models/feature_columns.pkl
# models/model_comparison.csv
# ============================================================

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
GRAPH_DIR = ROOT / "graphs"

st.set_page_config(
    page_title="Kolkata Weather AI",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------- THEME -----------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background:
      radial-gradient(circle at 85% 0%, rgba(37,99,235,.13), transparent 28%),
      #0b1120;
    color: #e5edf8;
}
[data-testid="stHeader"] { background: rgba(11,17,32,0.88); }
[data-testid="stSidebar"] {
    background: #101a2d;
    border-right: 1px solid rgba(148,163,184,.14);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #a9b8cc;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3 { letter-spacing: -0.035em; }
h1 { color: #f8fafc; font-weight: 800; }
h2, h3 { color: #eaf1fb; }
p, label, li { color: #b8c6d9; }
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17243a, #121d30);
    border: 1px solid rgba(148,163,184,.15);
    padding: 17px 18px;
    border-radius: 15px;
}
div[data-testid="stMetricLabel"] { color: #9fb0c8; }
div[data-testid="stMetricValue"] { color: #f8fafc; }
div[data-testid="stMetricDelta"] { color: #60a5fa; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(148,163,184,.15) !important;
    border-radius: 16px;
}
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    min-height: 2.7rem;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    border: 1px solid #93c5fd;
    color: white;
}
div[data-testid="stAlert"] { border-radius: 12px; }
.kicker {
    color: #60a5fa; text-transform: uppercase; letter-spacing: .16em;
    font-size: .72rem; font-weight: 800; margin-bottom: .35rem;
}
.hero {
    background: linear-gradient(115deg, #1d4ed8 0%, #2563eb 48%, #0f3c91 100%);
    border: 1px solid rgba(147,197,253,.28);
    border-radius: 22px; padding: 28px 30px; margin: 10px 0 22px 0;
    box-shadow: 0 18px 55px rgba(37,99,235,.18);
}
.hero h2 { color: white; font-size: 2rem; margin: 0; }
.hero p { color: #dbeafe; margin: .5rem 0 0 0; max-width: 780px; }
.panel-title { color: #eaf1fb; font-weight: 700; font-size: 1.05rem; }
.small-muted { color: #8fa2bd; font-size: .82rem; }
.pill {
    display:inline-block; padding:5px 10px; border-radius:999px;
    background:rgba(96,165,250,.13); color:#93c5fd;
    border:1px solid rgba(96,165,250,.25); font-size:.75rem; font-weight:700;
}
div[data-testid="stDataFrame"] { border: 1px solid rgba(148,163,184,.16); border-radius: 12px; }
hr { border-color: rgba(148,163,184,.16); }
</style>
""", unsafe_allow_html=True)

# ---------------------- DATA LOADING -------------------------

@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_pickle(path_string):
    path = Path(path_string)
    if path.exists():
        return joblib.load(path)
    return None

raw_path = DATA_DIR / "kolkata_weather_jan_june_2026.csv"
processed_path = DATA_DIR / "processed_weather.csv"
comparison_path = MODEL_DIR / "model_comparison.csv"
model_path = MODEL_DIR / "best_weather_model.pkl"
features_path = MODEL_DIR / "feature_columns.pkl"

raw_df = load_csv(str(raw_path))
processed_df = load_csv(str(processed_path))
comparison_df = load_csv(str(comparison_path))
best_model = load_pickle(str(model_path))
feature_columns = load_pickle(str(features_path))

# ---------------------- HELPERS ------------------------------

def page_header(eyebrow, title, subtitle):
    st.markdown(f'<div class="kicker">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="small-muted">{subtitle}</p>', unsafe_allow_html=True)

def nice_plot(fig, height=390):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#cbd5e1"),
        height=height,
        margin=dict(l=15, r=15, t=45, b=15),
        legend=dict(orientation="h", y=1.12, x=0)
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)", zerolinecolor="rgba(148,163,184,.12)")
    return fig

def missing_file_note(filename):
    st.info(f"`{filename}` was not found. Place it in the indicated project folder to enable this section.")

def numeric_df(frame):
    return frame.select_dtypes(include=np.number)

def get_dataset_date_range(frame):
    if frame is not None and "time" in frame.columns:
        dates = pd.to_datetime(frame["time"], errors="coerce").dropna()
        if len(dates):
            return dates.min().strftime("%d %b %Y"), dates.max().strftime("%d %b %Y")
    return "—", "—"

# ------------------------- SIDEBAR ----------------------------

with st.sidebar:
    st.markdown("## 🌦️ Kolkata Weather AI")
    st.markdown('<span class="pill">AI PROJECT</span>', unsafe_allow_html=True)
    st.markdown("")
    st.caption("NAVIGATION")
    page = st.radio(
        "Go to",
        [
            "Overview",
            "Dataset Explorer",
            "Preprocessing",
            "Exploratory Analysis",
            "Model Performance",
            "Weather Predictor"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("PROJECT DETAILS")
    st.markdown("**Location:** Kolkata, India")
    st.markdown("**Dataset:** Daily weather · 2021–2025")
    st.markdown("**Task:** Next-day temperature Predictor")
    st.markdown("---")
    st.caption("Built with Python · Pandas · Scikit-learn · Streamlit")

# --------------------------- PAGES ----------------------------

if page == "Overview":
    page_header("WEATHER INTELLIGENCE PLATFORM", "Kolkata Weather Dashboard",
                "Explore five years of historical weather observations and a machine-learning model that estimates the next day's average temperature.")

    st.markdown("""
    <div class="hero">
      <div class="kicker" style="color:#bfdbfe">PROJECT SPOTLIGHT</div>
      <h2>Understand the weather.<br>Explore the patterns. Predict tomorrow.</h2>
      <p>A complete student data-science project covering data preparation, exploratory analysis, regression modelling, model evaluation and an interactive prediction interface.</p>
    </div>
    """, unsafe_allow_html=True)

    if raw_df is not None:
        start, end = get_dataset_date_range(raw_df)
        numeric_cols = numeric_df(raw_df)
        avg_temp = raw_df["temperature_2m_mean"].mean() if "temperature_2m_mean" in raw_df else np.nan
        total_rain = raw_df["precipitation_sum"].sum() if "precipitation_sum" in raw_df else np.nan
        records = len(raw_df)
        columns = len(raw_df.columns)
    else:
        start, end, avg_temp, total_rain, records, columns = "—", "—", np.nan, np.nan, 0, 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weather records", f"{records:,}" if records else "Not loaded")
    c2.metric("Raw data columns", columns if columns else "—")
    c3.metric("Mean temperature", f"{avg_temp:.1f} °C" if pd.notna(avg_temp) else "—")
    c4.metric("Total precipitation", f"{total_rain:,.1f} mm" if pd.notna(total_rain) else "—")

    st.markdown("")
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("### Temperature through time")
        if raw_df is not None and {"time", "temperature_2m_mean"}.issubset(raw_df.columns):
            chart_df = raw_df.copy()
            chart_df["time"] = pd.to_datetime(chart_df["time"], errors="coerce")
            fig = px.line(chart_df, x="time", y="temperature_2m_mean",
                          labels={"time": "Date", "temperature_2m_mean": "Mean temperature (°C)"})
            st.plotly_chart(nice_plot(fig, 350), use_container_width=True)
        else:
            missing_file_note("dataset/kolkata_weather.csv")
    with right:
        st.markdown("### Project snapshot")
        st.markdown(f"""
        <div style="background:#142137;border:1px solid #263650;border-radius:15px;padding:18px">
          <div class="small-muted">OBSERVATION PERIOD</div>
          <h3 style="margin:4px 0 16px 0">{start} — {end}</h3>
          <div class="small-muted">PREDICTION TARGET</div>
          <h3 style="margin:4px 0 16px 0">Next-day mean temperature</h3>
          <div class="small-muted">MACHINE LEARNING TASK</div>
          <h3 style="margin:4px 0 0 0">Supervised regression</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if best_model is not None:
            st.success("Trained model artifact detected.")
        else:
            st.warning("Trained model not found yet. Complete model training to activate predictions.")

    st.markdown("### Explore the project")
    cards = st.columns(3)
    descriptions = [
        ("🗂️ Dataset Explorer", "Inspect rows, columns, data types and descriptive statistics."),
        ("📊 Exploratory Analysis", "Explore temperature, rainfall, wind and feature relationships."),
        ("🧠 Model Performance", "Compare regression models and understand prediction quality.")
    ]
    for col, (title, desc) in zip(cards, descriptions):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(desc)

elif page == "Dataset Explorer":
    page_header("01 · UNDERSTAND THE DATA", "Dataset Explorer",
                "Inspect the original weather dataset, its structure, data types and statistical properties.")

    if raw_df is None:
        missing_file_note("dataset/kolkata_weather.csv")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Rows", f"{raw_df.shape[0]:,}")
        b.metric("Columns", raw_df.shape[1])
        c.metric("Duplicate rows", int(raw_df.duplicated().sum()))
        d.metric("Missing cells", int(raw_df.isna().sum().sum()))

        tab1, tab2, tab3 = st.tabs(["Data preview", "Column information", "Statistics"])
        with tab1:
            st.dataframe(raw_df, use_container_width=True, height=430)
            st.download_button("Download original CSV", raw_df.to_csv(index=False).encode("utf-8"),
                               "kolkata_weather_original.csv", "text/csv")
        with tab2:
            info = pd.DataFrame({
                "Column": raw_df.columns,
                "Data type": [str(t) for t in raw_df.dtypes],
                "Non-null values": raw_df.notna().sum().values,
                "Missing values": raw_df.isna().sum().values,
                "Unique values": raw_df.nunique().values
            })
            st.dataframe(info, use_container_width=True, height=430)
        with tab3:
            st.dataframe(raw_df.describe(include="all").T, use_container_width=True)

        st.markdown("### Dataset notes")
        st.markdown(f"""
        - **Location:** Kolkata, West Bengal, India.
        - **Date range:** {get_dataset_date_range(raw_df)[0]} to {get_dataset_date_range(raw_df)[1]}.
        - **Observation frequency:** Daily, subject to checking for date gaps.
        - **Prediction task:** Estimate the following day's mean temperature.
        """)
        st.caption("The dataset summary above is calculated directly from the CSV loaded by this application.")

elif page == "Preprocessing":
    page_header("02 · PREPARE THE DATA", "Data Preprocessing",
                "A transparent record of the transformations used to prepare the modelling dataset.")

    st.markdown("### Preprocessing workflow")
    steps = [
        ("01", "Load and inspect", "Read the original CSV and inspect dimensions, data types, missing values and duplicate records."),
        ("02", "Date conversion", "Convert the time column to datetime and sort observations chronologically."),
        ("03", "Remove constant/context columns", "Remove city and fixed coordinates; sunrise and sunset are omitted from the initial feature set."),
        ("04", "Create prediction target", "Shift daily mean temperature by one row to create the next-day target. Remove the final row with no target."),
        ("05", "Feature engineering", "Extract month, day and weekday; encode month cyclically using sine and cosine."),
        ("06", "Prepare model inputs", "Treat weather_code as categorical and scale numeric inputs inside a scikit-learn pipeline.")
    ]
    for number, title, detail in steps:
        with st.container(border=True):
            x, y = st.columns([0.12, 0.88])
            x.markdown(f"### <span style='color:#60a5fa'>{number}</span>", unsafe_allow_html=True)
            y.markdown(f"**{title}**")
            y.caption(detail)

    st.markdown("### Processed dataset status")
    if processed_df is None:
        missing_file_note("dataset/processed_weather.csv")
    else:
        x, y, z = st.columns(3)
        x.metric("Processed rows", f"{len(processed_df):,}")
        y.metric("Processed columns", len(processed_df.columns))
        z.metric("Missing cells", int(processed_df.isna().sum().sum()))
        st.dataframe(processed_df.head(15), use_container_width=True)
        st.download_button("Download processed CSV", processed_df.to_csv(index=False).encode("utf-8"),
                        "processed_weather.csv", "text/csv")

    st.warning("Important: the target shift assumes consecutive daily observations. Check for missing dates before interpreting the target as 'tomorrow'. The workflow description should be kept in sync with the preprocessing notebook if you change its steps.")

elif page == "Exploratory Analysis":
    page_header("03 · DISCOVER PATTERNS", "Exploratory Data Analysis",
                "Explore historical temperature, precipitation, wind behaviour and relationships among numerical variables.")

    if raw_df is None:
        missing_file_note("dataset/kolkata_weather.csv")
    else:
        edf = raw_df.copy()
        if "time" in edf:
            edf["time"] = pd.to_datetime(edf["time"], errors="coerce")
            edf = edf.sort_values("time")
            edf["month"] = edf["time"].dt.month
            edf["month_name"] = edf["time"].dt.strftime("%b")

        tab1, tab2, tab3 = st.tabs(["Temperature", "Rainfall & wind", "Correlations"])
        with tab1:
            if "temperature_2m_mean" in edf:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(edf, x="temperature_2m_mean", nbins=30, marginal="box",
                                       labels={"temperature_2m_mean": "Mean temperature (°C)"},
                                       title="Temperature distribution")
                    st.plotly_chart(nice_plot(fig), use_container_width=True)
                with col2:
                    if "month" in edf:
                        monthly = edf.groupby("month", as_index=False)["temperature_2m_mean"].mean()
                        fig = px.bar(monthly, x="month", y="temperature_2m_mean",
                                     labels={"month": "Month number", "temperature_2m_mean": "Mean temperature (°C)"},
                                     title="Average temperature by month")
                        st.plotly_chart(nice_plot(fig), use_container_width=True)
                if "time" in edf:
                    fig = px.line(edf, x="time", y="temperature_2m_mean",
                                  labels={"time": "Date", "temperature_2m_mean": "Mean temperature (°C)"},
                                  title="Daily average temperature over time")
                    st.plotly_chart(nice_plot(fig, 420), use_container_width=True)
            else:
                st.info("Temperature column not found in the raw dataset.")
        with tab2:
            col1, col2 = st.columns(2)
            if "precipitation_sum" in edf:
                with col1:
                    fig = px.histogram(edf, x="precipitation_sum", nbins=30,
                                       labels={"precipitation_sum": "Precipitation (mm)"},
                                       title="Daily precipitation distribution")
                    st.plotly_chart(nice_plot(fig), use_container_width=True)
                with col2:
                    if "month" in edf:
                        monthly_rain = edf.groupby("month", as_index=False)["precipitation_sum"].mean()
                        fig = px.bar(monthly_rain, x="month", y="precipitation_sum",
                                    labels={"month": "Month number", "precipitation_sum": "Average precipitation (mm)"},
                                    title="Average precipitation by month")
                        st.plotly_chart(nice_plot(fig), use_container_width=True)
            if {"precipitation_sum", "temperature_2m_mean"}.issubset(edf.columns):
                fig = px.scatter(edf, x="precipitation_sum", y="temperature_2m_mean",
                                opacity=.6, labels={"precipitation_sum": "Precipitation (mm)",
                                "temperature_2m_mean": "Mean temperature (°C)"},
                                title="Temperature vs precipitation")
                st.plotly_chart(nice_plot(fig), use_container_width=True)
            wind_cols = [c for c in ["wind_speed_10m_max", "wind_gusts_10m_max"] if c in edf.columns]
            if wind_cols:
                st.markdown("#### Wind variables")
                st.dataframe(edf[wind_cols].describe().T, use_container_width=True)
        with tab3:
            corr = numeric_df(edf).corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="Correlation matrix")
            st.plotly_chart(nice_plot(fig, 650), use_container_width=True)
            st.caption("Correlation describes linear association. It does not establish cause and effect.")

elif page == "Model Performance":
    page_header("04 · EVALUATE THE MODELS", "Model Performance",
                "Compare the regression algorithms using the metrics saved during model training.")

    if comparison_df is None:
        missing_file_note("models/model_comparison.csv")
    else:
        display_df = comparison_df.copy()
        if "RMSE" in display_df:
            display_df = display_df.sort_values("RMSE")
        st.markdown("### Model leaderboard")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if {"Model", "RMSE"}.issubset(display_df.columns):
            fig = px.bar(display_df, x="Model", y="RMSE", color="Model",
                        title="RMSE comparison · lower is better",
                        labels={"RMSE": "Root Mean Squared Error (°C)"})
            st.plotly_chart(nice_plot(fig, 400), use_container_width=True)

        if {"Model", "R2 Score"}.issubset(display_df.columns):
            fig = px.bar(display_df, x="Model", y="R2 Score", color="Model",
                        title="R² comparison · higher is generally better")
            st.plotly_chart(nice_plot(fig, 400), use_container_width=True)

        if {"Model", "MAE"}.issubset(display_df.columns):
            fig = px.bar(display_df, x="Model", y="MAE", color="Model",
                        title="MAE comparison · lower is better")
            st.plotly_chart(nice_plot(fig, 400), use_container_width=True)

        if len(display_df):
            best_row = display_df.iloc[0]
            metric_cols = st.columns(3)
            if "Model" in best_row:
                metric_cols[0].metric("Lowest-RMSE model", str(best_row["Model"]))
            if "RMSE" in best_row:
                metric_cols[1].metric("Best RMSE", f'{best_row["RMSE"]:.3f} °C')
            if "R2 Score" in best_row:
                metric_cols[2].metric("Best R²", f'{best_row["R2 Score"]:.3f}')

    st.markdown("---")
    st.markdown("### Understanding the algorithms")
    model_notes = {
        "Dummy Regressor": "Baseline model that predicts a simple reference value. It helps check whether trained models outperform a naive approach.",
        "Linear Regression": "Fits a linear relationship between input weather variables and next-day temperature.",
        "Decision Tree": "Learns a sequence of feature-based splits and can capture nonlinear patterns.",
        "Random Forest": "Combines many decision trees to improve stability and reduce the variance of a single tree.",
        "Gradient Boosting": "Builds trees sequentially, with later trees learning from earlier prediction errors.",
        "Extra Trees": "Uses a highly randomized collection of trees and averages their predictions.",
        "Hist Gradient Boosting": "A histogram-based boosting implementation designed to train efficiently on larger datasets."
    }
    for _, row in (display_df.iterrows() if comparison_df is not None else []):
        model_name = str(row.get("Model", "Model"))
        with st.expander(model_name):
            st.write(model_notes.get(model_name, "Regression estimator used in this project."))
            if "RMSE" in row:
                st.caption(f"RMSE: {row['RMSE']:.3f} °C")
            if "MAE" in row:
                st.caption(f"MAE: {row['MAE']:.3f} °C")
            if "R2 Score" in row:
                st.caption(f"R²: {row['R2 Score']:.3f}")

    st.markdown("---")
    st.markdown("### Feature importance")
    if best_model is not None:
        try:
            estimator = best_model.named_steps["model"]
            prep = best_model.named_steps["preprocessor"]
            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
                transformed_names = prep.get_feature_names_out()
                importance_df = pd.DataFrame({
                    "Feature": transformed_names,
                    "Importance": importances
                }).sort_values("Importance", ascending=False).head(15)
                fig = px.bar(
                    importance_df.sort_values("Importance"),
                    x="Importance", y="Feature", orientation="h",
                    title="Top 15 features used by the selected tree model"
                )
                st.plotly_chart(nice_plot(fig, 470), use_container_width=True)
            else:
                st.caption("The selected model does not expose built-in tree feature importances. This is normal for models such as Linear Regression and some boosting estimators.")
        except Exception as exc:
            st.caption(f"Feature importance is unavailable for this saved model: {exc}")
    else:
        st.warning("The saved model was not found at models/best_weather_model.pkl.")

    st.markdown("### Model artifact")
    if best_model is not None:
        st.success("The saved best-model pipeline is available.")
        st.caption("The pipeline contains both feature preprocessing and the trained estimator.")
    else:
        st.warning("The saved model was not found at models/best_weather_model.pkl.")

elif page == "Weather Predictor":
    page_header("05 · INTERACTIVE PREDICTION", "Next-Day Temperature Predictor",
                "Enter today's weather observations to estimate tomorrow's average temperature.")

    if best_model is None or feature_columns is None:
        st.error("The trained model or feature list is missing. Run the model-training notebook and save both model artifacts first.")
    elif raw_df is None:
        missing_file_note("dataset/kolkata_weather.csv")
    else:
        latest = raw_df.iloc[-1]
        weather_options = sorted(raw_df["weather_code"].dropna().astype(str).unique().tolist()) if "weather_code" in raw_df else ["0"]
        latest_code = str(latest.get("weather_code", weather_options[0]))
        default_code_index = weather_options.index(latest_code) if latest_code in weather_options else 0

        st.markdown("""
        <div style="background:linear-gradient(110deg,#172d50,#14243b);border:1px solid #2a4770;border-radius:16px;padding:17px 20px;margin-bottom:18px">
          <b style="color:#bfdbfe">How it works</b>
          <div style="color:#b8c6d9;margin-top:5px">The model uses the weather inputs for the selected day and predicts the following day's mean temperature. Use realistic values within the range of the training data.</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("prediction_form"):
            selected_date = st.date_input("Date of today's weather", value=pd.to_datetime(latest.get("time", "2025-06-30")).date())
            weather_code = st.selectbox("Weather code", weather_options, index=default_code_index)

            st.markdown("#### Temperature and precipitation")
            c1, c2, c3 = st.columns(3)
            with c1:
                tmean = st.number_input("Mean temperature (°C)", value=float(latest.get("temperature_2m_mean", 28.0)), step=0.1)
                tmax = st.number_input("Maximum temperature (°C)", value=float(latest.get("temperature_2m_max", 32.0)), step=0.1)
            with c2:
                tmin = st.number_input("Minimum temperature (°C)", value=float(latest.get("temperature_2m_min", 25.0)), step=0.1)
                precip = st.number_input("Precipitation (mm)", min_value=0.0, value=float(latest.get("precipitation_sum", 0.0)), step=0.1)
            with c3:
                rain = st.number_input("Rainfall (mm)", min_value=0.0, value=float(latest.get("rain_sum", 0.0)), step=0.1)
                precip_hours = st.number_input("Precipitation hours", min_value=0.0, max_value=24.0, value=float(latest.get("precipitation_hours", 0.0)), step=0.5)

            st.markdown("#### Wind and sunshine")
            c1, c2, c3 = st.columns(3)
            with c1:
                wind_speed = st.number_input("Maximum wind speed (km/h)", min_value=0.0, value=float(latest.get("wind_speed_10m_max", 10.0)), step=0.1)
            with c2:
                wind_gusts = st.number_input("Maximum wind gusts (km/h)", min_value=0.0, value=float(latest.get("wind_gusts_10m_max", 15.0)), step=0.1)
            with c3:
                wind_direction = st.number_input("Dominant wind direction (°)", min_value=0, max_value=360, value=int(latest.get("wind_direction_10m_dominant", 180)))

            sunshine = st.number_input("Sunshine duration (seconds)", min_value=0.0, value=float(latest.get("sunshine_duration", 20000.0)), step=300.0)
            submitted = st.form_submit_button("🌡️ Predict tomorrow's temperature", use_container_width=True)

        if submitted:
            month = selected_date.month
            input_values = {
                "weather_code": str(weather_code),
                "temperature_2m_mean": tmean,
                "temperature_2m_max": tmax,
                "temperature_2m_min": tmin,
                "precipitation_sum": precip,
                "rain_sum": rain,
                "precipitation_hours": precip_hours,
                "wind_speed_10m_max": wind_speed,
                "wind_gusts_10m_max": wind_gusts,
                "wind_direction_10m_dominant": wind_direction,
                "sunshine_duration": sunshine,
                "month": month,
                "day": selected_date.day,
                "day_of_week": selected_date.weekday(),
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12)
            }
            try:
                input_frame = pd.DataFrame([{col: input_values[col] for col in feature_columns}])
                prediction = float(best_model.predict(input_frame)[0])
                tomorrow = pd.Timestamp(selected_date) + pd.Timedelta(days=1)
                st.markdown("")
                st.markdown(f"""
                <div class="hero" style="text-align:center">
                  <div class="kicker" style="color:#bfdbfe">PREDICTED NEXT-DAY MEAN TEMPERATURE</div>
                  <h2 style="font-size:3.4rem">{prediction:.2f} °C</h2>
                  <p style="margin:8px auto 0 auto">For {tomorrow.strftime('%A, %d %B %Y')}</p>
                </div>
                """, unsafe_allow_html=True)
                st.info("This is an educational machine-learning estimate based on historical patterns, not an official weather forecast.")
            except KeyError as exc:
                st.error(f"The saved feature list contains a feature not supplied by this app: {exc}. Check that the training notebook and app use the same feature columns.")
            except Exception as exc:
                st.error(f"Prediction could not be generated: {exc}")

# --------------------------- FOOTER ---------------------------

st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#71839d;font-size:.78rem">'
    'Kolkata Weather AI · Class XII Artificial Intelligence Project · '
    'Built with Python, Pandas, Scikit-learn and Streamlit'
    '</div>',
    unsafe_allow_html=True
)
