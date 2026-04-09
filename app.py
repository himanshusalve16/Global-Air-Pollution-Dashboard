"""
============================================================
  Air Quality Dashboard — India
  Streamlit app for exploring AQI & pollutant data
  Dataset: rohanrao/air-quality-data-in-india (Kaggle)
============================================================
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="India Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  CUSTOM CSS — refined dark industrial theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #2a2d3a;
}
section[data-testid="stSidebar"] * { color: #c9ccd8 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1d27 0%, #1f2232 100%);
    border: 1px solid #2e3148;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
div[data-testid="metric-container"] label { color: #8b8fa8 !important; font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f0f2ff !important; font-family: 'Space Mono', monospace; font-size: 1.5rem; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #6ee7b7 !important; }

/* Section headers */
h1 { font-family: 'Space Mono', monospace !important; color: #ffffff !important; letter-spacing: -0.02em; }
h2, h3 { font-family: 'DM Sans', sans-serif !important; color: #d4d8f0 !important; }

/* Divider */
hr { border-color: #2a2d3a !important; }

/* Plotly charts background */
.js-plotly-plot { border-radius: 12px; }

/* Expander */
details { background: #1a1d27 !important; border: 1px solid #2e3148 !important; border-radius: 8px !important; }

/* Download button */
div[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
}

/* Info boxes */
div[data-testid="stInfo"] { background: #1e2235 !important; border-left: 3px solid #4f46e5 !important; }

/* Caption text */
.caption-text { color: #8b8fa8; font-size: 0.82rem; margin-top: -4px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  AQI CATEGORY HELPER
# ──────────────────────────────────────────────
AQI_CATEGORIES = [
    (0,   50,  "Good",       "#22c55e"),
    (51,  100, "Satisfactory","#84cc16"),
    (101, 200, "Moderate",   "#eab308"),
    (201, 300, "Poor",       "#f97316"),
    (301, 400, "Very Poor",  "#ef4444"),
    (401, 9999,"Severe",     "#7f1d1d"),
]

def aqi_category(val):
    """Return (label, colour) for an AQI value."""
    if pd.isna(val):
        return ("N/A", "#555")
    for lo, hi, label, colour in AQI_CATEGORIES:
        if lo <= val <= hi:
            return label, colour
    return ("Severe", "#7f1d1d")

def colour_for_aqi(series: pd.Series) -> list:
    return [aqi_category(v)[1] for v in series]

# ──────────────────────────────────────────────
#  DATA LOADING & PREPARATION
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading dataset from Kaggle…")
def load_data():
    """Download dataset via kagglehub, auto-detect CSV, and prepare it."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("rohanrao/air-quality-data-in-india")
    except Exception as e:
        st.error(f"❌ Could not download dataset: {e}")
        st.stop()

    # Find all CSVs in the downloaded folder
    csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csv_files:
        st.error("No CSV files found in the downloaded dataset.")
        st.stop()

    # Prefer a file whose name contains 'city_day' or is the largest
    preferred = [f for f in csv_files if "city_day" in os.path.basename(f).lower()]
    target = preferred[0] if preferred else max(csv_files, key=os.path.getsize)

    df = pd.read_csv(target)

    # ── Normalise column names ──────────────────
    df.columns = df.columns.str.strip()

    # ── Date column ────────────────────────────
    date_col_candidates = [c for c in df.columns if "date" in c.lower()]
    if date_col_candidates:
        dc = date_col_candidates[0]
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df.rename(columns={dc: "Date"}, inplace=True)
        df.dropna(subset=["Date"], inplace=True)
        df.sort_values("Date", inplace=True)
        # Derived time columns
        df["Year"]       = df["Date"].dt.year
        df["Month"]      = df["Date"].dt.month
        df["Month_Name"] = df["Date"].dt.strftime("%b")
        df["Day"]        = df["Date"].dt.day
        df["Season"]     = df["Month"].map({
            12:"Winter", 1:"Winter", 2:"Winter",
            3:"Spring",  4:"Spring", 5:"Spring",
            6:"Summer",  7:"Summer", 8:"Summer",
            9:"Autumn",  10:"Autumn",11:"Autumn",
        })

    # ── Standardise City column ─────────────────
    city_candidates = [c for c in df.columns if c.lower() in ("city", "station", "location")]
    if city_candidates:
        df.rename(columns={city_candidates[0]: "City"}, inplace=True)

    # ── AQI column ──────────────────────────────
    aqi_candidates = [c for c in df.columns if "aqi" in c.lower() and "bucket" not in c.lower()]
    if aqi_candidates:
        df.rename(columns={aqi_candidates[0]: "AQI"}, inplace=True)

    # ── Numeric pollutant columns ───────────────
    non_numeric = {"City", "Date", "Year", "Month", "Month_Name", "Day", "Season"}
    pollutant_cols = [c for c in df.columns if c not in non_numeric and df[c].dtype in [np.float64, np.int64]]

    # Fill missing values: median per city for numerics
    for col in pollutant_cols:
        if "City" in df.columns:
            df[col] = df.groupby("City")[col].transform(lambda x: x.fillna(x.median()))
        df[col].fillna(df[col].median(), inplace=True)

    # AQI bucket column
    if "AQI" in df.columns:
        df["AQI_Category"] = df["AQI"].apply(lambda v: aqi_category(v)[0])

    return df, pollutant_cols

# ──────────────────────────────────────────────
#  PLOTLY DARK TEMPLATE DEFAULTS
# ──────────────────────────────────────────────
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(26,29,39,0.95)",
    plot_bgcolor="rgba(26,29,39,0.0)",
    font=dict(family="DM Sans, sans-serif", color="#c9ccd8"),
    margin=dict(l=24, r=24, t=40, b=24),
)

# ──────────────────────────────────────────────
#  LOAD DATA
# ──────────────────────────────────────────────
df_raw, POLLUTANTS = load_data()

# ──────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div style='padding:2rem 0 0.5rem 0;'>
  <h1 style='font-size:2.2rem; margin-bottom:0;'>🌫️ India Air Quality Dashboard</h1>
  <p style='color:#8b8fa8; font-size:1rem; margin-top:6px;'>
    Comprehensive analysis of AQI & pollutant levels across Indian cities
    &nbsp;|&nbsp; Dataset: <em>rohanrao/air-quality-data-in-india</em>
  </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ──────────────────────────────────────────────
#  SIDEBAR FILTERS
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Filters")
    st.caption("Use these controls to slice & explore the data.")
    st.divider()

    # City selector
    all_cities = sorted(df_raw["City"].dropna().unique().tolist()) if "City" in df_raw.columns else []
    default_cities = all_cities[:8] if len(all_cities) >= 8 else all_cities
    selected_cities = st.multiselect("🏙️ Select Cities", all_cities, default=default_cities)

    # Date range
    if "Date" in df_raw.columns:
        min_date = df_raw["Date"].min().date()
        max_date = df_raw["Date"].max().date()
        date_range = st.date_input("📅 Date Range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date = end_date = None

    # Pollutant selector
    visible_pollutants = [p for p in POLLUTANTS if p != "AQI"]
    selected_pollutants = st.multiselect("💨 Pollutants to Show",
                                          visible_pollutants,
                                          default=visible_pollutants[:4] if len(visible_pollutants) >= 4 else visible_pollutants)

    # AQI slider
    if "AQI" in df_raw.columns:
        aqi_min = int(df_raw["AQI"].min())
        aqi_max = int(df_raw["AQI"].max())
        aqi_range = st.slider("📊 AQI Range", aqi_min, aqi_max, (aqi_min, aqi_max))
    else:
        aqi_range = None

    st.divider()
    st.caption("Built for college project — Streamlit + Plotly")

# ──────────────────────────────────────────────
#  FILTER DATA
# ──────────────────────────────────────────────
df = df_raw.copy()
if selected_cities and "City" in df.columns:
    df = df[df["City"].isin(selected_cities)]
if start_date and "Date" in df.columns:
    df = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)]
if aqi_range and "AQI" in df.columns:
    df = df[(df["AQI"] >= aqi_range[0]) & (df["AQI"] <= aqi_range[1])]

# Guard: empty dataframe
if df.empty:
    st.warning("⚠️ No data matches the current filters. Please adjust the sidebar filters.")
    st.stop()

# ──────────────────────────────────────────────
#  KPI CARDS
# ──────────────────────────────────────────────
st.markdown("### 📌 Key Metrics")
k1, k2, k3, k4, k5 = st.columns(5)

avg_aqi = round(df["AQI"].mean(), 1) if "AQI" in df.columns else "N/A"
aqi_cat, aqi_col = aqi_category(avg_aqi if avg_aqi != "N/A" else None)

with k1:
    st.metric("Avg AQI", f"{avg_aqi}", f"{aqi_cat}")

if "AQI" in df.columns and "City" in df.columns:
    city_aqi = df.groupby("City")["AQI"].mean()
    worst_city  = city_aqi.idxmax()
    best_city   = city_aqi.idxmin()
    worst_val   = round(city_aqi.max(), 1)
    best_val    = round(city_aqi.min(), 1)
    with k2:
        st.metric("Most Polluted City", worst_city, f"AQI {worst_val}")
    with k3:
        st.metric("Cleanest City", best_city, f"AQI {best_val}")

with k4:
    st.metric("Cities Selected", len(selected_cities) if selected_cities else len(df["City"].unique()))

if "AQI" in df.columns:
    with k5:
        st.metric("Total Records", f"{len(df):,}")

st.divider()

# ──────────────────────────────────────────────
#  ROW 1 — AQI Trend (full width) + City Bar
# ──────────────────────────────────────────────
st.markdown("### 📈 AQI Trends Over Time")

if "Date" in df.columns and "AQI" in df.columns:
    # Monthly average per city
    trend_df = (df.groupby(["Date", "City"])["AQI"]
                  .mean().reset_index()
                  .sort_values("Date"))

    fig_trend = px.line(trend_df, x="Date", y="AQI", color="City",
                        labels={"AQI": "AQI", "Date": ""},
                        color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_trend.update_traces(line_width=1.8, opacity=0.9)
    fig_trend.update_layout(**PLOT_LAYOUT, height=340,
                             legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown("<p class='caption-text'>Daily AQI for each selected city. Spikes indicate pollution events.</p>",
                unsafe_allow_html=True)
else:
    st.info("Date or AQI column not found — trend chart skipped.")

st.divider()

# ──────────────────────────────────────────────
#  ROW 2 — City Bar + Box plot
# ──────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### 🏙️ Average AQI by City")
    if "AQI" in df.columns and "City" in df.columns:
        city_avg = df.groupby("City")["AQI"].mean().sort_values(ascending=False).reset_index()
        city_avg["Colour"] = colour_for_aqi(city_avg["AQI"])
        fig_bar = go.Figure(go.Bar(
            x=city_avg["AQI"], y=city_avg["City"],
            orientation="h",
            marker_color=city_avg["Colour"],
            text=city_avg["AQI"].round(1),
            textposition="outside",
        ))
        fig_bar.update_layout(**PLOT_LAYOUT, height=380,
                               xaxis_title="Average AQI", yaxis_title="",
                               yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("<p class='caption-text'>Bars are colour-coded by AQI category (green = good, red = severe).</p>",
                    unsafe_allow_html=True)

with col_b:
    st.markdown("#### 📦 AQI Distribution by City")
    if "AQI" in df.columns and "City" in df.columns:
        fig_box = px.box(df, x="City", y="AQI", color="City",
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         points=False)
        fig_box.update_layout(**PLOT_LAYOUT, height=380,
                               xaxis_title="", showlegend=False,
                               xaxis_tickangle=-35)
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("<p class='caption-text'>Wider boxes indicate more variability in air quality over time.</p>",
                    unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────
#  ROW 3 — Monthly Area + Seasonal Bar
# ──────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.markdown("#### 📅 Monthly AQI Trend (Area)")
    if "Month_Name" in df.columns and "AQI" in df.columns:
        month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly = (df.groupby("Month_Name")["AQI"]
                     .mean().reset_index()
                     .rename(columns={"AQI": "Avg_AQI"}))
        monthly["Month_Name"] = pd.Categorical(monthly["Month_Name"],
                                               categories=month_order, ordered=True)
        monthly.sort_values("Month_Name", inplace=True)
        fig_area = px.area(monthly, x="Month_Name", y="Avg_AQI",
                           color_discrete_sequence=["#818cf8"])
        fig_area.update_layout(**PLOT_LAYOUT, height=320,
                                xaxis_title="", yaxis_title="Avg AQI")
        st.plotly_chart(fig_area, use_container_width=True)
        st.markdown("<p class='caption-text'>Peaks in winter months are driven by lower wind speeds and crop burning.</p>",
                    unsafe_allow_html=True)

with col_d:
    st.markdown("#### 🌸 Seasonal Pollution Trends")
    if "Season" in df.columns and "AQI" in df.columns:
        season_df = df.groupby(["Season", "City"])["AQI"].mean().reset_index()
        fig_season = px.bar(season_df, x="Season", y="AQI", color="City",
                            barmode="group",
                            color_discrete_sequence=px.colors.qualitative.Safe)
        fig_season.update_layout(**PLOT_LAYOUT, height=320,
                                  xaxis_title="Season", yaxis_title="Avg AQI",
                                  legend=dict(orientation="h", y=-0.3, font_size=10))
        st.plotly_chart(fig_season, use_container_width=True)
        st.markdown("<p class='caption-text'>Winter consistently shows the highest AQI across most cities.</p>",
                    unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────
#  ROW 4 — Correlation Heatmap + Scatter
# ──────────────────────────────────────────────
col_e, col_f = st.columns(2)

with col_e:
    st.markdown("#### 🔥 Pollutant Correlation Heatmap")
    heat_cols = [p for p in POLLUTANTS if p in df.columns and df[p].notna().sum() > 10]
    if len(heat_cols) >= 2:
        corr = df[heat_cols].corr()
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns, y=corr.index,
            colorscale="RdBu", zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        ))
        fig_heat.update_layout(**PLOT_LAYOUT, height=370)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("<p class='caption-text'>Strong positive correlations (dark red) suggest co-emitted pollutants.</p>",
                    unsafe_allow_html=True)
    else:
        st.info("Not enough numeric pollutant columns for a heatmap.")

with col_f:
    st.markdown("#### 🔵 Scatter: Pollutant vs AQI")
    scatter_options = [p for p in POLLUTANTS if p != "AQI" and p in df.columns]
    if scatter_options:
        chosen_pol = st.selectbox("X-axis pollutant", scatter_options, key="scatter_sel")
        if "AQI" in df.columns:
            sdf = df[[chosen_pol, "AQI", "City"]].dropna()
            fig_scat = px.scatter(sdf, x=chosen_pol, y="AQI", color="City",
                                  opacity=0.55, trendline="ols",
                                  color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_scat.update_layout(**PLOT_LAYOUT, height=340)
            st.plotly_chart(fig_scat, use_container_width=True)
            st.markdown(f"<p class='caption-text'>Each dot is a city-day record. Trend line shows {chosen_pol}–AQI relationship.</p>",
                        unsafe_allow_html=True)
    else:
        st.info("No pollutant columns available for scatter.")

st.divider()

# ──────────────────────────────────────────────
#  TOP POLLUTED DAYS + SPIKE DETECTION
# ──────────────────────────────────────────────
st.markdown("### ⚠️ Advanced Insights")
ins1, ins2 = st.columns(2)

with ins1:
    st.markdown("#### 🔴 Top 10 Most Polluted Days")
    if "AQI" in df.columns and "Date" in df.columns:
        top_days = (df[["Date", "City", "AQI"]]
                    .dropna()
                    .sort_values("AQI", ascending=False)
                    .head(10)
                    .reset_index(drop=True))
        top_days.index += 1
        top_days["Date"] = top_days["Date"].dt.strftime("%Y-%m-%d")
        top_days["AQI Category"] = top_days["AQI"].apply(lambda v: aqi_category(v)[0])
        st.dataframe(top_days.style.background_gradient(subset=["AQI"], cmap="Reds"),
                     use_container_width=True, height=300)

with ins2:
    st.markdown("#### 📡 Pollution Spike Detection")
    if "AQI" in df.columns and "City" in df.columns and "Date" in df.columns:
        # Spike = AQI > mean + 2*std for that city
        spike_df = df[["Date", "City", "AQI"]].dropna().copy()
        stats = spike_df.groupby("City")["AQI"].agg(["mean", "std"]).reset_index()
        spike_df = spike_df.merge(stats, on="City")
        spike_df["Spike"] = spike_df["AQI"] > (spike_df["mean"] + 2 * spike_df["std"])
        spikes = spike_df[spike_df["Spike"]].sort_values("AQI", ascending=False).head(10)
        if not spikes.empty:
            spikes = spikes[["Date", "City", "AQI"]].reset_index(drop=True)
            spikes.index += 1
            spikes["Date"] = spikes["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(spikes.style.background_gradient(subset=["AQI"], cmap="Oranges"),
                         use_container_width=True, height=300)
            st.markdown("<p class='caption-text'>Spikes = AQI > (city mean + 2 standard deviations).</p>",
                        unsafe_allow_html=True)
        else:
            st.success("No significant pollution spikes detected in filtered range.")

st.divider()

# ──────────────────────────────────────────────
#  RANKING CHART
# ──────────────────────────────────────────────
st.markdown("#### 🏆 City Pollution Ranking")
if "AQI" in df.columns and "City" in df.columns:
    rank_df = (df.groupby("City")["AQI"]
                 .mean()
                 .sort_values(ascending=True)
                 .reset_index())
    rank_df["Colour"] = colour_for_aqi(rank_df["AQI"])
    rank_df["Rank"] = range(len(rank_df), 0, -1)

    fig_rank = go.Figure()
    fig_rank.add_trace(go.Bar(
        x=rank_df["AQI"], y=rank_df["City"],
        orientation="h",
        marker=dict(color=rank_df["Colour"], line_width=0),
        text=rank_df["AQI"].round(0).astype(int),
        textposition="outside",
    ))
    fig_rank.update_layout(**PLOT_LAYOUT, height=max(300, len(rank_df) * 32),
                            xaxis_title="Average AQI",
                            xaxis=dict(range=[0, rank_df["AQI"].max() * 1.18]))
    st.plotly_chart(fig_rank, use_container_width=True)
    st.markdown("<p class='caption-text'>Lower AQI = cleaner air. Colour: green → severe.</p>",
                unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────
#  CITY COMPARISON SECTION
# ──────────────────────────────────────────────
st.markdown("### 🔄 City-by-City Comparison")
if len(selected_cities) >= 2 and "Month_Name" in df.columns and "AQI" in df.columns:
    comp_df = (df.groupby(["City", "Month_Name"])["AQI"]
                 .mean().reset_index())
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    comp_df["Month_Name"] = pd.Categorical(comp_df["Month_Name"],
                                            categories=month_order, ordered=True)
    comp_df.sort_values("Month_Name", inplace=True)
    fig_comp = px.line(comp_df, x="Month_Name", y="AQI", color="City",
                       markers=True,
                       color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_comp.update_layout(**PLOT_LAYOUT, height=340,
                            xaxis_title="Month", yaxis_title="Avg AQI",
                            legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown("<p class='caption-text'>Monthly AQI cycle comparison across selected cities.</p>",
                unsafe_allow_html=True)
elif len(selected_cities) < 2:
    st.info("Select at least 2 cities in the sidebar to enable city comparison.")

st.divider()

# ──────────────────────────────────────────────
#  POLLUTANT TREND (selected pollutants)
# ──────────────────────────────────────────────
if selected_pollutants and "Date" in df.columns:
    st.markdown("### 💨 Selected Pollutant Trends")
    valid_pols = [p for p in selected_pollutants if p in df.columns]
    if valid_pols:
        pol_trend = df.groupby("Date")[valid_pols].mean().reset_index()
        fig_pol = go.Figure()
        for pol in valid_pols:
            fig_pol.add_trace(go.Scatter(x=pol_trend["Date"], y=pol_trend[pol],
                                          name=pol, mode="lines", line_width=1.5))
        fig_pol.update_layout(**PLOT_LAYOUT, height=320,
                               legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_pol, use_container_width=True)
        st.markdown("<p class='caption-text'>National daily averages for chosen pollutants.</p>",
                    unsafe_allow_html=True)
    st.divider()

# ──────────────────────────────────────────────
#  AQI CATEGORY PIE CHART
# ──────────────────────────────────────────────
if "AQI_Category" in df.columns:
    st.markdown("#### 🥧 AQI Category Breakdown")
    cat_counts = df["AQI_Category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    cat_colour_map = {l: c for _, _, l, c in AQI_CATEGORIES}
    fig_pie = px.pie(cat_counts, names="Category", values="Count",
                     color="Category",
                     color_discrete_map=cat_colour_map,
                     hole=0.45)
    fig_pie.update_layout(**PLOT_LAYOUT, height=360)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.divider()

# ──────────────────────────────────────────────
#  SIMPLE TREND FORECAST (rolling average)
# ──────────────────────────────────────────────
st.markdown("### 🔮 Trend Analysis (30-day Rolling Average)")
if "AQI" in df.columns and "Date" in df.columns:
    national_aqi = df.groupby("Date")["AQI"].mean().reset_index().sort_values("Date")
    national_aqi["Rolling_30"] = national_aqi["AQI"].rolling(30, min_periods=1).mean()
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=national_aqi["Date"], y=national_aqi["AQI"],
                                   name="Daily Avg", mode="lines",
                                   line=dict(color="#4f46e5", width=1), opacity=0.4))
    fig_roll.add_trace(go.Scatter(x=national_aqi["Date"], y=national_aqi["Rolling_30"],
                                   name="30-day Rolling Avg", mode="lines",
                                   line=dict(color="#f59e0b", width=2.5)))
    fig_roll.update_layout(**PLOT_LAYOUT, height=320,
                            legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_roll, use_container_width=True)
    st.markdown("<p class='caption-text'>The rolling average smooths short-term fluctuations to reveal the long-term pollution trend.</p>",
                unsafe_allow_html=True)
    st.divider()

# ──────────────────────────────────────────────
#  KEY FINDINGS SUMMARY
# ──────────────────────────────────────────────
st.markdown("### 📝 Key Findings Summary")
with st.expander("Expand to read insights", expanded=True):
    if "AQI" in df.columns and "City" in df.columns:
        overall_avg = df["AQI"].mean()
        cat, _ = aqi_category(overall_avg)
        worst = df.groupby("City")["AQI"].mean().idxmax()
        best  = df.groupby("City")["AQI"].mean().idxmin()

        monthly_avg_text = ""
        if "Month_Name" in df.columns:
            mn = df.groupby("Month_Name")["AQI"].mean()
            worst_month = mn.idxmax()
            best_month  = mn.idxmin()
            monthly_avg_text = f"- **{worst_month}** is the most polluted month; **{best_month}** is the cleanest.\n"

        seasonal_text = ""
        if "Season" in df.columns:
            sn = df.groupby("Season")["AQI"].mean()
            worst_season = sn.idxmax()
            seasonal_text = f"- **{worst_season}** is the most polluted season.\n"

        st.markdown(f"""
| Insight | Detail |
|---------|--------|
| Overall AQI | **{overall_avg:.1f}** ({cat}) |
| Most Polluted City | **{worst}** |
| Cleanest City | **{best}** |
| Records Analysed | **{len(df):,}** |
""")
        st.markdown(f"""
**Narrative:**
- The dataset covers **{len(selected_cities or df['City'].unique())}** cities with an average AQI of **{overall_avg:.1f}** (**{cat}**).
- **{worst}** consistently records the highest pollution levels.
- **{best}** enjoys relatively cleaner air.
{monthly_avg_text}{seasonal_text}- Pollution spikes correlate with winter temperature inversions, festival seasons (Diwali), and agricultural burning.
- Strong correlation between PM2.5 and PM10 confirms a shared particulate source.
        """)

st.divider()

# ──────────────────────────────────────────────
#  DOWNLOAD FILTERED DATA
# ──────────────────────────────────────────────
st.markdown("### 💾 Download Filtered Dataset")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️  Download as CSV",
    data=csv_bytes,
    file_name="india_air_quality_filtered.csv",
    mime="text/csv",
)
st.caption(f"Exporting {len(df):,} rows × {len(df.columns)} columns")

# ──────────────────────────────────────────────
#  RAW DATA PREVIEW
# ──────────────────────────────────────────────
with st.expander("🗃️ Raw Data Preview (first 200 rows)"):
    st.dataframe(df.head(200), use_container_width=True)

# ──────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#555; padding:2rem 0 1rem 0; font-size:0.8rem;'>
  India Air Quality Dashboard &nbsp;·&nbsp; Built with Streamlit & Plotly &nbsp;·&nbsp;
  Data: Kaggle / rohanrao
</div>
""", unsafe_allow_html=True)