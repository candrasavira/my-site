"""
Dashboard Volatilitas Kurs USD/IDR
Streamlit App — LSTM / GRU / CNN Forecasting
Forecast otomatis menggunakan model terbaik (RMSE Testing terkecil).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import json
from datetime import timedelta

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Volatilitas Kurs USD/IDR",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* ── Reset & Base ─────────────────────────────────────────────────────── */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

  /* ── App Background: soft off-white ──────────────────────────────────── */
  .stApp                        { background: #f0f2f5 !important; }
  .block-container              { padding: 0 2rem 3rem !important; }
  section[data-testid="stSidebar"] { background: #ffffff !important; }

  /* ── ALL text defaults → dark (override Streamlit's white) ───────────── */
  .stApp, .stApp *              { color: #374151; }
  h1, h2, h3, h4, h5, h6       { color: #111827 !important; }
  p, li, span, label, div      { color: #374151; }

  /* ── Streamlit-specific text overrides ───────────────────────────────── */
  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] span,
  [data-testid="stWidgetLabel"] label  { color: #374151 !important; font-weight: 500 !important; }

  [data-testid="stMarkdown"] p,
  [data-testid="stMarkdown"] span,
  [data-testid="stMarkdown"] li        { color: #374151 !important; }

  [data-testid="stText"],
  [data-testid="stCaptionContainer"] p { color: #6b7280 !important; }

  /* ── Inputs & Selects ────────────────────────────────────────────────── */
  input, select, textarea              { color: #111827 !important; background: #fff !important; }
  [data-baseweb="input"] input         { color: #111827 !important; }
  [data-baseweb="select"] *            { color: #111827 !important; background: #fff !important; }
  [data-baseweb="popover"] *           { color: #111827 !important; background: #fff !important; }
  [data-baseweb="calendar"] *          { color: #111827 !important; }

  /* Streamlit date/text/number inputs */
  .stDateInput > div, .stTextInput > div, .stNumberInput > div {
    background: #ffffff !important;
    border-color: #d1d5db !important;
  }
  .stDateInput input, .stTextInput input, .stNumberInput input {
    color: #111827 !important; background: #ffffff !important;
    border-color: #d1d5db !important;
  }

  /* ── Slider ──────────────────────────────────────────────────────────── */
  [data-testid="stSlider"] span,
  [data-testid="stSlider"] p      { color: #374151 !important; }

  /* ── Dataframe ───────────────────────────────────────────────────────── */
  [data-testid="stDataFrame"] *   { color: #374151 !important; }
  [data-testid="stDataFrame"] th  { color: #6b7280 !important; background: #f7f8fa !important; }

  /* ── Metric widget ───────────────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background: #ffffff !important; border: 1px solid #e2e6ed !important;
    border-radius: 10px !important; padding: 0.85rem 1rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
  }
  [data-testid="stMetricLabel"] p { color: #6b7280 !important; font-size: 0.73rem !important; }
  [data-testid="stMetricValue"]   { color: #111827 !important; font-weight: 700 !important; }
  [data-testid="stMetricDelta"]   { font-weight: 600 !important; font-size: 0.78rem !important; }
  [data-testid="stMetricDelta"] svg { display: none; }

  /* ── Buttons ─────────────────────────────────────────────────────────── */
  .stButton > button, [data-testid="stDownloadButton"] button {
    color: #374151 !important; background: #ffffff !important;
    border: 1px solid #d1d5db !important; border-radius: 8px !important;
  }
  .stButton > button:hover { background: #f3f4f6 !important; }

  /* ── File uploader ───────────────────────────────────────────────────── */
  [data-testid="stFileUploader"] span,
  [data-testid="stFileUploader"] p    { color: #6b7280 !important; }

  /* ── Alerts ──────────────────────────────────────────────────────────── */
  [data-testid="stAlert"] p { color: inherit !important; }

  /* ─────────────────────────────────────────────────────────────────────
     NAVBAR
  ───────────────────────────────────────────────────────────────────── */
  .navbar {
    background: #ffffff;
    border-bottom: 1px solid #e2e6ed;
    padding: 0 2rem;
    margin: -1rem -1rem 0 -1rem;
    height: 62px;
    display: flex; align-items: center; gap: 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    position: sticky; top: 0; z-index: 999;
  }
  .navbar-brand { display: flex; align-items: center; gap: 10px; }
  .navbar-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 1rem;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3); flex-shrink: 0;
  }
  .navbar-title {
    font-size: 1.05rem; font-weight: 700;
    color: #111827 !important; letter-spacing: -0.02em; line-height: 1.2;
  }
  .navbar-subtitle {
    font-size: 0.7rem; color: #6b7280 !important;
    font-weight: 400; letter-spacing: 0.01em;
  }
  .navbar-divider {
    width: 1px; height: 28px; background: #e2e6ed;
    margin: 0 1.25rem; flex-shrink: 0;
  }
  .navbar-meta { display: flex; align-items: center; gap: 1.25rem; flex: 1; }
  .navbar-chip {
    display: flex; align-items: center; gap: 5px;
    font-size: 0.73rem; color: #6b7280 !important;
    font-weight: 500; white-space: nowrap;
  }
  .navbar-chip-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #d1d5db; flex-shrink: 0;
  }
  .navbar-chip-dot.blue  { background: #3b82f6; }
  .navbar-chip-dot.green { background: #10b981; }
  .navbar-chip-dot.amber { background: #f59e0b; }
  .navbar-right { margin-left: auto; display: flex; align-items: center; gap: 0.75rem; }
  .badge-live {
    background: linear-gradient(135deg, #10b981, #059669);
    color: #fff !important; font-size: 0.62rem; font-weight: 700;
    padding: 3px 9px; border-radius: 999px; letter-spacing: 0.07em;
    text-transform: uppercase; box-shadow: 0 1px 4px rgba(16,185,129,0.35);
    display: flex; align-items: center; gap: 4px;
  }
  .badge-live::before {
    content: ''; width: 5px; height: 5px; border-radius: 50%;
    background: #fff; animation: pulse-dot 1.6s ease-in-out infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.75); }
  }
  .badge-model {
    background: #eff6ff; color: #2563eb !important; border: 1px solid #bfdbfe;
    font-size: 0.65rem; font-weight: 600; padding: 3px 9px; border-radius: 6px;
    letter-spacing: 0.04em; text-transform: uppercase; white-space: nowrap;
  }

  /* ─────────────────────────────────────────────────────────────────────
     KPI CARDS
  ───────────────────────────────────────────────────────────────────── */
  .kpi-row  { display: flex; gap: 0.85rem; margin: 1.25rem 0 1.5rem; flex-wrap: wrap; }
  .kpi-card {
    flex: 1; min-width: 155px;
    background: #ffffff; border: 1px solid #e2e6ed;
    border-radius: 12px; padding: 1rem 1.2rem;
    position: relative; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: box-shadow 0.15s ease;
  }
  .kpi-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.09); }
  .kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: var(--accent, #3b82f6); border-radius: 12px 12px 0 0;
  }
  .kpi-card .label {
    color: #6b7280 !important; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.09em;
  }
  .kpi-card .value {
    color: #111827 !important; font-size: 1.5rem; font-weight: 800;
    margin: 0.25rem 0 0; line-height: 1.1; letter-spacing: -0.02em;
  }
  .kpi-card .delta { font-size: 0.73rem; font-weight: 500; margin-top: 0.2rem; }
  .delta-up   { color: #059669 !important; font-weight: 600; }
  .delta-down { color: #dc2626 !important; font-weight: 600; }
  .delta-neu  { color: #6b7280 !important; }
  .kpi-badge {
    display: inline-block; font-size: 0.58rem; font-weight: 700;
    padding: 2px 7px; border-radius: 5px; margin-top: 5px;
    background: linear-gradient(135deg, #10b981, #059669);
    color: #fff !important; text-transform: uppercase; letter-spacing: 0.06em;
    box-shadow: 0 1px 3px rgba(16,185,129,0.3);
  }

  /* ─────────────────────────────────────────────────────────────────────
     SECTION TITLES
  ───────────────────────────────────────────────────────────────────── */
  .section-title {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.11em; color: #4b5563 !important; margin: 1.5rem 0 0.75rem;
    border-left: 3px solid #3b82f6; padding-left: 0.6rem;
  }

  /* ─────────────────────────────────────────────────────────────────────
     TABS
  ───────────────────────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 2px solid #e2e6ed;
    background: transparent; margin-top: 0.75rem;
  }
  .stTabs [data-baseweb="tab"] {
    color: #6b7280 !important; font-size: 0.81rem; font-weight: 500;
    padding: 0.6rem 1.2rem; border-radius: 0; border: none;
    background: transparent; border-bottom: 2px solid transparent;
    margin-bottom: -2px; transition: color 0.15s;
  }
  .stTabs [data-baseweb="tab"]:hover { color: #374151 !important; }
  .stTabs [aria-selected="true"] {
    color: #1d4ed8 !important;
    border-bottom: 2px solid #2563eb !important;
    background: transparent !important; font-weight: 600 !important;
  }

  /* ─────────────────────────────────────────────────────────────────────
     TABLES
  ───────────────────────────────────────────────────────────────────── */
  .styled-table { width: 100%; border-collapse: collapse; font-size: 0.81rem; }
  .styled-table thead tr {
    background: #f7f8fa; color: #6b7280 !important;
    text-transform: uppercase; font-size: 0.68rem; letter-spacing: 0.08em;
    border-bottom: 1px solid #e2e6ed;
  }
  .styled-table th, .styled-table td {
    padding: 0.6rem 0.9rem; border-bottom: 1px solid #f0f2f5;
    color: #374151 !important;
  }
  .styled-table tbody tr:hover { background: #f7f8fa; }
  .styled-table .highlight { color: #059669 !important; font-weight: 600; }
  .styled-table .rank-1    { color: #d97706 !important; font-weight: 700; }

  /* ─────────────────────────────────────────────────────────────────────
     INFO / WARN / BEST BOXES
  ───────────────────────────────────────────────────────────────────── */
  .info-box {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 8px; padding: 0.9rem 1.1rem;
    color: #1e40af !important; font-size: 0.8rem; margin: 0.75rem 0;
  }
  .warn-box {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 8px; padding: 0.9rem 1.1rem;
    color: #92400e !important; font-size: 0.8rem; margin: 0.75rem 0;
  }
  .best-box {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 8px; padding: 0.9rem 1.1rem;
    color: #15803d !important; font-size: 0.85rem; margin: 0.75rem 0;
  }

  /* ─────────────────────────────────────────────────────────────────────
     UPLOAD SECTION
  ───────────────────────────────────────────────────────────────────── */
  .upload-section {
    background: #ffffff; border: 2px dashed #d1d5db;
    border-radius: 12px; padding: 1.5rem; text-align: center; margin: 1rem 0;
  }

  /* ─────────────────────────────────────────────────────────────────────
     MISC
  ───────────────────────────────────────────────────────────────────── */
  code {
    background: #f1f5f9; color: #be185d !important;
    padding: 1px 5px; border-radius: 4px; font-size: 0.85em;
  }
  .streamlit-expanderHeader { color: #374151 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly light theme helper ─────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(family="Inter, sans-serif", color="#374151", size=12),
    xaxis=dict(gridcolor="#e5e9ef", linecolor="#d1d5db", showgrid=True, zeroline=False, tickfont=dict(color="#6b7280")),
    yaxis=dict(gridcolor="#e5e9ef", linecolor="#d1d5db", showgrid=True, zeroline=False, tickfont=dict(color="#6b7280")),
    legend=dict(bgcolor="#ffffff", bordercolor="#e2e6ed", borderwidth=1, font=dict(size=11, color="#374151")),
    margin=dict(l=55, r=25, t=45, b=50),
    hovermode="x unified",
)

MODEL_COLORS = {"LSTM": "#3b82f6", "GRU": "#10b981", "CNN": "#f59e0b"}

# ─── CI helper (identik dengan notebook) ──────────────────────────────────────
def compute_ci(pred_vol):
    """CI 95%: sigma = sqrt(|pred_vol|), batas = ±1.96 × sigma"""
    z = norm.ppf(0.975)
    sigma = np.sqrt(np.abs(pred_vol))
    return -z * sigma, z * sigma

def pct_in_ci(y_actual, lower, upper):
    n = min(len(y_actual), len(lower), len(upper))
    inside = np.sum((y_actual[:n] >= lower[:n]) & (y_actual[:n] <= upper[:n]))
    return 100 * inside / n

# ─── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {}

    try:
        df_kurs = pd.read_csv("data_kurs_bersih.csv", parse_dates=["Tanggal"])
        data["kurs"] = df_kurs.sort_values("Tanggal").reset_index(drop=True)
    except Exception as e:
        data["kurs"] = None

    try:
        data["forecast"] = pd.read_csv("forecast_30hari.csv", parse_dates=["Tanggal"])
    except Exception:
        data["forecast"] = None

    try:
        with open("metrics_summary.json") as f:
            data["metrics"] = json.load(f)
    except Exception:
        data["metrics"] = None

    try:
        data["rmse"] = pd.read_csv("comparison_rmse.csv")
    except Exception:
        data["rmse"] = None

    try:
        data["ci_pct"] = pd.read_csv("comparison_pct.csv")
    except Exception:
        data["ci_pct"] = None

    try:
        data["test_preds"] = pd.read_csv("predictions_testing.csv", parse_dates=["Tanggal"])
    except Exception:
        data["test_preds"] = None

    try:
        data["train_preds"] = pd.read_csv("predictions_training.csv")
    except Exception:
        data["train_preds"] = None

    return data

D = load_data()

# ─── Deteksi model terbaik & kolom forecast ────────────────────────────────────
def get_best_model():
    """
    Prioritas: baca 'best_model_name' dari metrics_summary.json (disimpan notebook).
    Fallback: hitung dari RMSE test terkecil.
    Terakhir: deteksi kolom di forecast_30hari.csv.
    """
    # 1. Dari JSON (paling akurat — disimpan langsung oleh notebook)
    if D["metrics"] and "best_model_name" in D["metrics"]:
        return D["metrics"]["best_model_name"]

    # 2. Hitung dari RMSE test
    if D["metrics"]:
        rmse_map = {m: D["metrics"]["RMSE"][m]["test"] for m in ["LSTM", "GRU", "CNN"]}
        return min(rmse_map, key=rmse_map.get)

    # 3. Deteksi kolom forecast CSV
    if D["forecast"] is not None:
        for m in ["LSTM", "GRU", "CNN"]:
            if f"Volatilitas_{m}" in D["forecast"].columns:
                return m

    return "GRU"  # default terakhir

BEST_MODEL = get_best_model()
BEST_COLOR = MODEL_COLORS[BEST_MODEL]
FC_COL     = f"Volatilitas_{BEST_MODEL}"   # nama kolom di forecast_30hari.csv

# ─── KPI Helpers ───────────────────────────────────────────────────────────────
def get_kpis():
    kpis = {}
    if D["kurs"] is not None:
        df = D["kurs"]
        kpis["kurs_last"] = df["Kurs"].iloc[-1]
        kpis["kurs_prev"] = df["Kurs"].iloc[-2] if len(df) > 1 else kpis["kurs_last"]
        kpis["vol_last"]  = df["Volatilitas"].iloc[-1]
        kpis["vol_prev"]  = df["Volatilitas"].iloc[-2] if len(df) > 1 else kpis["vol_last"]
    else:
        kpis["kurs_last"] = kpis["kurs_prev"] = kpis["vol_last"] = kpis["vol_prev"] = 0

    if D["metrics"]:
        m = D["metrics"]
        kpis["best_rmse"] = m["RMSE"][BEST_MODEL]["test"]
        kpis["best_ci"]   = m["CI_pct"][BEST_MODEL]["test"]
    else:
        kpis["best_rmse"] = kpis["best_ci"] = 0
    return kpis

kpis = get_kpis()

# ─── Header ────────────────────────────────────────────────────────────────────
date_str = (D["kurs"]["Tanggal"].iloc[-1].strftime("%d %b %Y")
            if D["kurs"] is not None else "—")
st.markdown(f"""
<nav class="navbar">
  <div class="navbar-brand">
    <div class="navbar-icon">📊</div>
    <div>
      <div class="navbar-title">Dashboard Volatilitas Kurs</div>
      <div class="navbar-subtitle">USD / IDR — Deep Learning Forecasting</div>
    </div>
  </div>
  <div class="navbar-divider"></div>
  <div class="navbar-meta">
    <div class="navbar-chip">
      <div class="navbar-chip-dot blue"></div>
      LSTM
    </div>
    <div class="navbar-chip">
      <div class="navbar-chip-dot green"></div>
      GRU
    </div>
    <div class="navbar-chip">
      <div class="navbar-chip-dot amber"></div>
      CNN
    </div>
    <div class="navbar-chip" style="margin-left:0.25rem;color:#9ca3af;">
      Data per {date_str}
    </div>
  </div>
  <div class="navbar-right">
    <span class="badge-model">Best: {BEST_MODEL}</span>
    <span class="badge-live">Live</span>
  </div>
</nav>
""", unsafe_allow_html=True)

# ─── Hero KPI Row ──────────────────────────────────────────────────────────────
kurs_delta = kpis["kurs_last"] - kpis["kurs_prev"]
vol_delta  = kpis["vol_last"]  - kpis["vol_prev"]
kurs_cls   = "delta-up"   if kurs_delta > 0 else "delta-down"
vol_cls    = "delta-up"   if vol_delta  > 0 else "delta-down"
kurs_arrow = "↑" if kurs_delta > 0 else "↓"
vol_arrow  = "↑" if vol_delta  > 0 else "↓"

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card" style="--accent:#3b82f6">
    <div class="label">Kurs Terakhir (IDR)</div>
    <div class="value">Rp {kpis["kurs_last"]:,.1f}</div>
    <div class="delta {kurs_cls}">{kurs_arrow} Rp {abs(kurs_delta):,.1f} vs hari lalu</div>
  </div>
  <div class="kpi-card" style="--accent:#f59e0b">
    <div class="label">Volatilitas Terakhir</div>
    <div class="value">{kpis["vol_last"]:.6f}</div>
    <div class="delta {vol_cls}">{vol_arrow} {abs(vol_delta):.6f} vs hari lalu</div>
  </div>
  <div class="kpi-card" style="--accent:{BEST_COLOR}">
    <div class="label">Model Terbaik (RMSE min)</div>
    <div class="value">{BEST_MODEL}</div>
    <div><span class="kpi-badge">Best Forecast</span></div>
  </div>
  <div class="kpi-card" style="--accent:#8b5cf6">
    <div class="label">RMSE Test ({BEST_MODEL})</div>
    <div class="value">{kpis["best_rmse"]:.6f}</div>
    <div class="delta delta-neu">CI 95% → {kpis["best_ci"]:.1f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈  Overview EDA",
    "📊  Perbandingan Model",
    "🎯  Selang Kepercayaan",
    f"🔮  Ramalan 30 Hari ({BEST_MODEL})",
    "⚙️  Hyperparameter",
    "🔄  Prediksi Data Sendiri",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview EDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if D["kurs"] is None:
        st.error("File `data_kurs_bersih.csv` tidak ditemukan.")
    else:
        df = D["kurs"].copy()

        st.markdown('<div class="section-title">Filter Rentang Tanggal</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            date_min = df["Tanggal"].min().date()
            date_max = df["Tanggal"].max().date()
            d_start = st.date_input("Dari", value=date_min, min_value=date_min, max_value=date_max, key="eda_start")
        with col_b:
            d_end = st.date_input("Sampai", value=date_max, min_value=date_min, max_value=date_max, key="eda_end")

        mask = (df["Tanggal"].dt.date >= d_start) & (df["Tanggal"].dt.date <= d_end)
        df_f = df[mask]

        st.markdown('<div class="section-title">Kurs Harian USD/IDR</div>', unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df_f["Tanggal"], y=df_f["Kurs"],
            mode="lines", name="Kurs USD/IDR",
            line=dict(color="#3b82f6", width=1.8),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.07)"
        ))
        fig1.update_layout(**CHART_LAYOUT, height=300,
                           yaxis_title="Kurs (IDR)", xaxis_title="Tanggal")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown('<div class="section-title">Log Return & Volatilitas (Squared Return)</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            fig2 = make_subplots(rows=2, cols=1, row_heights=[0.45, 0.55], vertical_spacing=0.18,
                                 subplot_titles=["Time Series Log Return", "Distribusi Log Return"])
            fig2.add_trace(go.Scatter(
                x=df_f["Tanggal"], y=df_f["Return"],
                mode="lines", name="Return", line=dict(color="#3b82f6", width=1),
            ), row=1, col=1)
            fig2.add_trace(go.Histogram(
                x=df_f["Return"], nbinsx=60, name="Frekuensi",
                marker_color="#3b82f6", opacity=0.75
            ), row=2, col=1)
            fig2.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="#f43f5e", row=2, col=1)
            fig2.update_layout(**{k: v for k, v in CHART_LAYOUT.items() if k != "margin"},
                               height=460, showlegend=False,
                               margin=dict(l=50, r=15, t=50, b=40))
            fig2.update_annotations(font=dict(size=11, color="#374151"))
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            fig3 = make_subplots(rows=2, cols=1, row_heights=[0.5, 0.5], vertical_spacing=0.18,
                                 subplot_titles=["Volatilitas (Squared Return)", "Log Volatilitas"])
            fig3.add_trace(go.Scatter(
                x=df_f["Tanggal"], y=df_f["Volatilitas"],
                mode="lines", name="Volatilitas",
                line=dict(color="#a855f7", width=1.2),
                fill="tozeroy", fillcolor="rgba(168,85,247,0.08)"
            ), row=1, col=1)
            log_vol = np.log(df_f["Volatilitas"].clip(lower=1e-10))
            fig3.add_trace(go.Scatter(
                x=df_f["Tanggal"], y=log_vol,
                mode="lines", name="Log Volatilitas",
                line=dict(color="#e67e22", width=1.0), opacity=0.85
            ), row=2, col=1)
            fig3.update_layout(**{k: v for k, v in CHART_LAYOUT.items() if k != "margin"},
                               height=460, showlegend=False,
                               margin=dict(l=50, r=15, t=50, b=40))
            fig3.update_annotations(font=dict(size=11, color="#374151"))
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="section-title">Top 5 Hari Volatilitas Tertinggi</div>', unsafe_allow_html=True)
        top5 = df_f.nlargest(5, "Volatilitas")[["Tanggal","Kurs","Return","Volatilitas"]].reset_index(drop=True)
        top5.index += 1
        top5["Tanggal"]     = top5["Tanggal"].dt.strftime("%d %b %Y")
        top5["Kurs"]        = top5["Kurs"].apply(lambda x: f"Rp {x:,.1f}")
        top5["Return"]      = top5["Return"].apply(lambda x: f"{x:.6f}")
        top5["Volatilitas"] = top5["Volatilitas"].apply(lambda x: f"{x:.8f}")

        tbl_html = "<table class='styled-table'><thead><tr><th>#</th><th>Tanggal</th><th>Kurs</th><th>Return</th><th>Volatilitas</th></tr></thead><tbody>"
        for i, row in top5.iterrows():
            rank_cls = "rank-1" if i == 1 else "highlight" if i == 2 else ""
            tbl_html += f"<tr><td class='{rank_cls}'>{i}</td><td>{row['Tanggal']}</td><td>{row['Kurs']}</td><td>{row['Return']}</td><td class='{rank_cls}'>{row['Volatilitas']}</td></tr>"
        tbl_html += "</tbody></table>"
        st.markdown(tbl_html, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Statistik Ringkas</div>', unsafe_allow_html=True)
        cs1, cs2, cs3, cs4 = st.columns(4)
        cs1.metric("Jumlah Hari",           f"{len(df_f):,}")
        cs2.metric("Return Rata-Rata",      f"{df_f['Return'].mean():.6f}")
        cs3.metric("Std Return",            f"{df_f['Return'].std():.6f}")
        cs4.metric("Volatilitas Rata-Rata", f"{df_f['Volatilitas'].mean():.8f}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Perbandingan Model
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    metrics = D["metrics"]
    if metrics is None:
        st.error("File `metrics_summary.json` tidak ditemukan.")
    else:
        models = ["LSTM", "GRU", "CNN"]

        # Banner model terbaik
        rmse_test_all = {m: metrics["RMSE"][m]["test"] for m in models}
        st.markdown(f"""
        <div class="best-box">
          🏆 <strong>Model Terbaik: {BEST_MODEL}</strong> &nbsp;|&nbsp;
          RMSE Testing terkecil: <strong>{rmse_test_all[BEST_MODEL]:.7f}</strong> &nbsp;|&nbsp;
          CI% Testing: <strong>{metrics["CI_pct"][BEST_MODEL]["test"]:.2f}%</strong> &nbsp;|&nbsp;
          Digunakan untuk Forecast 30 Hari
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="section-title">Performa Model Terbaik — {BEST_MODEL}</div>', unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(f"{BEST_MODEL} RMSE Train", f"{metrics['RMSE'][BEST_MODEL]['train']:.7f}")
        mc2.metric(f"{BEST_MODEL} RMSE Test",  f"{metrics['RMSE'][BEST_MODEL]['test']:.7f}")
        mc3.metric(f"{BEST_MODEL} CI% Train",  f"{metrics['CI_pct'][BEST_MODEL]['train']:.2f}%")
        mc4.metric(f"{BEST_MODEL} CI% Test",   f"{metrics['CI_pct'][BEST_MODEL]['test']:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-title">RMSE per Model (lebih kecil = lebih baik)</div>', unsafe_allow_html=True)
            train_vals = [metrics["RMSE"][m]["train"] for m in models]
            test_vals  = [metrics["RMSE"][m]["test"]  for m in models]

            # Highlight warna model terbaik lebih terang
            bar_colors = [MODEL_COLORS[m] if m == BEST_MODEL else
                          MODEL_COLORS[m].replace("#", "rgba(") + ",0.55)" for m in models]

            fig_rmse = go.Figure()
            fig_rmse.add_trace(go.Bar(
                x=models, y=train_vals, name="Training",
                marker_color=["rgba(59,130,246,0.45)","rgba(16,185,129,0.45)","rgba(245,158,11,0.45)"],
                text=[f"{v:.7f}" for v in train_vals], textposition="outside",
                textfont=dict(size=9, color="#6b7280")
            ))
            fig_rmse.add_trace(go.Bar(
                x=models, y=test_vals, name="Testing",
                marker_color=[MODEL_COLORS[m] for m in models],
                text=[f"{v:.7f}" for v in test_vals], textposition="outside",
                textfont=dict(size=9, color="#374151")
            ))
            # Tandai model terbaik
            fig_rmse.add_annotation(
                x=BEST_MODEL, y=min(test_vals) * 0.5,
                text="⭐ Best", showarrow=False,
                font=dict(color=BEST_COLOR, size=11, family="Inter")
            )
            fig_rmse.update_layout(**CHART_LAYOUT, barmode="group", height=370, yaxis_title="RMSE")
            st.plotly_chart(fig_rmse, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-title">% Data dalam CI 95% (lebih tinggi = lebih baik)</div>', unsafe_allow_html=True)
            ci_train = [metrics["CI_pct"][m]["train"] for m in models]
            ci_test  = [metrics["CI_pct"][m]["test"]  for m in models]

            fig_ci = go.Figure()
            fig_ci.add_shape(type="line", x0=0, x1=1, y0=95, y1=95,
                             xref="paper", yref="y",
                             line=dict(color="#f43f5e", width=1.5, dash="dash"))
            fig_ci.add_annotation(x=1, y=95, xref="paper", yref="y",
                                  text="Target 95%", showarrow=False,
                                  xanchor="right", font=dict(color="#f43f5e", size=10))
            fig_ci.add_trace(go.Bar(
                x=models, y=ci_train, name="Training",
                marker_color=["rgba(59,130,246,0.45)","rgba(16,185,129,0.45)","rgba(245,158,11,0.45)"],
                text=[f"{v:.1f}%" for v in ci_train], textposition="outside",
                textfont=dict(size=10, color="#6b7280")
            ))
            fig_ci.add_trace(go.Bar(
                x=models, y=ci_test, name="Testing",
                marker_color=[MODEL_COLORS[m] for m in models],
                text=[f"{v:.1f}%" for v in ci_test], textposition="outside",
                textfont=dict(size=10, color="#374151")
            ))
            fig_ci.update_layout(**CHART_LAYOUT, barmode="group", height=370,
                                 yaxis_title="Persentase (%)", yaxis_range=[0, 105])
            st.plotly_chart(fig_ci, use_container_width=True)

        st.markdown('<div class="section-title">Tabel Perbandingan Lengkap</div>', unsafe_allow_html=True)
        rows = []
        for m in models:
            r = metrics["RMSE"][m]
            c = metrics["CI_pct"][m]
            rows.append({
                "Model":      m,
                "Status":     f"⭐ Best Forecast" if m == BEST_MODEL else "",
                "RMSE Train": f"{r['train']:.8f}",
                "RMSE Test":  f"{r['test']:.8f}",
                "CI% Train":  f"{c['train']:.2f}%",
                "CI% Test":   f"{c['test']:.2f}%",
                "Memenuhi ≥95%": "✅" if c["test"] >= 95 else "❌",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Selang Kepercayaan
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    if D["test_preds"] is None:
        st.error("File `predictions_testing.csv` tidak ditemukan.")
    else:
        df_tp = D["test_preds"].copy()
        x_axis     = df_tp["Tanggal"] if "Tanggal" in df_tp.columns else np.arange(len(df_tp))
        ret_actual = df_tp["Return_Aktual"].values
        vol_actual = df_tp["Volatilitas_Aktual"].values

        st.markdown('<div class="section-title">Tampilkan CI 95% per Model</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          ℹ️ CI 95%: σ = √|prediksi_volatilitas|, batas = ±1.96 × σ.
          Plot menggunakan <strong>Return Aktual</strong> (bukan volatilitas) — sesuai metode notebook.
        </div>
        """, unsafe_allow_html=True)

        tc1, tc2, tc3 = st.columns(3)
        show_lstm = tc1.checkbox("LSTM", value=True,  key="ci_lstm")
        show_gru  = tc2.checkbox("GRU",  value=True,  key="ci_gru")
        show_cnn  = tc3.checkbox("CNN",  value=False, key="ci_cnn")

        fig_ci_plot = go.Figure()
        fig_ci_plot.add_trace(go.Scatter(
            x=x_axis, y=ret_actual, mode="lines", name="Return Aktual",
            line=dict(color="#f8fafc", width=1)
        ))

        for model, color, show in [("LSTM","#3b82f6",show_lstm),
                                    ("GRU","#10b981",show_gru),
                                    ("CNN","#f59e0b",show_cnn)]:
            if show and model in df_tp.columns:
                pred_vol = df_tp[model].values
                lo, hi   = compute_ci(pred_vol)
                n_plot   = min(len(lo), len(ret_actual))
                rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                x_fwd = list(x_axis[:n_plot]) + list(x_axis[:n_plot])[::-1]
                y_fwd = list(hi[:n_plot])     + list(lo[:n_plot])[::-1]
                fig_ci_plot.add_trace(go.Scatter(
                    x=x_fwd, y=y_fwd, fill="toself",
                    fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.20)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"CI 95% {model}", hoverinfo="skip"
                ))
                fig_ci_plot.add_trace(go.Scatter(
                    x=x_axis, y=pred_vol, mode="lines",
                    name=f"Pred Vol {model}",
                    line=dict(color=color, width=1.2, dash="dot")
                ))
                pct = pct_in_ci(ret_actual, lo, hi)
                st.markdown(
                    f'<div class="info-box" style="margin-bottom:0.3rem">'
                    f'<b style="color:{color}">{model}</b> — {pct:.2f}% return aktual dalam CI 95%</div>',
                    unsafe_allow_html=True
                )

        fig_ci_plot.update_layout(**CHART_LAYOUT, height=420,
                                  title="Selang Kepercayaan 95% — Data Testing",
                                  yaxis_title="Return / Prediksi Volatilitas",
                                  xaxis_title="Waktu")
        st.plotly_chart(fig_ci_plot, use_container_width=True)

        st.markdown('<div class="section-title">Volatilitas Prediksi vs Aktual — Data Testing</div>', unsafe_allow_html=True)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=x_axis, y=vol_actual, mode="lines", name="Aktual",
                                      line=dict(color="#f8fafc", width=1.5)))
        for m, c in MODEL_COLORS.items():
            if m in df_tp.columns:
                fig_pred.add_trace(go.Scatter(x=x_axis, y=df_tp[m], mode="lines", name=m,
                                              line=dict(color=c, width=1.2), opacity=0.85))
        fig_pred.update_layout(**CHART_LAYOUT, height=380,
                               yaxis_title="Volatilitas", xaxis_title="Waktu")
        st.plotly_chart(fig_pred, use_container_width=True)

        if D["train_preds"] is not None:
            df_tr = D["train_preds"].copy()
            st.markdown('<div class="section-title">Volatilitas Prediksi vs Aktual — Data Training</div>', unsafe_allow_html=True)
            x_tr = np.arange(len(df_tr))
            fig_tr = go.Figure()
            fig_tr.add_trace(go.Scatter(x=x_tr, y=df_tr["Volatilitas_Aktual"], mode="lines",
                                        name="Aktual", line=dict(color="#f8fafc", width=1.5)))
            for m, c in MODEL_COLORS.items():
                if m in df_tr.columns:
                    fig_tr.add_trace(go.Scatter(x=x_tr, y=df_tr[m], mode="lines", name=m,
                                                line=dict(color=c, width=1.2), opacity=0.85))
            fig_tr.update_layout(**CHART_LAYOUT, height=350,
                                 yaxis_title="Volatilitas", xaxis_title="Indeks Waktu")
            st.plotly_chart(fig_tr, use_container_width=True)

        st.markdown('<div class="section-title">Ringkasan % Data dalam CI 95%</div>', unsafe_allow_html=True)
        if D["metrics"]:
            ci_rows = []
            for m in ["LSTM", "GRU", "CNN"]:
                ci_rows.append({
                    "Model":                  m,
                    "CI% Training":           f"{D['metrics']['CI_pct'][m]['train']:.2f}%",
                    "CI% Testing":            f"{D['metrics']['CI_pct'][m]['test']:.2f}%",
                    "Memenuhi Target (≥95%)": "✅" if D["metrics"]["CI_pct"][m]["test"] >= 95 else "❌"
                })
            st.dataframe(pd.DataFrame(ci_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Ramalan 30 Hari (Model Terbaik — Dinamis)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    if D["forecast"] is None or D["kurs"] is None:
        st.error("File `forecast_30hari.csv` atau `data_kurs_bersih.csv` tidak ditemukan.")
    else:
        df_fc   = D["forecast"].copy()
        df_kurs = D["kurs"].copy()

        # Validasi kolom forecast
        if FC_COL not in df_fc.columns:
            # coba deteksi kolom Volatilitas_* apapun
            vol_cols = [c for c in df_fc.columns if c.startswith("Volatilitas_")]
            if vol_cols:
                fc_col_use = vol_cols[0]
                actual_best = fc_col_use.replace("Volatilitas_", "")
            else:
                st.error(f"Kolom `{FC_COL}` tidak ditemukan di forecast_30hari.csv. Kolom tersedia: {list(df_fc.columns)}")
                st.stop()
        else:
            fc_col_use   = FC_COL
            actual_best  = BEST_MODEL

        fc_color = MODEL_COLORS.get(actual_best, "#10b981")

        # Banner model terbaik
        st.markdown(f"""
        <div class="best-box">
          🏆 Forecast menggunakan <strong>Model {actual_best}</strong>
          (RMSE Testing terkecil dari LSTM, GRU, CNN) &nbsp;·&nbsp;
          30 hari ke depan menggunakan auto-regressive prediction (LOOKBACK = 10)
        </div>
        """, unsafe_allow_html=True)

        cutoff  = df_kurs["Tanggal"].max() - pd.DateOffset(months=3)
        df_hist = df_kurs[df_kurs["Tanggal"] >= cutoff]
        split_dt = df_hist["Tanggal"].max().strftime("%Y-%m-%d")

        st.markdown(f'<div class="section-title">Volatilitas Historis 3 Bulan + Forecast 30 Hari ({actual_best})</div>', unsafe_allow_html=True)

        fig_fc_plot = go.Figure()
        fig_fc_plot.add_trace(go.Scatter(
            x=df_hist["Tanggal"], y=df_hist["Volatilitas"],
            mode="lines", name="Historis",
            line=dict(color="#64748b", width=1.5),
            fill="tozeroy", fillcolor="rgba(100,116,139,0.08)"
        ))
        fig_fc_plot.add_trace(go.Scatter(
            x=df_fc["Tanggal"], y=df_fc[fc_col_use],
            mode="lines+markers", name=f"Forecast {actual_best}",
            line=dict(color=fc_color, width=2.2),
            marker=dict(size=5, color=fc_color),
            fill="tozeroy", fillcolor=f"rgba({','.join(str(int(fc_color[i:i+2],16)) for i in (1,3,5))},0.12)"
        ))
        fig_fc_plot.add_shape(type="line", x0=split_dt, x1=split_dt, y0=0, y1=1,
                              xref="x", yref="paper",
                              line=dict(color="#334155", width=1.5, dash="dash"))
        fig_fc_plot.add_annotation(x=split_dt, y=0.98, xref="x", yref="paper",
                                   text="Mulai Forecast", showarrow=False,
                                   xanchor="left", font=dict(color="#6b7280", size=10),
                                   bgcolor="#ffffff", bordercolor="#e2e6ed", borderwidth=1)
        fig_fc_plot.update_layout(**CHART_LAYOUT, height=380,
                                  yaxis_title="Volatilitas", xaxis_title="Tanggal")
        st.plotly_chart(fig_fc_plot, use_container_width=True)

        st.markdown(f'<div class="section-title">Zoom: 30 Hari Forecast + Anotasi Tiap 5 Hari ({actual_best})</div>', unsafe_allow_html=True)
        fig_zoom = go.Figure()
        fig_zoom.add_trace(go.Scatter(
            x=df_fc["Tanggal"], y=df_fc[fc_col_use],
            mode="lines+markers", name=f"Forecast {actual_best}",
            line=dict(color=fc_color, width=2),
            marker=dict(size=6, color=fc_color, symbol="circle")
        ))
        for i, row in df_fc.iterrows():
            if i % 5 == 0:
                fig_zoom.add_annotation(
                    x=row["Tanggal"], y=row[fc_col_use],
                    text=f"{row[fc_col_use]:.6f}",
                    showarrow=True, arrowhead=2, arrowcolor="#d1d5db",
                    arrowsize=1, ax=0, ay=-28,
                    font=dict(size=9, color="#6b7280"),
                    bgcolor="#ffffff", bordercolor="#e2e6ed", borderwidth=1
                )
        fig_zoom.update_layout(**CHART_LAYOUT, height=320,
                               yaxis_title="Volatilitas", xaxis_title="Tanggal")
        st.plotly_chart(fig_zoom, use_container_width=True)

        st.markdown('<div class="section-title">Tabel Nilai Forecast</div>', unsafe_allow_html=True)
        df_show = df_fc[["Tanggal", fc_col_use]].copy()
        df_show["Tanggal"] = df_show["Tanggal"].dt.strftime("%d %b %Y")
        df_show.columns    = ["Tanggal", f"Volatilitas Forecast ({actual_best})"]
        df_show.index      = range(1, len(df_show) + 1)

        col_tbl, col_dl = st.columns([3, 1])
        with col_tbl:
            st.dataframe(df_show, use_container_width=True)
        with col_dl:
            st.markdown("<br><br>", unsafe_allow_html=True)
            csv_out = df_fc.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Download CSV",
                data=csv_out,
                file_name=f"forecast_volatilitas_30hari_{actual_best}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown(f"""
            <div class="info-box">
              Model <strong>{actual_best}</strong> dipilih secara otomatis
              karena memiliki RMSE Testing terkecil di antara LSTM, GRU, dan CNN.
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Statistik Forecast</div>', unsafe_allow_html=True)
        sf1, sf2, sf3, sf4 = st.columns(4)
        sf1.metric("Min Volatilitas", f"{df_fc[fc_col_use].min():.8f}")
        sf2.metric("Max Volatilitas", f"{df_fc[fc_col_use].max():.8f}")
        sf3.metric("Rata-Rata",       f"{df_fc[fc_col_use].mean():.8f}")
        sf4.metric("Tren",
                   "⬇️ Menurun" if df_fc[fc_col_use].iloc[-1] < df_fc[fc_col_use].iloc[0]
                   else "⬆️ Meningkat")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Hyperparameter
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    if D["metrics"] is None:
        st.error("File `metrics_summary.json` tidak ditemukan.")
    else:
        bp = D["metrics"]["best_params"]

        st.markdown('<div class="section-title">Konfigurasi Hyperparameter Terbaik (Random Search CV)</div>', unsafe_allow_html=True)
        hp_c1, hp_c2, hp_c3 = st.columns(3)

        def render_hp_card(col, title, color, items, is_best=False):
            best_tag = " ⭐ Best Forecast" if is_best else ""
            col.markdown(f"""
            <div style='background:#ffffff;border:1px solid #e2e6ed;border-top:3px solid {color};
                        border-radius:10px;padding:1.2rem;box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
              <div style='color:{color};font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
                          text-transform:uppercase;margin-bottom:0.75rem;'>{title}{best_tag}</div>
            """, unsafe_allow_html=True)
            for k, v in items.items():
                col.markdown(f"""
                <div style='display:flex;justify-content:space-between;padding:0.35rem 0;
                            border-bottom:1px solid #f0f2f5;font-size:0.82rem;'>
                  <span style='color:#6b7280;'>{k}</span>
                  <span style='color:#111827;font-weight:600;'>{v}</span>
                </div>""", unsafe_allow_html=True)
            col.markdown("</div>", unsafe_allow_html=True)

        lstm = bp["LSTM"]
        render_hp_card(hp_c1, "LSTM", "#3b82f6", {
            "Optimizer":     lstm.get("optimizer", "adam").capitalize(),
            "Activation":    lstm.get("activation", "-").capitalize(),
            "Neurons":       str(lstm.get("n_neurons", "-")),
            "Dropout":       f"{lstm.get('dropout', 0):.4f}",
            "Learning Rate": f"{lstm.get('lr', 0):.7f}",
            "Avg RMSE (CV)": f"{lstm.get('avg_rmse', 0):.8f}",
        }, is_best=(BEST_MODEL == "LSTM"))

        gru = bp["GRU"]
        render_hp_card(hp_c2, "GRU", "#10b981", {
            "Units":         str(gru.get("units", "-")),
            "Batch Size":    str(gru.get("batch_size", "-")),
            "Dropout":       f"{gru.get('dropout', 0):.4f}",
            "Learning Rate": f"{gru.get('lr', 0):.7f}",
            "Avg RMSE (CV)": f"{gru.get('avg_rmse', 0):.8f}",
        }, is_best=(BEST_MODEL == "GRU"))

        cnn = bp["CNN"]
        render_hp_card(hp_c3, "CNN", "#f59e0b", {
            "Filters":        str(cnn.get("n_filters", "-")),
            "Kernel Size":    str(cnn.get("kernel_size", "-")),
            "Pool Size":      str(cnn.get("pool_size", "-")),
            "Conv Layers":    str(cnn.get("conv_layers", "-")),
            "Hidden Layers":  str(cnn.get("hidden_layers", "-")),
            "Layer Units":    str(cnn.get("layer_units", "-")),
            "Activation":     cnn.get("activation", "-").capitalize(),
            "Batch Size":     str(cnn.get("batch_size", "-")),
            "Learning Rate":  f"{cnn.get('lr', 0):.7f}",
            "Conv Dropout":   f"{cnn.get('conv_dropout', 0):.4f}",
            "Hidden Dropout": f"{cnn.get('hidden_dropout', 0):.4f}",
            "Avg RMSE (CV)":  f"{cnn.get('avg_rmse', 0):.8f}",
        }, is_best=(BEST_MODEL == "CNN"))

        st.markdown('<div class="section-title">Avg RMSE Cross-Validation (Random Search — 4 Kelompok)</div>', unsafe_allow_html=True)
        avg_rmses = {m: D["metrics"]["best_params"][m].get("avg_rmse", 0) for m in ["LSTM","GRU","CNN"]}
        fig_hp = go.Figure(go.Bar(
            x=list(avg_rmses.keys()), y=list(avg_rmses.values()),
            marker_color=[MODEL_COLORS[m] for m in avg_rmses],
            text=[f"{v:.8f}" for v in avg_rmses.values()],
            textposition="outside", textfont=dict(size=11, color="#374151")
        ))
        fig_hp.update_layout(**{k: v for k, v in CHART_LAYOUT.items() if k != "margin"},
                             height=300, yaxis_title="Avg RMSE (CV)",
                             showlegend=False, margin=dict(l=50, r=15, t=30, b=40))
        st.plotly_chart(fig_hp, use_container_width=True)

        st.markdown("""
        <div class="info-box">
          ⚙️ Hyperparameter tuning: <strong>Random Search 20 iterasi</strong> ×
          <strong>Sliding Window CV 4 kelompok</strong> (~2.5 tahun/kelompok, 2016–2026).
          Lookback = 10 hari. Model terbaik dipilih otomatis dari RMSE Testing terkecil.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Prediksi Data Sendiri
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">Upload Data Kurs Anda Sendiri</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      📌 <strong>Format CSV yang diterima:</strong><br>
      Minimal satu kolom tanggal (<code>Tanggal</code> / <code>Date</code>) dan satu kolom kurs
      (<code>Kurs</code>, <code>Close</code>, <code>Price</code>, atau <code>Terakhir</code>).<br>
      Contoh: <code>Tanggal,Kurs</code> → <code>2024-01-01,15000</code>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload file CSV kurs Anda", type=["csv"], key="user_upload")

    if uploaded_file is not None:
        try:
            df_user = pd.read_csv(uploaded_file)

            date_col  = next((c for c in df_user.columns if c.lower() in ["tanggal","date","waktu","time"]), None)
            price_col = next((c for c in df_user.columns if c.lower() in ["kurs","close","price","harga","nilai","terakhir"]), None)

            if date_col is None or price_col is None:
                st.markdown(f"""<div class="warn-box">
                ⚠️ Kolom tidak terdeteksi. Kolom ditemukan: <code>{list(df_user.columns)}</code>.
                Pilih secara manual di bawah.
                </div>""", unsafe_allow_html=True)
                date_col  = st.selectbox("Pilih kolom Tanggal",    df_user.columns, key="sel_date")
                price_col = st.selectbox("Pilih kolom Kurs/Harga", df_user.columns, key="sel_price")

            df_user[date_col]  = pd.to_datetime(df_user[date_col], dayfirst=True, errors="coerce")
            df_user[price_col] = pd.to_numeric(df_user[price_col].astype(str).str.replace(",",""), errors="coerce")
            df_user = df_user.dropna(subset=[date_col, price_col]).sort_values(date_col).reset_index(drop=True)

            if len(df_user) < 11:
                st.error("Data terlalu sedikit (minimal 11 baris untuk LOOKBACK=10).")
            else:
                df_user["Return"]      = np.log(df_user[price_col] / df_user[price_col].shift(1))
                df_user["Volatilitas"] = df_user["Return"] ** 2
                df_user = df_user.dropna().reset_index(drop=True)

                st.markdown('<div class="section-title">Preview Data yang Diupload</div>', unsafe_allow_html=True)
                st.markdown(f"**{len(df_user):,} baris valid** | {df_user[date_col].min().strftime('%d %b %Y')} — {df_user[date_col].max().strftime('%d %b %Y')}")
                st.dataframe(df_user[[date_col, price_col, "Return", "Volatilitas"]].tail(10),
                             use_container_width=True, hide_index=True)

                # ── Kurs Harian ──────────────────────────────────────────────
                st.markdown('<div class="section-title">Kurs Harian</div>', unsafe_allow_html=True)
                fig_u1 = go.Figure()
                fig_u1.add_trace(go.Scatter(
                    x=df_user[date_col], y=df_user[price_col],
                    mode="lines", name="Kurs",
                    line=dict(color="#3b82f6", width=1.8),
                    fill="tozeroy", fillcolor="rgba(59,130,246,0.07)"
                ))
                ma20 = df_user[price_col].rolling(20).mean()
                fig_u1.add_trace(go.Scatter(
                    x=df_user[date_col], y=ma20,
                    mode="lines", name="MA-20",
                    line=dict(color="#f59e0b", width=1.5, dash="dot")
                ))
                y_min = df_user[price_col].min() * 0.999
                y_max = df_user[price_col].max() * 1.001
                layout_no_yaxis = {k: v for k, v in CHART_LAYOUT.items() if k != "yaxis"}
                fig_u1.update_layout(**layout_no_yaxis, height=300,
                                     yaxis_title="Kurs (IDR)", xaxis_title="Tanggal",
                                     yaxis=dict(range=[y_min, y_max]))
                st.plotly_chart(fig_u1, use_container_width=True)

                col_u1, col_u2 = st.columns(2)
                with col_u1:
                    st.markdown('<div class="section-title">Volatilitas (Squared Return)</div>', unsafe_allow_html=True)
                    fig_u2 = go.Figure()
                    fig_u2.add_trace(go.Scatter(
                        x=df_user[date_col], y=df_user["Volatilitas"],
                        mode="lines", name="Volatilitas",
                        line=dict(color="#a855f7", width=1.2),
                        fill="tozeroy", fillcolor="rgba(168,85,247,0.08)"
                    ))
                    fig_u2.update_layout(**{k: v for k, v in CHART_LAYOUT.items() if k != "margin"},
                                        height=270, yaxis_title="Volatilitas",
                                        margin=dict(l=50, r=15, t=30, b=40))
                    st.plotly_chart(fig_u2, use_container_width=True)

                with col_u2:
                    st.markdown('<div class="section-title">Distribusi Log Return</div>', unsafe_allow_html=True)
                    fig_u3 = go.Figure()
                    fig_u3.add_trace(go.Histogram(
                        x=df_user["Return"], nbinsx=60,
                        marker_color="#3b82f6", opacity=0.75, name="Return"
                    ))
                    fig_u3.add_vline(x=0, line_dash="dash", line_color="#f43f5e", line_width=1.5)
                    fig_u3.update_layout(**{k: v for k, v in CHART_LAYOUT.items() if k != "margin"},
                                        height=270, xaxis_title="Log Return", yaxis_title="Frekuensi",
                                        margin=dict(l=50, r=15, t=30, b=40), showlegend=False)
                    st.plotly_chart(fig_u3, use_container_width=True)

                # ── Estimasi Kurs 30 Hari ke Depan ───────────────────────────
                st.markdown('<div class="section-title">Estimasi Kurs 10 Hari ke Depan</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="info-box">
                  ℹ️ Estimasi menggunakan <strong>Exponential Smoothing</strong> berbasis level kurs terakhir
                  dengan drift rata-rata harian. Untuk prediksi akurat dengan {BEST_MODEL},
                  load <code>{BEST_MODEL.lower()}_model.keras</code> dan <code>scaler_{BEST_MODEL.lower()}.pkl</code>
                  dari hasil notebook.
                </div>""", unsafe_allow_html=True)

                window = st.slider("Window rolling (hari)", min_value=5, max_value=60, value=30, step=5)

                # Hitung estimasi KURS (bukan volatilitas) agar skala sama
                alpha_price  = 0.3
                last_val     = df_user[price_col].iloc[-1]
                long_run_val = df_user[price_col].tail(window).mean()
                daily_drift  = df_user[price_col].diff().tail(window).mean()

                fc_prices = []
                cur_p = last_val
                for _ in range(10):
                    cur_p = alpha_price * (cur_p + daily_drift) + (1 - alpha_price) * long_run_val
                    fc_prices.append(cur_p)

                last_date  = df_user[date_col].max()
                bdays      = pd.bdate_range(start=last_date + timedelta(days=1), periods=10)
                df_kurs_fc = pd.DataFrame({"Tanggal": bdays, "Kurs_Est": fc_prices})

                # Confidence band ± 1.5 std dari rolling window
                std_val = df_user[price_col].tail(window).std()
                df_kurs_fc["Upper"] = df_kurs_fc["Kurs_Est"] + 1.5 * std_val
                df_kurs_fc["Lower"] = df_kurs_fc["Kurs_Est"] - 1.5 * std_val

                fig_uf = go.Figure()
                df_recent = df_user.tail(60)

                # Historis 60 hari terakhir
                fig_uf.add_trace(go.Scatter(
                    x=df_recent[date_col], y=df_recent[price_col],
                    mode="lines", name="Historis",
                    line=dict(color="#64748b", width=2)
                ))

                # Rolling MA historis
                df_user["Kurs_Roll"] = df_user[price_col].rolling(window).mean()
                fig_uf.add_trace(go.Scatter(
                    x=df_recent[date_col], y=df_user["Kurs_Roll"].tail(60),
                    mode="lines", name=f"MA-{window}d",
                    line=dict(color="#3b82f6", width=1.5, dash="dot")
                ))

                # Confidence band
                fig_uf.add_trace(go.Scatter(
                    x=pd.concat([df_kurs_fc["Tanggal"], df_kurs_fc["Tanggal"][::-1]]),
                    y=pd.concat([df_kurs_fc["Upper"], df_kurs_fc["Lower"][::-1]]),
                    fill="toself", fillcolor="rgba(234,179,8,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Confidence Band ±1.5σ", showlegend=True
                ))

                # Garis estimasi kurs
                fig_uf.add_trace(go.Scatter(
                    x=df_kurs_fc["Tanggal"], y=df_kurs_fc["Kurs_Est"],
                    mode="lines+markers", name="Estimasi 10 Hari",
                    line=dict(color=BEST_COLOR, width=2.5),
                    marker=dict(size=5)
                ))

                # Titik sambung historis → estimasi
                fig_uf.add_trace(go.Scatter(
                    x=[last_date, bdays[0]],
                    y=[last_val, fc_prices[0]],
                    mode="lines", showlegend=False,
                    line=dict(color=BEST_COLOR, width=2.5)
                ))

                last_dt_str = last_date.strftime("%Y-%m-%d")
                fig_uf.add_shape(type="line", x0=last_dt_str, x1=last_dt_str, y0=0, y1=1,
                                 xref="x", yref="paper",
                                 line=dict(color="#334155", width=1.5, dash="dash"))
                fig_uf.add_annotation(x=last_dt_str, y=0.98, xref="x", yref="paper",
                                      text="Mulai Estimasi", showarrow=False,
                                      xanchor="left", font=dict(color="#6b7280", size=10),
                                      bgcolor="#ffffff", bordercolor="#e2e6ed", borderwidth=1)
                fig_uf.update_layout(**CHART_LAYOUT, height=380,
                                     yaxis_title="Kurs (IDR)", xaxis_title="Tanggal")
                st.plotly_chart(fig_uf, use_container_width=True)

                su1, su2, su3, su4 = st.columns(4)
                su1.metric("Kurs Terakhir",      f"{last_val:,.1f}")
                su2.metric("Estimasi 10h (Avg)", f"{np.mean(fc_prices):,.1f}")
                su3.metric("Estimasi Min",        f"{min(fc_prices):,.1f}")
                su4.metric("Estimasi Maks",       f"{max(fc_prices):,.1f}")

                csv_proc = df_user[[date_col, price_col, "Return", "Volatilitas"]].to_csv(index=False).encode("utf-8")
                st.download_button("⬇️  Download Data yang Telah Diproses",
                                   data=csv_proc, file_name="data_kurs_diproses.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
    else:
        st.markdown("""
        <div class="upload-section">
          <div style='font-size:2.5rem;margin-bottom:0.5rem;'>📂</div>
          <div style='color:#6b7280;font-size:0.9rem;'>Tarik & lepas file CSV di sini, atau klik tombol di atas</div>
          <div style='color:#9ca3af;font-size:0.75rem;margin-top:0.4rem;'>Format: CSV dengan kolom Tanggal dan Kurs</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Contoh Format Data</div>', unsafe_allow_html=True)
        example_df = pd.DataFrame({
            "Tanggal": ["2024-01-01","2024-01-02","2024-01-03","2024-01-04","2024-01-05"],
            "Kurs": [15500, 15520, 15490, 15610, 15580]
        })
        st.dataframe(example_df, use_container_width=True, hide_index=True)
        example_csv = example_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️  Download Template CSV", data=example_csv,
                           file_name="template_kurs.csv", mime="text/csv")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;color:#9ca3af;font-size:0.72rem;margin-top:3rem;padding-top:1.5rem;
            border-top:1px solid #e5e7eb;'>
  Dashboard Volatilitas Kurs USD/IDR &nbsp;·&nbsp; LSTM / GRU / CNN &nbsp;·&nbsp;
  Forecast: Model Terbaik Otomatis ({BEST_MODEL}) &nbsp;·&nbsp; Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
