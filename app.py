import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
import re
import os
from groq import Groq

# Load .env file if python-dotenv is installed (local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Load API key — works for both local (.env) and Streamlit Cloud (st.secrets)
try:
    ENV_GROQ_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError, AttributeError):
    ENV_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DataLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — dark editorial aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0e1117;
    color: #e2e8f0;
}
.stApp { background-color: #0e1117; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2535;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] h2 { color: #e2e8f0 !important; font-size:1rem !important; }

/* ── Dashboard header ── */
.dash-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 0.5rem 0;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 1.2rem;
}
.dash-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.02em;
}
.dash-badge {
    background: #1e2535;
    border: 1px solid #2d3748;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
}

/* ── KPI Cards ── */
.kpi-card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.15s;
}
.kpi-card:hover { border-color: #3b82f6; transform: translateY(-2px); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #3b82f6);
}
.kpi-icon {
    font-size: 1.2rem;
    margin-bottom: 0.4rem;
    display: block;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
}
.kpi-label {
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}
.kpi-delta {
    font-size: 0.72rem;
    margin-top: 0.3rem;
    color: #10b981;
}

/* ── Chart Panel ── */
.chart-panel {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 1rem 1.2rem 0.5rem 1.2rem;
    margin-bottom: 1rem;
}
.panel-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.panel-title::before {
    content: '';
    width: 3px; height: 14px;
    background: #3b82f6;
    border-radius: 2px;
    display: inline-block;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2535;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161b27;
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
    border: 1px solid #1e2535;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px;
    color: #475569;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 6px 14px;
}
.stTabs [aria-selected="true"] {
    background: #1d4ed8 !important;
    color: #fff !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #1d4ed8;
    color: white;
    border: none;
    border-radius: 7px;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
    font-size: 0.85rem;
    transition: background 0.2s, transform 0.1s;
    width: 100%;
}
.stButton > button:hover { background: #2563eb; transform: translateY(-1px); }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input {
    background: #161b27 !important;
    border: 1px solid #1e2535 !important;
    color: #e2e8f0 !important;
    border-radius: 7px !important;
    font-size: 0.85rem !important;
}
.stTextInput input:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2535 !important;
    border-radius: 8px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #1e2535;
    border-radius: 10px;
    background: #161b27;
}

/* ── LinkedIn article ── */
.linkedin-article {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 2rem;
    font-size: 0.9rem;
    line-height: 1.8;
    color: #cbd5e1;
    white-space: pre-wrap;
    max-height: 600px;
    overflow-y: auto;
}

/* ── Status badges ── */
.badge-ok   { background:#052e16; color:#4ade80; border:1px solid #166534; border-radius:5px; padding:2px 8px; font-size:0.72rem; }
.badge-warn { background:#1c1917; color:#fb923c; border:1px solid #7c2d12; border-radius:5px; padding:2px 8px; font-size:0.72rem; }
.badge-info { background:#0f172a; color:#60a5fa; border:1px solid #1e3a5f; border-radius:5px; padding:2px 8px; font-size:0.72rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0e1117; }
::-webkit-scrollbar-thumb { background: #1e2535; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3b82f6; }

/* ── Alert overrides ── */
.stSuccess { background: #052e16 !important; border-left: 3px solid #16a34a !important; border-radius: 0 7px 7px 0 !important; }
.stWarning { background: #1c1917 !important; border-left: 3px solid #d97706 !important; border-radius: 0 7px 7px 0 !important; }
.stInfo    { background: #0f172a !important; border-left: 3px solid #3b82f6 !important; border-radius: 0 7px 7px 0 !important; }
.stError   { background: #1a0a0a !important; border-left: 3px solid #ef4444 !important; border-radius: 0 7px 7px 0 !important; }

/* ── Expander ── */
details { background: #161b27 !important; border: 1px solid #1e2535 !important; border-radius: 8px !important; }

/* ── Progress ── */
.stProgress > div > div { background: #3b82f6; border-radius: 4px; }

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background: #161b27 !important;
    border-color: #1e2535 !important;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        return pd.read_json(uploaded_file)
    elif name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    elif name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    else:
        raise ValueError(f"Unsupported file type: {name}")


def compute_summary(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    duplicates = df.duplicated().sum()

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "datetime_cols": dt_cols,
        "missing_total": int(missing.sum()),
        "missing_pct": float((missing.sum() / (len(df) * len(df.columns)) * 100).round(2)),
        "duplicates": int(duplicates),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "missing_by_col": missing[missing > 0].to_dict(),
        "missing_pct_by_col": missing_pct[missing_pct > 0].to_dict(),
    }
    return summary


def set_dark_plot_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#13131f",
        "axes.facecolor": "#1a1a2e",
        "axes.edgecolor": "#2a2a40",
        "axes.labelcolor": "#c9c7c0",
        "xtick.color": "#6b6880",
        "ytick.color": "#6b6880",
        "grid.color": "#1e1e2e",
        "text.color": "#c9c7c0",
        "font.family": "sans-serif",
    })


PALETTE = ["#7c6af7", "#c77dff", "#f72585", "#7bf1a8", "#f7a76c", "#48cae4", "#f9c74f"]


def generate_groq_response(client: Groq, prompt: str, system: str = "", model: str = "llama-3.3-70b-versatile") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=3000,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def build_data_context(df: pd.DataFrame, summary: dict) -> str:
    num_cols = summary["numeric_cols"]
    cat_cols = summary["categorical_cols"]

    ctx_parts = [
        f"Dataset: {summary['rows']} rows × {summary['columns']} columns",
        f"Numeric columns: {', '.join(num_cols) if num_cols else 'None'}",
        f"Categorical columns: {', '.join(cat_cols) if cat_cols else 'None'}",
        f"Missing values: {summary['missing_pct']}% overall",
        f"Duplicate rows: {summary['duplicates']}",
    ]

    if num_cols:
        desc = df[num_cols].describe().round(3).to_string()
        ctx_parts.append(f"\nNumeric statistics:\n{desc}")

    if cat_cols:
        for c in cat_cols[:5]:
            vc = df[c].value_counts().head(5)
            ctx_parts.append(f"\nTop values for '{c}': {vc.to_dict()}")

    return "\n".join(ctx_parts)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 DataLens AI")
    st.markdown("<p style='color:#475569; font-size:0.75rem; margin-top:-0.5rem'>Dataset Analyser + AI Insights</p>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls", "json", "parquet", "tsv"],
        help="CSV, Excel, JSON, Parquet, or TSV",
    )

    st.markdown("---")
    st.markdown("**Chart Settings**")
    chart_theme = st.selectbox("Color Theme", ["Violet/Purple", "Sunset", "Ocean", "Neon"])

    st.markdown("---")
    st.markdown("**LinkedIn Article**")
    article_tone = st.selectbox("Writing Tone", ["Professional", "Conversational", "Data-driven", "Storytelling"])
    article_audience = st.selectbox("Target Audience", ["General", "Data Scientists", "Business Leaders", "Recruiters"])

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + Groq")


# API key — loaded from .env or environment variable, never exposed to users
groq_key = ENV_GROQ_KEY

# ─────────────────────────────────────────────
# THEME PALETTE
# ─────────────────────────────────────────────

THEME_MAP = {
    "Violet/Purple": ["#7c6af7", "#c77dff", "#f72585", "#7bf1a8", "#f7a76c"],
    "Sunset":        ["#f72585", "#f7a76c", "#f9c74f", "#c77dff", "#7c6af7"],
    "Ocean":         ["#48cae4", "#0096c7", "#7bf1a8", "#023e8a", "#48cae4"],
    "Neon":          ["#39ff14", "#ff073a", "#00f5ff", "#ffff00", "#ff6fff"],
}
PALETTE = THEME_MAP.get(chart_theme, THEME_MAP["Violet/Purple"])


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert a #rrggbb hex string to an rgba() string Plotly accepts."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fig_to_html(fig) -> str:
    """Convert a Plotly figure to a self-contained HTML string — instant, no kaleido needed."""
    return fig.to_html(full_html=True, include_plotlyjs="cdn")


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────

st.markdown(f"""
<div class="dash-header">
    <div class="dash-title">📊 DataLens AI</div>
    <div style="display:flex; gap:8px; align-items:center;">
        <span class="dash-badge">v2.0</span>
        <span class="dash-badge">Groq · LLaMA 3.3</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────

if not uploaded_file:
    st.markdown('<div class="section-label">GET STARTED</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="kpi-card" style="--accent:#3b82f6">
            <span class="kpi-icon">📊</span>
            <div class="kpi-value" style="font-size:1rem; color:#60a5fa">Deep Analysis</div>
            <div class="kpi-label" style="margin-top:0.4rem; font-size:0.8rem; color:#475569; text-transform:none; letter-spacing:0">
                Distributions, correlations, outlier detection, missing value heatmaps
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card" style="--accent:#8b5cf6">
            <span class="kpi-icon">🤖</span>
            <div class="kpi-value" style="font-size:1rem; color:#a78bfa">Groq AI Insights</div>
            <div class="kpi-label" style="margin-top:0.4rem; font-size:0.8rem; color:#475569; text-transform:none; letter-spacing:0">
                LLaMA 3 powered summaries, recommendations, and chat with your data
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card" style="--accent:#10b981">
            <span class="kpi-icon">💼</span>
            <div class="kpi-value" style="font-size:1rem; color:#34d399">LinkedIn Article</div>
            <div class="kpi-label" style="margin-top:0.4rem; font-size:0.8rem; color:#475569; text-transform:none; letter-spacing:0">
                Auto-generate a compelling LinkedIn post with key insights from your data
            </div>
        </div>""", unsafe_allow_html=True)
    st.info("⬆️ Upload a dataset from the sidebar to get started")
    st.stop()


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

try:
    df = load_data(uploaded_file)
    summary = compute_summary(df)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

num_cols = summary["numeric_cols"]
cat_cols = summary["categorical_cols"]
dt_cols = summary["datetime_cols"]

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────

missing_badge = "badge-warn" if summary["missing_pct"] > 5 else "badge-ok"
dup_badge     = "badge-warn" if summary["duplicates"] > 0 else "badge-ok"

k1, k2, k3, k4, k5 = st.columns(5)
kpi_items = [
    (k1, "📁", summary["rows"],              "Total Rows",       "#3b82f6", f"{uploaded_file.name}"),
    (k2, "🧮", summary["columns"],           "Total Columns",    "#8b5cf6", f"{len(num_cols)} numeric · {len(cat_cols)} categorical"),
    (k3, "🔢", len(num_cols),                "Numeric Cols",     "#06b6d4", f"{len(cat_cols)} categorical"),
    (k4, "⚠️", f"{summary['missing_pct']}%", "Missing Values",   "#f59e0b", f"{summary['missing_total']} cells affected"),
    (k5, "🔁", summary["duplicates"],        "Duplicate Rows",   "#10b981", "clean" if summary["duplicates"]==0 else "needs review"),
]
for col, icon, val, label, accent, sub in kpi_items:
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="--accent:{accent}">
            <span class="kpi-icon">{icon}</span>
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-delta">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview", "📈 Distributions", "🔗 Correlations", "🤖 AI Insights", "💼 LinkedIn Article"
])


# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-label">DATA PREVIEW</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="chart-panel"><div class="panel-title">RAW DATA</div>', unsafe_allow_html=True)
        rows_to_show = st.slider("Rows to display", 5, min(100, len(df)), 10)
        st.dataframe(df.head(rows_to_show), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="chart-panel"><div class="panel-title">COLUMN INSPECTOR</div>', unsafe_allow_html=True)
        type_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.count().values,
            "Null %": (df.isnull().sum() / len(df) * 100).round(1).astype(str) + "%",
        })
        st.dataframe(type_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">STATISTICS & QUALITY</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="chart-panel"><div class="panel-title">NUMERIC STATISTICS</div>', unsafe_allow_html=True)
        if num_cols:
            st.dataframe(df[num_cols].describe().round(3), use_container_width=True)
        else:
            st.info("No numeric columns found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-panel"><div class="panel-title">MISSING VALUES</div>', unsafe_allow_html=True)
        if summary["missing_total"] > 0:
            missing_data = df.isnull().sum()[df.isnull().sum() > 0]
            fig = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                labels={"x": "Column", "y": "Missing Count"},
                color=missing_data.values,
                color_continuous_scale=["#3b82f6", "#f59e0b", "#ef4444"],
            )
            fig.update_layout(
                paper_bgcolor="#161b27", plot_bgcolor="#161b27",
                font_color="#94a3b8", showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0),
                height=260,
            )
            fig.update_xaxes(showgrid=False, tickfont=dict(size=11))
            fig.update_yaxes(gridcolor="#1e2535", zerolinecolor="#1e2535")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values — dataset is clean!")
        st.markdown('</div>', unsafe_allow_html=True)

    if cat_cols:
        st.markdown('<div class="section-label">CATEGORICAL BREAKDOWN</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-panel"><div class="panel-title">VALUE DISTRIBUTION</div>', unsafe_allow_html=True)
        selected_cat = st.selectbox("Select column", cat_cols)
        vc = df[selected_cat].value_counts().head(20)
        fig = px.bar(
            x=vc.values, y=vc.index, orientation="h",
            labels={"x": "Count", "y": selected_cat},
            color=vc.values,
            color_continuous_scale=["#1d4ed8", "#3b82f6", "#60a5fa"],
        )
        fig.update_layout(
            paper_bgcolor="#161b27", plot_bgcolor="#161b27",
            font_color="#94a3b8", showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(autorange="reversed"),
            height=max(300, len(vc) * 28),
        )
        fig.update_xaxes(gridcolor="#1e2535", zerolinecolor="#1e2535")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — DISTRIBUTIONS
# ══════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-label">DISTRIBUTION ANALYSIS</div>', unsafe_allow_html=True)

    if not num_cols:
        st.warning("No numeric columns available for distribution analysis.")
    else:
        sel_cols = st.multiselect(
            "Select numeric columns to plot",
            num_cols,
            default=num_cols[:min(4, len(num_cols))],
        )

        if sel_cols:
            # Histograms
            st.markdown('<div class="chart-panel"><div class="panel-title">HISTOGRAMS</div>', unsafe_allow_html=True)
            n = len(sel_cols)
            ncols = min(2, n)
            nrows = (n + ncols - 1) // ncols
            fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=sel_cols,
                                vertical_spacing=0.12, horizontal_spacing=0.08)
            for i, col in enumerate(sel_cols):
                r, c = divmod(i, ncols)
                fig.add_trace(
                    go.Histogram(x=df[col].dropna(), name=col, nbinsx=30,
                                 marker_color=PALETTE[i % len(PALETTE)],
                                 opacity=0.8, showlegend=False),
                    row=r+1, col=c+1,
                )
            fig.update_layout(
                paper_bgcolor="#161b27", plot_bgcolor="#161b27",
                font_color="#94a3b8", height=280*nrows,
                margin=dict(l=0, r=0, t=35, b=0),
            )
            for ax in fig.layout:
                if "xaxis" in ax or "yaxis" in ax:
                    fig.layout[ax].gridcolor = "#1e2535"
                    fig.layout[ax].zerolinecolor = "#1e2535"
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Box + Violin side by side
            bp_col, vp_col = st.columns(2)
            with bp_col:
                st.markdown('<div class="chart-panel"><div class="panel-title">BOX PLOTS · OUTLIER DETECTION</div>', unsafe_allow_html=True)
                fig2 = go.Figure()
                for i, col in enumerate(sel_cols):
                    fig2.add_trace(go.Box(
                        y=df[col].dropna(), name=col,
                        marker_color=PALETTE[i % len(PALETTE)],
                        line_color=PALETTE[i % len(PALETTE)],
                        fillcolor=hex_to_rgba(PALETTE[i % len(PALETTE)], 0.2),
                    ))
                fig2.update_layout(
                    paper_bgcolor="#161b27", plot_bgcolor="#161b27",
                    font_color="#94a3b8", showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0), height=320,
                )
                fig2.update_yaxes(gridcolor="#1e2535", zerolinecolor="#1e2535")
                fig2.update_xaxes(showgrid=False)
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with vp_col:
                st.markdown('<div class="chart-panel"><div class="panel-title">VIOLIN PLOTS · DISTRIBUTION SHAPE</div>', unsafe_allow_html=True)
                fig3 = go.Figure()
                for i, col in enumerate(sel_cols):
                    fig3.add_trace(go.Violin(
                        y=df[col].dropna(), name=col,
                        box_visible=True, meanline_visible=True,
                        fillcolor=hex_to_rgba(PALETTE[i % len(PALETTE)], 0.33),
                        line_color=PALETTE[i % len(PALETTE)],
                    ))
                fig3.update_layout(
                    paper_bgcolor="#161b27", plot_bgcolor="#161b27",
                    font_color="#94a3b8", showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0), height=320,
                )
                fig3.update_yaxes(gridcolor="#1e2535", zerolinecolor="#1e2535")
                fig3.update_xaxes(showgrid=False)
                st.plotly_chart(fig3, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Scatter plot
        if len(num_cols) >= 2:
            st.markdown('<div class="section-label">SCATTER EXPLORER</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-panel"><div class="panel-title">SCATTER PLOT WITH TRENDLINE</div>', unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)
            with sc1: x_col = st.selectbox("X axis", num_cols, index=0)
            with sc2: y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1))
            with sc3: color_col = st.selectbox("Color by", ["None"] + cat_cols + num_cols)

            color_arg = None if color_col == "None" else color_col
            fig_sc = px.scatter(df, x=x_col, y=y_col, color=color_arg,
                                opacity=0.65, color_discrete_sequence=PALETTE,
                                trendline="ols", trendline_color_override="#f59e0b")
            fig_sc.update_layout(
                paper_bgcolor="#161b27", plot_bgcolor="#161b27",
                font_color="#94a3b8",
                margin=dict(l=0, r=0, t=10, b=0), height=400,
            )
            fig_sc.update_xaxes(gridcolor="#1e2535", zerolinecolor="#1e2535")
            fig_sc.update_yaxes(gridcolor="#1e2535", zerolinecolor="#1e2535")
            st.plotly_chart(fig_sc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — CORRELATIONS
# ══════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-label">CORRELATION ANALYSIS</div>', unsafe_allow_html=True)

    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
    else:
        corr = df[num_cols].corr()

        heat_col, bar_col = st.columns([3, 2])

        with heat_col:
            st.markdown('<div class="chart-panel"><div class="panel-title">CORRELATION HEATMAP</div>', unsafe_allow_html=True)
            fig_heat = px.imshow(
                corr,
                color_continuous_scale=["#ef4444", "#161b27", "#3b82f6"],
                zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
            )
            fig_heat.update_layout(
                paper_bgcolor="#161b27", font_color="#94a3b8",
                margin=dict(l=0, r=0, t=10, b=0),
                height=max(380, len(num_cols) * 42),
            )
            fig_heat.update_coloraxes(colorbar_tickfont_color="#64748b")
            st.plotly_chart(fig_heat, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with bar_col:
            st.markdown('<div class="chart-panel"><div class="panel-title">TOP CORRELATED PAIRS</div>', unsafe_allow_html=True)
            corr_pairs = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack().reset_index()
            )
            corr_pairs.columns = ["Feature A", "Feature B", "Correlation"]
            corr_pairs["Abs"] = corr_pairs["Correlation"].abs()
            corr_pairs = corr_pairs.sort_values("Abs", ascending=False).drop(columns="Abs")
            corr_pairs["Correlation"] = corr_pairs["Correlation"].round(4)
            top_n = st.slider("Top N pairs", 5, min(30, len(corr_pairs)), 12)
            top_corr = corr_pairs.head(top_n)
            fig_bar = px.bar(
                top_corr,
                x="Correlation",
                y=[f"{a} × {b}" for a, b in zip(top_corr["Feature A"], top_corr["Feature B"])],
                orientation="h",
                color="Correlation",
                color_continuous_scale=["#ef4444", "#1e2535", "#3b82f6"],
                range_color=[-1, 1],
            )
            fig_bar.update_layout(
                paper_bgcolor="#161b27", plot_bgcolor="#161b27",
                font_color="#94a3b8", showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(autorange="reversed"),
                height=max(320, top_n * 28),
            )
            fig_bar.update_xaxes(gridcolor="#1e2535", range=[-1,1], zerolinecolor="#2a3040")
            fig_bar.update_yaxes(showgrid=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Scatter matrix
        if len(num_cols) <= 8:
            st.markdown('<div class="section-label">SCATTER MATRIX</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-panel"><div class="panel-title">PAIR PLOT</div>', unsafe_allow_html=True)
            show_cols = st.multiselect("Columns for pair plot", num_cols, default=num_cols[:min(4, len(num_cols))])
            if show_cols and len(show_cols) >= 2:
                color_dim = st.selectbox("Color by", ["None"] + cat_cols, key="pair_color")
                color_arg = None if color_dim == "None" else color_dim
                fig_pair = px.scatter_matrix(df, dimensions=show_cols, color=color_arg,
                                             color_discrete_sequence=PALETTE, opacity=0.6)
                fig_pair.update_traces(diagonal_visible=False)
                fig_pair.update_layout(
                    paper_bgcolor="#161b27", plot_bgcolor="#161b27",
                    font_color="#94a3b8", height=580,
                    margin=dict(l=0, r=0, t=20, b=0),
                )
                st.plotly_chart(fig_pair, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 4 — AI INSIGHTS
# ══════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-label">AI-POWERED INSIGHTS · GROQ LLAMA 3.3</div>', unsafe_allow_html=True)

    if not groq_key:
        st.error("⚙️ AI features are unavailable. The app administrator needs to set the `GROQ_API_KEY` environment variable on the server.")
    else:
        client = Groq(api_key=groq_key)
        data_ctx = build_data_context(df, summary)

        insight_type = st.selectbox(
            "Select insight type",
            [
                "📊 Executive Summary",
                "🔍 Data Quality Report",
                "📈 Key Trends & Patterns",
                "⚠️ Anomalies & Outliers",
                "💡 Business Recommendations",
                "🧠 Statistical Deep Dive",
            ]
        )

        custom_question = st.text_input(
            "Or ask a custom question about your data",
            placeholder="e.g. What is the most important factor driving sales?",
        )

        if st.button("🚀 Generate Insight"):
            with st.spinner("Groq LLaMA is analysing your data..."):
                if custom_question:
                    prompt = f"""You are an expert data analyst. Here is the dataset context:

{data_ctx}

Answer this question clearly and insightfully: {custom_question}

Be specific, reference actual column names and values from the data. Use bullet points where appropriate."""
                else:
                    type_map = {
                        "📊 Executive Summary": "Write a comprehensive executive summary of this dataset in 300–400 words. Cover what the data represents, key highlights, data quality, and what decisions could be informed by it.",
                        "🔍 Data Quality Report": "Write a thorough data quality report. Cover missing values, duplicates, data types, suspicious patterns, and recommendations for data cleaning.",
                        "📈 Key Trends & Patterns": "Identify and explain the top 5–7 key trends, patterns, and interesting findings in this dataset. Be specific with numbers.",
                        "⚠️ Anomalies & Outliers": "Identify potential anomalies, outliers, and unusual patterns in the data. Explain what they might mean and how to handle them.",
                        "💡 Business Recommendations": "Based on the data, provide 5–7 actionable business recommendations. Each should be concrete and backed by the data.",
                        "🧠 Statistical Deep Dive": "Provide a deep statistical analysis including distribution shapes, correlation insights, variance analysis, and statistical significance notes.",
                    }
                    prompt = f"""You are an expert data analyst. Here is the dataset context:

{data_ctx}

Task: {type_map[insight_type]}

Be analytical, specific, and insightful. Reference actual numbers and column names."""

                try:
                    response = generate_groq_response(
                        client, prompt,
                        system="You are an expert data analyst and statistician with 20 years of experience. Provide clear, actionable, data-backed insights."
                    )
                    st.markdown(
                        f"<div class='linkedin-article'>{response}</div>",
                        unsafe_allow_html=True,
                    )
                    st.download_button(
                        "⬇️ Download Insight",
                        response,
                        file_name="data_insight.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Groq API error: {e}")

        # Chat with data
        st.markdown("---")
        st.markdown("### 💬 Chat with Your Data")
        st.caption("Ask follow-up questions about your dataset")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            role_icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(f"**{role_icon} {msg['role'].title()}:** {msg['content']}")

        chat_input = st.text_input("Your question", key="chat_input", placeholder="Ask anything about the data...")
        if st.button("Send 📨") and chat_input:
            with st.spinner("Thinking..."):
                messages = [{"role": "system", "content": f"You are an expert data analyst. Dataset context:\n{data_ctx}"}]
                for m in st.session_state.chat_history[-6:]:
                    messages.append(m)
                messages.append({"role": "user", "content": chat_input})

                try:
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages,
                        max_tokens=800,
                    )
                    answer = resp.choices[0].message.content
                    st.session_state.chat_history.append({"role": "user", "content": chat_input})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
# TAB 5 — DASHBOARD + LINKEDIN ARTICLE
# ══════════════════════════════════════════════

with tab5:

    # ─── SECTION A: DASHBOARD (always visible) ───
    st.markdown('<div class="section-label">📊 DATA DASHBOARD</div>', unsafe_allow_html=True)

    if num_cols:
        import json as _json
        from plotly.utils import PlotlyJSONEncoder

        _tc   = len(df) * len(df.columns)
        _tm   = int(df.isnull().sum().sum())
        _td   = summary["duplicates"]
        _comp = round((1 - _tm / max(_tc, 1)) * 100, 1)
        _dpct = round(_td / max(len(df), 1) * 100, 1)
        _C    = "#ffffff"   # card background — white
        _B    = "#e2e8f0"   # border — light grey
        _T    = "#0f172a"   # primary text — near black
        _M    = "#475569"   # muted text — slate

        def _jdump(fig):
            return _json.dumps(fig, cls=PlotlyJSONEncoder)

        def _emb_js(fig, did):
            """Embed each chart as its own self-contained Plotly div."""
            return fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=did,
                config={"displayModeBar": False, "responsive": True},
                default_width="100%",
                default_height="240px",
            )

        # Gauge
        _fg = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=_comp,
            delta={"reference":90,"increasing":{"color":"#059669"},"decreasing":{"color":"#dc2626"}},
            number={"suffix":"%","font":{"size":32,"color":"#0f172a"}},
            gauge={"axis":{"range":[0,100],"tickcolor":_M,"tickfont":{"size":10,"color":_M}},
                   "bar":{"color":"#2563eb","thickness":0.25},"bgcolor":"#f8fafc","bordercolor":_B,
                   "steps":[{"range":[0,60],"color":"#fee2e2"},{"range":[60,80],"color":"#fef9c3"},
                             {"range":[80,100],"color":"#dcfce7"}],
                   "threshold":{"line":{"color":"#059669","width":3},"thickness":0.8,"value":90}},
        ))
        _fg.update_layout(paper_bgcolor=_C,font_color=_M,margin=dict(l=30,r=30,t=50,b=20),height=260,
            title=dict(text="DATA COMPLETENESS",font=dict(size=11,color=_M),x=0.5))

        # Donut
        _fd = go.Figure(go.Pie(
            labels=["Complete","Missing","Duplicates"],
            values=[max(0,_tc-_tm-_td),max(0,_tm),max(0,_td)],
            hole=0.65,marker_colors=["#2563eb","#f59e0b","#ef4444"],
            textinfo="percent+label",textfont_size=10,
        ))
        _fd.update_layout(paper_bgcolor=_C,font_color=_M,showlegend=False,
            margin=dict(l=10,r=10,t=50,b=20),height=260,
            title=dict(text="QUALITY BREAKDOWN",font=dict(size=11,color=_M),x=0.5),
            annotations=[dict(text=f"<b>{_comp}%</b><br>clean",x=0.5,y=0.5,
                              showarrow=False,font=dict(size=14,color=_T))])

        # Means
        _mv = df[num_cols[:8]].mean().sort_values(ascending=True)
        _fm = go.Figure(go.Bar(
            x=_mv.values, y=_mv.index, orientation="h",
            marker=dict(color=_mv.values,
                colorscale=[[0,"#93c5fd"],[0.5,"#3b82f6"],[1,"#1d4ed8"]],
                showscale=False),
            text=[f"{v:.1f}" for v in _mv.values],
            textposition="outside",
            textfont=dict(size=9, color="#0f172a")
        ))
        _fm.update_layout(paper_bgcolor=_C, plot_bgcolor="#f8fafc", font_color="#0f172a",
            margin=dict(l=100,r=60,t=50,b=20), height=260,
            title=dict(text="COLUMN MEANS", font=dict(size=11,color=_M), x=0.5),
            yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=9,color="#0f172a")),
            xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0", tickfont=dict(color="#0f172a")))

        # Light-theme safe palette
        _LP = ["#2563eb","#7c3aed","#059669","#d97706","#dc2626","#0891b2","#9333ea"]

        # Trend
        _ft = go.Figure()
        for _i,_col in enumerate(num_cols[:3]):
            _s = df[_col].dropna().reset_index(drop=True)
            _s = _s.iloc[::max(1,len(_s)//200)]
            _ft.add_trace(go.Scatter(
                x=_s.index, y=_s.values, mode="lines", name=_col,
                line=dict(color=_LP[_i%len(_LP)], width=2),
                fill="tozeroy",
                fillcolor=hex_to_rgba(_LP[_i%len(_LP)], 0.1)
            ))
        _ft.update_layout(
            paper_bgcolor=_C, plot_bgcolor="#f8fafc", font_color=_M,
            legend=dict(font=dict(size=9,color=_T), orientation="h", y=1.12, x=0),
            margin=dict(l=50,r=20,t=50,b=20), height=260, hovermode="x unified",
            title=dict(text="VALUE TRENDS", font=dict(size=11,color=_M), x=0.5)
        )
        _ft.update_xaxes(gridcolor="#e2e8f0", showticklabels=False, zerolinecolor="#e2e8f0")
        _ft.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0", tickfont=dict(color=_T))

        # Violin
        _fv = go.Figure()
        for _i,_col in enumerate(num_cols[:4]):
            _fv.add_trace(go.Violin(
                y=df[_col].dropna(), name=_col[:10],
                box_visible=True, meanline_visible=True,
                fillcolor=hex_to_rgba(_LP[_i%len(_LP)], 0.3),
                line_color=_LP[_i%len(_LP)],
                showlegend=False
            ))
        _fv.update_layout(
            paper_bgcolor=_C, plot_bgcolor="#f8fafc", font_color=_M,
            margin=dict(l=50,r=20,t=50,b=20), height=260,
            title=dict(text="DISTRIBUTION SPREAD", font=dict(size=11,color=_M), x=0.5)
        )
        _fv.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0", tickfont=dict(color=_T))
        _fv.update_xaxes(showgrid=False, tickfont=dict(size=9, color=_T))

        # Heatmap
        _cr = df[num_cols[:6]].corr()
        _fh = px.imshow(_cr,
            color_continuous_scale=["#dc2626","#f8fafc","#2563eb"],
            zmin=-1, zmax=1, text_auto=".1f", aspect="auto"
        )
        _fh.update_layout(
            paper_bgcolor=_C, font_color=_T,
            margin=dict(l=80,r=20,t=50,b=50), height=260,
            coloraxis_showscale=False,
            title=dict(text="CORRELATION HEATMAP", font=dict(size=11,color=_M), x=0.5)
        )
        _fh.update_xaxes(tickfont=dict(color=_T))
        _fh.update_yaxes(tickfont=dict(color=_T))

        # Missing %
        _mp = (df.isnull().sum()/len(df)*100).round(1)
        _fms = go.Figure(go.Bar(
            x=df.columns.tolist(), y=_mp.values,
            marker_color=["#ef4444" if v>20 else "#f59e0b" if v>5 else "#3b82f6" for v in _mp.values],
            text=[f"{v}%" for v in _mp.values],
            textposition="outside",
            textfont=dict(size=9, color="#0f172a")
        ))
        _fms.update_layout(paper_bgcolor=_C, plot_bgcolor="#f8fafc", font_color="#0f172a",
            margin=dict(l=20,r=20,t=50,b=100), height=280,
            title=dict(text="MISSING VALUES %", font=dict(size=11,color=_M), x=0.5),
            xaxis=dict(showgrid=False, tickangle=-45, tickfont=dict(size=8,color="#0f172a"), automargin=True),
            yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0",
                       range=[0,max(_mp.max()+20,20)], tickfont=dict(color="#0f172a")))

        if cat_cols:
            _vc = df[cat_cols[0]].value_counts().head(8)
            _fc = go.Figure(go.Bar(
                x=_vc.index.astype(str), y=_vc.values,
                marker_color=["#2563eb","#7c3aed","#059669","#d97706","#dc2626","#0891b2","#9333ea","#2563eb"][:len(_vc)],
                text=_vc.values, textposition="outside",
                textfont=dict(size=10, color="#0f172a")
            ))
            _fc.update_layout(paper_bgcolor=_C, plot_bgcolor="#f8fafc", font_color="#0f172a",
                margin=dict(l=20,r=20,t=50,b=100), height=280,
                title=dict(text=f"TOP: {cat_cols[0]}", font=dict(size=11,color=_M), x=0.5),
                xaxis=dict(showgrid=False, tickangle=-45, tickfont=dict(size=9,color="#0f172a"), automargin=True),
                yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0", tickfont=dict(color="#0f172a")))
        else:
            _pr = (_cr.abs().where(np.triu(np.ones(_cr.shape),k=1).astype(bool))
                   .stack().sort_values(ascending=False).head(8))
            _fc = go.Figure(go.Bar(
                x=[f"{a[:5]}x{b[:5]}" for a,b in _pr.index], y=_pr.values,
                marker_color=["#2563eb" if v>0.7 else "#d97706" if v>0.4 else "#94a3b8" for v in _pr.values],
                text=[f"{v:.2f}" for v in _pr.values],textposition="outside",
                textfont=dict(size=9,color="#0f172a")))
            _fc.update_layout(paper_bgcolor=_C,plot_bgcolor=_C,font_color="#0f172a",
                margin=dict(l=20,r=20,t=50,b=80),height=280,
                title=dict(text="TOP CORR PAIRS",font=dict(size=11,color=_M),x=0.5),
                xaxis=dict(showgrid=False,tickangle=-30,tickfont=dict(size=9,color="#0f172a")),
                yaxis=dict(gridcolor='#e2e8f0',range=[0,1.2],tickfont=dict(color="#0f172a")))

        # ── KPI Banner HTML ──
        _kpi_html = "".join([
            f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;'
            f'padding:14px;text-align:center;border-top:3px solid {c};">'
            f'<div style="font-size:1.6rem;font-weight:700;color:{c};font-family:monospace">{v}</div>'
            f'<div style="font-size:0.62rem;color:#475569;margin-top:4px;text-transform:uppercase;'
            f'letter-spacing:.08em">{lbl}</div></div>'
            for v,lbl,c in [
                (f"{len(df):,}",    "Total Rows",   "#2563eb"),
                (len(df.columns),   "Columns",      "#7c3aed"),
                (f"{_comp}%",       "Completeness", "#059669"),
                (len(num_cols),     "Numeric Cols", "#d97706"),
                (f"{_dpct}%",       "Duplicates",   "#dc2626"),
            ]
        ])

        # ── Streamlit KPI banner ──
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:12px;">'
            f'{_kpi_html}</div>',
            unsafe_allow_html=True
        )

        # ── Streamlit chart rows — instant display ──
        r1a, r1b, r1c = st.columns(3)
        with r1a: st.plotly_chart(_fg,  use_container_width=True, key="db_gauge")
        with r1b: st.plotly_chart(_fd,  use_container_width=True, key="db_donut")
        with r1c: st.plotly_chart(_fm,  use_container_width=True, key="db_means")

        r2a, r2b, r2c = st.columns(3)
        with r2a: st.plotly_chart(_ft,  use_container_width=True, key="db_trend")
        with r2b: st.plotly_chart(_fv,  use_container_width=True, key="db_violin")
        with r2c: st.plotly_chart(_fh,  use_container_width=True, key="db_hm")

        r3a, r3b = st.columns(2)
        with r3a: st.plotly_chart(_fms, use_container_width=True, key="db_miss")
        with r3b: st.plotly_chart(_fc,  use_container_width=True, key="db_cat")

        # ── Download ──
        _dsn = uploaded_file.name.rsplit(".",1)[0]
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        if "dashboard_html" not in st.session_state:
            st.session_state["dashboard_html"] = None

        if st.button("🛠️ Prepare Dashboard for Download", use_container_width=True):
            _figs = [_fg, _fd, _fm, _ft, _fv, _fh, _fms, _fc]
            _ids  = ["p1","p2","p3","p4","p5","p6","p7","p8"]

            # First chart includes plotly JS, rest share it
            _divs = []
            for _idx, (fig, did) in enumerate(zip(_figs, _ids)):
                _incl = "cdn" if _idx == 0 else False
                _ch = fig.to_html(
                    full_html=False,
                    include_plotlyjs=_incl,
                    div_id=did,
                    config={"displayModeBar": False, "responsive": True},
                    default_width="100%",
                    default_height="250px",
                )
                _divs.append(
                    f'<div style="background:#ffffff;border:1px solid #e2e8f0;'
                    f'border-radius:8px;padding:6px;overflow:hidden;">{_ch}</div>'
                )

            _css = (
                "*{box-sizing:border-box;margin:0;padding:0;}"
                "body{background:#ffffff;font-family:Arial,sans-serif;"
                "     padding:14px 18px;color:#0f172a;}"
                ".ttl{font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:2px;}"
                ".sub{font-size:0.68rem;color:#64748b;margin-bottom:10px;}"
                ".kpi{display:grid;grid-template-columns:repeat(5,1fr);"
                "     gap:8px;margin-bottom:10px;}"
                ".r3{display:grid;grid-template-columns:repeat(3,1fr);"
                "    gap:8px;margin-bottom:8px;}"
                ".r2{display:grid;grid-template-columns:repeat(2,1fr);"
                "    gap:8px;margin-bottom:8px;}"
                ".ft{text-align:center;color:#94a3b8;font-size:0.58rem;margin-top:6px;}"
                "@media print{"
                "  body{-webkit-print-color-adjust:exact !important;"
                "       print-color-adjust:exact !important;"
                "       background:#ffffff !important;}"
                "  .kpi,.r3,.r2{break-inside:avoid;}"
                "  @page{size:A3 landscape;margin:8mm;}"
                "}"
            )

            _print_js = """
<script>
function savePDF() {
    // Relayout all charts so they render fully
    var plots = document.querySelectorAll('.js-plotly-plot');
    plots.forEach(function(p) { try{ Plotly.relayout(p, {}); }catch(e){} });
    // Wait 1.5s for render then print
    setTimeout(function(){ window.print(); }, 1500);
}
// Auto-relayout on page load too
window.addEventListener('load', function(){
    setTimeout(function(){
        var plots = document.querySelectorAll('.js-plotly-plot');
        plots.forEach(function(p){ try{ Plotly.relayout(p, {}); }catch(e){} });
    }, 500);
});
</script>"""

            st.session_state["dashboard_html"] = (
                "<!DOCTYPE html><html><head>"
                "<meta charset='utf-8'>"
                f"<title>{_dsn} Dashboard</title>"
                f"<style>{_css}</style>"
                "</head><body>"
                "<div style='display:flex;justify-content:space-between;"
                "align-items:center;margin-bottom:4px;'>"
                f"<div class='ttl'>&#128202; {_dsn} &mdash; Data Dashboard</div>"
                "<button onclick='savePDF()' style='background:#2563eb;color:white;"
                "border:none;border-radius:6px;padding:5px 14px;cursor:pointer;"
                "font-size:0.78rem;font-weight:600;display:block;'>&#128438; Save as PDF</button>"
                "</div>"
                f"<div class='sub'>DataLens AI &middot; "
                f"{pd.Timestamp.now().strftime('%d %b %Y %H:%M')} &middot; "
                f"Open in Chrome, wait 3 sec, then click Save as PDF</div>"
                f"<div class='kpi'>{_kpi_html}</div>"
                f"<div class='r3'>{_divs[0]}{_divs[1]}{_divs[2]}</div>"
                f"<div class='r3'>{_divs[3]}{_divs[4]}{_divs[5]}</div>"
                f"<div class='r2'>{_divs[6]}{_divs[7]}</div>"
                "<div class='ft'>DataLens AI &middot; All charts are interactive</div>"
                f"{_print_js}"
                "</body></html>"
            )
            st.success("✅ Ready! Download → open in Chrome → wait 3 sec → click **Save as PDF**")

        if st.session_state.get("dashboard_html"):
            st.download_button(
                label="⬇️  Download Full Dashboard  —  All 8 Charts + KPIs in 1 File",
                data=st.session_state["dashboard_html"],
                file_name=f"{_dsn}_dashboard.html",
                mime="text/html",
                use_container_width=True,
            )
            st.caption("Open in Chrome → **Ctrl + P** → **Save as PDF** (A3 Landscape) ✅")

    else:
        st.info("Upload a dataset with numeric columns to generate the dashboard.")

    st.markdown("---")

    # ─── SECTION B: LINKEDIN ARTICLE GENERATOR ───
    st.markdown('<div class="section-label">✍️ LINKEDIN ARTICLE GENERATOR</div>', unsafe_allow_html=True)

    if not groq_key:
        st.error("⚙️ AI features unavailable. Set the GROQ_API_KEY environment variable on the server.")
    else:
        client = Groq(api_key=groq_key)

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            dataset_name = st.text_input("Dataset / Topic Name", placeholder="e.g. Global Sales Data 2024")
            your_name    = st.text_input("Your Name (optional)", placeholder="e.g. Jane Doe")
        with col_opt2:
            industry     = st.text_input("Industry / Domain", placeholder="e.g. Retail, Finance, Healthcare")
            focus_angle  = st.text_area("Key message / angle (optional)",
                placeholder="e.g. Focus on how customer age affects purchase frequency", height=80)

        include_emojis   = st.checkbox("Include emojis",          value=True)
        include_cta      = st.checkbox("Include Call-to-Action",   value=True)
        include_hashtags = st.checkbox("Include hashtags",         value=True)

        if st.button("✍️ Generate LinkedIn Article"):
            data_ctx     = build_data_context(df, summary)
            emoji_note   = "Use emojis strategically." if include_emojis   else "Do not use emojis."
            cta_note     = "End with a strong call-to-action."              if include_cta      else ""
            hashtag_note = "Add 8-12 relevant hashtags at the end."         if include_hashtags else ""

            prompt = f"""Write a compelling LinkedIn article about a data analysis I performed.
DATASET CONTEXT:
{data_ctx}
ARTICLE DETAILS:
- Dataset/Topic: {dataset_name or "My Dataset"}
- Industry: {industry or "General"}
- Author: {your_name or "a data professional"}
- Tone: {article_tone}  Audience: {article_audience}
- Angle: {focus_angle or "Surface the most interesting insights"}
STRUCTURE: Hook → Intro → 3 Key Findings (with numbers) → Implications → Recommendations
{cta_note} {hashtag_note}
STYLE: {emoji_note} Short paragraphs. 600-900 words. LinkedIn line breaks.
Write the full article now:"""

            with st.spinner("Generating article with Groq LLaMA..."):
                try:
                    article = generate_groq_response(client, prompt,
                        system="You are a top LinkedIn thought leader and data storyteller.",
                        model="llama-3.3-70b-versatile")
                    wc = len(article.split())
                    st.markdown('<div class="section-label">GENERATED ARTICLE</div>', unsafe_allow_html=True)
                    art_col, stat_col = st.columns([3,1])
                    with art_col:
                        st.markdown('<div class="chart-panel"><div class="panel-title">LINKEDIN POST PREVIEW</div>', unsafe_allow_html=True)
                        st.markdown(f"<div class='linkedin-article'>{article}</div>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with stat_col:
                        st.markdown(f"""
                        <div class="kpi-card" style="--accent:#3b82f6;margin-bottom:.7rem">
                            <span class="kpi-icon">&#128221;</span>
                            <div class="kpi-value">{wc}</div>
                            <div class="kpi-label">Words</div>
                            <div class="kpi-delta">~{max(1,wc//200)} min read</div>
                        </div>""", unsafe_allow_html=True)
                        dl1, dl2 = st.columns(2)
                        with dl1:
                            st.download_button("⬇️ .txt", article,
                                file_name="linkedin_article.txt", mime="text/plain",
                                use_container_width=True)
                        with dl2:
                            st.download_button("⬇️ .md",
                                f"# {dataset_name or 'Data Analysis'}\n\n{article}",
                                file_name="linkedin_article.md", mime="text/markdown",
                                use_container_width=True)
                except Exception as e:
                    st.error(f"Groq API error: {e}")

        with st.expander("💡 Tips for a viral LinkedIn post"):
            st.markdown("""
- **Post time**: Tuesday-Thursday, 8-10am or 5-6pm in your audience's timezone
- **First line** is critical — LinkedIn cuts off after 2-3 lines. Hook them immediately.
- **Images** increase engagement 2-3x. Use the dashboard download above!
- **Tag colleagues** or companies mentioned in your analysis
- **Engage early**: Reply to every comment in the first hour for maximum reach
- **Avoid links in body**: Put external links in the first comment instead
""")