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

# Load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #e8e6e0;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d18 !important;
    border-right: 1px solid #1e1e2e;
}
section[data-testid="stSidebar"] * { color: #c9c7c0 !important; }

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

h1 { font-size: 2.8rem !important; font-weight: 800 !important;
     background: linear-gradient(90deg, #7c6af7, #c77dff, #f72585);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

h2 { font-size: 1.6rem !important; font-weight: 700 !important; color: #c9c7c0 !important; }
h3 { font-size: 1.15rem !important; font-weight: 600 !important; color: #a09db5 !important; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #13131f, #1a1a2e);
    border: 1px solid #2a2a40;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #7c6af7; }

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #c77dff;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b6880;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* Section divider */
.section-header {
    display: flex; align-items: center; gap: 12px;
    margin: 2rem 0 1rem 0;
}
.section-tag {
    background: linear-gradient(90deg, #7c6af7, #f72585);
    border-radius: 4px; width: 4px; height: 28px;
    flex-shrink: 0;
}

/* LinkedIn article */
.linkedin-article {
    background: linear-gradient(135deg, #13131f, #1a1a2e);
    border: 1px solid #2a2a40;
    border-radius: 16px;
    padding: 2.5rem;
    font-family: 'Inter', sans-serif;
    line-height: 1.8;
    color: #d4d2cc;
    white-space: pre-wrap;
    max-height: 600px;
    overflow-y: auto;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #13131f;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e1e2e;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #6b6880;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #7c6af7, #c77dff) !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #7c6af7, #c77dff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    transition: opacity 0.2s, transform 0.1s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #2a2a40;
    border-radius: 14px;
    background: #13131f;
    padding: 1rem;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    overflow: hidden;
}

/* Input */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: #13131f !important;
    border: 1px solid #2a2a40 !important;
    color: #e8e6e0 !important;
    border-radius: 8px !important;
}
.stTextInput input:focus { border-color: #7c6af7 !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #13131f;
    border-radius: 8px;
    border: 1px solid #1e1e2e;
}

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #7c6af7, #f72585); border-radius: 4px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a40; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c6af7; }

/* Info boxes */
.stInfo { background: #13131f; border-left: 3px solid #7c6af7; border-radius: 0 8px 8px 0; }
.stWarning { background: #13131f; border-left: 3px solid #f7a76c; border-radius: 0 8px 8px 0; }
.stSuccess { background: #13131f; border-left: 3px solid #6af7a7; border-radius: 0 8px 8px 0; }
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
    st.markdown("## 🔍 DataLens AI")
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


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────

st.markdown("# DataLens AI")
st.markdown(
    "<p style='color:#6b6880; font-size:1.05rem; margin-top:-0.5rem;'>"
    "Intelligent dataset analysis · AI-powered insights · LinkedIn-ready storytelling"
    "</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────

if not uploaded_file:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem">📊</div>
            <div class="metric-label" style="margin-top:0.5rem; font-size:0.9rem; color:#c9c7c0">Deep Analysis</div>
            <p style="color:#6b6880; font-size:0.82rem; margin-top:0.3rem">
                Distribution plots, correlations, outlier detection, missing value heatmaps
            </p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem">🤖</div>
            <div class="metric-label" style="margin-top:0.5rem; font-size:0.9rem; color:#c9c7c0">Groq AI Insights</div>
            <p style="color:#6b6880; font-size:0.82rem; margin-top:0.3rem">
                LLaMA 3 powered narrative summaries, recommendations, and pattern detection
            </p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:2rem">💼</div>
            <div class="metric-label" style="margin-top:0.5rem; font-size:0.9rem; color:#c9c7c0">LinkedIn Article</div>
            <p style="color:#6b6880; font-size:0.82rem; margin-top:0.3rem">
                Auto-generate a compelling LinkedIn post with key insights from your data
            </p>
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

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

kpi_data = [
    (k1, summary["rows"], "Rows"),
    (k2, summary["columns"], "Columns"),
    (k3, len(num_cols), "Numeric"),
    (k4, f"{summary['missing_pct']}%", "Missing"),
    (k5, summary["duplicates"], "Duplicates"),
]
for col, val, label in kpi_data:
    with col:
        st.markdown(f"""
        <div class="metric-card" style="text-align:center">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

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
    st.markdown("## Dataset Overview")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Raw Data Preview")
        rows_to_show = st.slider("Rows to display", 5, min(100, len(df)), 10)
        st.dataframe(df.head(rows_to_show), use_container_width=True)

    with col_right:
        st.markdown("### Column Types")
        type_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.count().values,
            "Null %": (df.isnull().sum() / len(df) * 100).round(1).astype(str) + "%",
        })
        st.dataframe(type_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Numeric Statistics")
        if num_cols:
            st.dataframe(df[num_cols].describe().round(3), use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with col_b:
        st.markdown("### Missing Value Heatmap")
        if summary["missing_total"] > 0:
            missing_data = df.isnull().sum()[df.isnull().sum() > 0]
            fig = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                labels={"x": "Column", "y": "Missing Count"},
                color=missing_data.values,
                color_continuous_scale=["#7c6af7", "#f72585"],
            )
            fig.update_layout(
                paper_bgcolor="#13131f", plot_bgcolor="#1a1a2e",
                font_color="#c9c7c0", showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor="#1e1e2e")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values — clean dataset!")

    # Categorical value counts
    if cat_cols:
        st.markdown("### Categorical Column Insights")
        selected_cat = st.selectbox("Select column", cat_cols)
        vc = df[selected_cat].value_counts().head(20)
        fig = px.bar(
            x=vc.values, y=vc.index, orientation="h",
            labels={"x": "Count", "y": selected_cat},
            color=vc.values,
            color_continuous_scale=PALETTE[:2] + [PALETTE[2]],
        )
        fig.update_layout(
            paper_bgcolor="#13131f", plot_bgcolor="#1a1a2e",
            font_color="#c9c7c0", showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        fig.update_xaxes(gridcolor="#1e1e2e")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — DISTRIBUTIONS
# ══════════════════════════════════════════════

with tab2:
    st.markdown("## Distribution Analysis")

    if not num_cols:
        st.warning("No numeric columns available for distribution analysis.")
    else:
        sel_cols = st.multiselect(
            "Select numeric columns to plot",
            num_cols,
            default=num_cols[:min(4, len(num_cols))],
        )

        if sel_cols:
            # Histogram + KDE grid
            st.markdown("### Histograms with KDE")
            n = len(sel_cols)
            ncols = min(2, n)
            nrows = (n + ncols - 1) // ncols

            fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=sel_cols)
            for i, col in enumerate(sel_cols):
                r, c = divmod(i, ncols)
                data = df[col].dropna()
                fig.add_trace(
                    go.Histogram(
                        x=data, name=col, nbinsx=30,
                        marker_color=PALETTE[i % len(PALETTE)],
                        opacity=0.75,
                        showlegend=False,
                    ),
                    row=r + 1, col=c + 1,
                )
            fig.update_layout(
                paper_bgcolor="#13131f", plot_bgcolor="#1a1a2e",
                font_color="#c9c7c0",
                height=280 * nrows,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            for ax in fig.layout:
                if "xaxis" in ax or "yaxis" in ax:
                    fig.layout[ax].gridcolor = "#1e1e2e"
                    fig.layout[ax].zerolinecolor = "#2a2a40"
            st.plotly_chart(fig, use_container_width=True)

            # Box plots
            st.markdown("### Box Plots (Outlier Detection)")
            fig2 = go.Figure()
            for i, col in enumerate(sel_cols):
                fig2.add_trace(go.Box(
                    y=df[col].dropna(),
                    name=col,
                    marker_color=PALETTE[i % len(PALETTE)],
                    line_color=PALETTE[i % len(PALETTE)],
                    fillcolor=hex_to_rgba(PALETTE[i % len(PALETTE)], 0.2),
                ))
            fig2.update_layout(
                paper_bgcolor="#13131f", plot_bgcolor="#1a1a2e",
                font_color="#c9c7c0",
                showlegend=True,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            fig2.update_yaxes(gridcolor="#1e1e2e", zerolinecolor="#2a2a40")
            st.plotly_chart(fig2, use_container_width=True)

            # Violin plots
            if len(sel_cols) >= 1:
                st.markdown("### Violin Plots (Distribution Shape)")
                fig3 = go.Figure()
                for i, col in enumerate(sel_cols):
                    fig3.add_trace(go.Violin(
                        y=df[col].dropna(),
                        name=col,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=hex_to_rgba(PALETTE[i % len(PALETTE)], 0.33),
                        line_color=PALETTE[i % len(PALETTE)],
                    ))
                fig3.update_layout(
                    paper_bgcolor="#13131f", plot_bgcolor="#1a1a2e",
                    font_color="#c9c7c0",
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                fig3.update_yaxes(gridcolor="#1e1e2e")
                st.plotly_chart(fig3, use_container_width=True)

        # Scatter plot
        st.markdown("### Scatter Plot Explorer")
        if len(num_cols) >= 2:
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                x_col = st.selectbox("X axis", num_cols, index=0)
            with sc2:
                y_col = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1))
            with sc3:
                color_col = st.selectbox("Color by (optional)", ["None"] + cat_cols + num_cols)

            color_arg = None if color_col == "None" else color_col
            fig_sc = px.scatter(
                df, x=x_col, y=y_col, color=color_arg,
                opacity=0.65,
                color_discrete_sequence=PALETTE,
                trendline="ols",
                trendline_color_override="#f72585",
            )
            fig_sc.update_layout(
                paper_bgcolor="#13131f", plot_bgcolor="#1a1a2e",
                font_color="#c9c7c0",
                margin=dict(l=10, r=10, t=20, b=10),
            )
            fig_sc.update_xaxes(gridcolor="#1e1e2e", zerolinecolor="#2a2a40")
            fig_sc.update_yaxes(gridcolor="#1e1e2e", zerolinecolor="#2a2a40")
            st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — CORRELATIONS
# ══════════════════════════════════════════════

with tab3:
    st.markdown("## Correlation Analysis")

    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
    else:
        corr = df[num_cols].corr()

        # Heatmap
        st.markdown("### Correlation Heatmap")
        fig_heat = px.imshow(
            corr,
            color_continuous_scale=["#f72585", "#13131f", "#7c6af7"],
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig_heat.update_layout(
            paper_bgcolor="#13131f",
            font_color="#c9c7c0",
            margin=dict(l=10, r=10, t=20, b=10),
            height=max(400, len(num_cols) * 45),
        )
        fig_heat.update_coloraxes(colorbar_tickfont_color="#c9c7c0")
        st.plotly_chart(fig_heat, use_container_width=True)

        # Top correlations
        st.markdown("### Top Correlations")
        corr_pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        corr_pairs.columns = ["Feature A", "Feature B", "Correlation"]
        corr_pairs["Abs"] = corr_pairs["Correlation"].abs()
        corr_pairs = corr_pairs.sort_values("Abs", ascending=False).drop(columns="Abs")
        corr_pairs["Correlation"] = corr_pairs["Correlation"].round(4)

        top_n = st.slider("Show top N pairs", 5, min(50, len(corr_pairs)), 15)
        top_corr = corr_pairs.head(top_n)

        fig_bar = px.bar(
            top_corr,
            x="Correlation",
            y=[f"{a} × {b}" for a, b in zip(top_corr["Feature A"], top_corr["Feature B"])],
            orientation="h",
            color="Correlation",
            color_continuous_scale=["#f72585", "#13131f", "#7c6af7"],
            range_color=[-1, 1],
        )
        fig_bar.update_layout(
            paper_bgcolor="#13131f", plot_bgcolor="#1a1a2e",
            font_color="#c9c7c0",
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis=dict(autorange="reversed"),
            height=max(300, top_n * 30),
        )
        fig_bar.update_xaxes(gridcolor="#1e1e2e", range=[-1, 1], zerolinecolor="#2a2a40")
        fig_bar.update_yaxes(showgrid=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Pair plot (Plotly scatter matrix)
        if len(num_cols) <= 8:
            st.markdown("### Scatter Matrix (Pair Plot)")
            show_cols = st.multiselect("Columns for pair plot", num_cols, default=num_cols[:min(4, len(num_cols))])
            if show_cols and len(show_cols) >= 2:
                color_dim = st.selectbox("Color dimension", ["None"] + cat_cols, key="pair_color")
                color_arg = None if color_dim == "None" else color_dim
                fig_pair = px.scatter_matrix(
                    df, dimensions=show_cols, color=color_arg,
                    color_discrete_sequence=PALETTE,
                    opacity=0.6,
                )
                fig_pair.update_traces(diagonal_visible=False)
                fig_pair.update_layout(
                    paper_bgcolor="#13131f", plot_bgcolor="#1a1a2e",
                    font_color="#c9c7c0",
                    height=600,
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_pair, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — AI INSIGHTS
# ══════════════════════════════════════════════

with tab4:
    st.markdown("## AI-Powered Data Insights")

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
# TAB 5 — LINKEDIN ARTICLE
# ══════════════════════════════════════════════

with tab5:
    st.markdown("## LinkedIn Article Generator")
    st.markdown("Generate a publication-ready LinkedIn article based on your dataset analysis.")

    if not groq_key:
        st.error("⚙️ AI features are unavailable. The app administrator needs to set the `GROQ_API_KEY` environment variable on the server.")
    else:
        client = Groq(api_key=groq_key)

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            dataset_name = st.text_input("Dataset / Topic Name", placeholder="e.g. Global Sales Data 2024")
            your_name = st.text_input("Your Name (optional)", placeholder="e.g. Jane Doe")
        with col_opt2:
            industry = st.text_input("Industry / Domain", placeholder="e.g. Retail, Finance, Healthcare")
            focus_angle = st.text_area(
                "Key message / angle (optional)",
                placeholder="e.g. Focus on how customer age affects purchase frequency",
                height=80,
            )

        include_emojis = st.checkbox("Include emojis", value=True)
        include_cta = st.checkbox("Include Call-to-Action", value=True)
        include_hashtags = st.checkbox("Include hashtags", value=True)

        if st.button("✍️ Generate LinkedIn Article"):
            data_ctx = build_data_context(df, summary)

            emoji_note = "Use emojis strategically to make it engaging." if include_emojis else "Do not use emojis."
            cta_note = "End with a strong call-to-action asking readers to comment or share." if include_cta else ""
            hashtag_note = "Add 8–12 relevant hashtags at the end." if include_hashtags else ""

            prompt = f"""Write a compelling LinkedIn article about a data analysis I performed.

DATASET CONTEXT:
{data_ctx}

ARTICLE DETAILS:
- Dataset/Topic: {dataset_name or "My Dataset"}
- Industry: {industry or "General"}
- Author: {your_name or "a data professional"}
- Writing tone: {article_tone}
- Target audience: {article_audience}
- Key angle: {focus_angle or "Surface the most interesting insights"}

ARTICLE STRUCTURE:
1. Hook (attention-grabbing opening line or question)
2. Brief intro about why this data matters
3. Key Finding #1 (with specific numbers)
4. Key Finding #2 (with specific numbers)
5. Key Finding #3 (with specific numbers)
6. What this means / implications
7. Lessons learned / recommendations
{cta_note}
{hashtag_note}

STYLE RULES:
- Tone: {article_tone}
- {emoji_note}
- Short paragraphs (2–3 sentences max)
- Use line breaks liberally for LinkedIn readability
- Make it feel personal and insightful, not robotic
- Target length: 600–900 words

Write the full article now:"""

            with st.spinner("Crafting your LinkedIn article with Groq LLaMA..."):
                try:
                    article = generate_groq_response(
                        client, prompt,
                        system="You are a top LinkedIn thought leader and data storyteller. You write viral posts that combine data insights with engaging narratives.",
                        model="llama-3.3-70b-versatile",
                    )

                    st.markdown("---")
                    st.markdown("### 📄 Your LinkedIn Article")

                    # Word count
                    wc = len(article.split())
                    st.caption(f"~{wc} words · Estimated read time: {max(1, wc // 200)} min")

                    st.markdown(
                        f"<div class='linkedin-article'>{article}</div>",
                        unsafe_allow_html=True,
                    )

                    # Actions
                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button(
                            "⬇️ Download Article (.txt)",
                            article,
                            file_name="linkedin_article.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
                    with dl2:
                        md_article = f"# {dataset_name or 'Data Analysis'} — LinkedIn Article\n\n{article}"
                        st.download_button(
                            "⬇️ Download as Markdown",
                            md_article,
                            file_name="linkedin_article.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )

                    # Visual summary to accompany article
                    st.markdown("---")
                    st.markdown("### 📊 Chart to Share with Article")
                    st.caption("Download this chart to include as a visual in your LinkedIn post.")

                    if num_cols:
                        fig_share = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=["Top Numeric Distributions", "Data Quality Overview"],
                            specs=[[{"type": "xy"}, {"type": "domain"}]],
                        )
                        cols_to_show = num_cols[:3]
                        for i, col in enumerate(cols_to_show):
                            fig_share.add_trace(
                                go.Box(
                                    y=df[col].dropna(), name=col,
                                    marker_color=PALETTE[i % len(PALETTE)],
                                    line_color=PALETTE[i % len(PALETTE)],
                                ),
                                row=1, col=1,
                            )

                        # Missing values pie
                        miss = {c: df[c].isnull().sum() for c in df.columns}
                        total_missing = sum(miss.values())
                        total_present = len(df) * len(df.columns) - total_missing
                        fig_share.add_trace(
                            go.Pie(
                                labels=["Complete", "Missing"],
                                values=[total_present, max(1, total_missing)],
                                marker_colors=[PALETTE[0], PALETTE[2]],
                                hole=0.4,
                            ),
                            row=1, col=2,
                        )

                        fig_share.update_layout(
                            paper_bgcolor="#13131f",
                            plot_bgcolor="#1a1a2e",
                            font_color="#c9c7c0",
                            height=350,
                            title_text=f"📊 {dataset_name or 'Dataset'} — Quick Analysis",
                            title_font_color="#c77dff",
                            margin=dict(l=20, r=20, t=60, b=20),
                            showlegend=False,
                        )
                        for ax in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
                            if hasattr(fig_share.layout, ax):
                                getattr(fig_share.layout, ax).update(gridcolor="#1e1e2e", zerolinecolor="#2a2a40")

                        st.plotly_chart(fig_share, use_container_width=True)

                except Exception as e:
                    st.error(f"Groq API error: {e}")

        # Tips
        with st.expander("💡 Tips for a viral LinkedIn post"):
            st.markdown("""
- **Post time**: Tuesday–Thursday, 8–10am or 5–6pm in your audience's timezone
- **First line** is critical — LinkedIn cuts off after 2–3 lines. Hook them immediately.
- **Images** increase engagement 2–3×. Download the chart above to attach.
- **Tag colleagues** or companies mentioned in your analysis
- **Engage early**: Reply to every comment in the first hour for maximum reach
- **Avoid links in body**: Put external links in the first comment instead
""")