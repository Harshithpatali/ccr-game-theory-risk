import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.explainer import (
    interpret_overview,
    interpret_risk_monitor,
    interpret_game_theory,
    interpret_monte_carlo,
    interpret_stress,
    interpret_transitions,
    interpret_counterparty,
    glossary_markdown,
    PAGE_METHOD,
)

# ============================================================================
#  Config / compatibility helpers
# ============================================================================
# Backend URL is read from the environment only — never rendered in the UI.
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CCR Quant Risk",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _stretch_kwargs() -> dict:
    """Streamlit >= 1.49 renamed use_container_width -> width='stretch'."""
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
        if (major, minor) >= (1, 49):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


STRETCH = _stretch_kwargs()

FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"
ACCENT = "#2563eb"
GOOD, WARN, BAD = "#16a34a", "#f59e0b", "#dc2626"

BAND_LOOKUP = {
    "green": GOOD, "low": GOOD, "normal": GOOD, "stable": GOOD, "none": GOOD,
    "amber": WARN, "medium": WARN, "watch": WARN, "moderate": WARN,
    "orange": "#f97316", "elevated": "#f97316", "high": "#f97316",
    "red": BAD, "critical": BAD, "severe": BAD, "high risk": BAD, "breach": BAD,
}
BAND_FALLBACK = ["#64748b", "#0ea5e9", "#8b5cf6", "#ec4899", "#14b8a6"]

# ============================================================================
#  Global styling
# ============================================================================
st.markdown(
    """
    <style>
      :root { --ccr-accent: #2563eb; }
      .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1550px; }

      /* hero banner */
      .ccr-hero {
        background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 58%, #2563eb 100%);
        border-radius: 16px; padding: 20px 26px; margin-bottom: 18px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, .18);
      }
      .ccr-hero h1 { color: #fff; font-size: 1.45rem; margin: 0; font-weight: 700; letter-spacing: -.015em; }
      .ccr-hero p  { color: #c7d2fe; margin: 6px 0 0; font-size: .86rem; }

      /* KPI cards */
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px 12px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, .05);
      }
      div[data-testid="stMetricLabel"] p {
        font-size: .72rem; letter-spacing: .05em; text-transform: uppercase;
        color: #64748b; font-weight: 700;
      }
      div[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 700; color: #0f172a; }

      /* tabs */
      .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #e2e8f0; }
      .stTabs [data-baseweb="tab"] { height: 44px; padding: 0 16px; font-weight: 600; }
      .stTabs [aria-selected="true"] { color: var(--ccr-accent) !important; }

      /* expanders / tables */
      div[data-testid="stExpander"] details { border: 1px solid #e2e8f0; border-radius: 10px; }
      div[data-testid="stDataFrame"] { border-radius: 10px; }
      h3 { letter-spacing: -.01em; }

      /* interpretation panel — a soft, readable decision box */
      .ccr-interpretation {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        border: 1px solid #e2e8f0;
        border-left: 4px solid var(--ccr-accent);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
      }
      .ccr-interpretation h3 {
        margin: 0 0 8px;
        font-size: 1.05rem;
        color: #0f172a;
      }
      .ccr-interpretation p, .ccr-interpretation li {
        font-size: .9rem;
        color: #334155;
        line-height: 1.55;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
#  Data access
# ============================================================================
def get(path: str, timeout: int = 90):
    r = requests.get(BACKEND + path, timeout=timeout)
    r.raise_for_status()
    return r.json()


def style_fig(fig, height: int = 420, showlegend: bool = True):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=8, r=8, t=54, b=8),
        font=dict(family=FONT, size=12, color="#334155"),
        title=dict(font=dict(size=14.5, color="#0f172a"), x=0.005, xanchor="left"),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#e2e8f0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0", zeroline=False)
    return fig


def band_color_map(bands) -> dict:
    out, i = {}, 0
    for b in pd.unique(pd.Series(list(bands)).dropna()):
        key = str(b).strip().lower()
        if key in BAND_LOOKUP:
            out[b] = BAND_LOOKUP[key]
        else:
            out[b] = BAND_FALLBACK[i % len(BAND_FALLBACK)]
            i += 1
    return out


# ---------- formatting helpers ---------------------------------------------
def gbp(x, decimals: int = 1) -> str:
    """Human-readable GBP: £1.24B / £310.5M / £820K."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    ax = abs(x)
    if ax >= 1e9:
        return f"£{x / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"£{x / 1e6:,.{decimals}f}M"
    if ax >= 1e3:
        return f"£{x / 1e3:,.0f}K"
    return f"£{x:,.0f}"


def pct(x, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x * 100:,.{decimals}f}%"


def pretty(col: str) -> str:
    return str(col).replace("_", " ").strip().title()


def compact_mapping(d, top: int = 3) -> str:
    if not isinstance(d, dict):
        return str(d)
    items = sorted(
        d.items(),
        key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0,
        reverse=True,
    )[:top]
    parts = []
    for k, v in items:
        label = pretty(str(k))
        parts.append(f"{label} {v:.0%}" if isinstance(v, (int, float)) else f"{label} {v}")
    return " · ".join(parts) if parts else "–"


# ============================================================================
#  Reusable UI blocks
# ============================================================================
def interpretation_panel(title: str, text: str):
    """Decision-interpretation layer shown above charts, styled as a card."""
    body = text if isinstance(text, str) else str(text)
    st.markdown(
        f"""
        <div class="ccr-interpretation">
          <h3>{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Render the (possibly markdown) body inside the styled block using a
    # tighter container rather than a raw HTML string so markdown works.
    with st.container():
        st.markdown(body)
        if os.getenv("GROQ_API_KEY", "").strip():
            st.caption("Briefing refined with Groq · openai/gpt-oss-120b from live model outputs")
        else:
            st.caption(
                "Deterministic briefing from live model outputs. "
                "Set GROQ_API_KEY in .env for AI-polished wording."
            )
    st.divider()


def method_note(page_key: str):
    with st.expander("Methodology note (optional)", expanded=False):
        st.markdown(PAGE_METHOD.get(page_key, ""))


# ============================================================================
#  Sidebar — controls (no backend URL exposed)
# ============================================================================
with st.sidebar:
    st.markdown("#### ⚙️ Data")
    if st.button("🔄 Refresh data", **STRETCH):
        st.cache_data.clear()
        st.rerun()

# ============================================================================
#  Header
# ============================================================================
st.markdown(
    """
    <div class="ccr-hero">
      <h1>📐 CCR Quantitative Risk &amp; Bayesian Game Theory</h1>
      <p>Layer 1: quantitative results · Layer 2: stakeholder interpretation
      (Finding → Why it matters → Risk implication → Management consideration)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Load data once ----
try:
    m = get("/metrics")
    cps = get("/counterparties")
except Exception as exc:  # noqa: BLE001
    st.error(
        "Cannot reach the risk API. Please ensure the backend service is running, "
        "then press **Refresh data**.\n\n"
        f"_Details: `{exc}`_"
    )
    st.stop()

cp_df = pd.DataFrame(cps)
BAND_MAP = (
    band_color_map(cp_df["early_warning_band"])
    if "early_warning_band" in cp_df.columns
    else {}
)

# ---------- KPI strip ------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total EAD", gbp(m.get("total_ead", 0), 2))
k2.metric("Counterparties", f"{m.get('counterparties', len(cp_df)):,}")
k3.metric("Anomalies", f"{m.get('anomalies', 0):,}")
k4.metric("Trades", f"{m.get('trades', 0):,}")
k5.metric("Mean utilisation", pct(m.get("mean_utilisation", 0)))

interpretation_panel("Portfolio takeaway", interpret_overview(m, cps))

with st.expander("📖 Metric glossary"):
    st.markdown(glossary_markdown())

# ============================================================================
#  Sidebar — risk-appetite filters
# ============================================================================
with st.sidebar:
    st.divider()
    st.markdown("#### 🎯 Risk appetite filters")

    band_options = (
        sorted(cp_df["early_warning_band"].dropna().unique().tolist())
        if "early_warning_band" in cp_df.columns
        else []
    )
    sel_bands = st.multiselect("Early-warning bands", band_options, default=band_options)

    min_util = st.slider("Minimum limit utilisation", 0, 150, 0, step=5)

    search = st.text_input("Search counterparty", placeholder="e.g. CP0007").strip().upper()

    st.divider()
    st.caption("Filters apply to every tab. Clear a filter to see the full book.")

# ---------- apply filters --------------------------------------------------
fdf = cp_df.copy()
if sel_bands and "early_warning_band" in fdf.columns:
    fdf = fdf[fdf["early_warning_band"].isin(sel_bands)]
if "limit_utilisation" in fdf.columns:
    fdf = fdf[fdf["limit_utilisation"].fillna(0) * 100 >= min_util]
if search and "counterparty_id" in fdf.columns:
    fdf = fdf[fdf["counterparty_id"].astype(str).str.upper().str.contains(search, na=False)]

if fdf.empty:
    st.warning("No counterparties match the current filters — widen the risk-appetite settings.")
    st.stop()

st.caption(
    f"Showing **{len(fdf):,}** of **{len(cp_df):,}** counterparties "
    f"· total filtered EAD **{gbp(fdf['ead'].sum()) if 'ead' in fdf else '–'}**"
)

# ============================================================================
#  Tabs
# ============================================================================
tabs = st.tabs(
    [
        "📊 Risk Monitor",
        "🎲 Game Theory & Nash",
        "📈 Monte Carlo",
        "🌪️ Stress Testing",
        "🔀 Behaviour Transitions",
        "🔍 Drilldown",
    ]
)


# ---------------------------------------------------------------------------
#  1. RISK MONITOR
# ---------------------------------------------------------------------------
def render_risk_monitor(df: pd.DataFrame):
    required = ["counterparty_id", "ead", "limit", "limit_utilisation",
                "breach_probability", "early_warning_band"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Backend response missing field(s): {', '.join(missing)}")
        return

    left, right = st.columns([2.1, 1])

    with left:
        st.subheader("Risk matrix — where to look first")
        st.caption(
            "Horizontal = how full the limit is. Vertical = chance the limit is tested in 30 days. "
            "Bubble size = EAD. Colour = early-warning band."
        )
        d = df.copy()
        d["_bubble"] = d["ead"].fillna(0).clip(lower=0)
        if d["_bubble"].max() <= 0:
            d["_bubble"] = 1.0

        pb_cut = float(d["breach_probability"].median()) if len(d) else 0.25
        pb_cut = max(pb_cut, 0.05)

        fig = px.scatter(
            d,
            x="limit_utilisation",
            y="breach_probability",
            size="_bubble",
            color="early_warning_band",
            color_discrete_map=BAND_MAP,
            hover_name="counterparty_id",
            size_max=46,
            custom_data=["counterparty_id", "ead", "limit", "limit_utilisation"],
            labels={
                "limit_utilisation": "Limit utilisation",
                "breach_probability": "Breach probability (30d)",
                "early_warning_band": "Band",
            },
        )
        fig.update_traces(
            marker=dict(line=dict(width=1, color="white"), opacity=0.85),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Utilisation: %{x:.1%}<br>"
                "P(breach 30d): %{y:.1%}<br>"
                "EAD: %{customdata[1]:,.0f}<br>"
                "Limit: %{customdata[2]:,.0f}"
                "<extra></extra>"
            ),
        )

        x_max = max(1.05, float(d["limit_utilisation"].max()) * 1.10)
        y_max = max(pb_cut * 1.8, float(d["breach_probability"].max()) * 1.20, 0.05)

        fig.add_vline(x=0.80, line_width=1, line_dash="dash", line_color="#94a3b8")
        fig.add_hline(y=pb_cut, line_width=1, line_dash="dash", line_color="#94a3b8")
        fig.add_annotation(
            x=x_max * 0.98, y=y_max * 0.96, text="ESCALATE", showarrow=False,
            font=dict(color=BAD, size=11, family=FONT), xanchor="right", yanchor="top",
        )
        fig.add_annotation(
            x=x_max * 0.02, y=y_max * 0.96, text="WATCH", showarrow=False,
            font=dict(color=WARN, size=11, family=FONT), xanchor="left", yanchor="top",
        )
        fig.update_xaxes(range=[0, x_max], tickformat=".0%")
        fig.update_yaxes(range=[0, y_max], tickformat=".0%")
        st.plotly_chart(style_fig(fig, height=440), **STRETCH)

    with right:
        st.subheader("Exposure by band")
        st.caption("Share of portfolio EAD sitting in each early-warning band.")
        band_ead = (
            df.groupby("early_warning_band", as_index=False)["ead"].sum()
            .sort_values("ead", ascending=False)
        )
        pie = px.pie(
            band_ead,
            names="early_warning_band",
            values="ead",
            hole=0.62,
            color="early_warning_band",
            color_discrete_map=BAND_MAP,
        )
        pie.update_traces(
            textposition="outside",
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>EAD: %{value:,.0f}<br>%{percent}<extra></extra>",
        )
        pie.add_annotation(
            text=gbp(df["ead"].sum()), x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color="#0f172a", family=FONT),
        )
        st.plotly_chart(style_fig(pie, height=440, showlegend=True), **STRETCH)

    # ---------------- table ------------------------------------------------
    st.subheader("Portfolio risk table")
    st.caption("Click any column header to sort · use column menus to filter and search.")

    view = pd.DataFrame(
        {
            "Counterparty": df["counterparty_id"],
            "EAD (£M)": (df["ead"] / 1e6).round(2),
            "Limit (£M)": (df["limit"] / 1e6).round(2),
            "Utilisation %": (df["limit_utilisation"].fillna(0) * 100).clip(lower=0),
            "Anomaly": df.get("anomaly_score"),
            "EW score": df.get("early_warning_score"),
            "Band": df["early_warning_band"],
            "P(breach 30d) %": (df["breach_probability"].fillna(0) * 100),
            "Strategy": df.get("preferred_strategy"),
            "VOI": df.get("value_of_perfect_info"),
        }
    ).sort_values("P(breach 30d) %", ascending=False)

    st.dataframe(
        view,
        hide_index=True,
        height=460,
        column_config={
            "Counterparty": st.column_config.TextColumn(width="medium"),
            "EAD (£M)": st.column_config.NumberColumn(format="£ %.2f", width="small"),
            "Limit (£M)": st.column_config.NumberColumn(format="£ %.2f", width="small"),
            "Utilisation %": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=150, width="medium"
            ),
            "Anomaly": st.column_config.NumberColumn(format="%.3f", width="small"),
            "EW score": st.column_config.NumberColumn(format="%.3f", width="small"),
            "Band": st.column_config.TextColumn(width="small"),
            "P(breach 30d) %": st.column_config.NumberColumn(format="%.2f%%", width="small"),
            "Strategy": st.column_config.TextColumn(width="medium"),
            "VOI": st.column_config.NumberColumn(format="%.3f", width="small"),
        },
        **STRETCH,
    )

    st.download_button(
        "⬇️ Download risk table (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="ccr_portfolio_risk.csv",
        mime="text/csv",
    )


with tabs[0]:
    interpretation_panel("What the model found (Risk Monitor)", interpret_risk_monitor(fdf.to_dict("records")))
    method_note("risk_monitor")
    st.subheader("Quantitative results")
    render_risk_monitor(fdf)


# ---------------------------------------------------------------------------
#  2. GAME THEORY
# ---------------------------------------------------------------------------
def render_game_theory(df: pd.DataFrame):
    if "preferred_strategy" not in df.columns:
        st.warning("Backend response missing `preferred_strategy`.")
        return

    st.info(
        "Payoffs are illustrative modelling assumptions. Recalibrate with your credit and "
        "relationship teams before operational use."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("What the model prefers")
        st.caption("Counterparties by recommended pure strategy (highest expected utility).")
        strat = (
            df["preferred_strategy"].value_counts().rename_axis("Strategy")
            .reset_index(name="Counterparties")
        )
        donut = px.pie(
            strat,
            names="Strategy",
            values="Counterparties",
            hole=0.58,
            color_discrete_sequence=["#2563eb", "#0ea5e9", "#8b5cf6", "#f59e0b", "#14b8a6"],
        )
        donut.update_traces(
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>%{value} counterparties<br>%{percent}<extra></extra>",
        )
        st.plotly_chart(style_fig(donut, height=380), **STRETCH)

    with c2:
        st.subheader("Is better information worth it?")
        st.caption(
            "Value of perfect information (VOI): extra utility if we knew the counterparty's true "
            "behavioural type. Higher = more value in relationship intelligence."
        )
        if "value_of_perfect_info" in df.columns:
            hist = px.histogram(
                df,
                x="value_of_perfect_info",
                nbins=22,
                labels={"value_of_perfect_info": "Value of perfect information"},
            )
            hist.update_traces(
                marker=dict(color=ACCENT, line=dict(color="white", width=1)),
                hovertemplate="VOI %{x:.3f}<br>%{y} counterparties<extra></extra>",
            )
            hist.add_vline(
                x=float(df["value_of_perfect_info"].median()),
                line_dash="dash", line_color=WARN,
                annotation_text="median", annotation_position="top",
            )
            st.plotly_chart(style_fig(hist, height=380), **STRETCH)
        else:
            st.info("`value_of_perfect_info` not supplied by the backend.")

    st.subheader("Value of information vs breach risk")
    st.caption("Names in the top-right combine high breach risk with high value of better intelligence.")
    if {"value_of_perfect_info", "breach_probability"}.issubset(df.columns):
        sc = px.scatter(
            df,
            x="breach_probability",
            y="value_of_perfect_info",
            size="ead" if "ead" in df.columns else None,
            color="early_warning_band" if "early_warning_band" in df.columns else None,
            color_discrete_map=BAND_MAP,
            hover_name="counterparty_id",
            size_max=40,
            labels={
                "breach_probability": "Breach probability (30d)",
                "value_of_perfect_info": "Value of perfect information",
            },
        )
        sc.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.85))
        sc.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_fig(sc, height=400), **STRETCH)

    st.subheader("Detail by counterparty")
    detail_cols = [
        c
        for c in [
            "counterparty_id", "preferred_strategy", "nash_value",
            "value_of_perfect_info", "probabilities", "expected_utilities", "mixed_nash_bank",
        ]
        if c in df.columns
    ]
    detail = df[detail_cols].copy()
    for col in ("probabilities", "expected_utilities", "mixed_nash_bank"):
        if col in detail.columns:
            detail[col] = detail[col].apply(compact_mapping)

    st.dataframe(
        detail.rename(columns={c: pretty(c) for c in detail.columns}),
        hide_index=True,
        height=380,
        column_config={
            "Value Of Perfect Info": st.column_config.NumberColumn(format="%.4f"),
            "Nash Value": st.column_config.NumberColumn(format="%.4f"),
            "Probabilities": st.column_config.TextColumn(width="large"),
            "Expected Utilities": st.column_config.TextColumn(width="large"),
            "Mixed Nash Bank": st.column_config.TextColumn(width="large"),
        },
        **STRETCH,
    )


with tabs[1]:
    interpretation_panel("What the model found (Game theory → commercial decision)", interpret_game_theory(fdf.to_dict("records")))
    method_note("game_theory")
    render_game_theory(fdf)


# ---------------------------------------------------------------------------
#  3. MONTE CARLO
# ---------------------------------------------------------------------------
def render_monte_carlo(df: pd.DataFrame):
    if "breach_probability" not in df.columns:
        st.warning("Backend response missing `breach_probability`.")
        return

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("How often would limits be tested?")
        st.caption("Distribution of estimated 30-day limit-breach probabilities across the book.")
        hist = px.histogram(
            df,
            x="breach_probability",
            nbins=25,
            labels={"breach_probability": "Breach probability (30d)"},
        )
        hist.update_traces(
            marker=dict(color="#0ea5e9", line=dict(color="white", width=1)),
            hovertemplate="P(breach) %{x:.1%}<br>%{y} counterparties<extra></extra>",
        )
        hist.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_fig(hist, height=400), **STRETCH)

    with c2:
        st.subheader("Breach risk by early-warning band")
        st.caption("Box = interquartile range, line = median, dots = individual counterparties.")
        if "early_warning_band" in df.columns:
            box = px.box(
                df,
                x="early_warning_band",
                y="breach_probability",
                color="early_warning_band",
                color_discrete_map=BAND_MAP,
                points="all",
                labels={
                    "early_warning_band": "Band",
                    "breach_probability": "Breach probability (30d)",
                },
            )
            box.update_traces(marker=dict(size=5, opacity=0.7))
            box.update_yaxes(tickformat=".0%")
            st.plotly_chart(style_fig(box, height=400, showlegend=False), **STRETCH)

    st.subheader("Peak exposure under normal-market simulation")
    st.caption(
        "P95 vs P99 of simulated peak EAD. Points above the diagonal carry fatter tails; "
        "colour intensity = breach probability."
    )
    if {"p95_max_ead", "p99_max_ead"}.issubset(df.columns):
        d = df.copy()
        d["_size"] = d["ead"].fillna(0).clip(lower=0)
        if d["_size"].max() <= 0:
            d["_size"] = 1.0

        sc = px.scatter(
            d,
            x="p95_max_ead",
            y="p99_max_ead",
            size="_size",
            color="breach_probability",
            color_continuous_scale="YlOrRd",
            hover_name="counterparty_id",
            size_max=44,
            custom_data=["counterparty_id", "ead"],
            labels={"p95_max_ead": "P95 peak EAD", "p99_max_ead": "P99 peak EAD"},
        )
        sc.update_traces(
            marker=dict(line=dict(width=1, color="white"), opacity=0.85),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>P95: %{x:,.0f}<br>P99: %{y:,.0f}"
                "<br>Current EAD: %{customdata[1]:,.0f}<extra></extra>"
            ),
        )
        lim = float(max(d["p95_max_ead"].max(), d["p99_max_ead"].max()))
        sc.add_shape(
            type="line", x0=0, y0=0, x1=lim, y1=lim,
            line=dict(color="#94a3b8", dash="dot", width=1),
        )
        sc.update_xaxes(tickformat=",.2s", tickprefix="£")
        sc.update_yaxes(tickformat=",.2s", tickprefix="£")
        st.plotly_chart(style_fig(sc, height=430), **STRETCH)

        st.markdown("##### Largest tail exposures (top 10 by P99)")
        top = (
            d.nlargest(10, "p99_max_ead")[
                ["counterparty_id", "ead", "p95_max_ead", "p99_max_ead", "breach_probability"]
            ]
            .rename(
                columns={
                    "counterparty_id": "Counterparty",
                    "ead": "Current EAD",
                    "p95_max_ead": "P95 peak EAD",
                    "p99_max_ead": "P99 peak EAD",
                    "breach_probability": "P(breach 30d)",
                }
            )
            .reset_index(drop=True)
        )
        st.dataframe(
            top,
            hide_index=True,
            column_config={
                "Current EAD": st.column_config.NumberColumn(format="£ %,.0f"),
                "P95 peak EAD": st.column_config.NumberColumn(format="£ %,.0f"),
                "P99 peak EAD": st.column_config.NumberColumn(format="£ %,.0f"),
                "P(breach 30d)": st.column_config.NumberColumn(format="%.2f%%"),
            },
            **STRETCH,
        )
    else:
        st.info("Peak-exposure quantiles (`p95_max_ead`, `p99_max_ead`) not supplied by the backend.")


with tabs[2]:
    interpretation_panel("What the model found (Monte Carlo → exposure headroom)", interpret_monte_carlo(fdf.to_dict("records")))
    method_note("monte_carlo")
    render_monte_carlo(fdf)


# ---------------------------------------------------------------------------
#  4. STRESS TESTING
# ---------------------------------------------------------------------------
def render_stress():
    try:
        raw = get("/stress")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Stress endpoint error. _Details: `{exc}`_")
        raw = {"scenarios": [], "reverse_stress": {}}

    interpretation_panel(
        "What the model found (Stress → limits & headroom)",
        interpret_stress(raw, m),
    )
    method_note("stress")

    scenarios = raw.get("scenarios") or []
    if not scenarios:
        st.info("No stress scenarios returned by the backend.")
        st.subheader("Reverse stress (capacity)")
        st.json(raw.get("reverse_stress") or {})
        return

    s = pd.DataFrame(scenarios)
    base_ead = float(m.get("total_ead", 0) or 0)

    st.subheader("Named market scenarios")
    st.caption("How much total EAD rises and how many names go over limit under each shock.")

    c1, c2 = st.columns(2)

    with c1:
        if "incremental_ead" in s.columns:
            bar = px.bar(
                s.sort_values("incremental_ead"),
                x="incremental_ead",
                y="scenario",
                orientation="h",
                labels={"incremental_ead": "Incremental EAD (£M)", "scenario": ""},
            )
            bar.update_traces(
                marker=dict(
                    color=s.sort_values("incremental_ead")["incremental_ead"] / 1e6,
                    colorscale="Reds",
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{y}</b><br>Extra EAD: £%{x:,.0f}<extra></extra>",
            )
            bar.update_xaxes(tickformat=",.2s", tickprefix="£")
            st.plotly_chart(style_fig(bar, height=400, showlegend=False), **STRETCH)
        else:
            st.info("`incremental_ead` not supplied.")

    with c2:
        over_col = next(
            (c for c in s.columns if any(k in c.lower() for k in ("over", "breach", "count"))),
            None,
        )
        if over_col:
            bar2 = px.bar(
                s.sort_values(over_col),
                x=over_col,
                y="scenario",
                orientation="h",
                labels={over_col: pretty(over_col), "scenario": ""},
            )
            bar2.update_traces(
                marker=dict(color=WARN, line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>%{x:,.0f} counterparties<extra></extra>",
            )
            st.plotly_chart(style_fig(bar2, height=400, showlegend=False), **STRETCH)
        else:
            st.info("No 'names over limit' field returned.")

    # ---------------- formatted scenario table ----------------------------
    disp = pd.DataFrame()
    cfg = {}
    for c in s.columns:
        if c == "scenario":
            disp["Scenario"] = s[c]
        elif c == "incremental_ead":
            disp["Incremental EAD (£M)"] = s[c] / 1e6
            cfg["Incremental EAD (£M)"] = st.column_config.NumberColumn(format="£ %.1f")
        else:
            name = pretty(c)
            disp[name] = s[c]
            if pd.api.types.is_float_dtype(s[c]):
                cfg[name] = st.column_config.NumberColumn(format="%.3f")

    if base_ead > 0 and "incremental_ead" in s.columns:
        disp["Δ vs current book %"] = s["incremental_ead"] / base_ead * 100
        cfg["Δ vs current book %"] = st.column_config.NumberColumn(format="%.1f%%")

    st.dataframe(disp, hide_index=True, column_config=cfg, **STRETCH)

    # ---------------- reverse stress --------------------------------------
    st.subheader("Reverse stress (capacity view)")
    st.caption(
        "Uniform EAD multiplier that would push roughly a quarter of the book over its limit — "
        "a simple read on remaining headroom."
    )
    rs = raw.get("reverse_stress") or {}
    if isinstance(rs, dict) and rs:
        keys = list(rs.keys())
        cols = st.columns(min(len(keys), 4))
        for i, k in enumerate(keys):
            v = rs[k]
            value = f"{v:,.2f}" if isinstance(v, (int, float)) else str(v)
            cols[i % len(cols)].metric(pretty(k), value)
    else:
        st.info("No reverse-stress output returned.")


with tabs[3]:
    render_stress()


# ---------------------------------------------------------------------------
#  5. BEHAVIOUR TRANSITIONS
# ---------------------------------------------------------------------------
def render_transitions():
    try:
        tm = get("/transitions")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Transitions error. _Details: `{exc}`_")
        tm = {}

    interpretation_panel(
        "What the model found (Behaviour persistence)",
        interpret_transitions(tm),
    )
    method_note("transitions")

    if not tm:
        st.info("No transition matrix returned by the backend.")
        return

    mat = pd.DataFrame(tm).T
    states = list(mat.columns)
    mat = mat.reindex(index=states, columns=states).astype(float)

    st.subheader("How behaviour persists")
    st.caption("Rows = yesterday's state; columns = today's state. A strong diagonal means stickiness.")

    c1, c2 = st.columns([1.35, 1])

    with c1:
        heat = go.Figure(
            go.Heatmap(
                z=mat.values,
                x=[pretty(c) for c in mat.columns],
                y=[pretty(i) for i in mat.index],
                colorscale="Blues",
                zmin=0,
                zmax=max(1.0, float(np.nanmax(mat.values))),
                text=[[f"{v:.0%}" for v in row] for row in mat.values],
                texttemplate="%{text}",
                hovertemplate="Yesterday: %{y}<br>Today: %{x}<br>Probability: %{z:.1%}<extra></extra>",
                colorbar=dict(title="P", tickformat=".0%", thickness=12),
            )
        )
        heat.update_yaxes(autorange="reversed")
        st.plotly_chart(
            style_fig(heat, height=430, showlegend=False).update_layout(
                title="Behaviour transition probabilities  P(today | yesterday)",
                xaxis_title="Today", yaxis_title="Yesterday",
            ),
            **STRETCH,
        )

    with c2:
        st.markdown("##### Long-run (steady-state) mix")
        st.caption("Where the book settles if current transition behaviour persists.")
        try:
            P = mat.values.astype(float)
            row_sums = P.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            P = P / row_sums
            v = np.full(len(P), 1.0 / len(P))
            for _ in range(600):
                v = v @ P
            steady = pd.Series(v, index=[pretty(i) for i in mat.index]).sort_values()
            sb = px.bar(
                x=steady.values,
                y=steady.index,
                orientation="h",
                labels={"x": "Long-run share", "y": ""},
            )
            sb.update_traces(
                marker=dict(color="#8b5cf6", line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>%{x:.1%} of the time<extra></extra>",
            )
            sb.update_xaxes(tickformat=".0%")
            st.plotly_chart(style_fig(sb, height=430, showlegend=False), **STRETCH)
        except Exception:  # noqa: BLE001
            st.info("Steady-state could not be computed from the supplied matrix.")

    st.markdown("##### Transition matrix (raw)")
    st.dataframe(
        mat.rename(index=pretty, columns=pretty).style.format("{:.2%}"),
        **STRETCH,
    )


with tabs[4]:
    render_transitions()


# ---------------------------------------------------------------------------
#  6. DRILLDOWN
# ---------------------------------------------------------------------------
def render_drilldown(df: pd.DataFrame):
    if "counterparty_id" not in df.columns:
        st.warning("Backend response missing `counterparty_id`.")
        return

    # Pick the highest-breach-risk row as the default selection.
    default_idx = 0
    if "breach_probability" in df.columns and len(df):
        series = pd.to_numeric(df["breach_probability"], errors="coerce").fillna(-np.inf)
        if series.notna().any():
            # argmax on the numpy array gives the positional index,
            # which is what st.selectbox(..., index=...) expects.
            default_idx = int(series.to_numpy().argmax())

    ids = df["counterparty_id"].tolist()
    default_idx = max(0, min(default_idx, len(ids) - 1))
    cp = st.selectbox("Choose counterparty", ids, index=default_idx)

    try:
        r = get(f"/counterparty/{cp}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load counterparty detail. _Details: `{exc}`_")
        return

    interpretation_panel(
        f"Stakeholder briefing — {cp}",
        interpret_counterparty(r),
    )
    method_note("drilldown")

    # ---------- headline metrics ------------------------------------------
    util = float(r.get("limit_utilisation") or 0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("EAD", gbp(r.get("ead", 0), 2))
    m2.metric("Breach prob (30d)", pct(r.get("breach_probability", 0), 2))
    m3.metric("Warning band", str(r.get("early_warning_band", "–")))
    m4.metric("Value of information", f"{float(r.get('value_of_perfect_info') or 0):.3f}")

    c1, c2 = st.columns([1, 1])

    # ---------- utilisation gauge -----------------------------------------
    with c1:
        st.markdown("##### Limit utilisation")
        u = util * 100
        axis_max = max(120.0, round(u * 1.15, -1))
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=u,
                number={"suffix": "%", "font": {"size": 32, "family": FONT}},
                gauge={
                    "axis": {"range": [0, axis_max], "ticksuffix": "%"},
                    "bar": {"color": ACCENT, "thickness": 0.28},
                    "steps": [
                        {"range": [0, 60], "color": "#dcfce7"},
                        {"range": [60, 85], "color": "#fef3c7"},
                        {"range": [85, axis_max], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": BAD, "width": 3},
                        "thickness": 0.85,
                        "value": 100,
                    },
                },
            )
        )
        st.plotly_chart(style_fig(gauge, height=300, showlegend=False), **STRETCH)

    # ---------- behaviour probabilities ------------------------------------
    with c2:
        st.markdown("##### Behaviour probabilities")
        st.caption("Model's belief about the counterparty's behavioural type.")
        probs = r.get("probabilities") or {}
        if isinstance(probs, dict) and probs:
            pdf = (
                pd.DataFrame({"Type": [pretty(k) for k in probs], "Probability": list(probs.values())})
                .sort_values("Probability")
            )
            pb = px.bar(
                pdf, x="Probability", y="Type", orientation="h",
                labels={"Probability": "Probability", "Type": ""},
            )
            pb.update_traces(
                marker=dict(color="#0ea5e9", line=dict(width=0)),
                text=pdf["Probability"].map(lambda v: f"{v:.1%}"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x:.1%}<extra></extra>",
            )
            pb.update_xaxes(tickformat=".0%", range=[0, max(1.0, pdf["Probability"].max() * 1.25)])
            st.plotly_chart(style_fig(pb, height=300, showlegend=False), **STRETCH)
        else:
            st.info("No behaviour probabilities returned.")

    c3, c4 = st.columns(2)

    # ---------- expected utilities -----------------------------------------
    with c3:
        st.markdown("##### Expected utility of bank actions")
        eu = r.get("expected_utilities") or {}
        if isinstance(eu, dict) and eu:
            if any(isinstance(v, dict) for v in eu.values()):
                edf = pd.DataFrame(eu).T
                edf.index.name = "Action"
                long = (
                    edf.reset_index()
                    .melt(id_vars="Action", var_name="Counterparty type", value_name="Utility")
                )
                long["Action"] = long["Action"].map(pretty)
                long["Counterparty type"] = long["Counterparty type"].map(pretty)
                eb = px.bar(
                    long, x="Action", y="Utility", color="Counterparty type",
                    barmode="group", color_discrete_sequence=px.colors.qualitative.Set2,
                )
            else:
                edf = pd.DataFrame(
                    {"Action": [pretty(k) for k in eu], "Utility": list(eu.values())}
                )
                eb = px.bar(edf, x="Action", y="Utility", color="Utility",
                            color_continuous_scale="Blues")
            eb.update_traces(hovertemplate="<b>%{x}</b><br>Utility: %{y:.3f}<extra></extra>")
            st.plotly_chart(style_fig(eb, height=340), **STRETCH)
        else:
            st.info("No expected utilities returned.")

    # ---------- mixed Nash --------------------------------------------------
    with c4:
        st.markdown("##### Mixed Nash — recommended action mix")
        mn = r.get("mixed_nash_bank") or {}
        if isinstance(mn, dict) and mn and sum(
            v for v in mn.values() if isinstance(v, (int, float))
        ) > 0:
            mdf = pd.DataFrame(
                {"Action": [pretty(k) for k in mn], "Probability": list(mn.values())}
            )
            dough = px.pie(
                mdf, names="Action", values="Probability", hole=0.6,
                color_discrete_sequence=["#2563eb", "#0ea5e9", "#f59e0b", "#8b5cf6"],
            )
            dough.update_traces(
                textinfo="percent", textposition="outside",
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            )
            st.plotly_chart(style_fig(dough, height=340), **STRETCH)
        else:
            st.info("No mixed-Nash mix returned.")

    # ---------- early-warning drivers --------------------------------------
    contrib = r.get("early_warning_contributions") or {}
    if isinstance(contrib, dict) and contrib:
        st.markdown("##### What is driving the early warning?")
        st.caption("Positive contributions push the score up; negative contributions pull it down.")
        cdf = pd.DataFrame(
            {"Driver": [pretty(k) for k in contrib], "Contribution": list(contrib.values())}
        ).sort_values("Contribution")
        cdf["Direction"] = np.where(cdf["Contribution"] >= 0, "Increases risk", "Reduces risk")
        cb = px.bar(
            cdf, x="Contribution", y="Driver", orientation="h", color="Direction",
            color_discrete_map={"Increases risk": BAD, "Reduces risk": GOOD},
        )
        cb.update_traces(
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>"
        )
        cb.add_vline(x=0, line_width=1, line_color="#94a3b8")
        st.plotly_chart(style_fig(cb, height=340), **STRETCH)

    st.success(f"Model-preferred strategy: **{r.get('preferred_strategy', '–')}**")
    st.caption("Input to credit discussion — not an automatic instruction.")


with tabs[5]:
    render_drilldown(fdf)