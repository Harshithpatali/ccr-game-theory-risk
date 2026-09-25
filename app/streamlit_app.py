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

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

B = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="CCR Quant Risk", page_icon="📐", layout="wide")
st.title("📐 CCR Quantitative Risk & Bayesian Game Theory")
st.caption(
    "Layer 1: quantitative results · Layer 2: stakeholder interpretation "
    "(Finding → Why it matters → Risk implication → Management consideration). "
    "Optional live polish via Groq openai/gpt-oss-120b when GROQ_API_KEY is set."
)


def get(p, timeout=90):
    r = requests.get(B + p, timeout=timeout)
    r.raise_for_status()
    return r.json()


def interpretation_panel(title: str, text: str):
    """Decision-interpretation layer shown above charts."""
    with st.container():
        st.markdown(f"### {title}")
        st.markdown(text)
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


# ---- Load data once ----
try:
    m = get("/metrics")
    cps = get("/counterparties")
except Exception as e:
    st.error(f"Cannot reach API at {B}. Start the backend first. ({e})")
    st.stop()

# Header KPIs
a, b, c, d, e = st.columns(5)
a.metric("Total EAD", f"£{m['total_ead']/1e9:.2f}B")
b.metric("Counterparties", m["counterparties"])
c.metric("Anomalies", m["anomalies"])
d.metric("Trades", m["trades"])
e.metric("Mean utilisation", f"{m.get('mean_utilisation', 0):.1%}")

interpretation_panel(
    "Portfolio takeaway",
    interpret_overview(m, cps),
)

with st.expander("Metric glossary"):
    st.markdown(glossary_markdown())

tabs = st.tabs([
    "Risk Monitor",
    "Game Theory & Nash",
    "Monte Carlo",
    "Stress Testing",
    "Behaviour Transitions",
    "Drilldown",
])

df = pd.DataFrame(cps)

# ---- Risk Monitor ----
with tabs[0]:
    interpretation_panel("What the model found (Risk Monitor)", interpret_risk_monitor(cps))
    method_note("risk_monitor")

    st.subheader("Quantitative results")
    st.dataframe(
        df[
            [
                "counterparty_id", "ead", "limit", "limit_utilisation",
                "anomaly_score", "early_warning_score", "early_warning_band",
                "breach_probability", "preferred_strategy", "value_of_perfect_info",
            ]
        ].style.format({
            "ead": "{:,.0f}",
            "limit": "{:,.0f}",
            "limit_utilisation": "{:.1%}",
            "breach_probability": "{:.2%}",
            "early_warning_score": "{:.3f}",
            "anomaly_score": "{:.3f}",
            "value_of_perfect_info": "{:.3f}",
        }),
        use_container_width=True,
        height=400,
    )
    st.caption("Top-right of the chart = high utilisation and high breach probability → review first.")
    st.plotly_chart(
        px.scatter(
            df,
            x="limit_utilisation",
            y="breach_probability",
            size="ead",
            color="early_warning_band",
            hover_name="counterparty_id",
            title="Utilisation vs 30-day breach probability",
            labels={
                "limit_utilisation": "Limit utilisation",
                "breach_probability": "Breach probability (30d)",
            },
        ),
        use_container_width=True,
    )

# ---- Game Theory ----
with tabs[1]:
    interpretation_panel("What the model found (Game theory → commercial decision)", interpret_game_theory(cps))
    method_note("game_theory")
    st.info(
        "Payoffs are illustrative. Recalibrate with credit and relationship teams before operational use."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Preferred pure strategy (count of names)")
        st.plotly_chart(
            px.bar(
                df["preferred_strategy"].value_counts().reset_index(),
                x="preferred_strategy",
                y="count",
                labels={"preferred_strategy": "Strategy", "count": "Counterparties"},
            ),
            use_container_width=True,
        )
    with c2:
        st.subheader("Value of perfect information")
        st.caption("Higher VOI = more value in knowing the counterparty’s true behavioural type.")
        st.plotly_chart(
            px.histogram(df, x="value_of_perfect_info", nbins=20),
            use_container_width=True,
        )
    st.subheader("Quantitative detail")
    st.dataframe(
        df[["counterparty_id", "probabilities", "expected_utilities", "mixed_nash_bank", "nash_value", "preferred_strategy", "value_of_perfect_info"]],
        use_container_width=True,
    )

# ---- Monte Carlo ----
with tabs[2]:
    interpretation_panel("What the model found (Monte Carlo → exposure headroom)", interpret_monte_carlo(cps))
    method_note("monte_carlo")
    st.subheader("Distribution of 30-day breach probabilities")
    st.plotly_chart(
        px.histogram(
            df,
            x="breach_probability",
            nbins=25,
            labels={"breach_probability": "Breach probability"},
        ),
        use_container_width=True,
    )
    st.subheader("Peak exposure quantiles (P95 vs P99)")
    st.plotly_chart(
        px.scatter(
            df,
            x="p95_max_ead",
            y="p99_max_ead",
            size="ead",
            color="breach_probability",
            hover_name="counterparty_id",
            labels={"p95_max_ead": "P95 peak EAD", "p99_max_ead": "P99 peak EAD"},
        ),
        use_container_width=True,
    )

# ---- Stress ----
with tabs[3]:
    try:
        raw = get("/stress")
    except Exception as ex:
        st.error(f"Stress endpoint error: {ex}")
        raw = {"scenarios": [], "reverse_stress": {}}
    interpretation_panel(
        "What the model found (Stress → limits & headroom)",
        interpret_stress(raw, m),
    )
    method_note("stress")
    s = pd.DataFrame(raw.get("scenarios") or [])
    if not s.empty:
        st.subheader("Incremental EAD by scenario")
        st.plotly_chart(
            px.bar(
                s,
                x="scenario",
                y="incremental_ead",
                labels={"incremental_ead": "Extra EAD", "scenario": "Scenario"},
            ),
            use_container_width=True,
        )
        st.dataframe(s, use_container_width=True)
    st.subheader("Reverse stress (capacity)")
    st.json(raw.get("reverse_stress") or {})

# ---- Transitions ----
with tabs[4]:
    try:
        tm = get("/transitions")
    except Exception as ex:
        st.error(f"Transitions error: {ex}")
        tm = {}
    interpretation_panel(
        "What the model found (Behaviour persistence)",
        interpret_transitions(tm),
    )
    method_note("transitions")
    if tm:
        mat = pd.DataFrame(tm).T
        st.subheader("Transition matrix P(today | yesterday)")
        st.dataframe(mat.style.format("{:.1%}"), use_container_width=True)
        fig = go.Figure(
            data=go.Heatmap(
                z=mat.values,
                x=list(mat.columns),
                y=list(mat.index),
                colorscale="Blues",
                text=[[f"{v:.0%}" for v in row] for row in mat.values],
                texttemplate="%{text}",
            )
        )
        fig.update_layout(xaxis_title="Today", yaxis_title="Yesterday")
        st.plotly_chart(fig, use_container_width=True)

# ---- Drilldown ----
with tabs[5]:
    method_note("drilldown")
    cp = st.selectbox("Choose counterparty", df.counterparty_id)
    r = get("/counterparty/" + cp)
    interpretation_panel(
        f"Stakeholder briefing — {cp}",
        interpret_counterparty(r),
    )
    x, y, z, w = st.columns(4)
    x.metric("EAD", f"£{r['ead']/1e6:.2f}M")
    y.metric("Breach prob (30d)", f"{r['breach_probability']:.2%}")
    z.metric("Warning band", r["early_warning_band"])
    w.metric("VOI", f"{r['value_of_perfect_info']:.3f}")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Behaviour probabilities**")
        st.json(r["probabilities"])
        st.write("**Expected utilities**")
        st.json(r["expected_utilities"])
    with c2:
        st.write("**Mixed Nash (bank)**")
        st.json(r["mixed_nash_bank"])
        st.write("**Early-warning contributions**")
        st.json(r.get("early_warning_contributions", {}))
    st.success("Model-preferred strategy: **" + r["preferred_strategy"] + "**")
    st.caption("Input to credit discussion — not an automatic instruction.")
