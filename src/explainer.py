"""
Plain-language explanations for stakeholders.

Uses Groq OpenAI-compatible API (openai/gpt-oss-120b) when GROQ_API_KEY is set.
Falls back to curated static explanations so the dashboard is always usable.
"""
from __future__ import annotations
import os
import json
from functools import lru_cache
from typing import Optional

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------------------
# Static plain-language content (always available)
# ---------------------------------------------------------------------------

PAGE_INTROS = {
    "overview": (
        "**What is this dashboard?**\n\n"
        "This tool helps risk managers answer one practical question: "
        "*Is our exposure to a counterparty becoming dangerous, and what should we do about it?*\n\n"
        "We combine market data, statistical warning signals, probability models of how the "
        "counterparty might behave, and a decision framework (game theory) that recommends "
        "whether to keep the limit, cut it, or ask for more collateral. "
        "Numbers are in economic units; charts show patterns across all counterparties."
    ),
    "risk_monitor": (
        "**Risk Monitor — what you are looking at**\n\n"
        "- **EAD (Exposure at Default)**: Roughly how much we could lose if the counterparty "
        "defaulted today, after collateral.\n"
        "- **Limit utilisation**: EAD divided by the credit limit we set. Above 100% means we "
        "are already over the agreed limit.\n"
        "- **Anomaly score**: How unusual this counterparty's recent pattern is compared with "
        "the rest of the book (higher = more unusual).\n"
        "- **Early-warning band (Low / Moderate / High)**: A traffic-light score that blends "
        "utilisation, trend, credit-spread pressure and anomalies.\n"
        "- **Breach probability**: Estimated chance that, over the next 30 business days, "
        "peak exposure will exceed the limit under realistic market moves.\n\n"
        "The scatter plot puts utilisation on the horizontal axis and breach probability on "
        "the vertical axis. Points in the top-right are the ones that need attention first."
    ),
    "game_theory": (
        "**Game Theory & Nash — what this means in plain English**\n\n"
        "We treat the bank and the counterparty as players who each have choices:\n\n"
        "**Bank can:** keep the current limit · cut the limit · demand more collateral.\n\n"
        "**Counterparty may:** reduce exposure · stay stable · increase exposure "
        "(we estimate the probability of each from data).\n\n"
        "Each combination has a payoff (gain or pain) for the bank. We convert those payoffs "
        "into *utilities* that reflect risk aversion, then:\n"
        "- Rank pure strategies by expected utility → **preferred strategy**.\n"
        "- Compute a **mixed Nash** mix (how often a cautious bank would pick each action "
        "if the other side played optimally against us).\n"
        "- **Value of perfect information (VOI)**: how much better our decision would be if "
        "we knew the counterparty's true type with certainty. High VOI means better "
        "relationship intelligence is worth paying for.\n\n"
        "These payoffs are *illustrative* and must be recalibrated with your business before "
        "operational use."
    ),
    "monte_carlo": (
        "**Monte Carlo — what the simulation is doing**\n\n"
        "We simulate thousands of possible paths for each counterparty's exposure over the "
        "next 30 business days. Exposure can go up or down with market volatility; volatility "
        "itself mean-reverts (it does not stay high forever). We also allow a mild link "
        "between credit stress and rising exposure (wrong-way risk).\n\n"
        "- **Breach probability**: share of simulated paths where peak exposure exceeds the limit.\n"
        "- **P95 / P99 max EAD**: levels that peak exposure stays below in 95% / 99% of paths.\n"
        "- **Expected shortfall (99%)**: average peak in the worst 1% of paths — useful for "
        "capital and limit discussions.\n\n"
        "A high breach probability does not mean default is likely; it means the *limit* is "
        "likely to be tested under normal market noise."
    ),
    "stress": (
        "**Stress testing — what if markets move sharply?**\n\n"
        "We apply named scenarios (rates up, FX shock, equity sell-off, credit spread "
        "widening, and a combined severe case). Each scenario moves exposure through "
        "simple sensitivity factors, then applies a systemic multiplier.\n\n"
        "- **Incremental EAD**: how much total exposure rises under that scenario.\n"
        "- **Breaches**: how many counterparties would sit over their limit after the shock.\n\n"
        "**Reverse stress**: we ask the opposite question — *how large a uniform increase in "
        "EAD would push 25% of the book over limit?* That multiplier is a simple stress "
        "capacity measure for senior management."
    ),
    "transitions": (
        "**Behaviour transitions — how counterparties change over time**\n\n"
        "Each day we classify a counterparty's behaviour as: reduce exposure, stable, or "
        "increase exposure (from utilisation, volatility and spreads).\n\n"
        "The table and heatmap show the **Markov transition matrix**: the historical "
        "probability of moving from one state to another. For example, if "
        "\"increase → increase\" is high, aggressive counterparties tend to stay aggressive.\n\n"
        "We blend this memory with a statistical model of today's features so the "
        "probabilities used in the game are both data-driven and path-aware."
    ),
    "drilldown": (
        "**Counterparty drill-down**\n\n"
        "Pick one counterparty to see its full profile: size of exposure, breach probability, "
        "early-warning band, behavioural probabilities, expected utilities of each bank "
        "action, the mixed Nash recommendation, and which signals are driving the "
        "early-warning score.\n\n"
        "Use this page in committee discussions: the preferred strategy is the model's "
        "suggestion, not a hard rule — combine it with relationship knowledge and legal "
        "constraints."
    ),
}

METRIC_GLOSSARY = {
    "ead": "Exposure at Default — estimated loss if the counterparty defaults today, after collateral.",
    "pfe": "Potential Future Exposure — possible increase in exposure over the life of the trades.",
    "limit_utilisation": "Current EAD as a percentage of the credit limit. Over 100% means over limit.",
    "anomaly_score": "0–1 score of how unusual recent behaviour is versus the rest of the portfolio.",
    "early_warning_score": "Combined traffic-light score (utilisation, trend, spreads, anomalies, persistence).",
    "breach_probability": "Chance that peak exposure exceeds the limit within the next 30 business days.",
    "preferred_strategy": "Bank action with highest expected utility given estimated counterparty behaviour.",
    "value_of_perfect_info": "Extra utility we would gain if we knew the counterparty type with certainty.",
    "mixed_nash_bank": "Cautious mixed recommendation: weights on keep / cut / collateralise.",
    "p95_max_ead": "Peak exposure level not exceeded in 95% of simulated paths.",
    "p99_max_ead": "Peak exposure level not exceeded in 99% of simulated paths.",
}


def _call_groq(system: str, user: str, max_tokens: int = 500) -> Optional[str]:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return None
    try:
        r = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


SYSTEM_PROMPT = (
    "You are a clear risk-communication assistant for bank stakeholders "
    "(CROs, credit officers, non-quants). Explain counterparty credit risk "
    "metrics in plain British English. Use short paragraphs and bullet points. "
    "Do not invent numbers. Do not use jargon without a one-line plain definition. "
    "Never recommend illegal or non-compliant actions. Be concise (under 180 words)."
)


def explain_page(page_key: str, context: Optional[dict] = None) -> str:
    """
    Return plain-language explanation for a dashboard page.
    Tries Groq; falls back to static PAGE_INTROS.
    """
    base = PAGE_INTROS.get(page_key, "")
    if not os.getenv("GROQ_API_KEY", "").strip():
        return base

    ctx = ""
    if context:
        # keep payload small
        slim = {k: context[k] for k in list(context)[:12]}
        ctx = "\nCurrent snapshot (JSON):\n" + json.dumps(slim, default=str)[:1200]

    user = (
        f"Explain this dashboard section for a non-quant stakeholder.\n"
        f"Section key: {page_key}\n"
        f"Static outline to respect and expand slightly:\n{base}\n"
        f"{ctx}\n"
        "Write a friendly, accurate briefing. Do not repeat the outline word-for-word."
    )
    live = _call_groq(SYSTEM_PROMPT, user, max_tokens=400)
    return live if live else base


def explain_counterparty(row: dict) -> str:
    """Plain-language briefing for one counterparty."""
    static = (
        f"**{row.get('counterparty_id', 'Counterparty')}**\n\n"
        f"- Exposure (EAD): about £{float(row.get('ead', 0))/1e6:.1f}m against a limit of "
        f"£{float(row.get('limit', 0))/1e6:.1f}m "
        f"({float(row.get('limit_utilisation', 0)):.0%} utilised).\n"
        f"- Early-warning band: **{row.get('early_warning_band', 'n/a')}** "
        f"(score {float(row.get('early_warning_score', 0)):.2f}).\n"
        f"- Estimated 30-day chance of testing the limit: "
        f"**{float(row.get('breach_probability', 0)):.1%}**.\n"
        f"- Model's preferred action: **{row.get('preferred_strategy', 'n/a')}** "
        f"(value of better information on their behaviour: "
        f"{float(row.get('value_of_perfect_info', 0)):.2f}).\n\n"
        "Combine this with relationship knowledge and credit policy before acting."
    )
    if not os.getenv("GROQ_API_KEY", "").strip():
        return static

    user = (
        "Write a short stakeholder briefing for this single counterparty. "
        "Use the numbers given; do not invent others.\n"
        + json.dumps(
            {
                k: row.get(k)
                for k in (
                    "counterparty_id",
                    "ead",
                    "limit",
                    "limit_utilisation",
                    "early_warning_band",
                    "early_warning_score",
                    "breach_probability",
                    "preferred_strategy",
                    "probabilities",
                    "value_of_perfect_info",
                    "mixed_nash_bank",
                )
                if k in row
            },
            default=str,
        )
    )
    live = _call_groq(SYSTEM_PROMPT, user, max_tokens=350)
    return live if live else static


def glossary_markdown() -> str:
    lines = ["**Metric glossary (plain language)**\n"]
    for k, v in METRIC_GLOSSARY.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)
