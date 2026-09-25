"""
Stakeholder decision-interpretation layer.

Layer 1 = quantitative metrics (computed elsewhere).
Layer 2 = Finding → Why it matters → Risk implication → Management consideration.

Uses Groq openai/gpt-oss-120b when GROQ_API_KEY is set to polish briefings from
actual model outputs. Always falls back to deterministic, number-aware templates.
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "You are a senior counterparty-credit risk communicator writing for CROs, "
    "credit officers and non-quant stakeholders at a bank. "
    "Use plain British English. Structure every answer exactly as:\n"
    "**Finding:** ...\n"
    "**Why it matters:** ...\n"
    "**Risk implication:** ...\n"
    "**Management consideration:** ...\n"
    "Use only numbers provided in the user message. Do not invent figures. "
    "Do not use unexplained jargon. Keep under 220 words. "
    "Be specific about which counterparties or scenarios drive the result."
)


def _gbp(x: float, scale: str = "auto") -> str:
    x = float(x or 0)
    if scale == "B" or (scale == "auto" and abs(x) >= 1e9):
        return f"£{x/1e9:.2f}B"
    if scale == "M" or (scale == "auto" and abs(x) >= 1e6):
        return f"£{x/1e6:.1f}M"
    return f"£{x:,.0f}"


def _pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * float(x or 0):.{digits}f}%"


def _call_groq(user: str, max_tokens: int = 450) -> Optional[str]:
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.25,
                "max_tokens": max_tokens,
            },
            timeout=35,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _polish(template: str, topic: str, payload: dict) -> str:
    """Optionally refine a number-aware template with Groq."""
    user = (
        f"Topic: {topic}\n"
        f"Draft briefing (preserve all numbers and structure; improve clarity only):\n"
        f"{template}\n\n"
        f"Supporting data (JSON, do not invent beyond this):\n"
        f"{json.dumps(payload, default=str)[:2500]}"
    )
    live = _call_groq(user)
    return live if live else template


# ---------------------------------------------------------------------------
# Portfolio / overview
# ---------------------------------------------------------------------------

def interpret_overview(metrics: dict, counterparties: List[dict]) -> str:
    total_ead = float(metrics.get("total_ead", 0))
    n_cp = int(metrics.get("counterparties", 0) or len(counterparties))
    anomalies = int(metrics.get("anomalies", 0))
    mean_u = float(metrics.get("mean_utilisation", 0))

    high = [c for c in counterparties if c.get("early_warning_band") == "High"]
    mod = [c for c in counterparties if c.get("early_warning_band") == "Moderate"]
    high_ead = sum(float(c.get("ead", 0)) for c in high)
    mod_ead = sum(float(c.get("ead", 0)) for c in mod)
    elevated_share = (high_ead + mod_ead) / total_ead if total_ead else 0.0

    top = sorted(counterparties, key=lambda c: float(c.get("breach_probability", 0)), reverse=True)[:3]
    top_line = "; ".join(
        f"{c.get('counterparty_id')} (util {_pct(c.get('limit_utilisation'))}, "
        f"breach {_pct(c.get('breach_probability'))})"
        for c in top
    ) or "n/a"

    template = (
        f"**Finding:** Portfolio EAD is **{_gbp(total_ead)}** across **{n_cp}** counterparties "
        f"(mean utilisation {_pct(mean_u)}). "
        f"**{len(high)}** names are in the High early-warning band and **{len(mod)}** in Moderate, "
        f"together carrying about **{_pct(elevated_share)}** of total EAD. "
        f"Anomaly flags on the panel: **{anomalies}** observation-days.\n\n"
        f"**Why it matters:** Elevated warning bands concentrate risk where utilisation and/or "
        f"predicted limit pressure are already high — so average book metrics can understate "
        f"where management attention is needed.\n\n"
        f"**Risk implication:** Exposure is not evenly distributed; a minority of names can "
        f"drive most of the near-term limit and collateral risk.\n\n"
        f"**Management consideration:** Prioritise credit review on the highest breach-probability "
        f"names first. Top of the list today: {top_line}."
    )
    payload = {
        "metrics": metrics,
        "n_high": len(high),
        "n_moderate": len(mod),
        "elevated_ead_share": elevated_share,
        "top_breach": [
            {
                "id": c.get("counterparty_id"),
                "util": c.get("limit_utilisation"),
                "breach": c.get("breach_probability"),
                "ead": c.get("ead"),
            }
            for c in top
        ],
    }
    return _polish(template, "portfolio overview", payload)


# ---------------------------------------------------------------------------
# Risk monitor
# ---------------------------------------------------------------------------

def interpret_risk_monitor(counterparties: List[dict]) -> str:
    if not counterparties:
        return "**Finding:** No counterparty data available."

    n = len(counterparties)
    high = [c for c in counterparties if c.get("early_warning_band") == "High"]
    over = [c for c in counterparties if float(c.get("limit_utilisation", 0)) >= 1.0]
    high_breach = [c for c in counterparties if float(c.get("breach_probability", 0)) >= 0.10]
    top = sorted(
        counterparties,
        key=lambda c: (
            0 if c.get("early_warning_band") != "High" else 1,
            float(c.get("breach_probability", 0)),
            float(c.get("limit_utilisation", 0)),
        ),
        reverse=True,
    )[:5]

    evidence = "\n".join(
        f"- **{c.get('counterparty_id')}**: util {_pct(c.get('limit_utilisation'))}, "
        f"30d breach {_pct(c.get('breach_probability'))}, "
        f"EAD {_gbp(c.get('ead'))}, band {c.get('early_warning_band')}, "
        f"preferred action `{c.get('preferred_strategy')}`"
        for c in top
    )

    template = (
        f"**Finding:** Of **{n}** counterparties, **{len(high)}** sit in the High early-warning band, "
        f"**{len(over)}** are already at or above limit utilisation, and **{len(high_breach)}** have "
        f"an estimated ≥10% chance of testing their limit within 30 business days.\n\n"
        f"**Why it matters:** The scatter chart’s top-right quadrant (high utilisation and high "
        f"breach probability) is where limit capacity is most likely to be consumed under ordinary "
        f"market moves — not only under extreme stress.\n\n"
        f"**Risk implication:** These names drive near-term operational risk (limit breaches, "
        f"collateral calls, committee time) even if portfolio average utilisation looks acceptable.\n\n"
        f"**Management consideration:** Investigate the following priority set for limit, "
        f"collateral or relationship action:\n{evidence}"
    )
    payload = {
        "n": n,
        "n_high": len(high),
        "n_over_limit": len(over),
        "n_high_breach": len(high_breach),
        "priority": [
            {
                "id": c.get("counterparty_id"),
                "util": c.get("limit_utilisation"),
                "breach": c.get("breach_probability"),
                "band": c.get("early_warning_band"),
                "strategy": c.get("preferred_strategy"),
            }
            for c in top
        ],
    }
    return _polish(template, "risk monitor", payload)


# ---------------------------------------------------------------------------
# Game theory
# ---------------------------------------------------------------------------

def interpret_game_theory(counterparties: List[dict]) -> str:
    if not counterparties:
        return "**Finding:** No game-theory results available."

    from collections import Counter

    prefs = Counter(c.get("preferred_strategy") or "unknown" for c in counterparties)
    n = len(counterparties)
    dominant, dominant_n = prefs.most_common(1)[0]
    voi_vals = [float(c.get("value_of_perfect_info") or 0) for c in counterparties]
    mean_voi = sum(voi_vals) / max(len(voi_vals), 1)
    high_voi = sorted(counterparties, key=lambda c: float(c.get("value_of_perfect_info") or 0), reverse=True)[:3]

    # Example name for commercial translation
    example = max(
        counterparties,
        key=lambda c: float((c.get("probabilities") or {}).get("increase_exposure", 0)),
    )
    probs = example.get("probabilities") or {}
    eus = example.get("expected_utilities") or {}
    pref = example.get("preferred_strategy")
    p_inc = float(probs.get("increase_exposure", 0))
    p_stable = float(probs.get("stable", 0))
    p_red = float(probs.get("reduce_exposure", 0))

    eu_lines = ", ".join(f"{k}: {float(v):.2f}" for k, v in eus.items()) if eus else "n/a"

    template = (
        f"**Finding:** Across **{n}** counterparties the most frequent preferred action is "
        f"**`{dominant}`** ({dominant_n} names, {_pct(dominant_n / n)} of the book). "
        f"Average value of perfect information (VOI) is **{mean_voi:.2f}** utility units.\n\n"
        f"**Commercial translation (example {example.get('counterparty_id')}):** "
        f"The model estimates behavioural probabilities of increase exposure **{_pct(p_inc)}**, "
        f"stable **{_pct(p_stable)}**, reduce exposure **{_pct(p_red)}**. "
        f"Expected utilities under the assumed payoff structure: {eu_lines}. "
        f"Therefore the preferred pure strategy is **`{pref}`** — not because of a label called "
        f"“Nash”, but because that action has the highest expected payoff given those probabilities "
        f"and the bank’s risk aversion.\n\n"
        f"**Why it matters:** Game theory here is a structured way to turn “what might the "
        f"counterparty do?” into “which of keep limit / cut limit / ask for more collateral "
        f"looks best under our assumptions?”\n\n"
        f"**Risk implication:** Where VOI is high, uncertainty about behaviour is costly — "
        f"better relationship intelligence can change the recommended action.\n\n"
        f"**Management consideration:** For names where the preferred action is `reduce_limit` or "
        f"`increase_collateral`, credit should test whether exposure reduction or extra collateral "
        f"is realistic before behaviour shifts. Highest-VOI names: "
        + ", ".join(
            f"{c.get('counterparty_id')} (VOI {float(c.get('value_of_perfect_info') or 0):.2f})"
            for c in high_voi
        )
        + ". Payoffs are illustrative and must be recalibrated before operational use."
    )
    payload = {
        "preference_counts": dict(prefs),
        "mean_voi": mean_voi,
        "example": {
            "id": example.get("counterparty_id"),
            "probabilities": probs,
            "expected_utilities": eus,
            "preferred": pref,
            "mixed_nash": example.get("mixed_nash_bank"),
        },
    }
    return _polish(template, "game theory commercial interpretation", payload)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def interpret_monte_carlo(counterparties: List[dict]) -> str:
    if not counterparties:
        return "**Finding:** No Monte Carlo results available."

    breach_vals = [float(c.get("breach_probability") or 0) for c in counterparties]
    mean_b = sum(breach_vals) / len(breach_vals)
    high_b = [c for c in counterparties if float(c.get("breach_probability") or 0) >= 0.05]
    # Tail headroom: P99 vs limit
    tight = []
    for c in counterparties:
        lim = float(c.get("limit") or 0) or 1.0
        p99 = float(c.get("p99_max_ead") or 0)
        ead = float(c.get("ead") or 0)
        tight.append(
            {
                "id": c.get("counterparty_id"),
                "ead": ead,
                "p99": p99,
                "limit": lim,
                "p99_util": p99 / lim,
                "headroom_to_p99": lim - p99,
                "breach": float(c.get("breach_probability") or 0),
            }
        )
    tight = sorted(tight, key=lambda x: x["p99_util"], reverse=True)[:5]

    lines = "\n".join(
        f"- **{t['id']}**: current EAD {_gbp(t['ead'])}, P99 peak {_gbp(t['p99'])}, "
        f"limit {_gbp(t['limit'])} → P99 utilisation {_pct(t['p99_util'])}, "
        f"30d breach {_pct(t['breach'])}"
        for t in tight
    )

    template = (
        f"**Finding:** Average estimated 30-day limit-breach probability across the book is "
        f"**{_pct(mean_b)}**. **{len(high_b)}** counterparties have breach probability ≥5%. "
        f"For the tightest names, simulated P99 peak exposure can sit close to or above today’s limit "
        f"even when current EAD looks comfortable.\n\n"
        f"**Why it matters:** Today’s EAD is a point estimate. Monte Carlo shows how far exposure "
        f"can travel under realistic volatility (including mean-reverting vol and mild wrong-way risk). "
        f"P95/P99 are peak levels that are not exceeded in 95%/99% of simulated paths.\n\n"
        f"**Risk implication:** The gap between current EAD and P99 peak is *tail exposure that is "
        f"invisible if you only look at today’s mark*. Limits and collateral must cover that path, "
        f"not only the spot number.\n\n"
        f"**Management consideration:** Ask whether limit and collateral capacity provide enough "
        f"headroom for P99 peak exposure on these names:\n{lines}"
    )
    payload = {"mean_breach": mean_b, "n_high_breach": len(high_b), "tightest": tight}
    return _polish(template, "monte carlo exposure interpretation", payload)


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------

def interpret_stress(stress_payload: dict, metrics: Optional[dict] = None) -> str:
    scenarios = stress_payload.get("scenarios") or stress_payload
    if isinstance(scenarios, dict) and "scenarios" in scenarios:
        scenarios = scenarios["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        return "**Finding:** No stress results available."

    base_ead = float((metrics or {}).get("total_ead") or 0)
    # pick combined / worst by incremental
    ranked = sorted(scenarios, key=lambda s: float(s.get("incremental_ead") or 0), reverse=True)
    worst = ranked[0]
    comb = next((s for s in scenarios if "combined" in str(s.get("scenario", "")).lower()), worst)

    inc = float(comb.get("incremental_ead") or 0)
    stressed = float(comb.get("stressed_ead") or (base_ead + inc))
    breaches = int(comb.get("breaches") or 0)
    breach_rate = float(comb.get("breach_rate") or 0)
    pct_up = (inc / base_ead) if base_ead else 0.0

    rev = stress_payload.get("reverse_stress") or {}
    mult = float(rev.get("ead_multiplier") or 0)
    rev_inc = float(rev.get("implied_incremental_ead") or 0)

    template = (
        f"**Finding:** Under the **{comb.get('scenario')}** scenario, portfolio EAD rises by "
        f"**{_gbp(inc)}**"
        + (f" (**{_pct(pct_up)}** vs current {_gbp(base_ead)})" if base_ead else "")
        + f" to about **{_gbp(stressed)}**. "
        f"**{breaches}** counterparties move above their credit limits"
        + (f" (breach rate {_pct(breach_rate)})" if breach_rate else "")
        + ".\n\n"
        f"**Why it matters:** This is not a forecast; it is a structured what-if on simultaneous "
        f"moves in rates, FX, equity and credit. It shows how much *headroom* the current limit "
        f"structure really has.\n\n"
        f"**Risk implication:** The book’s limit framework is sensitive to multi-factor "
        f"deterioration — a large step-up in EAD plus many names over limit implies operational "
        f"and capital pressure if markets gap.\n\n"
        f"**Management consideration:** Review the counterparties that breach under this scenario "
        f"for collateral, limit or hedging capacity. "
        + (
            f"Reverse-stress capacity: a uniform EAD multiplier of **{mult:.2f}×** "
            f"(about {_gbp(rev_inc)} extra EAD) would put roughly "
            f"{_pct(float(rev.get('target_breach_rate') or 0.25))} of names over limit — "
            f"a simple measure of portfolio headroom."
            if mult
            else "Run reverse stress with the API to quantify headroom."
        )
    )
    payload = {"combined": comb, "reverse_stress": rev, "base_ead": base_ead, "top_scenarios": ranked[:3]}
    return _polish(template, "stress testing capital and limits", payload)


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def interpret_transitions(tm: dict) -> str:
    if not tm:
        return "**Finding:** No transition matrix available."

    # tm is nested dict state -> state -> prob
    stick = {}
    for prev, row in tm.items():
        if isinstance(row, dict) and prev in row:
            stick[prev] = float(row[prev])
    stickiest = max(stick.items(), key=lambda kv: kv[1]) if stick else ("n/a", 0.0)
    most_volatile = min(stick.items(), key=lambda kv: kv[1]) if stick else ("n/a", 0.0)

    inc_to_inc = float((tm.get("increase_exposure") or {}).get("increase_exposure") or 0)
    red_to_red = float((tm.get("reduce_exposure") or {}).get("reduce_exposure") or 0)

    template = (
        f"**Finding:** Behaviour is persistent. Counterparties already in "
        f"**increase_exposure** stay there with probability **{_pct(inc_to_inc)}**; "
        f"those in **reduce_exposure** stay with **{_pct(red_to_red)}**. "
        f"Stickiest state overall: **{stickiest[0]}** ({_pct(stickiest[1])}); "
        f"least sticky: **{most_volatile[0]}** ({_pct(most_volatile[1])}).\n\n"
        f"**Why it matters:** The game-theory layer does not treat behaviour as a one-off coin flip. "
        f"Markov memory means today’s type is informative about tomorrow — so early warnings "
        f"compound if aggressive behaviour tends to continue.\n\n"
        f"**Risk implication:** Names currently classified as increasing exposure are more likely "
        f"to keep pressing limits; waiting for “mean reversion” is not supported by the empirical "
        f"transition rates.\n\n"
        f"**Management consideration:** For high-utilisation names already in increase_exposure, "
        f"accelerate limit/collateral discussion rather than assuming they will self-correct."
    )
    return _polish(template, "behaviour transitions", {"matrix": tm, "stickiness": stick})


# ---------------------------------------------------------------------------
# Single counterparty drill-down
# ---------------------------------------------------------------------------

def interpret_counterparty(row: dict) -> str:
    cid = row.get("counterparty_id", "Counterparty")
    ead = float(row.get("ead") or 0)
    lim = float(row.get("limit") or 0)
    util = float(row.get("limit_utilisation") or 0)
    band = row.get("early_warning_band", "n/a")
    ew = float(row.get("early_warning_score") or 0)
    breach = float(row.get("breach_probability") or 0)
    p99 = float(row.get("p99_max_ead") or 0)
    pref = row.get("preferred_strategy", "n/a")
    voi = float(row.get("value_of_perfect_info") or 0)
    probs = row.get("probabilities") or {}
    eus = row.get("expected_utilities") or {}
    contrib = row.get("early_warning_contributions") or {}

    p_line = ", ".join(f"{k} {_pct(v)}" for k, v in probs.items()) or "n/a"
    eu_line = ", ".join(f"{k} {float(v):.2f}" for k, v in eus.items()) or "n/a"
    top_driver = max(contrib.items(), key=lambda kv: float(kv[1]))[0] if contrib else "n/a"

    headroom = lim - p99
    template = (
        f"**Finding:** **{cid}** has EAD **{_gbp(ead)}** against limit **{_gbp(lim)}** "
        f"(utilisation **{_pct(util)}**). Early-warning band **{band}** (score {ew:.2f}); "
        f"30-day limit-breach probability **{_pct(breach)}**. "
        f"Simulated P99 peak EAD **{_gbp(p99)}** "
        f"({'above' if p99 > lim else 'within'} limit; headroom {_gbp(headroom)}).\n\n"
        f"**Behaviour & decision:** Estimated type probabilities: {p_line}. "
        f"Expected utilities of bank actions: {eu_line}. "
        f"Preferred action under current assumptions: **`{pref}`**. "
        f"Value of knowing the true type with certainty (VOI): **{voi:.2f}**.\n\n"
        f"**Why it matters:** Spot EAD alone understates path risk; breach probability and P99 "
        f"show whether the limit is likely to be tested. The preferred strategy is the "
        f"highest-expected-utility choice given those behavioural odds — a commercial prompt "
        f"for limit vs collateral vs monitor, not an automatic order.\n\n"
        f"**Risk implication:** Primary early-warning driver is **{top_driver}**. "
        f"If utilisation and breach probability are both elevated, operational limit pressure "
        f"can appear before a credit event.\n\n"
        f"**Management consideration:** Credit/relationship owners should validate whether "
        f"`{pref}` is feasible (exposure reduction, collateral, or documented decision to maintain), "
        f"and whether P99 headroom is acceptable under policy."
    )
    payload = {
        "id": cid,
        "ead": ead,
        "limit": lim,
        "util": util,
        "band": band,
        "breach": breach,
        "p99": p99,
        "preferred": pref,
        "voi": voi,
        "probabilities": probs,
        "expected_utilities": eus,
        "contributions": contrib,
    }
    return _polish(template, f"counterparty briefing {cid}", payload)


# ---------------------------------------------------------------------------
# Methodology gloss (secondary expander)
# ---------------------------------------------------------------------------

PAGE_METHOD = {
    "overview": "Combines exposure, warnings, behaviour probabilities and decision utilities into one portfolio view.",
    "risk_monitor": "Table and scatter use utilisation, anomaly ensemble, early-warning score and Monte Carlo breach probability.",
    "game_theory": "Bayesian expected utilities under CARA; mixed Nash / maximin for robustness; VOI on counterparty type.",
    "monte_carlo": "GBM exposure paths with OU stochastic volatility and wrong-way correlation; peak EAD distribution.",
    "stress": "Multi-factor sensitivities plus systemic multiplier; reverse-stress EAD multiplier for target breach rate.",
    "transitions": "Empirical first-order Markov matrix of behavioural states, blended with multinomial logit.",
    "drilldown": "Full stack for one name: warnings, MC peaks, behaviour posterior, game utilities, strategy.",
}

METRIC_GLOSSARY = {
    "ead": "Exposure at Default — estimated loss if the counterparty defaults today, after collateral.",
    "limit_utilisation": "Current EAD as a share of the credit limit. ≥100% means already over limit.",
    "anomaly_score": "How unusual recent behaviour is versus the rest of the book (higher = more unusual).",
    "early_warning_band": "Traffic light (Low / Moderate / High) from utilisation, trend, spreads, anomalies, persistence.",
    "breach_probability": "Share of 30-day simulated paths where peak exposure exceeds the limit.",
    "preferred_strategy": "Bank action with highest expected utility given estimated counterparty behaviour.",
    "value_of_perfect_info": "Extra utility if we knew the counterparty’s true behavioural type with certainty.",
    "p99_max_ead": "Peak exposure level not exceeded in 99% of simulated paths.",
    "mixed_nash_bank": "Robust mix of keep / cut / collateralise if the other side played against us.",
}


def glossary_markdown() -> str:
    lines = ["**Metric glossary**\n"]
    for k, v in METRIC_GLOSSARY.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


# Backwards-compatible aliases used by older streamlit imports
def explain_page(page_key: str, context: Optional[dict] = None) -> str:
    return PAGE_METHOD.get(page_key, "Quantitative CCR decision-support view.")


def explain_counterparty(row: dict) -> str:
    return interpret_counterparty(row)


PAGE_INTROS = PAGE_METHOD
