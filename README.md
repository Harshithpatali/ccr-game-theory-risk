# CCR Quantitative Risk & Bayesian Game Theory

Portfolio / educational project that layers **rigorous applied mathematics and statistics** on synthetic counterparty credit risk (CCR) data.

## Research question
Can probabilistic modelling and game-theoretic analysis improve counterparty risk assessment by jointly modelling the likelihood of adverse exposure events **and** the strategic responses of counterparties and risk managers?

## Mathematics & statistics (v2)

1. **Exposure (SA-CCR style)**  
   Replacement cost RC, supervisory add-on with maturity scaling, multiplier, EAD = α(RC + PFE), α = 1.4.

2. **Statistical features & anomaly detection**  
   - Rolling %Δ EAD, collateral coverage, EWMA volatility, utilisation autocorrelation  
   - Ensemble: Isolation Forest + Local Outlier Factor + robust Mahalanobis (MinCovDet)  
   - Logistic mapping of the ensemble z-score to an anomaly probability

3. **Behavioural probabilities**  
   - Multinomial logistic regression on the feature set  
   - Empirical first-order Markov transition matrix of behavioural state  
   - Convex blend (logit + Markov) → Bayesian posterior over types

4. **Game theory (Bayesian game of incomplete information)**  
   - Bank strategies: maintain limit / reduce limit / increase collateral  
   - Counterparty types: reduce / stable / increase exposure  
   - Monetary payoff matrix calibrated to breach loss, opportunity cost, franchise value  
   - **CARA utility** (constant absolute risk aversion)  
   - Pure-strategy Bayesian expected utility  
   - **Mixed-strategy Nash** via support enumeration + indifference conditions  
   - **Value of perfect information (VOI)** on counterparty type

5. **Monte Carlo**  
   - Geometric Brownian motion for exposure with **Ornstein–Uhlenbeck stochastic volatility**  
   - Correlated Brownian drivers (wrong-way risk parameter ρ)  
   - P(max EAD > limit), P95/P99 peak exposure, expected shortfall of the peak

6. **Stress testing**  
   - Multi-factor linear sensitivities (rates, FX, equity, credit) + systemic multiplier  
   - Incremental EAD and breach counts  
   - Reverse-stress EAD multiplier that produces a target portfolio breach rate

7. **Early-warning composite**  
   Weighted logistic score of anomaly, utilisation, trend, spread and persistence with explicit contribution decomposition.

## Dataset
Synthetic longitudinal panel (real bank CCR data is confidential):
- 5 000 trades, 80 counterparties, 4 EMEA legal entities  
- 180 daily observations per counterparty  
- MTM, collateral, notional, maturity, PFE/EAD, credit limits  
- Market factors and a latent behavioural state for model training

## Stakeholder explanations
Every dashboard tab opens with a plain-language guide (what the page means, what to look at first).
Optional live briefings use **Groq** model `openai/gpt-oss-120b` when you set:

```bash
# in .env
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b
```

Without a key, curated static explanations are always shown.

## Run
```bash
pip install -r requirements.txt
python -m scripts.generate_data
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Second terminal:
```bash
streamlit run app/streamlit_app.py --server.port 8501
```
Or:
```bash
docker compose up --build
```

## API highlights
- `GET /counterparties` – full risk + game + MC + early-warning panel  
- `GET /transitions` – Markov behaviour transition matrix  
- `GET /stress` – multi-factor scenarios + reverse stress  
- `GET /counterparty/{id}` – drill-down

## Interview positioning
"My background is applied mathematics. I built a quantitative decision-support layer over CCR data that uses ensemble anomaly detection, multinomial logit + Markov behavioural modelling, multi-factor Monte Carlo with stochastic volatility, and a Bayesian game with CARA utilities and mixed-strategy Nash equilibria to evaluate alternative risk-management responses and the value of better information about counterparty type."

This is a portfolio / educational project, **not** a regulatory capital implementation. Payoffs and betas are illustrative and must be recalibrated with domain experts before any operational use.
