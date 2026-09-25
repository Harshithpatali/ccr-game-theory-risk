import numpy as np

def breach_probability(ead, limit, vol=.20, credit_vol=.01, corr=.45, n=30000, days=30, seed=42):
    rng=np.random.default_rng(seed); dt=1/252
    z1=rng.normal(size=(n,days)); z2=rng.normal(size=(n,days))
    zc=corr*z1+np.sqrt(1-corr**2)*z2
    sigma=max(float(vol),.03); cs=max(float(credit_vol),.001)
    log_inc=(-.5*sigma**2)*dt + sigma*np.sqrt(dt)*z1 + cs*np.sqrt(dt)*zc
    paths=float(ead)*np.exp(np.cumsum(log_inc,axis=1)); mx=paths.max(axis=1)
    return {"breach_probability":float(np.mean(mx>limit)),"p95_max_ead":float(np.quantile(mx,.95)),
            "p99_max_ead":float(np.quantile(mx,.99)),"expected_max_ead":float(mx.mean()),
            "expected_shortfall_99":float(mx[mx>=np.quantile(mx,.99)].mean()),"n_simulations":n}
