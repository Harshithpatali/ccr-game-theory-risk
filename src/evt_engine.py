import numpy as np
from scipy.stats import genpareto

def fit_evt(excesses, threshold_quantile=.95):
    x=np.asarray(excesses,dtype=float); x=x[np.isfinite(x)]
    u=float(np.quantile(x,threshold_quantile)); excess=x[x>u]-u
    if len(excess)<30:
        var99=float(np.quantile(x,.99)); tail=x[x>=var99]
        es=float(tail.mean()) if len(tail) else var99
        return {"threshold":u,"shape":None,"scale":None,"tail_count":int(len(excess)),"var_99":var99,"es_99":es}
    shape,loc,scale=genpareto.fit(excess,floc=0)
    n=len(x); nu=len(excess); p=.99; tail_prob=nu/n
    if abs(shape)>1e-8: q=u + scale/shape*((tail_prob/(1-p))**shape-1)
    else: q=u + scale*np.log(tail_prob/(1-p))
    es=(q + (scale-shape*u)/(1-shape)) if shape<1 else np.inf
    return {"threshold":u,"shape":float(shape),"scale":float(scale),"tail_count":int(nu),"var_99":float(q),"es_99":float(es)}
