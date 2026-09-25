import numpy as np, pandas as pd
from .config import DATA_DIR, DAILY_FILE, TRADES_FILE

def generate_portfolio(n_trades=5000,n_days=180,seed=42):
    r=np.random.default_rng(seed)
    cps=[f"CP{i:04d}" for i in range(1,81)]
    entities=["JPM_EMEA_UK","JPM_EMEA_DE","JPM_EMEA_FR","JPM_EMEA_CH"]
    master=pd.DataFrame({"counterparty_id":cps,
        "counterparty_type":r.choice(["Bank","Hedge Fund","Asset Manager","Corporate","Sovereign","CCP"],len(cps)),
        "rating":r.choice(["AAA","AA","A","BBB","BB"],len(cps),p=[.08,.22,.35,.27,.08]),
        "credit_limit":r.uniform(20e6,150e6,len(cps))})
    products=r.choice(["Interest Rate Swap","FX Forward","Equity Option","Commodity Future","Securities Financing"],n_trades,p=[.30,.25,.18,.12,.15])
    ac=pd.Series(products).map({"Interest Rate Swap":"Interest Rate","FX Forward":"FX","Equity Option":"Equity","Commodity Future":"Commodity","Securities Financing":"SFT"}).to_numpy()
    af=pd.Series(ac).map({"Interest Rate":.005,"FX":.04,"Equity":.08,"Commodity":.15,"SFT":.03}).to_numpy()
    notional=r.lognormal(np.log(4e6),1,n_trades); mtm=r.normal(0,notional*.035)
    collateral=np.maximum(r.normal(notional*.012,notional*.008),0)
    margin=r.choice(["Margined","Unmargined"],n_trades,p=[.72,.28]); collateral=np.where(margin=="Margined",collateral,0)
    trades=pd.DataFrame({"trade_id":[f"T{i:07d}" for i in range(n_trades)],"counterparty_id":r.choice(cps,n_trades),
        "legal_entity":r.choice(entities,n_trades),"product":products,"asset_class":ac,"notional":notional,
        "mtm":mtm,"collateral":collateral,"margining":margin,"maturity_years":r.uniform(.05,7,n_trades),
        "add_on_factor":af}).merge(master,on="counterparty_id")
    rows=[]
    for d in pd.date_range("2025-01-01",periods=n_days):
        vol=max(.6,1+r.normal(0,.12))
        for i,cp in enumerate(cps):
            lim=float(master.loc[master.counterparty_id.eq(cp),"credit_limit"].iloc[0])
            ead=float(np.exp(r.normal(np.log(35e6),.35))*vol)
            pfe=ead*r.uniform(.35,.85); coll=ead*r.uniform(.15,.70)
            rows.append({"date":d,"counterparty_id":cp,"legal_entity":entities[i%4],"ead":ead,"pfe":pfe,
                "collateral":coll,"credit_limit":lim,"limit_utilisation":ead/lim,
                "market_volatility":vol,"fx_return":r.normal(0,.01),"rate_change":r.normal(0,.004),
                "equity_return":r.normal(0,.012),"credit_spread_change":r.normal(0,.01)})
    daily=pd.DataFrame(rows)
    latent=2.4*daily.limit_utilisation+1.2*daily.market_volatility+1.5*daily.credit_spread_change.abs()-daily.collateral.div(daily.ead).clip(0,1)
    daily["behaviour_state"]=pd.qcut(latent.rank(method="first"),3,labels=["reduce_exposure","stable","increase_exposure"]).astype(str)
    DATA_DIR.mkdir(exist_ok=True); trades.to_csv(TRADES_FILE,index=False); daily.to_csv(DAILY_FILE,index=False)
    return trades,daily
def load_data():
    if not DAILY_FILE.exists(): generate_portfolio()
    return pd.read_csv(TRADES_FILE),pd.read_csv(DAILY_FILE,parse_dates=["date"])
