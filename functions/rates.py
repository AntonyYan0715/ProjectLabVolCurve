import numpy as np
import pandas as pd
import statsmodels.api as sm


def syn_forward(df):
    """Call - Put = Synthetic Future."""
    ret = pd.DataFrame()
    ranks = df["RANK"].unique()
    ts = df["TS"].values[0]
    exp = df["EXP"].values[0]
    bus_days = df["BUS_DAYS"].values[0]
    cal_days = df["CAL_DAYS"].values[0]
    S_0 = df["UNDERLYING_PRICE"].values[0]
    ser_list = []
    
    for r in ranks:
        try:
            slice = df[df["RANK"]==r]
            row = pd.Series(dtype=float)

            row["TS"]       = ts
            row["RANK"]     = r
            row["STRIKE"]   = slice["STRIKE"].values[0]
            row["EXP"]      = exp
            row["BUS_DAYS"] = bus_days
            row["CAL_DAYS"] = cal_days
            row["UNDERLYING_PRICE"] = S_0
            row["ACTIVE"] = (slice.loc[slice["TYPE"]=="C", "ASK_CLOSE"].values[0] 
                               - slice.loc[slice["TYPE"]=="P", "BID_CLOSE"].values[0])
            row["PASSIVE"] = (slice.loc[slice["TYPE"]=="C", "BID_CLOSE"].values[0] 
                               - slice.loc[slice["TYPE"]=="P", "ASK_CLOSE"].values[0])
            row["MID"] = (row["PASSIVE"] + row["ACTIVE"]) / 2
            row["VOLUME"] = (slice.loc[slice["TYPE"]=="C", "VOLUME"].values[0] 
                               + slice.loc[slice["TYPE"]=="P", "VOLUME"].values[0])
            row["OPEN_INT"] = (slice.loc[slice["TYPE"]=="C", "OPEN_INT"].values[0] 
                               + slice.loc[slice["TYPE"]=="P", "OPEN_INT"].values[0])

            ser_list.append(row)
        except:
            pass
    
    ret = pd.concat(ser_list, axis=1)
        
    return ret.T

def reg_imp_rate(df, side="ACTIVE"):
    moneyness = np.abs(df["STRIKE"] - df["UNDERLYING_PRICE"])
    moneyness = ((moneyness.max()-moneyness))

    close = df["UNDERLYING_PRICE"].values[0]
    
    endo = df[side]
    exog = sm.add_constant(df["STRIKE"])

    res_ols = sm.OLS(endo, exog).fit()
    res_wls_oi = sm.WLS(endo, exog, weights=df["OPEN_INT"]).fit()
    res_wls_vo = sm.WLS(endo, exog, weights=df["VOLUME"]).fit()
    res_wls_mo = sm.WLS(endo, exog, weights=moneyness).fit()

    keys = ["OLS", "Open_Int", "Volume", "Moneyness"]

    T = df["CAL_DAYS"].values[0] / 360
    params = pd.concat([res_ols.params, res_wls_oi.params, 
                        res_wls_vo.params, res_wls_mo.params], 
                       keys=keys, axis=1)
    params.rename(index={"const":"F_0"}, inplace=True)
    params.loc["RATE",:] = -np.log(-params.loc["STRIKE",:]) / T
    params.loc["F_T",:]  = -params.loc["F_0",:] / params.loc["STRIKE",:]
    return params

# Below: deprecated

def put_call_imp_rates(df):
    """Uses each Synthetic Future struck at K to get the rate implied
    using Put-Call Parity."""
    ret = pd.DataFrame()
    strikes = df["STRIKE"].unique()
    ts = df["TS"].values[0]
    exp = df["EXP"].values[0]
    exp_days = df["EXP_DAYS"].values[0]
    S_0 = df["UNDERLYING_PRICE"].values[0]
    exp_t = df["EXP_DAYS"].values[0] / 252
    
    for K in strikes:
        slice = df[df["STRIKE"]==K]
        row = pd.Series(dtype=float)
        
        row["TS"]     = ts
        row["RANK"]   = slice["RANK"].values[0]
        row["STRIKE"] = K
        row["EXP"]    = exp
        row["EXP_DAYS"] = exp_days
        row["UNDERLYING_PRICE"] = S_0
        row["BID_CLOSE"] = -np.log((S_0 - slice["BID_CLOSE"].values[0]) / K) / exp_t
        row["ASK_CLOSE"] = -np.log((S_0 - slice["ASK_CLOSE"].values[0]) / K) / exp_t
        ret[K] = row
        
    return ret.T

def rate_from_futures(S_0, EXP, V_LO, K_LO, V_HI, K_HI):
    """Computes the rate implied by two futures"""
    return np.log((K_HI-((K_HI-K_LO)*V_HI/(V_HI-V_LO)))/S_0)/EXP

def two_fut_imp_rates(df):
    """Gathers Futures that sandwich the underlying price. Computes implied
    rates from pairs with strikes closest to the underlying."""
    ret = pd.DataFrame()
    ranks = df["RANK"].unique()
    ts = df["TS"].values[0]
    exp = df["EXP"].values[0]
    exp_days = df["EXP_DAYS"].values[0]
    S_0 = df["UNDERLYING_PRICE"].values[0]
    exp_t = df["EXP_DAYS"].values[0] / 252
    
    for i in range(len(ranks)//2):
        near = df[df["RANK"]==(2*i)]
        far = df[df["RANK"]==(2*i+1)]
        if (near["STRIKE"].values[0] > far["STRIKE"].values[0]):
            lo = far; hi = near
        else: lo = near; hi = far
        lo_b = lo["BID_CLOSE"].values[0]
        lo_a = lo["ASK_CLOSE"].values[0]
        lo_K = lo["STRIKE"].values[0]
        hi_b = hi["BID_CLOSE"].values[0]
        hi_a = hi["ASK_CLOSE"].values[0]
        hi_K = hi["STRIKE"].values[0]
        row  = pd.Series(dtype=float)
        
        row["TS"]        = ts
        row["RANK"]      = i
        row["STRIKE_LO"] = lo["STRIKE"].values[0]
        row["STRIKE_HI"] = hi["STRIKE"].values[0]
        row["EXP"]       = exp
        row["EXP_DAYS"] = exp_days
        row["UNDERLYING_PRICE"] = S_0
        row["BID_CLOSE"] = rate_from_futures(S_0, exp_t, lo_b, lo_K, hi_a, hi_K)
        row["ASK_CLOSE"] = rate_from_futures(S_0, exp_t, lo_a, lo_K, hi_a, hi_K)
        ret[i] = row
        
    return ret.T