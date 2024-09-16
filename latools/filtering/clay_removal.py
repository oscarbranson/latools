import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import zscore

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def linear(x, m, c):
    return m * x + c

def clay_removal(data_dict):
    
    dat = pd.DataFrame.from_dict(data_dict)
    clay_tracers = list(dat.columns)
    
    sub = dat.copy().dropna()

    # calculate clay score
    sub['clay'] = zscore(sub).mean(axis=1)

    # sort by clay score
    ssub = sub.dropna().sort_values('clay', ascending=True)
    
    # calculate cumulative mean
    msub = ssub.cumsum() / np.arange(1, len(ssub)+1).reshape(-1,1)
    
    # fit piecewise linear
    changepoints = []
    for c in clay_tracers:
        mp, mcov = curve_fit(piecewise_linear, msub['clay'].values, msub[c].values, p0=[msub['clay'].mean(), 0, 0, 0])

        # check that slopes are sufficiently different
        if 0.8 < mp[-2] / mp[-1] < 1.2:
            continue
        
        changepoints.append(mp[0])
    
    if len(changepoints) == 0:
        return np.ones_like(dat.index, dtype=bool)
        
    # calculate average changepoint
    changepoint = np.mean(changepoints)
    
    ind = msub['clay'] < changepoint

    dat['filt'] = False
    dat.loc[ind.index[ind], 'filt'] = True
        
    return dat['filt'].values