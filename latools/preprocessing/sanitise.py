import pandas as pd
import numpy as np

def Neptune_xlsx(file):
    raw = pd.read_excel(file, skiprows=16)
    
    keep = []
    for i, row in raw.iterrows():
        if isinstance(row['Cycle'], int):
            keep.append(i)
    
    dat = raw.loc[keep]
    dat = dat.dropna()
    
    types = {k: float for k in dat.columns}
    types['Cycle'] = int
    types['Time'] = str
    dat = dat.astype(types)
    
    # convert timescale to seconds
    timescale = pd.to_datetime(dat['Time'], format='%H:%M:%S:%f')
    timedelta = timescale - timescale.min()

    day_transition = np.argwhere(np.diff(timedelta).astype(float) < 0).flatten()
    if day_transition.size > 0:
        for d in day_transition:
            timedelta[d + 1:] += pd.Timedelta('1 day')

    seconds = timedelta.dt.total_seconds()
    seconds -= seconds.min()
    
    dat['Time'] = seconds
    del dat['Cycle']
    
    outpath = file.replace('xlsx', 'csv')
    dat.to_csv(outpath, index=False)
    
    return f"Saved to: {outpath}"