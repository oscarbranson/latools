import numpy as np
import pandas as pd

from ..processes import read_data
from ..helpers.signal import bool_transitions

def parse(csv_file, dataformat, laserlog_file, on_pad=[5, 3], off_pad=[3, 5], align_bkg=10, align_nstd=12):
    
    _, _, dat, meta = read_data(csv_file, dataformat=dataformat, name_mode='file')
    
    laserlog = pd.read_csv(laserlog_file, parse_dates=['Timestamp'])
    laserlog.columns = [s.strip() for s in laserlog.columns]
    sample_names = laserlog.Comment.dropna().values
    
    laserlog.Comment.ffill(inplace=True)
    
    # calculate seconds
    laserlog['orig_seconds'] = (laserlog.Timestamp - laserlog.Timestamp.min()).dt.seconds
    
    # align data and laserlog
    
    # identify inital background mean and std
    ind = dat.Time < align_bkg
    mu_init = dat.total_counts[ind].mean()
    std_init = dat.total_counts[ind].std()
    
    # identify time of first signal arrival
    first_signal_ind = np.argwhere(dat.total_counts > mu_init + align_nstd * std_init)[0][0]
    first_signal_seconds = dat.Time[first_signal_ind - 1]
    
    # identify time of first laser-on
    first_laser_on_seconds = laserlog.loc[laserlog['Laser State'] == 'On', 'orig_seconds'].min()
    
    time_offset = first_signal_seconds - first_laser_on_seconds
    
    laserlog['seconds'] = laserlog['orig_seconds'] + time_offset
    
    dat.laserlog = laserlog  # save to analysis object
    
    # separate signal, background and transitions
    sig = np.zeros(dat.Time.shape, dtype=bool)
    
    laseron = False
    for i, row in laserlog.iterrows():
        if laseron:
            sig[dat.Time > row.seconds] = False
            laseron = False
        if row['Laser State'] == 'On':
            sig[dat.Time > row.seconds] = True
            laseron = True

    bkg = ~sig
        
    # remove transitions    
    trn_inds = bool_transitions(sig)
    trn = np.zeros(bkg.shape, dtype=bool)

    # get split locations
    split_inds = []
    for i in range(1, len(trn_inds) - 1, 2):
        split_inds.append((trn_inds[i] + trn_inds[i+1]) // 2)

    trn_on = True  # first transition is always 'on'
    for t in dat.Time[trn_inds]:
        if trn_on:
            trn[(dat.Time >= t - on_pad[0]) & (dat.Time <= t + on_pad[1])] = True
            trn_on = False
        else:    
            trn[(dat.Time >= t - off_pad[0]) & (dat.Time <= t + off_pad[1])] = True
            trn_on = True

    sig = sig & ~trn
    bkg = bkg & ~trn 

    # return D_obj with ranges
    sections = {}
    for i, s in enumerate(sample_names):
        if i == 0:
            ind = dat.Time < dat.Time[split_inds[0]]
        elif i == len(sample_names) - 1:
            ind = dat.Time >= dat.Time[split_inds[-1]]
        else:
            ind = (dat.Time >= dat.Time[split_inds[i-1]]) & (dat.Time < dat.Time[split_inds[i]])

            
        if s in sections:
            s += '.1'
        
        sections[s] = {
            'Time': dat.Time[ind],
            'rawdata': {k: v[ind] for k, v in dat.rawdata.items()},
            'total_counts': dat.total_counts[ind],
            'bkg': bkg,
            'sig': sig,
            'trn': trn,
        }
    
    analytes = set(dat.rawdata.keys())

    for sample, data in sections.items():
        yield '', sample, analytes, data, meta