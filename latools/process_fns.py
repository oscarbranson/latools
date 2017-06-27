import os
import re
import numpy as np
import warnings

from io import BytesIO
from latools.helpers import Bunch


# Functions to work with laser ablation signals

# Despiking functions
def noise_despike(sig, win=3, nlim=24., maxiter=4):
    """
    Apply standard deviation filter to remove anomalous values.

    Parameters
    ----------
    win : int
        The window used to calculate rolling statistics.
    nlim : float
        The number of standard deviations above the rolling
        mean above which data are considered outliers.

    Returns 
    -------
    None
    """
    if win % 2 != 1:
        win += 1  # win must be odd

    kernel = np.ones(win) / win  # make convolution kernel
    over = np.ones(len(sig), dtype=bool)  # initialize bool array
    # pad edges to avoid edge-effects
    npad = int((win - 1) / 2)
    over[:npad] = False
    over[-npad:] = False
    # set up monitoring
    nloops = 0
    # do the despiking
    while any(over) and (nloops < maxiter):
        rmean = np.convolve(sig, kernel, 'valid')  # mean by convolution
        rstd = rmean**0.5  # std = sqrt(signal), because count statistics
        # identify where signal > mean + std * nlim (OR signa < mean - std * nlim)
        over[npad:-npad] = (sig[npad:-npad] > rmean + nlim * rstd)  # | (sig[npad:-npad] < rmean - nlim * rstd)
        # if any are over, replace them with mean of neighbours
        if any(over):
            # replace with values either side
            # sig[over] = sig[np.roll(over, -1) | np.roll(over, 1)].reshape((sum(over), 2)).mean(1)
            # replace with mean
            sig[npad:-npad][over[npad:-npad]] = rmean[over[npad:-npad]]
            nloops += 1
        # repeat until no more removed.
    return sig


def expdecay_despike(sig, expdecay_coef, tstep, maxiter=3, silent=True):
    """
    THERE'S SOMETHING WRONG WITH THIS FUNCTION. REMOVES TOO MUCH DATA!

    Apply exponential decay filter to remove unrealistically low values.

    Parameters
    ----------
    exponent : float
        Exponent used in filter
    tstep : float
        The time increment between data points.
    maxiter : int
        The maximum number of iterations to
        apply the filter

    Returns
    -------
    None
    """
    lo = np.ones(len(sig), dtype=bool)  # initialize bool array
    nloop = 0  # track number of iterations
    # do the despiking
    while any(lo) and (nloop <= maxiter):
        # find values that are lower than allowed by the washout
        # characteristics of the laser cell.
        lo = sig < np.roll(sig * np.exp(expdecay_coef * tstep), 1)
        if any(lo):
            prev = sig[np.roll(lo, -1)]
            sig[lo] = prev
            nloop += 1

    if nloop >= maxiter and not silent:
        raise warnings.warn(('\n***maxiter ({}) exceeded during expdecay_despike***\n\n'.format(maxiter) +
                             'This is probably because the expdecay_coef is too small.\n'))
    return sig


def read_data(data_file, dataformat, name_mode):
    with open(data_file) as f:
        lines = f.readlines()

    if 'meta_regex' in dataformat.keys():
        meta = Bunch()
        for k, v in dataformat['meta_regex'].items():
            out = re.search(v[-1], lines[int(k)]).groups()
            for i in np.arange(len(v[0])):
                meta[v[0][i]] = out[i]

    # sample name
    if name_mode == 'file_names':
        sample = os.path.basename(data_file).split('.')[0]
    elif name_mode == 'metadata_names':
        sample = meta['name']
    else:
        sample = 0

    # column and analyte names
    columns = np.array(lines[dataformat['column_id']['name_row']].strip().split(dataformat['column_id']['delimiter']))
    if 'pattern' in dataformat['column_id'].keys():
        pr = re.compile(dataformat['column_id']['pattern'])
        analytes = [pr.match(c).groups()[0] for c in columns if pr.match(c)]

    # do any required pre-formatting
    if 'preformat_replace' in dataformat.keys():
        with open(data_file) as f:
                fbuffer = f.read()
        for k, v in dataformat['preformat_replace'].items():
            fbuffer = re.sub(k, v, fbuffer)
        # dead data
        read_data = np.genfromtxt(BytesIO(fbuffer.encode()),
                                  **dataformat['genfromtext_args']).T
    else:
        # read data
        read_data = np.genfromtxt(data_file,
                                  **dataformat['genfromtext_args']).T

    # data dict
    dind = np.ones(read_data.shape[0], dtype=bool)
    dind[dataformat['column_id']['timecolumn']] = False

    data = Bunch()
    data['Time'] = read_data[dataformat['column_id']['timecolumn']]
    data['rawdata'] = Bunch(zip(analytes, read_data[dind]))
    data['total_counts'] = read_data[dind].sum(0)

    return sample, analytes, data, meta
