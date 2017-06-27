import numpy as np
import warnings


# Functions to work with laser ablation signals
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
            sig[npad:-npad][over[npad:-npad]] = rmean[over[npad:-npad]]
            nloops += 1
        # repeat until no more removed.
    return sig


def expdecay_despike(sig, expdecay_coef, tstep, maxiter=3, silent=True):
    """
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
