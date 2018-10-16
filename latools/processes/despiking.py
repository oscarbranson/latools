import numpy as np

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
        # identify where signal > mean + std * nlim (OR signa < mean - std *
        # nlim)
        # | (sig[npad:-npad] < rmean - nlim * rstd)
        over[npad:-npad] = (sig[npad:-npad] > rmean + nlim * rstd)
        # if any are over, replace them with mean of neighbours
        if any(over):
            # replace with values either side
            # sig[over] = sig[np.roll(over, -1) | np.roll(over, 1)].reshape((sum(over), 2)).mean(1)
            # replace with mean
            sig[npad:-npad][over[npad:-npad]] = rmean[over[npad:-npad]]
            nloops += 1
        # repeat until no more removed.
    return sig


def expdecay_despike(sig, expdecay_coef, tstep, maxiter=3):
    """
    Apply exponential decay filter to remove physically impossible data based on instrumental washout.

    The filter is re-applied until no more points are removed, or maxiter is reached.

    Parameters
    ----------
    exponent : float
        Exponent used in filter
    tstep : float
        The time increment between data points.
    maxiter : int
        The maximum number of times the filter should be applied.

    Returns
    -------
    None
    """
    # determine rms noise of data
    noise = np.std(sig[:5])  # initially, calculated based on first 5 points
    # expand the selection up to 50 points, unless it dramatically increases 
    # the std (i.e. catches the 'laser on' region)
    for i in [10, 20, 30, 50]:
        inoise = np.std(sig[:i])
        if inoise < 1.5 * noise:
            noise = inoise
    rms_noise3 = 3 * noise

    i = 0
    f = True
    while (i < maxiter) and f:
        # calculate low and high possibles values based on exponential decay
        siglo = np.roll(sig * np.exp(tstep * expdecay_coef), 1)
        sighi = np.roll(sig * np.exp(-tstep * expdecay_coef), -1)

        # identify points that are outside these limits, beyond what might be explained
        # by noise in the data
        loind = (sig < siglo - rms_noise3) & (sig < np.roll(sig, -1) - rms_noise3)
        hiind = (sig > sighi + rms_noise3) & (sig > np.roll(sig, 1) + rms_noise3)

        # replace all such values with their preceding
        sig[loind] = sig[np.roll(loind, -1)]
        sig[hiind] = sig[np.roll(hiind, -1)]

        f = any(np.concatenate([loind, hiind]))
        i += 1

    return sig