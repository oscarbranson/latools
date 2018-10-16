import numpy as np
import uncertainties.unumpy as un
from scipy.stats import pearsonr

def nan_pearsonr(x, y):
    xy = np.vstack([x, y])
    xy = xy[:, ~np.any(np.isnan(xy),0)]
    n = len(x)
    if xy.shape[-1] < n // 2:
        return np.nan, np.nan
        
    return pearsonr(xy[0], xy[1])

def R2calc(meas, model, force_zero=False):
    if force_zero:
        SStot = np.sum(meas**2)
    else:
        SStot = np.sum((meas - np.nanmean(meas))**2)
    SSres = np.sum((meas - model)**2)
    return 1 - (SSres / SStot)


# uncertainties unpackers
def unpack_uncertainties(uarray):
    """
    Convenience function to unpack nominal values and uncertainties from an
    ``uncertainties.uarray``.

    Returns:
        (nominal_values, std_devs)
    """
    try:
        return un.nominal_values(uarray), un.std_devs(uarray)
    except:
        return uarray, None


def nominal_values(a):
    try:
        return un.nominal_values(a)
    except:
        return a


def std_devs(a):
    try:
        return un.std_devs(a)
    except:
        return a

def gauss_weighted_stats(x, yarray, x_new, fwhm):
    """
    Calculate gaussian weigted moving mean, SD and SE.

    Parameters
    ----------
    x : array-like
        The independent variable
    yarray : (n,m) array
        Where n = x.size, and m is the number of
        dependent variables to smooth.
    x_new : array-like
        The new x-scale to interpolate the data
    fwhm : int
        FWHM of the gaussian kernel.

    Returns
    -------
    (mean, std, se) : tuple
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # create empty mask array
    mask = np.zeros((x.size, yarray.shape[1], x_new.size))
    # fill mask
    for i, xni in enumerate(x_new):
        mask[:, :, i] = gauss(x[:, np.newaxis], 1, xni, sigma)
    # normalise mask
    nmask = mask / mask.sum(0)  # sum of each gaussian = 1

    # calculate moving average
    av = (nmask * yarray[:, :, np.newaxis]).sum(0)  # apply mask to data
    # sum along xn axis to get means

    # calculate moving sd
    diff = np.power(av - yarray[:, :, np.newaxis], 2)
    std = np.sqrt((diff * nmask).sum(0))
    # sqrt of weighted average of data-mean

    # calculate moving se
    se = std / np.sqrt(mask.sum(0))
    # max amplitude of weights is 1, so sum of weights scales
    # a fn of how many points are nearby. Use this as 'n' in
    # SE calculation.

    return av, std, se


def gauss(x, *p):
    """ Gaussian function.

    Parameters
    ----------
    x : array_like
        Independent variable.
    *p : parameters unpacked to A, mu, sigma
        A = amplitude, mu = centre, sigma = width

    Return
    ------
    array_like
        gaussian descriped by *p.
    """
    A, mu, sigma = p
    return A * np.exp(-0.5 * (-mu + x)**2 / sigma**2)


# Statistical Functions
def stderr(a):
    """
    Calculate the standard error of a.
    """
    return np.nanstd(a) / np.sqrt(sum(np.isfinite(a)))


# Robust Statistics. See:
#   - https://en.wikipedia.org/wiki/Robust_statistics
#   - http://www.cscjp.co.jp/fera/document/ANALYSTVol114Decpgs1693-97_1989.pdf
#   - http://www.rsc.org/images/robust-statistics-technical-brief-6_tcm18-214850.pdf
#   - http://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/h15.htm

def H15_mean(x):
    """
    Calculate the Huber (H15) Robust mean of x.

    For details, see:
        http://www.cscjp.co.jp/fera/document/ANALYSTVol114Decpgs1693-97_1989.pdf
        http://www.rsc.org/images/robust-statistics-technical-brief-6_tcm18-214850.pdf
    """
    mu = np.nanmean(x)
    sd = np.nanstd(x) * 1.134
    sig = 1.5

    hi = x > mu + sig * sd
    lo = x < mu - sig * sd

    if any(hi | lo):
        x[hi] = mu + sig * sd
        x[lo] = mu - sig * sd
        return H15_mean(x)
    else:
        return mu


def H15_std(x):
    """
    Calculate the Huber (H15) Robust standard deviation of x.

    For details, see:
        http://www.cscjp.co.jp/fera/document/ANALYSTVol114Decpgs1693-97_1989.pdf
        http://www.rsc.org/images/robust-statistics-technical-brief-6_tcm18-214850.pdf
    """
    mu = np.nanmean(x)
    sd = np.nanstd(x) * 1.134
    sig = 1.5

    hi = x > mu + sig * sd
    lo = x < mu - sig * sd

    if any(hi | lo):
        x[hi] = mu + sig * sd
        x[lo] = mu - sig * sd
        return H15_std(x)
    else:
        return sd


def H15_se(x):
    """
    Calculate the Huber (H15) Robust standard deviation of x.

    For details, see:
        http://www.cscjp.co.jp/fera/document/ANALYSTVol114Decpgs1693-97_1989.pdf
        http://www.rsc.org/images/robust-statistics-technical-brief-6_tcm18-214850.pdf
    """
    sd = H15_std(x)
    return sd / np.sqrt(sum(np.isfinite(x)))
