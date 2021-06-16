"""
Functions for calculating statistics and handling uncertainties.

(c) Oscar Branson : https://github.com/oscarbranson
"""

import numpy as np
import uncertainties.unumpy as un
import scipy.interpolate as interp
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

def get_total_n_points(d):
    """
    Returns the total number of data points in values of dict.

    Paramters
    ---------
    d : dict
    """
    n = 0
    for di in d.values():
        n += len(di)
    return n

def get_total_time_span(d):
    """
    Returns total length of analysis.
    """

    tmax = 0
    for di in d.values():
        if di.uTime.max() > tmax:
            tmax = di.uTime.max()
    
    return tmax

class un_interp1d(object):
    """
    object for handling interpolation of values with uncertainties.
    """

    def __init__(self, x, y, fill_value=np.nan, **kwargs):
        if isinstance(fill_value, tuple):
            nom_fill = tuple([un.nominal_values(v) for v in fill_value])
            std_fill = tuple([un.std_devs(v) for v in fill_value])
        else:
            nom_fill = std_fill = fill_value
        self.nom_interp = interp.interp1d(un.nominal_values(x),
                                          un.nominal_values(y),
                                          fill_value=nom_fill, **kwargs)
        self.std_interp = interp.interp1d(un.nominal_values(x),
                                          un.std_devs(y),
                                          fill_value=std_fill, **kwargs)

    def new(self, xn):
        yn = self.nom_interp(xn)
        yn_err = self.std_interp(xn)
        return un.uarray(yn, yn_err)

    def new_nom(self, xn):
        return self.nom_interp(xn)

    def new_std(self, xn):
        return self.std_interp(xn)

def stack_keys(ddict, keys, extra=None):
    """
    Combine elements of ddict into an array of shape (len(ddict[key]), len(keys)).

    Useful for preparing data for sklearn.

    Parameters
    ----------
    ddict : dict
        A dict containing arrays or lists to be stacked.
        Must be of equal length.
    keys : list or str
        The keys of dict to stack. Must be present in ddict.
    extra : list (optional)
        A list of additional arrays to stack. Elements of extra
        must be the same length as arrays in ddict.
        Extras are inserted as the first columns of output.
    """
    if isinstance(keys, str):
        d = [ddict[keys]]
    else:
        d = [ddict[k] for k in keys]
    if extra is not None:
        d = extra + d
    return np.vstack(d).T

