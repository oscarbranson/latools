"""
Functions for automatically distinguishing between signal and background
in LA-ICPMS data.

(c) Oscar Branson : https://github.com/oscarbranson
"""
import warnings
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

from ..helpers.utils import Bunch
from ..helpers.signal import fastgrad, fastsmooth, findmins, bool_2_indices, bool_transitions
from ..helpers.stat_fns import gauss

warnings.filterwarnings("ignore")

def split_kmeans(X, transform=None, scaleX=True, sample_weight=None):
    if transform is not None:
        X = transform(X)
    if scaleX:
        X = scale(X)
    
    init = np.percentile(X, [5, 50], 0)
    
    return KMeans(2, init=init).fit_predict(X, sample_weight=sample_weight)

def polynomial_background(x, y, order=3, noise_level=2, mode='low'):
    """
    Fit a polynomial background to the minima or maxima of the signal.

    Parameters
    ----------
    x : array-like
        the x data
    y : array-like
        the y data
    order : int, optional
        the order of the polynomial, by default 3
    noise_level : int, optional
        the noise level above or below which to remove data, by default 2
    mode : str, optional
        whether to fit the minima ('low') or maxima ('high') of the data, by default 'low'

    Returns
    -------
    tuple
        containing (polynomial coefficients, standard deviation of residuals)
    """
    p = np.polyfit(x, y, order)
    pred = np.polyval(p, x)
    resid = y - pred
    resid_std = np.std(resid)
    
    if mode == 'low':
        f = resid < noise_level * resid_std
    if mode == 'high':
        f = resid > -noise_level * resid_std
        
    n_excluded = np.sum(~f)
        
    if n_excluded != 0:
        return polynomial_background(x[f], y[f], order=order, noise_level=noise_level)
    else:
        return p, resid_std

def split_polynomial(x, y, order=3, noise_level=3, std_above_baseline=5):
    
    p, resid_std = polynomial_background(x, y, order=order, noise_level=noise_level, mode='low')
    
    resid = y - np.polyval(p, x)
    trans = resid > resid_std * std_above_baseline
    
    return trans


def log_nozero(a, **kwargs):
    a[a <= 0] = a[a > 0].min()
    return np.log(a)


def autorange(xvar, sig, gwin=7, swin=None, win=30,
              on_mult=(1.5, 1.), off_mult=(1., 1.5),
              transform='log', thresh=None, min_points=None,
              signal_id_mode='kmeans',
              poly_noise_level=3, poly_order=3, std_above_baseline=5):
    """
    Automatically separates signal and background in an on/off data stream.

    **Step 1: Thresholding.**
    KMeans clustering is used to identify data where the laser is 'on' vs
    where the laser is 'off'.

    **Step 2: Transition Removal.**
    The width of the transition regions between signal and background are
    then determined, and the transitions are excluded from analysis. The
    width of the transitions is determined by fitting a gaussian to the
    smoothed first derivative of the analyte trace, and determining its
    width at a point where the gaussian intensity is at at `conf` time the
    gaussian maximum. These gaussians are fit to subsets of the data
    centered around the transitions regions determined in Step 1, +/- `win`
    data points. The peak is further isolated by finding the minima and
    maxima of a second derivative within this window, and the gaussian is
    fit to the isolated peak.

    Parameters
    ----------
    xvar : array-like
        Independent variable (usually time).
    sig : array-like
        Dependent signal, of shape (nsamples, nfeatures). Should be clear 
        distinction between laser 'on' and 'off' regions.
    gwin : int
        The window used for calculating first derivative.
        Defaults to 7.
    swin : int
        The window used for signal smoothing. If None, signal is not smoothed.
    win : int
        The width (c +/- win) of the transition data subsets.
        Defaults to 20.
    on_mult and off_mult : tuple, len=2
        Control the width of the excluded transition regions, which is defined
        relative to the peak full-width-half-maximum (FWHM) of the transition
        gradient. The region n * FHWM below the transition, and m * FWHM above
        the tranision will be excluded, where (n, m) are specified in `on_mult`
        and `off_mult`.
        `on_mult` and `off_mult` apply to the off-on and on-off transitions,
        respectively.
        Defaults to (1.5, 1) and (1, 1.5).
    transform : str
        How to transform the data. Default is 'log'.
    signal_id_mode : str
        How to identify signal and background - either 'kmeans' or 'polynomial'. 
        Default is 'kmeans'.

    Returns
    -------
    fbkg, fsig, ftrn, failed : tuple
        where fbkg, fsig and ftrn are boolean arrays the same length as sig,
        that are True where sig is background, signal and transition, respecively.
        failed contains a list of transition positions where gaussian fitting
        has failed.
    """
    failed = []
    sig = np.asanyarray(sig)

    # smooth signal
    if swin is not None:
        sigs = fastsmooth(sig, swin)
    else:
        sigs = sig

    # transform signal
    if transform == 'log':
        tsigs = log_nozero(sigs)
    else:
        tsigs = sigs

    if thresh is not None:
        if transform == 'log':
            thresh = np.log(thresh)
        fsig = tsigs > thresh
    elif signal_id_mode == 'kmeans':
        if tsigs.ndim == 1:
            scale = False
            tsigs = tsigs.reshape(-1, 1)
        else:
            scale = True
        
        fsig = split_kmeans(tsigs, scaleX=scale).astype(bool)
    
    elif signal_id_mode == 'polynomial':
        fsig = split_polynomial(xvar, tsigs, order=poly_order, noise_level=poly_noise_level, std_above_baseline=std_above_baseline)
    else:
        raise ValueError(f"signal_id_mode must be 'kmeans' or 'polynomial', not '{signal_id_mode}'")
        
    fsig[0] = False  # the first value must always be background
    fbkg = ~fsig

    # remove transitions by fitting a gaussian to the gradients of
    # each transition

    # 1. determine the approximate index of each transition
    zeros = bool_2_indices(fsig)
    
    # remove any regions that are smaller than min_points
    if min_points is not None:
        too_small = np.diff(zeros) < min_points
        for ts in zeros[too_small.flatten()]:
            fsig[ts[0]:ts[1]+1] = False
            fbkg[ts[0]:ts[1]+1] = True
        zeros = zeros[(~too_small).flatten()]
    if zeros is not None:
        zeros = zeros.flatten()
        if sigs.ndim > 1:
            sigs = sigs.sum(axis=1)

        # 2. calculate the absolute gradient of the target trace.
        grad = abs(fastgrad(sigs, gwin))  # gradient of untransformed data.

        for z in zeros:  # for each approximate transition
            # isolate the data around the transition
            if z - win < 0:
                lo = gwin // 2
                hi = int(z + win)
            elif z + win > (len(sig) - gwin // 2):
                lo = int(z - win)
                hi = len(sig) - gwin // 2
            else:
                lo = int(z - win)
                hi = int(z + win)

            xs = xvar[lo:hi]
            ys = grad[lo:hi]

            # determine type of transition (on/off)
            mid = (hi + lo) // 2
            tp = sigs[mid + 3] > sigs[mid - 3]  # True if 'on' transition.

            # fit a gaussian to the first derivative of each
            # transition. Initial guess parameters:
            #   - A: maximum gradient in data
            #   - mu: c
            #   - width: 2 * time step
            # The 'sigma' parameter of curve_fit:
            # This weights the fit by distance from c - i.e. data closer
            # to c are more important in the fit than data further away
            # from c. This allows the function to fit the correct curve,
            # even if the data window has captured two independent
            # transitions (i.e. end of one ablation and start of next)
            # ablation are < win time steps apart).
            centre = xvar[z]  # center of transition
            width = (xvar[1] - xvar[0]) * 2

            try:
                pg, _ = curve_fit(gauss, xs, ys,
                                  p0=(np.nanmax(ys),
                                      centre,
                                      width),
                                  sigma=(xs - centre)**2 + .01)
                # get the x positions when the fitted gaussian is at 'conf' of
                # maximum
                # determine transition FWHM
                fwhm = abs(2 * pg[-1] * np.sqrt(2 * np.log(2)))
                # apply on_mult or off_mult, as appropriate.
                if tp:
                    lim = np.array([-fwhm, fwhm]) * on_mult + pg[1]
                else:
                    lim = np.array([-fwhm, fwhm]) * off_mult + pg[1]

                fbkg[(xvar > lim[0]) & (xvar < lim[1])] = False
                fsig[(xvar > lim[0]) & (xvar < lim[1])] = False

            except RuntimeError:
                failed.append([centre, tp])
                pass

    ftrn = ~fbkg & ~fsig

    # if there are any failed transitions, exclude the mean transition width
    # either side of the failures
    if len(failed) > 0:
        trns = xvar[bool_2_indices(ftrn)]
        tr_mean = (trns[:, 1] - trns[:, 0]).mean() / 2
        for f, tp in failed:
            if tp:
                ind = (xvar >= f - tr_mean *
                       on_mult[0]) & (xvar <= f + tr_mean * on_mult[0])
            else:
                ind = (xvar >= f - tr_mean *
                       off_mult[0]) & (xvar <= f + tr_mean * off_mult[0])
            fsig[ind] = False
            fbkg[ind] = False
            ftrn[ind] = False

    return fbkg, fsig, ftrn, [f[0] for f in failed]

def autorange_components(t, sig, transform='log', gwin=7, swin=None,
                         win=30, on_mult=(1.5, 1.), off_mult=(1., 1.5),
                         thresh=None):
    """
    Returns the components underlying the autorange algorithm.

    Returns
    -------
    t : array-like
        Time axis (independent variable)
    sig : array-like
        Raw signal (dependent variable)
    sigs : array-like
        Smoothed signal (swin)
    tsig : array-like
        Transformed raw signal (transform)
    tsigs : array-like
        Transformed smoothed signal (transform, swin)
    kde_x : array-like
        kernel density estimate of smoothed signal.
    yd : array-like
        bins of kernel density estimator.
    g : array-like
        gradient of smoothed signal (swin, gwin)
    trans : dict
        per-transition data.
    thresh : float
        threshold identified from kernel density plot
    """
    failed = []
    # smooth signal
    if swin is not None:
        sigs = fastsmooth(sig, swin)
    else:
        sigs = sig

    # transform signal
    if transform == 'log':
        tsigs = np.log10(sigs)
        tsig = np.log10(sig)
    else:
        tsigs = sigs
        tsig = sig

    if thresh is None:
        bins = 50
        kde_x = np.linspace(tsigs.min(), tsigs.max(), bins)

        kde = gaussian_kde(tsigs)
        yd = kde.pdf(kde_x)
        mins = findmins(kde_x, yd)  # find minima in kde

        if len(mins) > 0:
            bkg = tsigs < (mins[0])  # set background as lowest distribution
            thresh = mins[0]
        else:
            bkg = np.ones(tsigs.size, dtype=bool)
    else:
        bkg = tsigs < thresh

    # assign rough background and signal regions based on kde minima
    fbkg = bkg
    fsig = ~bkg

    # remove transitions by fitting a gaussian to the gradients of
    # each transition

    # 1. determine the approximate index of each transition
    zeros = bool_2_indices(fsig)

    # 2. calculate the absolute gradient of the target trace.
    g = abs(fastgrad(sigs, gwin))  # gradient of untransformed data.

    if zeros is not None:
        zeros = zeros.flatten()
        trans = dict(zeros=zeros.flatten(),
                     lohi=[],
                     pgs=[],
                     excl=[],
                     tps=[],
                     failed=[],
                     xs=[],
                     ys=[])

        for z in zeros:  # for each approximate transition
            # isolate the data around the transition
            if z - win < 0:
                lo = gwin // 2
                hi = int(z + win)
            elif z + win > (len(sig) - gwin // 2):
                lo = int(z - win)
                hi = len(sig) - gwin // 2
            else:
                lo = int(z - win)
                hi = int(z + win)

            xs = t[lo:hi]
            ys = g[lo:hi]

            trans['xs'].append(xs)
            trans['ys'].append(ys)

            trans['lohi'].append([lo, hi])

            # determine type of transition (on/off)
            mid = (hi + lo) // 2
            tp = sigs[mid + 3] > sigs[mid - 3]  # True if 'on' transition.
            trans['tps'].append(tp)

            c = t[z]  # center of transition
            width = (t[1] - t[0]) * 2  # initial width guess
            try:
                pg, _ = curve_fit(gauss, xs, ys,
                                  p0=(np.nanmax(ys),
                                      c,
                                      width),
                                  sigma=(xs - c)**2 + .01)
                trans['pgs'].append(pg)
                fwhm = abs(2 * pg[-1] * np.sqrt(2 * np.log(2)))
                # apply on_mult or off_mult, as appropriate.
                if tp:
                    lim = np.array([-fwhm, fwhm]) * on_mult + pg[1]
                else:
                    lim = np.array([-fwhm, fwhm]) * off_mult + pg[1]
                trans['excl'].append(lim)

                fbkg[(t > lim[0]) & (t < lim[1])] = False
                fsig[(t > lim[0]) & (t < lim[1])] = False
                failed.append(False)
            except RuntimeError:
                failed.append(True)
                trans['lohi'].append([np.nan, np.nan])
                trans['pgs'].append([np.nan, np.nan, np.nan])
                trans['excl'].append([np.nan, np.nan])
                trans['tps'].append(tp)
                pass
    else:
        zeros = []
    
    return t, sig, sigs, tsig, tsigs, kde_x, yd, g, trans, thresh