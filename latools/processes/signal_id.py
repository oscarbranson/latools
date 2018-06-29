import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit

from ..helpers.helpers import Bunch, fastgrad, fastsmooth, findmins, bool_2_indices
from ..helpers.stat_fns import gauss

def autorange(t, sig, gwin=7, swin=None, win=30,
              on_mult=(1.5, 1.), off_mult=(1., 1.5),
              nbin=10, transform='log', thresh=None):
    """
    Automatically separates signal and background in an on/off data stream.

    **Step 1: Thresholding.**
    The background signal is determined using a gaussian kernel density
    estimator (kde) of all the data. Under normal circumstances, this
    kde should find two distinct data distributions, corresponding to
    'signal' and 'background'. The minima between these two distributions
    is taken as a rough threshold to identify signal and background
    regions. Any point where the trace crosses this thrshold is identified
    as a 'transition'.

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
    t : array-like
        Independent variable (usually time).
    sig : array-like
        Dependent signal, with distinctive 'on' and 'off' regions.
    gwin : int
        The window used for calculating first derivative.
        Defaults to 7.
    swin : int
        The window used for signal smoothing. If None, ``gwin // 2``.
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

    Returns
    -------
    fbkg, fsig, ftrn, failed : tuple
        where fbkg, fsig and ftrn are boolean arrays the same length as sig,
        that are True where sig is background, signal and transition, respecively.
        failed contains a list of transition positions where gaussian fitting
        has failed.
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
    else:
        tsigs = sigs

    if thresh is None:
        bins = 50
        kde_x = np.linspace(tsigs.min(), tsigs.max(), bins)

        kde = gaussian_kde(tsigs)
        yd = kde.pdf(kde_x)
        mins = findmins(kde_x, yd)  # find minima in kde

        if len(mins) > 0:
            bkg = tsigs < (mins[0])  # set background as lowest distribution
        else:
            bkg = np.ones(tsigs.size, dtype=bool)
        # bkg[0] = True  # the first value must always be background
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
            c = t[z]  # center of transition
            width = (t[1] - t[0]) * 2

            try:
                pg, _ = curve_fit(gauss, xs, ys,
                                  p0=(np.nanmax(ys),
                                      c,
                                      width),
                                  sigma=(xs - c)**2 + .01)
                # get the x positions when the fitted gaussian is at 'conf' of
                # maximum
                # determine transition FWHM
                fwhm = abs(2 * pg[-1] * np.sqrt(2 * np.log(2)))
                # apply on_mult or off_mult, as appropriate.
                if tp:
                    lim = np.array([-fwhm, fwhm]) * on_mult + pg[1]
                else:
                    lim = np.array([-fwhm, fwhm]) * off_mult + pg[1]

                fbkg[(t > lim[0]) & (t < lim[1])] = False
                fsig[(t > lim[0]) & (t < lim[1])] = False

            except RuntimeError:
                failed.append([c, tp])
                pass

    ftrn = ~fbkg & ~fsig

    # if there are any failed transitions, exclude the mean transition width
    # either side of the failures
    if len(failed) > 0:
        trns = t[bool_2_indices(ftrn)]
        tr_mean = (trns[:, 1] - trns[:, 0]).mean() / 2
        for f, tp in failed:
            if tp:
                ind = (t >= f - tr_mean *
                       on_mult[0]) & (t <= f + tr_mean * on_mult[0])
            else:
                ind = (t >= f - tr_mean *
                       off_mult[0]) & (t <= f + tr_mean * off_mult[0])
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