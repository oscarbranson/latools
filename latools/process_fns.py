import os
import re
import numpy as np
import warnings
import matplotlib.pyplot as plt

from io import BytesIO
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit

from .helpers import Bunch, fastgrad, fastsmooth, findmins, bool_2_indices
from .stat_fns import gauss


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
    """
    Load data_file described by a dataformat dict.

    Parameters
    ----------
    data_file : str
        Path to data file, including extension.
    dataformat : dict
        A dataformat dict, see example below.
    name_mode : str
        How to identyfy sample names. If 'file_names' uses the
        input name of the file, stripped of the extension. If
        'metadata_names' uses the 'name' attribute of the 'meta'
        sub-dictionary in dataformat. If any other str, uses this
        str as the sample name.

    Example dataformat
    -------------------
        {'genfromtext_args': {'delimiter': ',',
                              'skip_header': 4},  # passed directly to np.genfromtxt
         'column_id': {'name_row': 3,  # which row contains the column names
                       'delimiter': ',',  # delimeter between column names
                       'timecolumn': 0,  # which column contains the 'time' variable
                       'pattern': '([A-z]{1,2}[0-9]{1,3})'},  # a regex pattern which captures the column names
         'meta_regex': {  # a dict of (line_no: ([descriptors], [regexs])) pairs
                        0: (['path'], '(.*)'),
                        2: (['date', 'method'],  # MUST include date
                            '([A-Z][a-z]+ [0-9]+ [0-9]{4}[ ]+[0-9:]+ [amp]+).* ([A-z0-9]+\.m)')
                        }
        }

    Returns
    -------
    sample, analytes, data, meta : tuple
    """
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
        sample = name_mode

    # column and analyte names
    columns = np.array(lines[dataformat['column_id']['name_row']].strip().split(
        dataformat['column_id']['delimiter']))
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

    # convert raw data into counts
    # TODO: Is this correct? Should actually be per-analyte dwell?
    if 'unit' in dataformat:
        if dataformat['unit'] == 'cps':
            tstep = data['Time'][1] - data['Time'][0]
            read_data[dind] *= tstep
        else:
            pass
    data['rawdata'] = Bunch(zip(analytes, read_data[dind]))
    data['total_counts'] = read_data[dind].sum(0)

    return sample, analytes, data, meta


def autorange(t, sig, gwin=7, win=30,
              on_mult=(1.5, 1.), off_mult=(1., 1.5)):
    """
    Automatically separates signal and background in an on/off data stream.

    Step 1: Thresholding
    The background signal is determined using a gaussian kernel density
    estimator (kde) of all the data. Under normal circumstances, this
    kde should find two distinct data distributions, corresponding to
    'signal' and 'background'. The minima between these two distributions
    is taken as a rough threshold to identify signal and background
    regions. Any point where the trace crosses this thrshold is identified
    as a 'transition'.

    Step 2: Transition Removal
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
        The window used for signal smoothing and calculationg of first
        derivative.
        Defaults to 7.
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

    Returns
    -------
    fbkg, fsig, ftrn, failed : tuple
        where fbkg, fsig and ftrn are boolean arrays the same length as sig,
        that are True where sig is background, signal and transition, respecively.
        failed contains a list of transition positions where gaussian fitting
        has failed.
    """

    failed = []

    bins = 50
    kde_x = np.linspace(sig.min(), sig.max(), bins)

    kde = gaussian_kde(sig)
    yd = kde.pdf(kde_x)
    mins = findmins(kde_x, yd)  # find minima in kde

    sigs = fastsmooth(sig, gwin)

    bkg = sigs < (mins[0])  # set background as lowest distribution
    # bkg[0] = True  # the first value must always be background

    # assign rough background and signal regions based on kde minima
    fbkg = bkg
    fsig = ~bkg

    # remove transitions by fitting a gaussian to the gradients of
    # each transition
    # 1. calculate the absolute gradient of the target trace.
    g = abs(fastgrad(sig, gwin))
    # 2. determine the approximate index of each transition
    zeros = bool_2_indices(fsig).flatten()

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
            fwhm = 2 * pg[-1] * np.sqrt(2 * np.log(2))
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


def autorange_plot(t, sig, gwin=7, win=30,
                   on_mult=(1.5, 1.), off_mult=(1., 1.5)):
    """
    Function for visualising the autorange mechanism.

    Parameters
    ----------
    t : array-like
        Independent variable (usually time).
    sig : array-like
        Dependent signal, with distinctive 'on' and 'off' regions.
    gwin : int
        The window used for signal smoothing and calculationg of first
        derivative.
        Defaults to 7.
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

    Returns
    -------
    fig, axes
    """

    # perform autorange calculations
    bins = 50
    kde_x = np.linspace(sig.min(), sig.max(), bins)

    kde = gaussian_kde(sig)
    yd = kde.pdf(kde_x)
    mins = findmins(kde_x, yd)  # find minima in kde

    sigs = fastsmooth(sig, gwin)

    bkg = sigs < (mins[0])  # set background as lowest distribution
    # bkg[0] = True  # the first value must always be background

    # assign rough background and signal regions based on kde minima
    fbkg = bkg
    fsig = ~bkg

    g = abs(fastgrad(sig, gwin))
    # 2. determine the approximate index of each transition
    zeros = bool_2_indices(fsig).flatten()

    lohi = []
    pgs = []
    excl = []
    tps = []
    failed = []

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

        lohi.append([lo, hi])

        # determine type of transition (on/off)
        mid = (hi + lo) // 2
        tp = sigs[mid + 3] > sigs[mid - 3]  # True if 'on' transition.
        tps.append(tp)

        c = t[z]  # center of transition
        width = (t[1] - t[0]) * 2  # initial width guess
        try:
            pg, _ = curve_fit(gauss, xs, ys,
                              p0=(np.nanmax(ys),
                                  c,
                                  width),
                              sigma=(xs - c)**2 + .01)
            pgs.append(pg)
            fwhm = 2 * pg[-1] * np.sqrt(2 * np.log(2))
            # apply on_mult or off_mult, as appropriate.
            if tp:
                lim = np.array([-fwhm, fwhm]) * on_mult + pg[1]
            else:
                lim = np.array([-fwhm, fwhm]) * off_mult + pg[1]
            excl.append(lim)

            fbkg[(t > lim[0]) & (t < lim[1])] = False
            fsig[(t > lim[0]) & (t < lim[1])] = False
            failed.append(False)
        except RuntimeError:
            failed.append(True)
            lohi.append([np.nan, np.nan])
            pgs.append([np.nan, np.nan, np.nan])
            excl.append([np.nan, np.nan])
            tps.append(tp)
            pass

    # make plot
    nrows = 2 + len(zeros) // 2 + len(zeros) % 2

    fig, axs = plt.subplots(nrows, 2, figsize=(6, 4 + 1.5 * nrows))

    # Trace
    ax1, ax2, ax3, ax4 = axs.flat[:4]
    ax4.set_visible(False)

    # widen ax1 & 3
    for ax in [ax1, ax3]:
        p = ax.axes.get_position()
        p2 = [p.x0, p.y0, p.width * 1.75, p.height]
        ax.axes.set_position(p2)

    # move ax3 up
    p = ax3.axes.get_position()
    p2 = [p.x0, p.y0 + 0.15 * p.height, p.width, p.height]
    ax3.axes.set_position(p2)

    # truncate ax2
    p = ax2.axes.get_position()
    p2 = [p.x0 + p.width * 0.6, p.y0, p.width * 0.4, p.height]
    ax2.axes.set_position(p2)

    # plot traces and gradient
    ax1.plot(t, sig, c='k', lw=1)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Signal')
    ax3.plot(t, g, c='k', lw=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Gradient')

    # plot kde
    ax2.fill_betweenx(kde_x, yd, color=(0, 0, 0, 0.2))
    ax2.plot(yd, kde_x, c='k')
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticklabels([])
    ax2.set_xlabel('Data\nDensity')

    # limit
    for ax in [ax1, ax2]:
        ax.axhline(mins[0], c='k', ls='dashed', alpha=0.4)

    # zeros
    for z in zeros:
        ax1.axvline(t[z], c='r', alpha=0.5)
        ax3.axvline(t[z], c='r', alpha=0.5)

    # plot individual transitions
    n = 1
    for (lo, hi), lim, tp, pg, fail, ax in zip(lohi, excl, tps, pgs, failed, axs.flat[4:]):
        # plot region on gradient axis
        ax3.axvspan(t[lo], t[hi], color='r', alpha=0.1, zorder=-2)

        # plot individual transitions
        x = t[lo:hi]
        y = g[lo:hi]
        ys = sig[lo:hi]
        ax.scatter(x, y, c='k', marker='x', zorder=-1, s=10)
        ax.set_yticklabels([])

        tax = ax.twinx()
        tax.plot(x, ys, c='k', alpha=0.3, zorder=-5)
        tax.set_yticklabels([])

        # plot fitted gaussian
        xn = np.linspace(x.min(), x.max(), 100)
        ax.plot(xn, gauss(xn, *pg), c='r', alpha=0.5)

        # plot center and excluded region
        ax.axvline(pg[1], c='b', alpha=0.5)
        ax.axvspan(*lim, color='b', alpha=0.1, zorder=-2)

        if tp:
            ax.text(.05, .95, '{} (on)'.format(n), ha='left',
                    va='top', transform=ax.transAxes)
        else:
            ax.text(.95, .95, '{} (off)'.format(n), ha='right',
                    va='top', transform=ax.transAxes)

        if ax.is_last_row():
            ax.set_xlabel('Time (s)')
        if ax.is_first_col():
            ax.set_ylabel('Gradient (x)')
        if ax.is_last_col():
            tax.set_ylabel('Signal (line)')

        if fail:
            ax.axes.set_facecolor((1, 0, 0, 0.2))
            ax.text(.5, .5, 'FAIL', ha='center', va='center',
                    fontsize=16, color=(1, 0, 0, 0.5), transform=ax.transAxes)

        n += 1

    # should never be, but just in case...
    if len(zeros) % 2 == 1:
        axs.flat[-1].set_visible = False

    return fig, axs

# def rolling_mean_std(sig, win=3):
#     npad = int((win - 1) / 2)
#     sumkernel = np.ones(win)
#     kernel = sumkernel / win
#     mean = np.convolve(sig, kernel, 'same')
#     mean[:npad] = sig[:npad]
#     mean[-npad:] = sig[-npad:]
#     # calculate sqdiff
#     sqdiff = (sig - mean)**2
#     sumsqdiff = np.convolve(sqdiff, sumkernel, 'same')
#     std = np.sqrt(sumsqdiff / win)

#     return mean[5], sqdiff[4:7], sumsqdiff[5], std[5]

#     # print(np.convolve(a, sumkernel, 'valid'))


# if __name__ == '__main__':
#     a = np.random.normal(5, 1, 100)
#     a[30] = 10
#     print(rolling_mean_std(a))

#     print(np.convolve(np.ones(10), np.ones(3), 'same'))
#     asub = a[4:7]
#     print(asub)
#     masub = np.mean(asub)
#     sqdiff = (asub - masub)**2
#     sumsqdiff = np.sum(sqdiff)
#     std = np.sqrt(sumsqdiff / 3)
#     print(masub, sumsqdiff, std)
