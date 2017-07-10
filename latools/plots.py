import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit

from tqdm import tqdm

from .helpers import fastgrad, fastsmooth, findmins, bool_2_indices, rangecalc, unitpicker
from.stat_fns import nominal_values, gauss


def crossplot(dat, keys=None, lognorm=True,
              bins=25, figsize=(12, 12),
              colourful=True, focus_stage=None, denominator=None,
              mode='hist2d', cmap=None, **kwargs):
    """
    Plot analytes against each other.

    The number of plots is n**2 - n, where n = len(keys).

    Parameters
    ----------
    keys : optional, array_like or str
        The analyte(s) to plot. Defaults to all analytes.
    lognorm : bool
        Whether or not to log normalise the colour scale
        of the 2D histogram.
    bins : int
        The number of bins in the 2D histogram.

    Returns
    -------
    (fig, axes)
    """
    if keys is None:
        keys = dat.keys()

    if figsize[0] < 1.5 * len(keys):
        figsize = [1.5 * len(keys)] * 2

    numvars = len(keys)
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars,
                             figsize=(12, 12))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # set up colour scales
    if colourful:
        cmlist = ['Blues', 'BuGn', 'BuPu', 'GnBu',
                  'Greens', 'Greys', 'Oranges', 'OrRd',
                  'PuBu', 'PuBuGn', 'PuRd', 'Purples',
                  'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']
    else:
        cmlist = ['Greys']

    if cmap is None and mode == 'scatter':
        cmap = {k: 'k' for k in dat.keys()}

    while len(cmlist) < len(keys):
        cmlist *= 2

    # isolate nominal_values for all keys
    focus = {k: nominal_values(dat[k]) for k in keys}
    # determine units for all keys
    udict = {a: unitpicker(np.nanmean(focus[a]),
                           focus_stage=focus_stage,
                           denominator=denominator) for a in keys}
    # determine ranges for all analytes
    rdict = {a: (np.nanmin(focus[a] * udict[a][0]),
                 np.nanmax(focus[a] * udict[a][0])) for a in keys}

    for i, j in tqdm(zip(*np.triu_indices_from(axes, k=1)), desc='Drawing Plots',
                     total=sum(range(len(keys)))):
        # get analytes
        ai = keys[i]
        aj = keys[j]

        # remove nan, apply multipliers
        pi = focus[ai] * udict[ai][0]
        pj = focus[aj] * udict[aj][0]

        # determine normalisation shceme
        if lognorm:
            norm = mpl.colors.LogNorm()
        else:
            norm = None

        # draw plots
        if mode == 'hist2d':
            # remove nan
            pi = pi[~np.isnan(pi)]
            pj = pj[~np.isnan(pj)]

            axes[i, j].hist2d(pj, pi, bins,
                              norm=norm,
                              cmap=plt.get_cmap(cmlist[i]))
            axes[j, i].hist2d(pi, pj, bins,
                              norm=norm,
                              cmap=plt.get_cmap(cmlist[j]))
        elif mode == 'scatter':
            axes[i, j].scatter(pj, pi, s=10,
                               c=cmap[ai], lw=0.5, edgecolor='k',
                               alpha=0.4)
            axes[j, i].scatter(pi, pj, s=10,
                               c=cmap[aj], lw=0.5, edgecolor='k',
                               alpha=0.4)
        else:
            raise ValueError("invalid mode. Must be 'hist2d' or 'scatter'.")

        axes[i, j].set_ylim(*rdict[ai])
        axes[i, j].set_xlim(*rdict[aj])

        axes[j, i].set_ylim(*rdict[aj])
        axes[j, i].set_xlim(*rdict[ai])

    # diagonal labels
    for a, n in zip(keys, np.arange(len(keys))):
        axes[n, n].annotate(a + '\n' + udict[a][1], (0.5, 0.5),
                            xycoords='axes fraction',
                            ha='center', va='center', fontsize=8)
        axes[n, n].set_xlim(*rdict[a])
        axes[n, n].set_ylim(*rdict[a])
    # switch on alternating axes
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        for label in axes[j, i].get_xticklabels():
            label.set_rotation(90)
        axes[i, j].yaxis.set_visible(True)

    return fig, axes


def histograms(dat, keys=None, bins=25, logy=False, cmap=None):
    """
    Plot histograms of all items in dat.

    Parameters
    ----------
    dat : dict
        Data in {key: array} pairs.
    keys : arra-like
        The keys in dat that you want to plot. If None,
        all are plotted.
    bins : int
        The number of bins in each histogram (default = 25)
    logy : bool
        If true, y axis is a log scale.
    cmap : dict
        The colours that the different items should be. If None,
        all are grey.

    Returns
    -------
    fig, axes
    """
    if keys is None:
        keys = dat.keys()

    nrow = len(keys) // 4
    if len(keys) % 4 > 0:
        nrow += 1
    fig, axs = plt.subplots(nrow, 4, figsize=[8, nrow * 2])

    pn = 0
    for k, ax in zip(keys, axs.flat):
        tmp = nominal_values(dat[k])
        x = tmp[~np.isnan(tmp)]

        if cmap is not None:
            c = cmap[k]
        else:
            c = (0, 0, 0, 0.5)
        ax.hist(x, bins=bins, color=c)

        if logy:
            ax.set_yscale('log')
            ylab = '$log_{10}(n)$'
        else:
            ylab = 'n'

        ax.set_ylim(1, ax.get_ylim()[1])

        if ax.is_first_col():
            ax.set_ylabel(ylab)

        ax.set_yticklabels([])

        ax.text(.95, .95, k, ha='right', va='top', transform=ax.transAxes)

        pn += 1

    for ax in axs.flat[pn:]:
        ax.set_visible(False)
    fig.tight_layout()

    return fig, axs


def autorange_plot(t, sig, gwin=7, swin=None, win=30,
                   on_mult=(1.5, 1.), off_mult=(1., 1.5),
                   nbin=10):
    """
    Function for visualising the autorange mechanism.

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
        The window ised for signal smoothing. If None, gwin // 2.
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
    nbin : ind
        Used to calculate the number of bins in the data histogram.
        bins = len(sig) // nbin

    Returns
    -------
    fig, axes
    """
    if swin is None:
        swin = gwin // 2

    # perform autorange calculations
    # bins = 50
    bins = sig.size // nbin
    kde_x = np.linspace(sig.min(), sig.max(), bins)

    kde = gaussian_kde(sig)
    yd = kde.pdf(kde_x)
    mins = findmins(kde_x, yd)  # find minima in kde

    sigs = fastsmooth(sig, swin)

    if len(mins) > 0:
        bkg = sigs < (mins[0])  # set background as lowest distribution
    else:
        bkg = np.ones(sig.size, dtype=bool)
        mins = [np.nan]
    # bkg[0] = True  # the first value must always be background

    # assign rough background and signal regions based on kde minima
    fbkg = bkg
    fsig = ~bkg

    g = abs(fastgrad(sigs, gwin))  # calculate gradient of signal
    # 2. determine the approximate index of each transition
    zeros = bool_2_indices(fsig)

    if zeros is not None:
        zeros = zeros.flatten()
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
    else:
        zeros = []

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

    if len(zeros) > 0:
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
            ax.set_ylim(rangecalc(y))

            tax = ax.twinx()
            tax.plot(x, ys, c='k', alpha=0.3, zorder=-5)
            tax.set_yticklabels([])
            tax.set_ylim(rangecalc(ys))

            # plot fitted gaussian
            xn = np.linspace(x.min(), x.max(), 100)
            ax.plot(xn, gauss(xn, *pg), c='r', alpha=0.5)

            # plot center and excluded region
            ax.axvline(pg[1], c='b', alpha=0.5)
            ax.axvspan(*lim, color='b', alpha=0.1, zorder=-2)

            ax1.axvspan(*lim, color='b', alpha=0.1, zorder=-2)
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

