"""
Plotting functions.

(c) Oscar Branson : https://github.com/oscarbranson
"""

import itertools, re, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from IPython import display
from pandas import IndexSlice as idx

from tqdm import tqdm

from .helpers import fastgrad, fastsmooth, findmins, bool_2_indices, rangecalc, unitpicker, pretty_element, calc_grads
from .stat_fns import nominal_values, gauss, R2calc, unpack_uncertainties

def calc_nrow(n, ncol):
    if n % ncol is 0:
        nrow = n / ncol
    else:
        nrow = n // ncol + 1
    
    return int(nrow)
    
def tplot(self, analytes=None, figsize=[10, 4], scale='log', filt=None,
              ranges=False, stats=False, stat='nanmean', err='nanstd',
              focus_stage=None, err_envelope=False, ax=None):
        """
        Plot analytes as a function of Time.

        Parameters
        ----------
        analytes : array_like
            list of strings containing names of analytes to plot.
            None = all analytes.
        figsize : tuple
            size of final figure.
        scale : str or None
           'log' = plot data on log scale
        filt : bool, str or dict
            False: plot unfiltered data.
            True: plot filtered data over unfiltered data.
            str: apply filter key to all analytes
            dict: apply key to each analyte in dict. Must contain all
            analytes plotted. Can use self.filt.keydict.
        ranges : bool
            show signal/background regions.
        stats : bool
            plot average and error of each trace, as specified by `stat` and
            `err`.
        stat : str
            average statistic to plot.
        err : str
            error statistic to plot.

        Returns
        -------
        figure, axis
        """
        if type(analytes) is str:
            analytes = [analytes]
        if analytes is None:
            analytes = self.analytes

        if focus_stage is None:
            focus_stage = self.focus_stage
        
        # exclude internal standard from analytes
        if focus_stage in ['ratios', 'calibrated']:
            analytes = [a for a in analytes if a != self.internal_standard]

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([.1, .12, .77, .8])
            ret = True
        else:
            fig = ax.figure
            ret = False

        for a in analytes:
            x = self.Time
            y, yerr = unpack_uncertainties(self.data[focus_stage][a])

            if scale is 'log':
                ax.set_yscale('log')
                y[y == 0] = np.nan

            if filt:
                ind = self.filt.grab_filt(filt, a)
                xf = x.copy()
                yf = y.copy()
                yerrf = yerr.copy()
                if any(~ind):
                    xf[~ind] = np.nan
                    yf[~ind] = np.nan
                    yerrf[~ind] = np.nan
                if any(~ind):
                    ax.plot(x, y, color=self.cmap[a], alpha=.2, lw=0.6)
                ax.plot(xf, yf, color=self.cmap[a], label=a)
                if err_envelope:
                    ax.fill_between(xf, yf - yerrf, yf + yerrf, color=self.cmap[a],
                                    alpha=0.2, zorder=-1)
            else:
                ax.plot(x, y, color=self.cmap[a], label=a)
                if err_envelope:
                    ax.fill_between(x, y - yerr, y + yerr, color=self.cmap[a],
                                    alpha=0.2, zorder=-1)

            # Plot averages and error envelopes
            if stats and hasattr(self, 'stats'):
                warnings.warn('\nStatistic plotting is broken.\nCheck progress here: https://github.com/oscarbranson/latools/issues/18')
                pass
                # sts = self.stats[sig][0].size
                # if sts > 1:
                #     for n in np.arange(self.n):
                #         n_ind = ind & (self.ns == n + 1)
                #         if sum(n_ind) > 2:
                #             x = [self.Time[n_ind][0], self.Time[n_ind][-1]]
                #             y = [self.stats[sig][self.stats['analytes'] == a][0][n]] * 2

                #             yp = ([self.stats[sig][self.stats['analytes'] == a][0][n] +
                #                   self.stats[err][self.stats['analytes'] == a][0][n]] * 2)
                #             yn = ([self.stats[sig][self.stats['analytes'] == a][0][n] -
                #                   self.stats[err][self.stats['analytes'] == a][0][n]] * 2)

                #             ax.plot(x, y, color=self.cmap[a], lw=2)
                #             ax.fill_between(x + x[::-1], yp + yn,
                #                             color=self.cmap[a], alpha=0.4,
                #                             linewidth=0)
                # else:
                #     x = [self.Time[0], self.Time[-1]]
                #     y = [self.stats[sig][self.stats['analytes'] == a][0]] * 2
                #     yp = ([self.stats[sig][self.stats['analytes'] == a][0] +
                #           self.stats[err][self.stats['analytes'] == a][0]] * 2)
                #     yn = ([self.stats[sig][self.stats['analytes'] == a][0] -
                #           self.stats[err][self.stats['analytes'] == a][0]] * 2)

                #     ax.plot(x, y, color=self.cmap[a], lw=2)
                #     ax.fill_between(x + x[::-1], yp + yn, color=self.cmap[a],
                #                     alpha=0.4, linewidth=0)

        if ranges:
            for lims in self.bkgrng:
                ax.axvspan(*lims, color='k', alpha=0.1, zorder=-1)
            for lims in self.sigrng:
                ax.axvspan(*lims, color='r', alpha=0.1, zorder=-1)

        ax.text(0.01, 0.99, self.sample + ' : ' + focus_stage,
                transform=ax.transAxes,
                ha='left', va='top')

        ax.set_xlabel('Time (s)')
        ax.set_xlim(np.nanmin(x), np.nanmax(x))

        # y label
        ud = {'rawdata': 'counts',
              'despiked': 'counts',
              'bkgsub': 'background corrected counts',
              'ratios': 'counts/{:s} count',
              'calibrated': 'mol/mol {:s}',
              'mass_fraction': 'Mass Fraction'}
        if focus_stage in ['ratios', 'calibrated']:
            ud[focus_stage] = ud[focus_stage].format(self.internal_standard)
        ax.set_ylabel(ud[focus_stage])

        # if interactive:
        #     ax.legend()
        #     plugins.connect(fig, plugins.MousePosition(fontsize=14))
        #     display.clear_output(wait=True)
        #     display.display(fig)
        #     input('Press [Return] when finished.')
        # else:
        ax.legend(bbox_to_anchor=(1.15, 1))

        if ret:
            return fig, ax

def gplot(self, analytes=None, win=25, figsize=[10, 4],
              ranges=False, focus_stage=None, ax=None, recalc=True):
        """
        Plot analytes gradients as a function of Time.

        Parameters
        ----------
        analytes : array_like
            list of strings containing names of analytes to plot.
            None = all analytes.
        win : int
            The window over which to calculate the rolling gradient.
        figsize : tuple
            size of final figure.
        ranges : bool
            show signal/background regions.

        Returns
        -------
        figure, axis
        """

        if type(analytes) is str:
            analytes = [analytes]
        if analytes is None:
            analytes = self.analytes

        if focus_stage is None:
            focus_stage = self.focus_stage

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes([.1, .12, .77, .8])
            ret = True
        else:
            fig = ax.figure
            ret = False

        x = self.Time
        if recalc or not self.grads_calced:
            self.grads = calc_grads(x, self.data[focus_stage], analytes, win)
            self.grads_calce = True

        for a in analytes:
            ax.plot(x, self.grads[a], color=self.cmap[a], label=a)

        if ranges:
            for lims in self.bkgrng:
                ax.axvspan(*lims, color='k', alpha=0.1, zorder=-1)
            for lims in self.sigrng:
                ax.axvspan(*lims, color='r', alpha=0.1, zorder=-1)

        ax.text(0.01, 0.99, self.sample + ' : ' + self.focus_stage + ' : gradient',
                transform=ax.transAxes,
                ha='left', va='top')

        ax.set_xlabel('Time (s)')
        ax.set_xlim(np.nanmin(x), np.nanmax(x))

        # y label
        ud = {'rawdata': 'counts/s',
              'despiked': 'counts/s',
              'bkgsub': 'background corrected counts/s',
              'ratios': 'counts/{:s} count/s',
              'calibrated': 'mol/mol {:s}/s',
              'mass_fraction': 'Mass Fraction/s'}
        if focus_stage in ['ratios', 'calibrated']:
            ud[focus_stage] = ud[focus_stage].format(self.internal_standard)
        ax.set_ylabel(ud[focus_stage])
        # y tick format

        def yfmt(x, p):
            return '{:.0e}'.format(x)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(yfmt))

        ax.legend(bbox_to_anchor=(1.15, 1))

        ax.axhline(0, color='k', lw=1, ls='dashed', alpha=0.5)

        if ret:
            return fig, ax

def crossplot(dat, keys=None, lognorm=True, bins=25, figsize=(12, 12),
              colourful=True, focus_stage=None, denominator=None,
              mode='hist2d', cmap=None, **kwargs):
    """
    Plot analytes against each other.

    The number of plots is n**2 - n, where n = len(keys).

    Parameters
    ----------
    dat : dict
        A dictionary of key: data pairs, where data is the same
        length in each entry.
    keys : optional, array_like or str
        The keys of dat to plot. Defaults to all keys.
    lognorm : bool
        Whether or not to log normalise the colour scale
        of the 2D histogram.
    bins : int
        The number of bins in the 2D histogram.
    figsize : tuple
    colourful : bool

    Returns
    -------
    (fig, axes)
    """
    if keys is None:
        keys = list(dat.keys())

    numvar = len(keys)
    if figsize[0] < 1.5 * numvar:
        figsize = [1.5 * numvar] * 2
    
    fig, axes = plt.subplots(nrows=numvar, ncols=numvar,
                             figsize=figsize)
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
                               color=cmap[ai], lw=0.5, edgecolor='k',
                               alpha=0.4)
            axes[j, i].scatter(pi, pj, s=10,
                               color=cmap[aj], lw=0.5, edgecolor='k',
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
    for i, j in zip(range(numvar), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        for label in axes[j, i].get_xticklabels():
            label.set_rotation(90)
        axes[i, j].yaxis.set_visible(True)

    return fig, axes


def histograms(dat, keys=None, bins=25, logy=False, cmap=None, ncol=4):
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

    ncol = int(ncol)
    nrow = calc_nrow(len(keys), ncol)

    fig, axs = plt.subplots(nrow, 4, figsize=[ncol * 2, nrow * 2])

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
                   nbin=10, thresh=None):
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

    sigs = fastsmooth(sig, swin)

    # perform autorange calculations
    
    # bins = 50
    bins = sig.size // nbin
    kde_x = np.linspace(sig.min(), sig.max(), bins)

    kde = gaussian_kde(sigs)
    yd = kde.pdf(kde_x)
    mins = findmins(kde_x, yd)  # find minima in kde

    if thresh is not None:
        mins = [thresh]
    if len(mins) > 0:
        bkg = sigs < (mins[0])  # set background as lowest distribution
    else:
        bkg = np.ones(sig.size, dtype=bool)
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
                fwhm = abs(2 * pg[-1] * np.sqrt(2 * np.log(2)))
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
    ax1.plot(t, sig, color='k', lw=1)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Signal')
    ax3.plot(t, g, color='k', lw=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Gradient')

    # plot kde
    ax2.fill_betweenx(kde_x, yd, color=(0, 0, 0, 0.2))
    ax2.plot(yd, kde_x, color='k')
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticklabels([])
    ax2.set_xlabel('Data\nDensity')

    # limit
    for ax in [ax1, ax2]:
        ax.axhline(mins[0], color='k', ls='dashed', alpha=0.4)

    if len(zeros) > 0:
        # zeros
        for z in zeros:
            ax1.axvline(t[z], color='r', alpha=0.5)
            ax3.axvline(t[z], color='r', alpha=0.5)

        # plot individual transitions
        n = 1
        for (lo, hi), lim, tp, pg, fail, ax in zip(lohi, excl, tps, pgs, failed, axs.flat[4:]):
            # plot region on gradient axis
            ax3.axvspan(t[lo], t[hi], color='r', alpha=0.1, zorder=-2)

            # plot individual transitions
            x = t[lo:hi]
            y = g[lo:hi]
            ys = sig[lo:hi]
            ax.scatter(x, y, color='k', marker='x', zorder=-1, s=10)
            ax.set_yticklabels([])
            ax.set_ylim(rangecalc(y))

            tax = ax.twinx()
            tax.plot(x, ys, color='k', alpha=0.3, zorder=-5)
            tax.set_yticklabels([])
            tax.set_ylim(rangecalc(ys))

            # plot fitted gaussian
            xn = np.linspace(x.min(), x.max(), 100)
            ax.plot(xn, gauss(xn, *pg), color='r', alpha=0.5)

            # plot center and excluded region
            ax.axvline(pg[1], color='b', alpha=0.5)
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

def calibration_plot(self, analytes=None, datarange=True, loglog=False, ncol=3, srm_group=None, save=True):
    """
    Plot the calibration lines between measured and known SRM values.

    Parameters
    ----------
    analytes : optional, array_like or str
        The analyte(s) to plot. Defaults to all analytes.
    datarange : boolean
        Whether or not to show the distribution of the measured data
        alongside the calibration curve.
    loglog : boolean
        Whether or not to plot the data on a log - log scale. This is
        useful if you have two low standards very close together,
        and want to check whether your data are between them, or
        below them.

    Returns
    -------
    (fig, axes)
    """

    if isinstance(analytes, str):
        analytes = [analytes]

    if analytes is None:
        analytes = [a for a in self.analytes if self.internal_standard not in a]

    if srm_group is not None:
        srm_groups = {int(g): t for g, t in self.stdtab.loc[:, ['group', 'gTime']].values}
        try:
            gTime = srm_groups[srm_group]
        except KeyError:
            text = ('Invalid SRM group selection. Valid options are:\n' +
                    ' Key:  Time Centre\n' + 
                    '\n'.join(['   {:}:  {:.1f}s'.format(k, v) for k, v in srm_groups.items()]))
            print(text)
    else:
        gTime = None

    ncol = int(ncol)
    n = len(analytes)
    nrow = calc_nrow(n + 1, ncol)

    axes = []

    if not datarange:
        fig = plt.figure(figsize=[4.1 * ncol, 3 * nrow])
    else:
        fig = plt.figure(figsize=[4.7 * ncol, 3 * nrow])
        self.get_focus()

    gs = mpl.gridspec.GridSpec(nrows=int(nrow), ncols=int(ncol),
                            hspace=0.35, wspace=0.3)

    mdict = self.srm_mdict

    for g, a in zip(gs, analytes):
        if not datarange:
            ax = fig.add_axes(g.get_position(fig))
            axes.append((ax,))
        else:
            f = 0.8
            p0 = g.get_position(fig)
            p1 = [p0.x0, p0.y0, p0.width * f, p0.height]
            p2 = [p0.x0 + p0.width * f, p0.y0, p0.width * (1 - f), p0.height]
            ax = fig.add_axes(p1)
            axh = fig.add_axes(p2)
            axes.append((ax, axh))
        
        if gTime is None:
            sub = idx[a]
        else:
            sub = idx[a, :, :, gTime]
        x = self.srmtabs.loc[sub, 'meas_mean'].values
        xe = self.srmtabs.loc[sub, 'meas_err'].values
        y = self.srmtabs.loc[sub, 'srm_mean'].values
        ye = self.srmtabs.loc[sub, 'srm_err'].values
        srm = self.srmtabs.loc[sub].index.get_level_values('SRM')
        
        # plot calibration data
        for s, m in mdict.items():
            ind = srm == s
            ax.errorbar(x[ind], y[ind], xerr=xe[ind], yerr=ye[ind],
                        color=self.cmaps[a], alpha=0.6,
                        lw=0, elinewidth=1, marker=m, #'o',
                        capsize=0, markersize=5, label='_')

        # work out axis scaling
        if not loglog:
            xmax = np.nanmax(x + xe)
            ymax = np.nanmax(y + ye)
            if any(x - xe < 0):
                xmin = np.nanmin(x - xe)
                xpad = (xmax - xmin) * 0.05
                xlim = [xmin - xpad, xmax + xpad]
            else:
                xlim = [0, xmax * 1.05]

            if any(y - ye < 0):
                ymin = np.nanmin(y - ye)
                ypad = (ymax - ymin) * 0.05
                ylim = [ymin - ypad, ymax + ypad]
            else:
                ylim = [0, ymax * 1.05]

        else:
            xd = self.srmtabs.loc[a, 'meas_mean'][self.srmtabs.loc[a, 'meas_mean'] > 0].values
            yd = self.srmtabs.loc[a, 'srm_mean'][self.srmtabs.loc[a, 'srm_mean'] > 0].values

            xlim = [10**np.floor(np.log10(np.nanmin(xd))),
                    10**np.ceil(np.log10(np.nanmax(xd)))]
            ylim = [10**np.floor(np.log10(np.nanmin(yd))),
                    10**np.ceil(np.log10(np.nanmax(yd)))]

            # scale sanity checks
            if xlim[0] == xlim[1]:
                xlim[0] = ylim[0]
            if ylim[0] == ylim[1]:
                ylim[0] = xlim[0]

            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # visual warning if any values < 0
        if xlim[0] < 0:
            ax.axvspan(xlim[0], 0, color=(1,0.8,0.8), zorder=-1)
        if ylim[0] < 0:
            ax.axhspan(ylim[0], 0, color=(1,0.8,0.8), zorder=-1)
        if any(x < 0) or any(y < 0):
            ax.text(.5, .5, 'WARNING: Values below zero.', color='r', weight='bold',
                    ha='center', va='center', rotation=40, transform=ax.transAxes, alpha=0.6)

        # calculate line and R2
        if loglog:
            x = np.logspace(*np.log10(xlim), 100)
        else:
            x = np.array(xlim)
        
        if gTime is None:
            coefs = self.calib_params.loc[:, a]
        else:
            coefs = self.calib_params.loc[gTime, a]
        
        m = np.nanmean(coefs['m'])
        m_nom = nominal_values(m)
        # calculate case-specific paramers
        if 'c' in coefs:
            c = np.nanmean(coefs['c'])
            c_nom = nominal_values(c)
            # calculate R2
            ym = self.srmtabs.loc[a, 'meas_mean'] * m_nom + c_nom
            R2 = R2calc(self.srmtabs.loc[a, 'srm_mean'], ym, force_zero=False)
            # generate line and label
            line = x * m_nom + c_nom
            label = 'y = {:.2e} x'.format(m)
            if c > 0:
                label += '\n+ {:.2e}'.format(c)
            else:
                label += '\n {:.2e}'.format(c)
        else:
            # calculate R2
            ym = self.srmtabs.loc[a, 'meas_mean'] * m_nom
            R2 = R2calc(self.srmtabs.loc[a, 'srm_mean'], ym, force_zero=True)
            # generate line and label
            line = x * m_nom
            label = 'y = {:.2e} x'.format(m)

        # plot line of best fit
        ax.plot(x, line, color=(0, 0, 0, 0.5), ls='dashed')

        # add R2 to label
        if round(R2, 3) == 1:
            label = '$R^2$: >0.999\n' + label
        else:
            label = '$R^2$: {:.3f}\n'.format(R2) + label

        ax.text(.05, .95, pretty_element(a), transform=ax.transAxes,
                weight='bold', va='top', ha='left', size=12)
        ax.set_xlabel('counts/counts ' + self.internal_standard)
        ax.set_ylabel('mol/mol ' + self.internal_standard)
        # write calibration equation on graph happens after data distribution

        # plot data distribution historgram alongside calibration plot
        if datarange:
            # isolate data
            meas = nominal_values(self.focus[a])
            meas = meas[~np.isnan(meas)]

            # check and set y scale
            if np.nanmin(meas) < ylim[0]:
                if loglog:
                    mmeas = meas[meas > 0]
                    ylim[0] = 10**np.floor(np.log10(np.nanmin(mmeas)))
                else:
                    ylim[0] = 0
                ax.set_ylim(ylim)

            m95 = np.percentile(meas[~np.isnan(meas)], 95) * 1.05
            if m95 > ylim[1]:
                if loglog:
                    ylim[1] = 10**np.ceil(np.log10(m95))
                else:
                    ylim[1] = m95

            # hist
            if loglog:
                bins = np.logspace(*np.log10(ylim), 30)
            else:
                bins = np.linspace(*ylim, 30)

            axh.hist(meas, bins=bins, orientation='horizontal',
                        color=self.cmaps[a], lw=0.5, alpha=0.5)

            if loglog:
                axh.set_yscale('log')
            axh.set_ylim(ylim)  # ylim of histogram axis
            ax.set_ylim(ylim)  # ylim of calibration axis
            axh.set_xticks([])
            axh.set_yticklabels([])

        # write calibration equation on graph
        cmax = np.nanmax(y)
        if cmax / ylim[1] > 0.5:
            ax.text(0.98, 0.04, label, transform=ax.transAxes,
                    va='bottom', ha='right')
        else:
            ax.text(0.02, 0.75, label, transform=ax.transAxes,
                    va='top', ha='left')

    if srm_group is None:
        title = 'All SRMs'
    else:
        title = 'SRM Group {:} (centre at {:.1f}s)'.format(srm_group, gTime)
    axes[0][0].set_title(title, loc='left', weight='bold', fontsize=12)
            
    # SRM legend
    ax = fig.add_axes(gs[-1].get_position(fig))
    for lab, m in mdict.items():
        ax.scatter([],[],marker=m, label=lab, color=(0,0,0,0.6))
    ax.legend()
    ax.axis('off')

    if save:
        fig.savefig(self.report_dir + '/calibration.pdf')

    return fig, axes

# def calibration_drift_plot(self, analytes=None, ncol=3, save=True):
#     """
#     Plot calibration slopes through the entire session.

#     Parameters
#     ----------
#     self : latools.analyse
#         Analyse object, containing 
#     analytes : optional, array_like or str
#         The analyte(s) to plot. Defaults to all analytes.
#     ncol : int
#         Number of columns of plots
#     save : bool
#         Whether or not to save the plot.

#     Returns
#     -------
#     (fig, axes)
#     """
#     if not hasattr(self, 'calib_params'):
#         raise ValueError('Please run calibrate before making this plot.')

#     if analytes is None:
#         analytes = [a for a in self.analytes if self.internal_standard not in a]

#     ncol = int(ncol)
#     n = len(analytes)
#     nrow = calc_nrow(n, ncol)

#     axes = []

#     fig = plt.figure(figsize=[6 * ncol, 3 * nrow])

#     gs = mpl.gridspec.GridSpec(nrows=int(nrow), ncols=int(ncol),
#                                hspace=0.35, wspace=0.3)

#     cp = self.calib_params

#     for g, a in zip(gs, analytes):
#         ax = fig.add_axes(g.get_position(fig))
#         axes.append(ax)

#         ax.plot(cp.index, nominal_values(cp.loc[:, (a, 'm')]), color=self.cmaps[a])
#         ax.fill_between(cp.index, 
#                         nominal_values(cp.loc[:, (a, 'm')]) - std_devs(cp.loc[:, (a, 'm')]),
#                         nominal_values(cp.loc[:, (a, 'm')]) + std_devs(cp.loc[:, (a, 'm')]),
#                         color=self.cmaps[a], alpha=0.2, lw=0)


#         ax.text(.05, .95, pretty_element(a), transform=ax.transAxes,
#                 weight='bold', va='top', ha='left', size=12)
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('mol/mol ' + self.internal_standard)
    
#     if save:
#         fig.savefig(self.report_dir + '/calibration_drift.pdf')

#     return fig, axes

def filter_report(Data, filt=None, analytes=None, savedir=None, nbin=5):
    """
    Visualise effect of data filters.

    Parameters
    ----------
    filt : str
        Exact or partial name of filter to plot. Supports
        partial matching. i.e. if 'cluster' is specified, all
        filters with 'cluster' in the name will be plotted.
        Defaults to all filters.
    analyte : str
        Name of analyte to plot.
    save : str
        file path to save the plot

    Returns
    -------
    (fig, axes)
    """
    if filt is None or filt == 'all':
        sets = Data.filt.sets
    else:
        sets = {k: v for k, v in Data.filt.sets.items() if any(filt in f for f in v)}

    regex = re.compile('^([0-9]+)_([A-Za-z0-9-]+)_'
                    '([A-Za-z0-9-]+)[_$]?'
                    '([a-z0-9]+)?')

    cm = plt.cm.get_cmap('Spectral')
    ngrps = len(sets)

    if analytes is None:
        analytes = Data.analytes
    elif isinstance(analytes, str):
        analytes = [analytes]

    axes = []
    for analyte in analytes:
        if analyte != Data.internal_standard:
            fig = plt.figure()

            for i in sorted(sets.keys()):
                filts = sets[i]
                nfilts = np.array([re.match(regex, f).groups() for f in filts])
                fgnames = np.array(['_'.join(a) for a in nfilts[:, 1:3]])
                fgrp = np.unique(fgnames)[0]

                fig.set_size_inches(10, 3.5 * ngrps)
                h = .8 / ngrps

                y = nominal_values(Data.focus[analyte])
                yh = y[~np.isnan(y)]

                m, u = unitpicker(np.nanmax(y),
                                denominator=Data.internal_standard,
                                focus_stage=Data.focus_stage)

                axs = tax, hax = (fig.add_axes([.1, .9 - (i + 1) * h, .6, h * .98]),
                                fig.add_axes([.7, .9 - (i + 1) * h, .2, h * .98]))
                axes.append(axs)

                # get variables
                fg = sets[i]
                cs = cm(np.linspace(0, 1, len(fg)))
                fn = ['_'.join(x) for x in nfilts[:, (0, 3)]]
                an = nfilts[:, 0]
                bins = np.linspace(np.nanmin(y), np.nanmax(y), len(yh) // nbin) * m

                if 'DBSCAN' in fgrp:
                    # determine data filters
                    core_ind = Data.filt.components[[f for f in fg
                                                    if 'core' in f][0]]
                    other = np.array([('noise' not in f) & ('core' not in f)
                                    for f in fg])
                    tfg = fg[other]
                    tfn = fn[other]
                    tcs = cm(np.linspace(0, 1, len(tfg)))

                    # plot all data
                    hax.hist(m * yh, bins, alpha=0.2, orientation='horizontal',
                            color='k', lw=0)
                    # legend markers for core/member
                    tax.scatter([], [], s=20, label='core', color='w', lw=0.5, edgecolor='k')
                    tax.scatter([], [], s=7.5, label='member', color='w', lw=0.5, edgecolor='k')
                    # plot noise
                    try:
                        noise_ind = Data.filt.components[[f for f in fg
                                                        if 'noise' in f][0]]
                        tax.scatter(Data.Time[noise_ind], m * y[noise_ind],
                                    lw=1, color='k', s=10, marker='x',
                                    label='noise', alpha=0.6)
                    except:
                        pass

                    # plot filtered data
                    for f, c, lab in zip(tfg, tcs, tfn):
                        ind = Data.filt.components[f]
                        tax.scatter(Data.Time[~core_ind & ind],
                                    m * y[~core_ind & ind], lw=.5, color=c, s=5, edgecolor='k')
                        tax.scatter(Data.Time[core_ind & ind],
                                    m * y[core_ind & ind], lw=.5, color=c, s=15, edgecolor='k',
                                    label=lab)
                        hax.hist(m * y[ind][~np.isnan(y[ind])], bins, color=c, lw=0.1,
                                orientation='horizontal', alpha=0.6)

                else:
                    # plot all data
                    tax.scatter(Data.Time, m * y, color='k', alpha=0.2, lw=0.1,
                                s=20, label='excl')
                    hax.hist(m * yh, bins, alpha=0.2, orientation='horizontal',
                             color='k', lw=0)

                    # plot filtered data
                    for f, c, lab in zip(fg, cs, fn):
                        ind = Data.filt.components[f]
                        tax.scatter(Data.Time[ind], m * y[ind],
                                    edgecolor=(0,0,0,0), color=c, s=15, label=lab)
                        hax.hist(m * y[ind][~np.isnan(y[ind])], bins, color=c, lw=0.1,
                                orientation='horizontal', alpha=0.6)

                if 'thresh' in fgrp and analyte in fgrp:
                    tax.axhline(Data.filt.params[fg[0]]['threshold'] * m,
                                ls='dashed', zorder=-2, alpha=0.5, color='k')
                    hax.axhline(Data.filt.params[fg[0]]['threshold'] * m,
                                ls='dashed', zorder=-2, alpha=0.5, color='k')

                # formatting
                for ax in axs:
                    mn = np.nanmin(y) * m
                    mx = np.nanmax(y) * m
                    rn = mx - mn
                    ax.set_ylim(mn - .05 * rn, mx + 0.05 * rn)

                # legend
                hn, la = tax.get_legend_handles_labels()
                hax.legend(hn, la, loc='upper right', scatterpoints=1)

                tax.text(.02, .98, Data.sample + ': ' + fgrp, size=12,
                        weight='bold', ha='left', va='top',
                        transform=tax.transAxes)
                tax.set_ylabel(pretty_element(analyte) + ' (' + u + ')')
                tax.set_xticks(tax.get_xticks()[:-1])
                hax.set_yticklabels([])

                if i < ngrps - 1:
                    tax.set_xticklabels([])
                    hax.set_xticklabels([])
                else:
                    tax.set_xlabel('Time (s)')
                    hax.set_xlabel('n')

        if isinstance(savedir, str):
            fig.savefig(savedir + '/' + Data.sample + '_' +
                        analyte + '.pdf')
            plt.close(fig)

    return fig, axes

def correlation_plot(self, corr=None):
    if len(self.correlations) == 0:
        raise ValueError("No calculated correlations. Run threshold_correlation first.")
    
    if corr is None:
        if len(self.correlations) == 1:
            corr = list(self.correlations.keys())[0]
    
    if corr not in self.correlations:
        raise ValueError("{:} not founself. Please use one of [{:}]".format(corr, [', '.join(c) for c in self.correlations.keys()]))
    
    x_analyte, y_analyte, window = corr.split('_')
    r, p = self.correlations[corr]
    
    fig, axs = plt.subplots(3, 1, figsize=[7, 5], sharex=True)
    
    # plot analytes
    ax = axs[0]
        
    ax.plot(self.Time, nominal_values(self.focus[x_analyte]), color=self.cmap[x_analyte], label=x_analyte)
    ax.plot(self.Time, nominal_values(self.focus[y_analyte]), color=self.cmap[y_analyte], label=y_analyte)
    
    ax.set_yscale('log')
    ax.legend()
    ax.set_ylabel('Signals')
    
    # plot r
    ax = axs[1]
    ax.plot(self.Time, r)
    ax.set_ylabel('Pearson R')
    
    # plot p
    ax = axs[2]
    ax.plot(self.Time, p)
    ax.set_ylabel('pignificance Level (p)')

    ax.set_xlabel('Time (s)')
    
    fig.tight_layout()
    
    return fig, axs