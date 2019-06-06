import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from .stats import fmt_RSS
from .plots import element_colour, rangecalc, rangecalcx, bland_altman, get_panel_bounds

def fmt_el(el):
    e = re.match('.*?([A-z]+).*?', el).groups()[0]
    m = re.match('.*?([0-9]+).*?', el).groups()[0]
    return e + m

def comparison_plots(df, els=['Mg', 'Sr', 'Al', 'Mn', 'Fe', 'Cu', 'Zn', 'B']):
    """
    Function for plotting Test User and LAtools data comparison.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing reference ('X/Ca_r'), test user 
        ('X/Ca_t') and LAtools ('X123') data.
    els : list
        list of elements (names only) to plot.
    """
    
    # get corresponding analyte and ratio names
    As = []
    Rs = []
    analytes = [c for c in df.columns if ('/' not in c)]
    ratios = [c for c in df.columns if ('/' in c)]

    for e in els:
        if e == 'Sr':
            As.append('88Sr')
        elif e == 'Mg':
            As.append('24Mg')
        else:
            As.append([a for a in analytes if e in a][0])
        Rs.append([r for r in ratios if e in r][0])

    fig, axs = plt.subplots(len(els), 2, figsize=(5, len(els) * 2))
    
    for i, (e, a) in enumerate(zip(Rs, As)):
        c = element_colour(fmt_el(a))
        u = 'mmol/mol'

        lax, hax = axs[i]
        
        x = df.loc[:, e].values
        yl = df.loc[:, a].values
        
        # calculate residuals
        rl = yl - x
        
        # plot residuals
        lax.scatter(x, yl, c=c, s=15, lw=0.5, edgecolor='k', alpha=0.5)
        
        # plot PDFs
        rl = rl[~np.isnan(rl)]
        lims = np.percentile(rl, [99, 1])
        lims += np.ptp(lims * np.array((-1.25, 1.25))
        bins = np.linspace(*lims, 100)
        kdl = stats.gaussian_kde(rl, .4)
        hax.fill_between(bins, kdl(bins), facecolor=c, alpha=0.7, edgecolor='k', lw=0.5, label='LAtools')
        hax.set_ylim([0, hax.get_ylim()[-1]])
        hax.set_xlim(lims)
        hax.axvline(0, c='k', ls='dashed', alpha=0.6)
        hax.set_yticklabels([])
        hax.set_ylabel('Density')
        
        # axis labels, annotations and limits
        lax.set_ylabel(e + ' ('+ u + ')')
        lax.text(.02,.98,fmt_RSS(rl), fontsize=8,
                 ha='left', va='top', transform=lax.transAxes)

        xlim = np.percentile(x[~np.isnan(x)], [0, 98])

        lax.set_xlim(xlim)
        lax.set_ylim(xlim)
        lax.plot(xlim, xlim, c='k', ls='dashed', alpha=0.6)
        
        for ax in axs[i]:
            if ax.is_last_row():
                hax.set_xlabel('Residual')
                lax.set_xlabel('Iolite User')
                hax.legend(fontsize=8)

            if ax.is_first_row():
                lax.set_title('LAtools', loc='left')
            
    fig.tight_layout()
    return fig, axs

def residual_plots(df, rep_stats=None, els=['Mg', 'Sr', 'Al', 'Mn', 'Fe', 'Cu', 'Zn', 'B']):
    """
    Function for plotting Test User and LAtools data comparison.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing reference ('X/Ca_r'), test user 
        ('X/Ca_t') and LAtools ('X123') data.
    rep_stats : dict
        Reproducibility stats of the reference data produced by
        `pairwise_reproducibility`
    els : list
        list of elements (names only) to plot.
    """
    
    # get corresponding analyte and ratio names
    As = []
    Rs = []
    analytes = [c for c in df.columns if ('/' not in c)]
    ratios = [c for c in df.columns if ('/' in c)]

    for e in els:
        if e == 'Sr':
            As.append('88Sr')
        elif e == 'Mg':
            As.append('24Mg')
        else:
            As.append([a for a in analytes if e in a][0])
        Rs.append([r for r in ratios if e in r][0])
    
    fig, axs = plt.subplots(len(els), 2, figsize=(5, len(els) * 2))
    
    for i, (e, a) in enumerate(zip(Rs, As)):
        lax, hax = axs[i]
        
        x = df.loc[:, e].values
        yl = df.loc[:, a].values
        
        c = element_colour(fmt_el(a))
        u = 'mmol/mol'
        
        # calculate residuals
        rl = yl - x
        
        # plot residuals
        lax.scatter(x, rl, c=c, s=15, lw=0.5, edgecolor='k', alpha=0.5)
        
        # plot PDFs
        rl = rl[~np.isnan(rl)]
        lims = np.percentile(rl, [99, 1])
        lims += np.ptp(lims) * np.array((-1.25, 1.25))
        bins = np.linspace(*lims, 100)
        kdl = stats.gaussian_kde(rl, .4)
        hax.fill_betweenx(bins, kdl(bins), facecolor=c, alpha=0.7, edgecolor='k', lw=0.5, label='LAtools')
        hax.set_xlim([0, hax.get_xlim()[-1]])
                
        # axis labels, annotations and limits
        lax.set_ylabel(e + ' ('+ u + ')')
        lax.text(.02,.02,fmt_RSS(rl), fontsize=8,
                 ha='left', va='bottom', transform=lax.transAxes)

        xlim = np.percentile(x[~np.isnan(x)], [0, 98])
        lax.set_xlim(xlim)
        
        for ax in axs[i]:
            ax.set_ylim(lims)
            # zero line and 2SD precision
            ax.axhline(0, c='k', ls='dashed', alpha=0.6)
            if rep_stats is not None:
                ax.axhspan(-rep_stats[e][0] * 2, rep_stats[e][0] * 2, color=(0,0,0,0.2), zorder=-1)
            
            if not ax.is_first_col():
                ax.set_yticklabels([])
                
            if ax.is_last_row():
                hax.set_xlabel('Density')
                lax.set_xlabel('Iolite User')

            if ax.is_first_row():
                lax.set_title('LAtools', loc='left')
            
    fig.tight_layout()

    return fig, axs

def bland_altman_plots(df, rep_stats=None, els=['Mg', 'Sr', 'Al', 'Mn', 'Fe', 'Cu', 'Zn', 'B']):
    # get corresponding analyte and ratio names
    As = []
    Rs = []
    analytes = [c for c in df.columns if ('/' not in c)]
    ratios = [c for c in df.columns if ('/' in c)]

    for e in els:
        if e == 'Sr':
            As.append('88Sr')
        elif e == 'Mg':
            As.append('24Mg')
        else:
            As.append([a for a in analytes if e in a][0])
        Rs.append([r for r in ratios if e in r][0])
    
    cols = 2
    rows = len(els) // cols
    bounds = [.05,.05,.9,.9]
    p = 0.66
    
    frame = [.25, .25, .75, .75]

    baframe = [frame[0], frame[1], frame[2] * p, frame[3]]
    rframe = [frame[0] + frame[2] * p * 1.02, frame[1], frame[2] * (1 - p * 1.02), frame[3]]

    fig = plt.figure(figsize=(cols * 2.8, rows * 1.5))
    axs = []
    
    for i, (e, a) in enumerate(zip(Rs, As)):
        c = element_colour(fmt_el(a))
        u = 'mmol/mol'
        
        row = i // cols
        col = i % cols

        lax = fig.add_axes(get_panel_bounds(row, col, bounds, rows, cols, baframe))
        dax = fig.add_axes(get_panel_bounds(row, col, bounds, rows, cols, rframe))
        axs.append([lax, dax])

        dax.set_yticklabels([])
        dax.set_xticklabels([])

        x = df.loc[:, e].values
        yl = df.loc[:, a].values
        r = yl - x

        ylim = rangecalcx(r, pad=0.5)
        lax.set_ylim(ylim)
        dax.set_ylim(ylim)

        lax.text(.03, .02, e, transform=lax.transAxes, ha='left', va='bottom')

        # draw Bland-Altman plots
        if rep_stats is None:
            CI = None
        else:
            CI = rep_stats[e][0]
        bland_altman(x, yl, interval=.75, indep_conf=CI, ax=lax, c=c)

        if row == rows:
            lax.set_xlabel('Mean')
        else:
            lax.set_xlabel('')
        
        if col == 0:
            lax.set_ylabel('Residual')
        else:
            lax.set_ylabel('')

        # draw residual PDFs
        # remove NaNs
        r = r[~np.isnan(r)]
        # calculate bins
        bins = np.linspace(*ylim, 100)
        # calculate KDEs
        kde = stats.gaussian_kde(r, .4)
        # draw KDEs
        dax.fill_betweenx(bins, kde(bins), facecolor=c, alpha=0.5, edgecolor='k', lw=0.75)
        # limits and horizontal line
        dax.set_xlim([0, dax.get_xlim()[-1] * 1.1])
        dax.axhline(0, ls='dashed', c='k', alpha=0.6, zorder=-1)

        # lax.set_ylabel(e + ' ('+ u + ')\nResidual')
        
        # if lax.is_last_row():
        #     lax.set_xlabel('Mean')
        # else:
        #     lax.set_xlabel('')

        # if lax.is_first_row() and lax.is_first_col():
        #     lax.set_title('LAtools', loc='left')

    for ax in axs[-2:]:
       ax[0].set_xlabel('X/Ca')
       ax[1].set_xlabel('Resid.\nDens')
        
    return fig, np.array(axs)