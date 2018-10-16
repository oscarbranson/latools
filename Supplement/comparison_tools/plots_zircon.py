import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from .stats import fmt_RSS
from .plots import rangecalcx, bland_altman, get_panel_bounds

def fmt_el(el):
    e = re.match('.*?([A-z]+).*?', el).groups()[0]
    m = re.match('.*?([0-9]+).*?', el).groups()[0]
    return e + m

def bland_altman_plots(df, rep_stats=None, els=None, c=(0,0,0,0.6)):
    if els is None:
        els = ['Li', 'Mg', 'Al', 'P', 'Ti', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Sm',
               'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Pb', 'Th',
               'U']
    
    cols = 4
    rows = len(els) // cols
    bounds = [.05,.05,.9,.9]
    p = 0.66
    
    frame = [.25, .25, .75, .75]

    baframe = [frame[0], frame[1], frame[2] * p, frame[3]]
    rframe = [frame[0] + frame[2] * p * 1.02, frame[1], frame[2] * (1 - p * 1.02), frame[3]]

    fig = plt.figure(figsize=(cols * 2.8, rows * 1.5))
    axs = []

    for i, e in enumerate(els):
        # lax = axs.flatten()[i]
        u = 'ppm'

        row = i // cols
        col = i % cols

        lax = fig.add_axes(get_panel_bounds(row, col, bounds, rows, cols, baframe))
        dax = fig.add_axes(get_panel_bounds(row, col, bounds, rows, cols, rframe))
        axs.append([lax, dax])

        dax.set_yticklabels([])
        dax.set_xticklabels([])

        lax.text(.03, .02, e, transform=lax.transAxes, ha='left', va='bottom')

        x1 = df.loc[:, e + '_la'].values
        x2 = df.loc[:, e + '_rd'].values
        r = x2 - x1

        ylim = rangecalcx(r, pad=0.5)
        lax.set_ylim(ylim)
        dax.set_ylim(ylim)
        
        # draw Bland-Altman plots
        if rep_stats is None:
            CI = None
        else:
            CI = rep_stats[e][0]

        bland_altman(x1, x2, interval=.75, indep_conf=CI, ax=lax, c=c)
        
        # lax.set_ylabel(e + ' ('+ u + ')\nResidual')
        
        if row == (rows - 1):
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


    for ax in axs[-4:]:
        ax[0].set_xlabel('[X]')
        ax[1].set_xlabel('Resid.\nDens')
    #     lax.set_title(e[-3:], loc='left')

    #     # if lax.is_first_row() and lax.is_first_col():
    #         # lax.set_title('LAtools', loc='left')
            
    # fig.tight_layout()

    return fig, np.array(axs)