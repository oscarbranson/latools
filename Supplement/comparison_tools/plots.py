import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from .stats import fmt_RSS

def element_colour(el):
    cdict = {'B11': [0.58039216, 0.40392157, 0.74117647, 1.],
             'Mg24': [0.12156863, 0.46666667, 0.70588235, 1.],
             'Mg25': [0.68235294, 0.78039216, 0.90980392, 1.],
             'Al27': [0.49803922, 0.49803922, 0.49803922, 1.],
             'Mn55': [0.54901961, 0.3372549 , 0.29411765, 1.],
             'Fe57': [0.76862745, 0.61176471, 0.58039216, 1.],
             'Cu63': [0.89019608, 0.46666667, 0.76078431, 1.],
             'Zn66': [0.96862745, 0.71372549, 0.82352941, 1.],
             'Sr88': [1., 0.49803922, 0.05490196, 1.],
             'Ba138': [1., 0.73333333, 0.47058824, 1.]}
    return cdict[el]

def rangecalc(x, y=None, pad=0.05):
    """
    Calculate padded range limits for axes.
    """        
    mn = np.nanmin([np.nanmin(x), np.nanmin(y)])
    mx = np.nanmax([np.nanmax(x), np.nanmax(y)])
    rn = mx - mn
    
    return (mn - pad * rn, mx + pad * rn)

def rangecalcx(x, pad=0.05):
    """
    Calculate padded range limits for axes.
    """        
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    rn = mx - mn

    return (mn - pad * rn, mx + pad * rn)

def get_panel_bounds(row, col, bounds=[.1,.1,.8,.8], rows=4, cols=4, frame=[.1,.1,.9,.9]):
    pw = bounds[2] / cols
    ph = bounds[3] / rows
    pl = bounds[0] + col * pw
    pb = bounds[1] + bounds[3] - (row + 1) * ph

    al = pl + frame[0] * pw
    ab = pb + frame[1] * ph
    aw = pw * frame[2]
    ah = ph * frame[3]

    return [al, ab, aw, ah]

def comparison_plots(df, els=['Mg', 'Sr', 'Ba', 'Al', 'Mn']):
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
    analytes = [c for c in df.columns if ('_r' not in c) and ('_t' not in c)]
    ratios = [c for c in df.columns if ('_r' in c)]

    for e in els:
        if e == 'Sr':
            As.append('Sr88')
        elif e == 'Mg':
            As.append('Mg24')
        else:
            As.append([a for a in analytes if e in a][0])
        Rs.append([r for r in ratios if e in r][0][:-2])
    
    fig, axs = plt.subplots(len(els), 3, figsize=(6.5, len(els) * 2))
    
    for i, (e, a) in enumerate(zip(Rs, As)):
        if a == 'Ba138':
            m = 1e3
            u = '$\mu$mol/mol'
        else:
            m = 1
            u = 'mmol/mol'
        
        c = element_colour(a)
        
        tax, lax, hax = axs[i]
        
        x = df.loc[:, e + '_r'].values * m
        yt = df.loc[:, e + '_t'].values * m
        yl = df.loc[:, a].values * m
        
        # calculate residuals
        rt = yt - x
        rl = yl - x
        
        # plot residuals
        tax.scatter(x, yt, c=c, s=15, lw=0.5, edgecolor='k', alpha=0.5)
        lax.scatter(x, yl, c=c, s=15, lw=0.5, edgecolor='k', alpha=0.5)
        
        # plot PDFs
        rt = rt[~np.isnan(rt)]
        rl = rl[~np.isnan(rl)]
        lims = np.percentile(np.hstack([rt, rl]), [99, 1])
        lims += np.ptp(lims) * np.array((-1.25, 1.25))
        bins = np.linspace(*lims, 100)
        kdt = stats.gaussian_kde(rt, .4)
        kdl = stats.gaussian_kde(rl, .4)
        hax.fill_between(bins, kdl(bins), facecolor=c, alpha=0.7, edgecolor='k', lw=0.5, label='LAtools')
        hax.fill_between(bins, kdt(bins), facecolor=c, alpha=0.4, edgecolor='k', lw=0.5, label='Test User')
        hax.set_ylim([0, hax.get_ylim()[-1]])
        hax.set_xlim(lims)
        hax.axvline(0, c='k', ls='dashed', alpha=0.6)
        # hax.set_yticklabels([])
        hax.set_ylabel('Density')
        
        # axis labels, annotations and limits
        tax.set_ylabel(e + ' ('+ u + ')')
        tax.text(.02,.98,fmt_RSS(rt), fontsize=8,
                 ha='left', va='top', transform=tax.transAxes)
        lax.text(.02,.98,fmt_RSS(rl), fontsize=8,
                 ha='left', va='top', transform=lax.transAxes)

        xlim = np.percentile(x[~np.isnan(x)], [0, 98])
        for ax in [tax, lax]:
            ax.set_xlim(xlim)
            ax.set_ylim(xlim)
            
            ax.plot(xlim, xlim, c='k', ls='dashed', alpha=0.6)
        
        for ax in axs[i]:
            if ax.is_last_row():
                hax.set_xlabel('Residual')
                tax.set_xlabel('Reference User')
                lax.set_xlabel('Reference User')
                hax.legend(fontsize=8)

            if ax.is_first_row():
                tax.set_title('Manual Test User', loc='left')
                lax.set_title('LAtools Test User', loc='left')
            
    fig.tight_layout()
    return fig, axs
def residual_plots(df, rep_stats=None, els=['Mg', 'Sr', 'Ba', 'Al', 'Mn']):

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
    analytes = [c for c in df.columns if ('_r' not in c) and ('_t' not in c)]
    ratios = [c for c in df.columns if ('_r' in c)]

    for e in els:
        if e == 'Sr':
            As.append('Sr88')
        elif e == 'Mg':
            As.append('Mg24')
        else:
            As.append([a for a in analytes if e in a][0])
        Rs.append([r for r in ratios if e in r][0][:-2])
    
    fig, axs = plt.subplots(len(els), 3, figsize=(6.5, len(els) * 2))
    
    for i, (e, a) in enumerate(zip(Rs, As)):
        if a == 'Ba138':
            m = 1e3
            u = '$\mu$mol/mol'
        else:
            m = 1
            u = 'mmol/mol'
        
        tax, lax, hax = axs[i]
        
        x = df.loc[:, e + '_r'].values * m
        yt = df.loc[:, e + '_t'].values * m
        yl = df.loc[:, a].values * m
        
        # calculate residuals
        rt = yt - x
        rl = yl - x
        
        # plot residuals
        tax.scatter(x, rt, c=element_colour(a), s=15, lw=0.5, edgecolor='k', alpha=0.5)
        lax.scatter(x, rl, c=element_colour(a), s=15, lw=0.5, edgecolor='k', alpha=0.5)
        
        # plot PDFs
        rt = rt[~np.isnan(rt)]
        rl = rl[~np.isnan(rl)]
        lims = np.percentile(np.hstack([rt, rl]), [99, 1])
        lims += np.ptp(lims) * np.array((-1.25, 1.25))
        bins = np.linspace(*lims, 100)
        kdt = stats.gaussian_kde(rt, .4)
        kdl = stats.gaussian_kde(rl, .4)
        hax.fill_betweenx(bins, kdl(bins), facecolor=element_colour(a), alpha=0.7, edgecolor='k', lw=0.5, label='LAtools')
        hax.fill_betweenx(bins, kdt(bins), facecolor=element_colour(a), alpha=0.4, edgecolor='k', lw=0.5, label='Test User')
        hax.set_xlim([0, hax.get_xlim()[-1]])
                
        # axis labels, annotations and limits
        tax.set_ylabel(e + ' ('+ u + ')')
        tax.text(.02,.02,fmt_RSS(rt), fontsize=8,
                 ha='left', va='bottom', transform=tax.transAxes)
        lax.text(.02,.02,fmt_RSS(rl), fontsize=8,
                 ha='left', va='bottom', transform=lax.transAxes)

        xlim = np.percentile(x[~np.isnan(x)], [0, 98])
        for ax in [tax, lax]:
            ax.set_xlim(xlim)
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
                tax.set_xlabel('Reference User')
                lax.set_xlabel('Reference User')

            if ax.is_first_row():
                tax.set_title('Manual Test User', loc='left')
                lax.set_title('LAtools Test User', loc='left')
            
    fig.tight_layout()

    return fig, axs

def bland_altman(x, y, interval=None, indep_conf=None, ax=None, c=None, **kwargs):
    """
    Draw a Bland-Altman plot of x and y data.
    
    https://en.wikipedia.org/wiki/Bland%E2%80%93Altman_plot
    
    Parameters
    ----------
    x, y : array-like
        x and y data to compare.
    interval : float
        Percentile band to draw on the residuals.
    indep_conf : float
        Independently determined confidence interval
        to draw on the plot
    ax : matplotlib.axesobject
        The axis on which to draw the plot
    **kwargs
        Passed to ax.scatter
    """
    ret = False
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ret = True
        
    # NaN screening
    ind = ~(np.isnan(x) | np.isnan(y))
    x = x[ind]
    y = y[ind]
    
    xy_mean = (x + y) / 2
    xy_resid = (y - x)

    ax.scatter(xy_mean, xy_resid, lw=0.5, edgecolor='k', alpha=0.6, c=c, s=15, **kwargs)

    # markup
    ax.axhline(0, ls='dashed', c='k', alpha=0.6, zorder=-1)
    
    ax.axhline(np.median(xy_resid), ls='dashed', c=c, alpha=0.8)
    
    if interval is not None:
        perc = 100 - interval * 100
        ints = [perc / 2, 100 - perc / 2]
        lims = np.percentile(xy_resid, ints)
        ax.axhspan(*lims, color=c, alpha=0.1, zorder=-3)
    
    if indep_conf is not None:
        ax.axhspan(-indep_conf, indep_conf, color=(0,0,0,0.1), zorder=-2)

    # labels
    ax.set_ylabel('y - x')
    ax.set_xlabel('mean (x, y)')
    
    if ret:
        return fig, ax

def bland_altman_plots(df, rep_stats=None, els=['Mg', 'Sr', 'Ba', 'Al', 'Mn']):
    # get corresponding analyte and ratio names
    As = []
    Rs = []
    analytes = [c for c in df.columns if ('_r' not in c) and ('_t' not in c)]
    ratios = [c for c in df.columns if ('_r' in c)]

    for e in els:
        if e == 'Sr':
            As.append('Sr88')
        elif e == 'Mg':
            As.append('Mg24')
        else:
            As.append([a for a in analytes if e in a][0])
        Rs.append([r for r in ratios if e in r][0][:-2])
    
    fig, axs = plt.subplots(len(els), 3, figsize=(6.5, len(els) * 2))
    
    for i, (e, a) in enumerate(zip(Rs, As)):
        if a == 'Ba138':
            m = 1e3
            u = '$\mu$mol/mol'
        else:
            m = 1
            u = 'mmol/mol'
        
        tax, lax, hax = axs[i]
        c=element_colour(a)

        x = df.loc[:, e + '_r'].values * m
        yt = df.loc[:, e + '_t'].values * m
        yl = df.loc[:, a].values * m
        
        # draw Bland-Altman plots
        if rep_stats is None:
            CI = None
        else:
            CI = rep_stats[e][0]
        bland_altman(x, yt, interval=.75, indep_conf=CI, ax=tax, c=c)
        bland_altman(x, yl, interval=.75, indep_conf=CI, ax=lax, c=c)
        
        xlim = (min(tax.get_xlim()[0], lax.get_xlim()[0]), max(tax.get_xlim()[1], lax.get_xlim()[1]))
        tax.set_xlim(xlim)
        lax.set_xlim(xlim)

        ylim = rangecalc(tax.get_ylim(), lax.get_ylim())

        # draw residual PDFs
        # calculate residuals
        rt = yt - x
        rl = yl - x
        # remove NaNs
        rt = rt[~np.isnan(rt)]
        rl = rl[~np.isnan(rl)]
        # calculate bins
        bins = np.linspace(*ylim, 100)
        # calculate KDEs
        kdt = stats.gaussian_kde(rt, .4)
        kdl = stats.gaussian_kde(rl, .4)
        # draw KDEs
        hax.fill_betweenx(bins, kdl(bins), facecolor=element_colour(a), alpha=0.8, edgecolor='k', lw=0.75, label='LAtools', zorder=-1)
        hax.fill_betweenx(bins, kdt(bins), facecolor=element_colour(a), alpha=0.4, edgecolor='k', lw=0.75, label='Manual', zorder=1)
        # limits and horizontal line
        hax.set_xlim([0, hax.get_xlim()[-1]])
        hax.axhline(0, ls='dashed', c='k', alpha=0.6, zorder=-1)

        for ax in axs[i]:
            ax.set_ylim(ylim)
                        
            if ax.is_first_col():
                ax.set_ylabel(e + ' ('+ u + ')\nResidual')
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            
            if ax.is_last_row():
                tax.set_xlabel('Mean')
                lax.set_xlabel('Mean')
                hax.set_xlabel('Residual Density')
                hax.legend()
            else:
                ax.set_xlabel('')

            if ax.is_first_row():
                tax.set_title('Manual Test User', loc='left')
                lax.set_title('LAtools Test User', loc='left')
                hax.set_title('Residuals', loc='left')

    fig.tight_layout()

    return fig, axs