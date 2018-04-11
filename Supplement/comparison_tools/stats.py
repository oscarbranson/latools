import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def fmt_RSS(x):
    """
    Calculate RSS and format as string.
    """
    return 'RSS: {sumsq:.2f}'.format(sumsq=np.sqrt(np.nansum((x)**2)))

def pairwise_reproducibility(df, plot=False):
    """
    Calculate the reproducibility of LA-ICPMS based on unique pairs of repeat analyses.
    
    Pairwise differences are fit with a half-Cauchy distribution, and the median and 
    95% confidence limits are returned for each analyte.
    
    Parameters
    ----------
    df : pandas.DataFrame
        A dataset
    
    plot : bool
        Whether or not to plot the resulting error distributions.
    
    Returns
    -------
    pdiffs : pandas.DataFrame
        Unique pairwise differences for all analytes.
    rep_dists : dict of scipy.stats.halfcauchy
        Half-Cauchy distribution objects fitted to the
        differences.
    rep_stats : dict of tuples
        The 50% and 95% quantiles of the half-cauchy
        distribution.
    (fig, axs) : matplotlib objects
        The figure. If not made, returnes (None, None) placeholder
    
    """
    
    ans = df.columns.values
    pdifs = []
    
    # calculate differences between unique pairs
    for ind, d in df.groupby(level=0):
        d.index = d.index.droplevel(0)

        difs = []
        for i, r in d.iterrows():
            t = d.loc[i+1:, :]
            difs.append(t[ans] - r[ans])

        pdifs.append(pd.concat(difs))
    pdifs = pd.concat(pdifs).abs()

    # calculate stats
    rep_stats = {}
    rep_dists = {}
    errfn = stats.halfcauchy
    
    for a in ans:
        d = pdifs.loc[:, a].dropna().values
        hdist = errfn.fit(d, floc=0)
        rep_dists[a] = errfn(*hdist)
        rep_stats[a] = rep_dists[a].ppf((0.5, 0.95))
    
    # make plot
    if not plot:
        return pdifs, rep_dists, rep_stats, (None, None)
    
    fig, axs = plt.subplots(1, len(ans), figsize=[len(ans) * 2, 2])
    for a, ax in zip(ans, axs):
        d = pdifs.loc[:, a].dropna().values
        hist, edges, _ = ax.hist(d, 30)
        ax.plot(edges, rep_dists[a].pdf(edges) * (sum(hist) * np.mean(np.diff(edges))))
        ax.set_title(a, loc='left')

    return pdifs, rep_dists, rep_stats, (fig, axs)

def comparison_stats(df, els=['Mg', 'Sr', 'Ba', 'Al', 'Mn']):
    """
    Compute comparison stats for test and LAtools data.
    
    Population-level similarity assessed by a Kolmogorov-Smirnov test.
    
    Individual similarity assessed by a pairwise Wilcoxon signed rank test.
    
    Trends in residuals assessed by regression analysis, where significance of
    the slope and intercept is determined by t-tests (both relative to zero).
    
    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing reference ('X/Ca_r'), test user 
        ('X/Ca_t') and LAtools ('X123') data.
    els : list
        list of elements (names only) to plot.
    
    Returns
    -------
    pandas.DataFrame
    
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
        
    yt_stats = []
    yl_stats = []
    
    for i, (e, a) in enumerate(zip(Rs, As)):
        if a == 'Ba138':
            m = 1e3
            u = '$\mu$mol/mol'
        else:
            m = 1
            u = 'mmol/mol'
        
        x = df.loc[:, e + '_r'].values * m
        yt = df.loc[:, e + '_t'].values * m
        yl = df.loc[:, a].values * m
        
        yt_stats.append(summary_stats(x, yt, e))
        yl_stats.append(summary_stats(x, yl, e))
    
    yt_stats = pd.concat(yt_stats).T
    yl_stats = pd.concat(yl_stats).T
    
    return pd.concat([yt_stats, yl_stats], keys=['Test User', 'LAtools']).T

def summary_stats(x, y, nm=None):
    """
    Compute summary statistics for paired x, y data.

    Tests
    -----

    Parameters
    ----------
    x, y : array-like
        Data to compare
    nm : str (optional)
        Index value of created dataframe.

    Returns
    -------
    pandas dataframe of statistics.
    """
    # create datafrane for results
    if isinstance(nm, str):
        nm = [nm]
    # cols = pd.MultiIndex.from_arrays([['', 'Pairwise', 'Pairwise', cat, cat, cat, cat],
    #                                   ['N', 'W', 'p', 'Median', 'IQR', 'W', 'p']])
#     cols = ['Median', 'IQR', 'CI95', 'L95', 'LQ', 'UQ', 'U95', 'N',
#             'Wilcoxon_stat', 'Wilcoxon_p',
#             'KS_stat', 'KS_p',
#             'LR_slope', 'LR_intercept', 'LR_slope_tvalue', 'LR_intercept_tvalue', 'LR_slope_p', 'LR_intercept_p', 'LR_R2adj']
#     out = pd.DataFrame(index=nm, columns=cols)
    
    cols = pd.MultiIndex.from_tuples([('Residual Summary', 'N'),
                                      ('Residual Summary', 'Median'),
                                      ('Residual Summary', 'LQ'),
                                      ('Residual Summary', 'IQR'),
                                      ('Residual Summary', 'UQ'),
                                      ('Residual Regression', 'Slope'),
                                      ('Residual Regression', 'Slope t'),
                                      ('Residual Regression', 'Slope p'),
                                      ('Residual Regression', 'Intercept'),
                                      ('Residual Regression', 'Intercept t'),
                                      ('Residual Regression', 'Intercept p'),
                                      ('Residual Regression', 'R2'),
                                      ('Kolmogorov-Smirnov', 'KS'),
                                      ('Kolmogorov-Smirnov', 'p')])
    
    out = pd.DataFrame(index=nm, columns=cols)
    

    # remove nan values
    ind = ~(np.isnan(x) | np.isnan(y))
    x = x[ind]
    y = y[ind]

    # calculate residuals
    r = y - x

    # summary statistics
    cat = 'Residual Summary'
    out.loc[:, (cat, 'N')] = len(x)
    out.loc[:, (cat, 'Median')] = np.median(r)
    out.loc[:, [(cat, 'LQ'), (cat, 'UQ')]] = np.percentile(r, [25, 75])
    out.loc[:, (cat, 'IQR')] = out.loc[:, (cat, 'UQ')] - out.loc[:, (cat, 'LQ')]

    # non-paired test for same distribution
    cat = 'Kolmogorov-Smirnov'
    ks = stats.ks_2samp(x, y)
    out.loc[:, (cat, 'KS')] = ks.statistic
    out.loc[:, (cat, 'p')] = ks.pvalue

    # regression analysis of residuals - slope should be 0, intercept should be 0
    cat = 'Residual Regression'
    X = sm.add_constant(x)
    reg = sm.OLS(r, X, missing='drop')
    fit = reg.fit()
    
    out.loc[:, [(cat, 'Intercept'), (cat, 'Slope')]] = fit.params
    out.loc[:, [(cat, 'Intercept t'), (cat, 'Slope t')]] = fit.tvalues
    out.loc[:, (cat, 'R2')] = fit.rsquared
    out.loc[:, [(cat, 'Intercept p'), (cat, 'Slope p')]] = fit.pvalues

    return out