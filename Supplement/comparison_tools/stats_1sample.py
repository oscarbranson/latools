import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from .stats import fmt_RSS, pairwise_reproducibility, summary_stats

def comparison_stats(df, els=['Mg', 'Sr', 'Al', 'Mn', 'Fe', 'Cu', 'Zn', 'B']):
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
        
    yl_stats = []
    
    for i, (e, a) in enumerate(zip(Rs, As)):
        x = df.loc[:, e].values
        yl = df.loc[:, a].values
        
        yl_stats.append(summary_stats(x, yl, e))
    
    yl_stats = pd.concat(yl_stats).T
    
    return yl_stats.T