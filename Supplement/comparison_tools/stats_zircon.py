import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from .stats import fmt_RSS, pairwise_reproducibility, summary_stats

def comparison_stats(df, els=None):
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
    if els is None:
        els = ['Li', 'Mg', 'Al', 'P', 'Ti', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Sm',
               'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Pb', 'Th',
               'U']
        
    yl_stats = []
    
    for i, e in enumerate(els):
        x = df.loc[:, e + '_rd'].values
        yl = df.loc[:, e + '_la'].values
        
        yl_stats.append(summary_stats(x, yl, e))
    
    yl_stats = pd.concat(yl_stats).T
    
    return yl_stats.T