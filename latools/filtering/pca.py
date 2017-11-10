"""
Functions for exploring LA-ICPMS data with PCA.
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA

def pca_calc(nc, d):
    """
    Calculates pca of d.
    
    Parameters
    ----------
    nc : int
        Number of components
    d : np.ndarray
        An NxM array, containing M observations of N variables.
        Data must be floats. Can contain NaN values.
    
    Returns
    -------
    pca, dt : tuple
        fitted PCA object, and transformed d (same size as d).
    """
    
    # check for and remove nans
    ind = ~np.apply_along_axis(any, 1, np.isnan(d))

    if any(~ind):
        pcs = np.full((d.shape[0], nc), np.nan)
        d = d[ind, :]

    pca = PCA(nc).fit(d)
    
    if any(~ind):
        pcs[ind, :] = pca.transform(d)
    else:
        pcs = pca.transform(d)
    
    return pca, pcs


def pca_plot(pca, dt, xlabs=None, mode='scatter', lognorm=True):
    """
    Plot a fitted PCA, and all components.
    """
    
    nc = pca.n_components
    f = np.arange(pca.n_features_)
    cs = list(itertools.combinations(range(nc), 2))
    
    ind = ~np.apply_along_axis(any, 1, np.isnan(dt))

    cylim = (pca.components_.min(), pca.components_.max())
    yd = cylim[1] - cylim[0]
    
    # Make figure
    fig, axs = plt.subplots(nc, nc, figsize=[3 * nc, nc * 3], tight_layout=True)
            
    for x, y in zip(*np.triu_indices(nc)):
        if x == y:
            tax = axs[x, y]
            tax.bar(f, pca.components_[x], 0.8)
            tax.set_xticks([])
            tax.axhline(0, zorder=-1, c=(0,0,0,0.6))

            # labels            
            tax.set_ylim(cylim[0] - 0.2 * yd,
                         cylim[1] + 0.2 * yd)

            for xi, yi, lab in zip(f, pca.components_[x], xlabs):
                if yi > 0:
                    yo = yd * 0.03
                    va = 'bottom'
                else:
                    yo = yd * -0.02
                    va = 'top'

                tax.text(xi, yi + yo, lab, ha='center', va=va, rotation=90, fontsize=8)

        else:
            xv = dt[ind, x]
            yv = dt[ind, y]

            if mode == 'scatter':
                axs[x, y].scatter(xv, yv, alpha=0.2)
                axs[y, x].scatter(yv, xv, alpha=0.2)
            if mode == 'hist2d':
                if lognorm:
                    norm = mpl.colors.LogNorm()
                else:
                    norm = None
                axs[x, y].hist2d(xv, yv, 50, cmap=plt.cm.Blues, norm=norm)
                axs[y, x].hist2d(yv, xv, 50, cmap=plt.cm.Blues, norm=norm)

        if x == 0:
            axs[y, x].set_ylabel('PC{:.0f}'.format(y + 1))
        if y == nc - 1:
            axs[y, x].set_xlabel('PC{:.0f}'.format(x + 1))
        
    return fig, axs, xv, yv