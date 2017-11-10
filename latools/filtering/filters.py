"""
Functions used for filtering data, or modifying existing filters.
"""
import numpy as np
from ..helpers import bool_2_indices

# Additional filter functions
def exclude_downhole(filt, threshold=2):
    """
    Exclude all data after the first excluded portion.

    This makes sense for spot measurements where, because
    of the signal mixing inherent in LA-ICPMS, once a
    contaminant is ablated, it will always be present to
    some degree in signals from further down the ablation
    pit.

    Parameters
    ----------
    filt : boolean array
    threshold : int

    Returns
    -------
    filter : boolean array
    """
    cfilt = filt.copy()

    inds = bool_2_indices(~filt)

    rem = (np.diff(inds) >= threshold)[:, 0]

    if any(rem):
        if inds[rem].shape[0] > 1:
            limit = inds[rem][1, 0]
            cfilt[limit:] = False
    
    return cfilt

def defrag(filt, threshold=3, mode='include'):
    """
    'Defragment' a filter.

    Parameters
    ----------
    filt : boolean array
        A filter
    threshold : int
        Consecutive values equal to or below this threshold
        length are considered fragments, and will be removed.
    mode : str
        Wheter to change False fragments to True ('include')
        or True fragments to False ('exclude')

    Returns
    -------
    defragmented filter : boolean array
    """
    if bool_2_indices(filt) is None:
        return filt

    if mode == 'include':
        inds = bool_2_indices(~filt) + 1
        rep = True
    if mode == 'exclude':
        inds = bool_2_indices(filt) + 1
        rep = False

    rem = (np.diff(inds) <= threshold)[:, 0]

    cfilt = filt.copy()
    if any(rem):
        for lo, hi in inds[rem]:
            cfilt[lo:hi] = rep

    return cfilt