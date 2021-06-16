"""
Helper functions for signal separation and identification.

(c) Oscar Branson : https://github.com/oscarbranson
"""
import numpy as np
from .stat_fns import nominal_values
from .utils import Bunch

def bool_transitions(a):
    """
    Return indices where a boolean array changes from True to False
    """
    return np.where(a[:-1] != a[1:])[0]

def bool_2_indices(a):
    """
    Convert boolean array into a 2D array of (start, stop) pairs.
    """
    if any(a):
        lims = []
        lims.append(np.where(a[:-1] != a[1:])[0])

        if a[0]:
            lims.append([0])
        if a[-1]:
            lims.append([len(a) - 1])
        lims = np.concatenate(lims)
        lims.sort()

        return np.reshape(lims, (lims.size // 2, 2))
    else:
        return None

def enumerate_bool(bool_array, nstart=0):
    """
    Consecutively numbers contiguous booleans in array.

    i.e. a boolean sequence, and resulting numbering
    T F T T T F T F F F T T F
    0-1 1 1 - 2 ---3 3 -

    where ' - '

    Parameters
    ----------
    bool_array : array_like
        Array of booleans.
    nstart : int
        The number of the first boolean group.
    """
    ind = bool_2_indices(bool_array)
    ns = np.full(bool_array.size, nstart, dtype=int)
    for n, lims in enumerate(ind):
        ns[lims[0]:lims[-1] + 1] = nstart + n + 1
    return ns

def tuples_2_bool(tuples, x):
    """
    Generate boolean array from list of limit tuples.

    Parameters
    ----------
    tuples : array_like
        [2, n] array of (start, end) values
    x : array_like
        x scale the tuples are mapped to

    Returns
    -------
    array_like
        boolean array, True where x is between each pair of tuples.
    """
    if np.ndim(tuples) == 1:
        tuples = [tuples]

    out = np.zeros(x.size, dtype=bool)
    for l, u in tuples:
        out[(x > l) & (x < u)] = True
    return out

def rolling_window(a, window, window_mode='mid', pad=None):
    """
    Returns (win, len(a)) rolling - window array of data.

    Parameters
    ----------
    a : array_like
        Array to calculate the rolling window of
    window : int
        Description of `window`.
    window_mode : str
        Describes the jusitification of the rolling window relative to the
        returned values. Can be 'left', 'mid' or 'right'.
    pad : same as dtype(a)
        How to pad the ends of the array such that shape[0] of the returned array
        is the same as len(a). Can be 'ends', 'mean_ends' or 'repeat_ends'. 'ends'
        just extends the start or end value across all the extra windows. 'mean_ends'
        extends the mean value of the end windows. 'repeat_ends' repeats the end window
        to completion.

    Returns
    -------
    array_like
        An array of shape (n, window), where n is either len(a) - window
        if pad is None, or len(a) if pad is not None.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    out = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    # pad shape
    if window_mode == 'mid':
        if window % 2 == 0:    
            npre = window // 2 - 1
            npost = window // 2
        else:
            npre = npost = window // 2
    elif window_mode == 'right':
        npre = window - 1
        npost = 0
    elif window_mode == 'left':
        npre = 0
        npost = window - 1
    else:
        raise ValueError("`window_mode` must be 'left', 'mid' or 'right'.")

    if isinstance(pad, str):
        if pad == 'ends':
            prepad = np.full((npre, window), a[0])
            postpad = np.full((npost, window), a[-1])
        elif pad == 'mean_ends':
            prepad = np.full((npre, window), np.mean(a[:(window // 2)]))
            postpad = np.full((npost, window), np.mean(a[-(window // 2):]))
        elif pad == 'repeat_ends':
            prepad = np.full((npre, window), out[0])
            postpad = np.full((npost, window), out[-1])
        else:
            raise ValueError("`pad` must be either 'ends', 'mean_ends' or 'repeat_ends'.")
        return np.concatenate((prepad, out, postpad))
    elif pad is not None:
        pre_blankpad = np.empty(((npre, window)))
        pre_blankpad[:] = pad
        post_blankpad = np.empty(((npost, window)))
        post_blankpad[:] = pad
        return np.concatenate([pre_blankpad, out, post_blankpad])
    else:
        return out

def fastsmooth(a, win=11):
    """
    Returns rolling - window smooth of a.

    Function to efficiently calculate the rolling mean of a numpy
    array using 'stride_tricks' to split up a 1D array into an ndarray of
    sub - sections of the original array, of dimensions [len(a) - win, win].

    Parameters
    ----------
    a : array_like
        The 1D array to calculate the rolling gradient of.
    win : int
        The width of the rolling window.

    Returns
    -------
    array_like
        Gradient of a, assuming as constant integer x - scale.
    """
    # check to see if 'window' is odd (even does not work)
    if win % 2 == 0:
        win += 1  # add 1 to window if it is even.
    kernel = np.ones(win) / win
    npad = int((win - 1) / 2)
    spad = np.full(npad + 1, np.mean(a[:(npad + 1)]))
    epad = np.full(npad - 1, np.mean(a[-(npad - 1):]))
    return np.concatenate([spad, np.convolve(a, kernel, 'valid'), epad])

def fastgrad(a, win=11, win_mode='mid'):
    """
    Returns rolling - window gradient of a.

    Function to efficiently calculate the rolling gradient of a numpy
    array using 'stride_tricks' to split up a 1D array into an ndarray of
    sub - sections of the original array, of dimensions [len(a) - win, win].

    Parameters
    ----------
    a : array_like
        The 1D array to calculate the rolling gradient of.
    win : int
        The width of the rolling window.
    win_mode : str
        Describes the jusitification of the rolling window relative to the
        returned values. Can be 'left', 'mid' or 'right'.

    Returns
    -------
    array_like
        Gradient of a, assuming as constant integer x - scale.
    """
    # check to see if 'window' is odd (even does not work)
    if win % 2 == 0:
        win += 1  # add 1 to window if it is even.
    # trick for efficient 'rolling' computation in numpy
    # shape = a.shape[:-1] + (a.shape[-1] - win + 1, win)
    # strides = a.strides + (a.strides[-1], )
    # wins = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    wins = rolling_window(a, win, pad='ends', window_mode=win_mode)
    # apply rolling gradient to data
    a = map(lambda x: np.polyfit(np.arange(win), x, 1)[0], wins)

    return np.array(list(a))

def calc_grads(x, dat, keys=None, win=5, win_mode='mid'):
    """
    Calculate gradients of values in dat.
    
    Parameters
    ----------
    x : array like
        Independent variable for items in dat.
    dat : dict
        {key: dependent_variable} pairs
    keys : str or array-like
        Which keys in dict to calculate the gradient of.
    win : int
        The side of the rolling window for gradient calculation
    win_mode : str
        Describes the jusitification of the rolling window relative to the
        returned values. Can be 'left', 'mid' or 'right'.

    Returns
    -------
    dict of gradients
    """
    if keys is None:
        keys = dat.keys()

    def grad(xy):
        idx = np.isfinite(xy[1])
        if sum(idx) > 2:
            try:
                return np.polyfit(xy[0][idx], xy[1][idx], 1)[0]
            except ValueError:
                return np.nan
        else:
            return np.nan

    xs = rolling_window(x, win, pad='repeat_ends', window_mode=win_mode)

    grads = Bunch()
    for k in keys:
        d = nominal_values(rolling_window(dat[k], win, pad='repeat_ends', window_mode=win_mode))

        grads[k] = np.array(list(map(grad, zip(xs, d))))

    return grads

def findmins(x, y):
    """ Function to find local minima.

    Parameters
    ----------
    x, y : array_like
        1D arrays of the independent (x) and dependent (y) variables.

    Returns
    -------
    array_like
        Array of points in x where y has a local minimum.
    """
    return x[np.r_[False, y[1:] < y[:-1]] & np.r_[y[:-1] < y[1:], False]]