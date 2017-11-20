import warnings
import numpy as np
import sklearn.cluster as cl

def cluster_meanshift(data, bandwidth=None, bin_seeding=False, **kwargs):
    """
    Identify clusters using Meanshift algorithm.

    Parameters
    ----------
    data : array_like
        array of size [n_samples, n_features].
    bandwidth : float or None
        If None, bandwidth is estimated automatically using
        sklean.cluster.estimate_bandwidth
    bin_seeding : bool
        Setting this option to True will speed up the algorithm.
        See sklearn documentation for full description.

    Returns
    -------
    dict
        boolean array for each identified cluster.
    """
    if bandwidth is None:
        bandwidth = cl.estimate_bandwidth(data)

    ms = cl.MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding, **kwargs)
    ms.fit(data)

    labels = ms.labels_

    return labels, [np.nan]

def cluster_kmeans(data, n_clusters, **kwargs):
    """
    Identify clusters using K - Means algorithm.

    Parameters
    ----------
    data : array_like
        array of size [n_samples, n_features].
    n_clusters : int
        The number of clusters expected in the data.

    Returns
    -------
    dict
        boolean array for each identified cluster.
    """
    km = cl.KMeans(n_clusters, **kwargs)
    kmf = km.fit(data)

    labels = kmf.labels_

    return labels, [np.nan]

def cluster_DBSCAN(data, eps=None, min_samples=None,
                   n_clusters=None, maxiter=200, **kwargs):
    """
    Identify clusters using DBSCAN algorithm.

    Parameters
    ----------
    data : array_like
        array of size [n_samples, n_features].
    eps : float
        The minimum 'distance' points must be apart for them to be in the
        same cluster. Defaults to 0.3. Note: If the data are normalised
        (they should be for DBSCAN) this is in terms of total sample
        variance.  Normalised data have a mean of 0 and a variance of 1.
    min_samples : int
        The minimum number of samples within distance `eps` required
        to be considered as an independent cluster.
    n_clusters : int
        The number of clusters expected. If specified, `eps` will be
        incrementally reduced until the expected number of clusters is
        found.
    maxiter : int
        The maximum number of iterations DBSCAN will run.

    Returns
    -------
    dict
        boolean array for each identified cluster and core samples.
    """
    if n_clusters is None:
        if eps is None:
            eps = 0.3
        db = cl.DBSCAN(eps=eps, min_samples=min_samples, **kwargs).fit(data)
    else:
        clusters = 0
        eps_temp = 1 / .95
        niter = 0
        while clusters < n_clusters:
            clusters_last = clusters
            eps_temp *= 0.95
            db = cl.DBSCAN(eps=eps_temp, min_samples=min_samples, **kwargs).fit(data)
            clusters = (len(set(db.labels_)) -
                        (1 if -1 in db.labels_ else 0))
            if clusters < clusters_last:
                eps_temp *= 1 / 0.95
                db = cl.DBSCAN(eps=eps_temp, min_samples=min_samples, **kwargs).fit(data)
                clusters = (len(set(db.labels_)) -
                            (1 if -1 in db.labels_ else 0))
                warnings.warn(('\n\n***Unable to find {:.0f} clusters in '
                                'data. Found {:.0f} with an eps of {:.2e}'
                                '').format(n_clusters, clusters, eps_temp))
                break
            niter += 1
            if niter == maxiter:
                warnings.warn(('\n\n***Maximum iterations ({:.0f}) reached'
                                ', {:.0f} clusters not found.\nDeacrease '
                                'min_samples or n_clusters (or increase '
                                'maxiter).').format(maxiter, n_clusters))
                break

    labels = db.labels_

    core_samples_mask = np.zeros_like(labels)
    core_samples_mask[db.core_sample_indices_] = True

    return labels, core_samples_mask