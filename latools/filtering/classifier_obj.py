import numpy as np
from sklearn import preprocessing
import sklearn.cluster as cl

from latools.helpers.stat_fns import nominal_values

class classifier(object):
    def __init__(self, analytes, sort_by=0):
        """
        Object to fit then apply a classifier.

        Parameters
        ----------
        analytes : str or array-like
            The analytes used by the clustring algorithm

        Returns
        -------
        classifier object
        """
        if isinstance(analytes, str):
            self.analytes = [analytes]
        else:
            self.analytes = analytes
        self.sort_by = sort_by
        return

    def format_data(self, data, scale=True):
        """
        Function for converting a dict to an array suitable for sklearn.

        Parameters
        ----------
        data : dict
            A dict of data, containing all elements of
            `analytes` as items.
        scale : bool
            Whether or not to scale the data. Should always be
            `True`, unless used by `classifier.fitting_data`
            where a scaler hasn't been created yet.

        Returns
        -------
        A data array suitable for use with `sklearn.cluster`.
        """
        if len(self.analytes) == 1:
            # if single analyte
            d = nominal_values(data[self.analytes[0]])
            ds = np.array(list(zip(d, np.zeros(len(d)))))
        else:
            # package multiple analytes
            d = [nominal_values(data[a]) for a in self.analytes]
            ds = np.vstack(d).T

        # identify all nan values
        finite = np.isfinite(ds).sum(1) == ds.shape[1]
        # remember which values are sampled
        sampled = np.arange(data[self.analytes[0]].size)[finite]
        # remove all nan values
        ds = ds[finite]

        if scale:
            ds = self.scaler.transform(ds)

        return ds, sampled

    def fitting_data(self, data):
        """
        Function to format data for cluster fitting.

        Parameters
        ----------
        data : dict
            A dict of data, containing all elements of
            `analytes` as items.

        Returns
        -------
        A data array for initial cluster fitting.
        """
        ds_fit, _ = self.format_data(data, scale=False)

        # define scaler
        self.scaler = preprocessing.StandardScaler().fit(ds_fit)

        # scale data and return
        return self.scaler.transform(ds_fit)

    def fit_kmeans(self, data, n_clusters, **kwargs):
        """
        Fit KMeans clustering algorithm to data.

        Parameters
        ----------
        data : array-like
            A dataset formatted by `classifier.fitting_data`.
        n_clusters : int
            The number of clusters in the data.
        **kwargs
            passed to `sklearn.cluster.KMeans`.

        Returns
        -------
        Fitted `sklearn.cluster.KMeans` object.
        """
        km = cl.KMeans(n_clusters=n_clusters, **kwargs)
        km.fit(data)
        return km

    def fit_meanshift(self, data, bandwidth=None, bin_seeding=False, **kwargs):
        """
        Fit MeanShift clustering algorithm to data.

        Parameters
        ----------
        data : array-like
            A dataset formatted by `classifier.fitting_data`.
        bandwidth : float
            The bandwidth value used during clustering.
            If none, determined automatically. Note:
            the data are scaled before clutering, so
            this is not in the same units as the data.
        bin_seeding : bool
            Whether or not to use 'bin_seeding'. See
            documentation for `sklearn.cluster.MeanShift`.
        **kwargs
            passed to `sklearn.cluster.MeanShift`.

        Returns
        -------
        Fitted `sklearn.cluster.MeanShift` object.
        """
        if bandwidth is None:
            bandwidth = cl.estimate_bandwidth(data)
        ms = cl.MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
        ms.fit(data)
        return ms

    def fit(self, data, method='kmeans', **kwargs):
        """
        fit classifiers from large dataset.

        Parameters
        ----------
        data : dict
            A dict of data for clustering. Must contain
            items with the same name as analytes used for
            clustering.
        method : str
            A string defining the clustering method used. Can be:

            * 'kmeans' : K-Means clustering algorithm
            * 'meanshift' : Meanshift algorithm

        n_clusters : int
            *K-Means only*. The numebr of clusters to identify
        bandwidth : float
            *Meanshift only.*
            The bandwidth value used during clustering.
            If none, determined automatically. Note:
            the data are scaled before clutering, so
            this is not in the same units as the data.
        bin_seeding : bool
            *Meanshift only.*
            Whether or not to use 'bin_seeding'. See
            documentation for `sklearn.cluster.MeanShift`.
        **kwargs :
            passed to `sklearn.cluster.MeanShift`.

        Returns
        -------
        list
        """
        self.method = method
        ds_fit = self.fitting_data(data)
        mdict = {'kmeans': self.fit_kmeans,
                 'meanshift': self.fit_meanshift}
        clust = mdict[method]

        self.classifier = clust(data=ds_fit, **kwargs)

        # sort cluster centers by value of first column, to avoid random variation.
        c0 = self.classifier.cluster_centers_.T[self.sort_by]
        self.classifier.cluster_centers_ = self.classifier.cluster_centers_[np.argsort(c0)]

        # recalculate the labels, so it's consistent with cluster centers
        self.classifier.labels_ = self.classifier.predict(ds_fit)
        self.classifier.ulabels_ = np.unique(self.classifier.labels_)

        return

    def predict(self, data):
        """
        Label new data with cluster identities.

        Parameters
        ----------
        data : dict
            A data dict containing the same analytes used to
            fit the classifier.
        sort_by : str
            The name of an analyte used to sort the resulting
            clusters. If None, defaults to the first analyte
            used in fitting.

        Returns
        -------
        array of clusters the same length as the data.
        """
        size = data[self.analytes[0]].size
        ds, sampled = self.format_data(data)

        # predict clusters
        cs = self.classifier.predict(ds)
        # map clusters to original index
        clusters = self.map_clusters(size, sampled, cs)

        return clusters

    def map_clusters(self, size, sampled, clusters):
        """
        Translate cluster identity back to original data size.

        Parameters
        ----------
        size : int
            size of original dataset
        sampled : array-like
            integer array describing location of finite values
            in original data.
        clusters : array-like
            integer array of cluster identities

        Returns
        -------
        list of cluster identities the same length as original
        data. Where original data are non-finite, returns -2.

        """
        ids = np.zeros(size, dtype=int)
        ids[:] = -2

        ids[sampled] = clusters

        return ids

    def sort_clusters(self, data, cs, sort_by):
        """
        Sort clusters by the concentration of a particular analyte.

        Parameters
        ----------
        data : dict
            A dataset containing sort_by as a key.
        cs : array-like
            An array of clusters, the same length as values of data.
        sort_by : str
            analyte to sort the clusters by

        Returns
        -------
        array of clusters, sorted by mean value of sort_by analyte.
        """
        # label the clusters according to their contents
        sdat = data[sort_by]

        means = []
        nclusts = np.arange(cs.max() + 1)
        for c in nclusts:
            means.append(np.nanmean(sdat[cs == c]))

        # create ranks
        means = np.array(means)
        rank = np.zeros(means.size)
        rank[np.argsort(means)] = np.arange(means.size)

        csn = cs.copy()
        for c, o in zip(nclusts, rank):
            csn[cs == c] = o

        return csn
