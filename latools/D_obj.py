"""
The Data object, used to contain single laser ablation data files.
"""
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import sklearn.cluster as cl
import scipy.interpolate as interp
import uncertainties.unumpy as un

from IPython import display
from scipy.stats import gaussian_kde, pearsonr
from sklearn import preprocessing

import latools.processes as proc

from .filtering import filters
from .filtering import clustering
from .filtering.filt_obj import filt
from .filtering.signal_optimiser import signal_optimiser, optimisation_plot

from .helpers import plot
from .helpers.helpers import (bool_2_indices, rolling_window, Bunch,
                              calc_grads, unitpicker, pretty_element,
                              findmins, stack_keys)
from .helpers.logging import _log
from .helpers.stat_fns import nominal_values, std_devs, unpack_uncertainties, nan_pearsonr

class D(object):
    """
    Container for data from a single laser ablation analysis.

    Parameters
    ----------
    data_file : str
        The path to a data file.
    errorhunt : bool
        Whether or not to print each data file name before
        import. This is useful for tracing which data file
        is causing the import to fail.
    dataformat : str or dict
        Either a path to a data format file, or a
        dataformat dict. See documentation for more details.

    Attributes
    ----------
    sample : str
        Sample name.
    meta : dict
        Metadata extracted from the csv header. Contents varies,
        depending on your `dataformat`.
    analytes : array_like
        A list of analytes measured.
    data : dict
        A dictionary containing the raw data, and modified data
        from each processing stage. Entries can be:

            * 'rawdata': created during initialisation.
            * 'despiked': created by `despike`
            * 'signal': created by `autorange`
            * 'background': created by `autorange`
            * 'bkgsub': created by `bkg_correct`
            * 'ratios': created by `ratio`
            * 'calibrated': created by `calibrate`

    focus : dict
        A dictionary containing one item from `data`. This is the
        currently 'active' data that processing functions will
        work on. This data is also directly available as class
        attributes with the same names as the items in `focus`.
    focus_stage : str
        Identifies which item in `data` is currently assigned to `focus`.
    cmap : dict
        A dictionary containing hex colour strings corresponding
        to each measured analyte.
    bkg, sig, trn : array_like, bool
        Boolean arrays identifying signal, background and transition
        regions. Created by `autorange`.
    bkgrng, sigrng, trnrng : array_like
        An array of shape (n, 2) containing pairs of values that
        describe the Time limits of background, signal and transition
        regions.
    ns : array_like
        An integer array the same length as the data, where each analysis
        spot is labelled with a unique number. Used for separating
        analysis spots when calculating sample statistics.
    filt : filt object
        An object for storing, selecting and creating data filters.F
    """

    def __init__(self, data_file, dataformat=None, errorhunt=False, cmap=None, internal_standard=None, name='file_names'):
        if errorhunt:
            # errorhunt prints each csv file name before it tries to load it,
            # so you can tell which file is failing to load.
            print(data_file)
        params = locals()
        del(params['self'])
        self.log = ['__init__ :: args=() kwargs={}'.format(str(params))]

        self.file = data_file
        self.internal_standard = internal_standard

        self.sample, self.analytes, self.data, self.meta = proc.read_data(data_file, dataformat, name)

        # calculate total counts
        self.data['total_counts'] = sum(self.data['rawdata'].values())

        # add placeholder for gradient info
        self.grads = Bunch()
        self.grads_calced = False

        # add placeholder for local correlations
        self.correlations = Bunch()

        # assign time information to attribute level
        self.Time = self.data['Time']
        self.tstep = self.Time[1] - self.Time[0]
        self.uTime = self.Time  # placeholder for uTime

        # set focus to rawdata
        self.setfocus('rawdata')

        # make a colourmap for plotting
        self.cmap = \
            dict(zip(self.analytes,
                        [mpl.colors.rgb2hex(c) for c
                        in plt.cm.Dark2(np.linspace(0, 1,
                                                    len(self.analytes)))]))
        # update colourmap with provided values
        if isinstance(cmap, dict):
            for k, v in cmap.items():
                if k in self.cmap.keys():
                    self.cmap[k] = v

        # set up flags
        self.sig = np.array([False] * self.Time.size)
        self.bkg = np.array([False] * self.Time.size)
        self.trn = np.array([False] * self.Time.size)
        self.ns = np.zeros(self.Time.size)
        self.bkgrng = np.array([]).reshape(0, 2)
        self.sigrng = np.array([]).reshape(0, 2)

        # set up filtering environment
        self.filt = filt(self.Time.size, self.analytes)

        if errorhunt:
            print('   -> OK')

        return

    @_log
    def setfocus(self, focus):
        """
        Set the 'focus' attribute of the data file.

        The 'focus' attribute of the object points towards data from a
        particular stage of analysis. It is used to identify the 'working
        stage' of the data. Processing functions operate on the 'focus'
        stage, so if steps are done out of sequence, things will break.

        Names of analysis stages:

        * 'rawdata': raw data, loaded from csv file when object
          is initialised.
        * 'despiked': despiked data.
        * 'signal'/'background': isolated signal and background data,
          padded with np.nan. Created by self.separate, after
          signal and background regions have been identified by
          self.autorange.
        * 'bkgsub': background subtracted data, created by
          self.bkg_correct
        * 'ratios': element ratio data, created by self.ratio.
        * 'calibrated': ratio data calibrated to standards, created by
          self.calibrate.

        Parameters
        ----------
        focus : str
            The name of the analysis stage desired.

        Returns
        -------
        None
        """
        self.focus = self.data[focus]
        self.focus_stage = focus

        self.__dict__.update(self.focus)
        # for k in self.focus.keys():
        # setattr(self, k, self.focus[k])

    @_log
    def despike(self, expdecay_despiker=True, exponent=None,
                noise_despiker=True, win=3, nlim=12., maxiter=3):
        """
        Applies expdecay_despiker and noise_despiker to data.

        Parameters
        ----------
        expdecay_despiker : bool
            Whether or not to apply the exponential decay filter.
        exponent : None or float
            The exponent for the exponential decay filter. If None,
            it is determined automatically using `find_expocoef`.
        noise_despiker : bool
            Whether or not to apply the standard deviation spike filter.
        win : int
            The rolling window over which the spike filter calculates
            the trace statistics.
        nlim : float
            The number of standard deviations above the rolling mean
            that data are excluded.
        maxiter : int
            The max number of times that the fitler is applied.

        Returns
        -------
        None

        """
        if not hasattr(self, 'despiked'):
            self.data['despiked'] = Bunch()

        out = {}
        for a, v in self.focus.items():
            if 'time' not in a.lower():
                sig = v.copy()  # copy data
                if expdecay_despiker:
                    if exponent is not None:
                        sig = proc.expdecay_despike(sig, exponent, self.tstep, maxiter)
                    else:
                        warnings.warn('exponent is None - either provide exponent, or run at `analyse`\nlevel to automatically calculate it.')
                
                if noise_despiker:
                    sig = proc.noise_despike(sig, int(win), nlim, maxiter)
                out[a] = sig

        self.data['despiked'].update(out)
        # recalculate total counts
        self.data['total_counts'] = sum(self.data['despiked'].values())
        self.setfocus('despiked')
        return

    @_log
    def autorange(self, analyte='total_counts', gwin=5, swin=3, win=30,
                  on_mult=[1., 1.], off_mult=[1., 1.5],
                  ploterrs=True, transform='log', **kwargs):
        """
        Automatically separates signal and background data regions.

        Automatically detect signal and background regions in the laser
        data, based on the behaviour of a single analyte. The analyte used
        should be abundant and homogenous in the sample.

        **Step 1: Thresholding.**
        The background signal is determined using a gaussian kernel density
        estimator (kde) of all the data. Under normal circumstances, this
        kde should find two distinct data distributions, corresponding to
        'signal' and 'background'. The minima between these two distributions
        is taken as a rough threshold to identify signal and background
        regions. Any point where the trace crosses this thrshold is identified
        as a 'transition'.

        **Step 2: Transition Removal.**
        The width of the transition regions between signal and background are
        then determined, and the transitions are excluded from analysis. The
        width of the transitions is determined by fitting a gaussian to the
        smoothed first derivative of the analyte trace, and determining its
        width at a point where the gaussian intensity is at at `conf` time the
        gaussian maximum. These gaussians are fit to subsets of the data
        centered around the transitions regions determined in Step 1, +/- `win`
        data points. The peak is further isolated by finding the minima and
        maxima of a second derivative within this window, and the gaussian is
        fit to the isolated peak.

        Parameters
        ----------
        analyte : str
            The analyte that autorange should consider. For best results,
            choose an analyte that is present homogeneously in high
            concentrations.
        gwin : int
            The smoothing window used for calculating the first derivative.
            Must be odd.
        win : int
            Determines the width (c +/- win) of the transition data subsets.
        on_mult and off_mult : tuple, len=2
            Factors to control the width of the excluded transition regions.
            A region n times the full - width - half - maximum of the transition
            gradient will be removed either side of the transition center.
            `on_mult` and `off_mult` refer to the laser - on and laser - off
            transitions, respectively. See manual for full explanation.
            Defaults to (1.5, 1) and (1, 1.5).


        Returns
        -------
        Outputs added as instance attributes. Returns None.
        bkg, sig, trn : iterable, bool
            Boolean arrays identifying background, signal and transision
            regions
        bkgrng, sigrng and trnrng : iterable
            (min, max) pairs identifying the boundaries of contiguous
            True regions in the boolean arrays.
        """
        if analyte is None:
            # sig = self.focus[self.internal_standard]
            sig = self.data['total_counts']
        elif analyte == 'total_counts':
            sig = self.data['total_counts']
        elif analyte in self.analytes:
            sig = self.focus[analyte]
        else:
            raise ValueError('Invalid analyte.')

        (self.bkg, self.sig,
         self.trn, failed) = proc.autorange(self.Time, sig, gwin=gwin, swin=swin, win=win,
                                            on_mult=on_mult, off_mult=off_mult,
                                            transform=transform)

        self.mkrngs()

        errs_to_plot = False
        if len(failed) > 0:
            errs_to_plot = True
            plotlines = []
            for f in failed:
                if f != self.Time[-1]:
                    plotlines.append(f)
            # warnings.warn(("\n\nSample {:s}: ".format(self.sample) +
            #                "Transition identification at " +
            #                "{:.1f} failed.".format(f) +
            #                "\n  **This is not necessarily a problem**"
            #                "\nBut please check the data plots and make sure " +
            #                "everything is OK.\n"))

        if ploterrs and errs_to_plot and len(plotlines) > 0:
            f, ax = self.tplot(ranges=True)
            for pl in plotlines:
                ax.axvline(pl, c='r', alpha=0.6, lw=3, ls='dashed')
            return f, plotlines
        else:
            return

    def autorange_plot(self, analyte='total_counts', gwin=7, swin=None, win=20,
                       on_mult=[1.5, 1.], off_mult=[1., 1.5],
                       transform='log'):
        """
        Plot a detailed autorange report for this sample.
        """
        if analyte is None:
            # sig = self.focus[self.internal_standard]
            sig = self.data['total_counts']
        elif analyte == 'total_counts':
            sig = self.data['total_counts']
        elif analyte in self.analytes:
            sig = self.focus[analyte]
        else:
            raise ValueError('Invalid analyte.')

        if transform == 'log':
            sig = np.log10(sig)

        fig, axs = plot.autorange_plot(t=self.Time, sig=sig, gwin=gwin,
                                       swin=swin, win=win, on_mult=on_mult,
                                       off_mult=off_mult)

        return fig, axs

    def mkrngs(self):
        """
        Transform boolean arrays into list of limit pairs.

        Gets Time limits of signal/background boolean arrays and stores them as
        sigrng and bkgrng arrays. These arrays can be saved by 'save_ranges' in
        the analyse object.
        """
        bbool = bool_2_indices(self.bkg)
        if bbool is not None:
            self.bkgrng = self.Time[bbool]
        else:
            self.bkgrng = [[np.nan, np.nan]]
        sbool = bool_2_indices(self.sig)
        if sbool is not None:
            self.sigrng = self.Time[sbool]
        else:
            self.sigrng = [[np.nan, np.nan]]
        tbool = bool_2_indices(self.trn)
        if tbool is not None:
            self.trnrng = self.Time[tbool]
        else:
            self.trnrng = [[np.nan, np.nan]]

        self.ns = np.zeros(self.Time.size)
        n = 1
        for i in range(len(self.sig) - 1):
            if self.sig[i]:
                self.ns[i] = n
            if self.sig[i] and ~self.sig[i + 1]:
                n += 1
        self.n = int(max(self.ns))  # record number of traces

        return

    @_log
    def bkg_subtract(self, analyte, bkg, ind=None, focus_stage='despiked'):
        """
        Subtract provided background from signal (focus stage).

        Results is saved in new 'bkgsub' focus stage

        Returns
        -------
        None
        """
        if 'bkgsub' not in self.data.keys():
            self.data['bkgsub'] = Bunch()

        self.data['bkgsub'][analyte] = self.data[focus_stage][analyte] - bkg

        if ind is not None:
            self.data['bkgsub'][analyte][ind] = np.nan

        return

    @_log
    def correct_spectral_interference(self, target_analyte, source_analyte, f):
        """
        Correct spectral interference.

        Subtract interference counts from target_analyte, based on the
        intensity of a source_analayte and a known fractional contribution (f).

        Correction takes the form:
        target_analyte -= source_analyte * f

        Only operates on background-corrected data ('bkgsub'). 
        
        To undo a correction,
        rerun `self.bkg_subtract()`.

        Parameters
        ----------
        target_analyte : str
            The name of the analyte to modify.
        source_analyte : str
            The name of the analyte to base the correction on.
        f : float
            The fraction of the intensity of the source_analyte to
            subtract from the target_analyte. Correction is:
            target_analyte - source_analyte * f

        Returns
        -------
        None
        """

        if target_analyte not in self.analytes:
            raise ValueError('target_analyte: {:} not in available analytes ({:})'.format(target_analyte, ', '.join(self.analytes)))

        if source_analyte not in self.analytes:
            raise ValueError('source_analyte: {:} not in available analytes ({:})'.format(source_analyte, ', '.join(self.analytes)))

        self.data['bkgsub'][target_analyte] -= self.data['bkgsub'][source_analyte] * f

    @_log
    def ratio(self, internal_standard=None):
        """
        Divide all analytes by a specified internal_standard analyte.

        Parameters
        ----------
        internal_standard : str
            The analyte used as the internal_standard.

        Returns
        -------
        None
        """
        if internal_standard is not None:
            self.internal_standard = internal_standard

        self.data['ratios'] = Bunch()
        for a in self.analytes:
            self.data['ratios'][a] = (self.data['bkgsub'][a] /
                                      self.data['bkgsub'][self.internal_standard])
        self.setfocus('ratios')
        return

    @_log
    def calibrate(self, calib_ps, analytes=None):
        """
        Apply calibration to data.

        The `calib_dict` must be calculated at the `analyse` level,
        and passed to this calibrate function.

        Parameters
        ----------
        calib_dict : dict
            A dict of calibration values to apply to each analyte.

        Returns
        -------
        None
        """
        # can have calibration function stored in self and pass *coefs?
        if analytes is None:
            analytes = self.analytes

        if 'calibrated' not in self.data.keys():
            self.data['calibrated'] = Bunch()

        for a in analytes:
            m = calib_ps[a]['m'].new(self.uTime)

            if 'c' in calib_ps[a]:
                c = calib_ps[a]['c'].new(self.uTime)
            else:
                c = 0

            self.data['calibrated'][a] = self.data['ratios'][a] * m + c

        if self.internal_standard not in analytes:
            self.data['calibrated'][self.internal_standard] = \
                np.empty(len(self.data['ratios'][self.internal_standard]))

        self.setfocus('calibrated')
        return

    # Function for calculating sample statistics
    @_log
    def sample_stats(self, analytes=None, filt=True,
                     stat_fns={},
                     eachtrace=True):
        """
        Calculate sample statistics

        Returns samples, analytes, and arrays of statistics
        of shape (samples, analytes). Statistics are calculated
        from the 'focus' data variable, so output depends on how
        the data have been processed.

        Parameters
        ----------
        analytes : array_like
            List of analytes to calculate the statistic on
        filt : bool or str
            The filter to apply to the data when calculating sample statistics.
                bool: True applies filter specified in filt.switches.
                str: logical string specifying a partucular filter
        stat_fns : dict
            Dict of {name: function} pairs. Functions that take a single
            array_like input, and return a single statistic. Function should
            be able to cope with NaN values.
        eachtrace : bool
            True: per - ablation statistics
            False: whole sample statistics

        Returns
        -------
        None
        """
        if analytes is None:
                analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        self.stats = Bunch()
        self.stats['analytes'] = analytes

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for n, f in stat_fns.items():
                self.stats[n] = []
                for a in analytes:
                    ind = self.filt.grab_filt(filt, a)
                    dat = nominal_values(self.focus[a])
                    if eachtrace:
                        sts = []
                        for t in np.arange(self.n) + 1:
                            sts.append(f(dat[ind & (self.ns == t)]))
                        self.stats[n].append(sts)
                    else:
                        self.stats[n].append(f(dat[ind]))
                self.stats[n] = np.array(self.stats[n])
        return

    @_log
    def ablation_times(self):
        """
        Function for calculating the ablation time for each
        ablation.

        Returns
        -------
            dict of times for each ablation.
        """
        ats = {}
        for n in np.arange(self.n) + 1:
            t = self.Time[self.ns == n]
            ats[n - 1] = t.max() - t.min()
        return ats

    # Data Selections Tools
    @_log
    def filter_threshold(self, analyte, threshold):
        """
        Apply threshold filter.

        Generates threshold filters for the given analytes above and below
        the specified threshold.

        Two filters are created with prefixes '_above' and '_below'.
            '_above' keeps all the data above the threshold.
            '_below' keeps all the data below the threshold.

        i.e. to select data below the threshold value, you should turn the
        '_above' filter off.

        Parameters
        ----------
        analyte : TYPE
            Description of `analyte`.
        threshold : TYPE
            Description of `threshold`.

        Returns
        -------
        None
        """
        params = locals()
        del(params['self'])

        # generate filter
        below, above = filters.threshold(self.focus[analyte], threshold)

        setn = self.filt.maxset + 1

        self.filt.add(analyte + '_thresh_below',
                        below,
                        'Keep below {:.3e} '.format(threshold) + analyte,
                        params, setn=setn)
        self.filt.add(analyte + '_thresh_above',
                        above,
                        'Keep above {:.3e} '.format(threshold) + analyte,
                        params, setn=setn)

    @_log
    def filter_gradient_threshold(self, analyte, win, threshold, recalc=True):
        """
        Apply gradient threshold filter.

        Generates threshold filters for the given analytes above and below
        the specified threshold.

        Two filters are created with prefixes '_above' and '_below'.
            '_above' keeps all the data above the threshold.
            '_below' keeps all the data below the threshold.

        i.e. to select data below the threshold value, you should turn the
        '_above' filter off.

        Parameters
        ----------
        analyte : str
            Description of `analyte`.
        threshold : float
            Description of `threshold`.
        win : int
            Window used to calculate gradients (n points)
        recalc : bool
            Whether or not to re-calculate the gradients.

        Returns
        -------
        None
        """
        params = locals()
        del(params['self'])

        # calculate absolute gradient
        if recalc or not self.grads_calced:
            self.grads = calc_grads(self.Time, self.focus,
                                    [analyte], win)
            self.grads_calced = True

        below, above = filters.threshold(abs(self.grads[analyte]), threshold)

        setn = self.filt.maxset + 1

        self.filt.add(analyte + '_gthresh_below',
                        below,
                        'Keep gradient below {:.3e} '.format(threshold) + analyte,
                        params, setn=setn)
        self.filt.add(analyte + '_gthresh_above',
                        above,
                        'Keep gradient above {:.3e} '.format(threshold) + analyte,
                        params, setn=setn)

    @_log
    def filter_clustering(self, analytes, filt=False, normalise=True,
                          method='meanshift', include_time=False,
                          sort=None, min_data=10, **kwargs):
        """
        Applies an n - dimensional clustering filter to the data.

        Available Clustering Algorithms

        * 'meanshift': The `sklearn.cluster.MeanShift` algorithm.
          Automatically determines number of clusters
          in data based on the `bandwidth` of expected
          variation.
        * 'kmeans': The `sklearn.cluster.KMeans` algorithm. Determines
          the characteristics of a known number of clusters
          within the data. Must provide `n_clusters` to specify
          the expected number of clusters.
        * 'DBSCAN': The `sklearn.cluster.DBSCAN` algorithm. Automatically
          determines the number and characteristics of clusters
          within the data based on the 'connectivity' of the
          data (i.e. how far apart each data point is in a
          multi - dimensional parameter space). Requires you to
          set `eps`, the minimum distance point must be from
          another point to be considered in the same cluster,
          and `min_samples`, the minimum number of points that
          must be within the minimum distance for it to be
          considered a cluster. It may also be run in automatic
          mode by specifying `n_clusters` alongside
          `min_samples`, where eps is decreased until the
          desired number of clusters is obtained.
        
        For more information on these algorithms, refer to the
        documentation.

        Parameters
        ----------
        analytes : str
            The analyte(s) that the filter applies to.
        filt : bool
            Whether or not to apply existing filters to the data before
            calculating this filter.
        normalise : bool
            Whether or not to normalise the data to zero mean and unit
            variance. Reccomended if clustering based on more than 1 analyte.
            Uses `sklearn.preprocessing.scale`.
        method : str
            Which clustering algorithm to use (see above).
        include_time : bool
            Whether or not to include the Time variable in the clustering
            analysis. Useful if you're looking for spatially continuous
            clusters in your data, i.e. this will identify each spot in your
            analysis as an individual cluster.
        sort : bool, str or array-like
            Whether or not to label the resulting clusters according to their
            contents. If used, the cluster with the lowest values will be
            labelled from 0, in order of increasing cluster mean value.analytes.
            The sorting rules depend on the value of 'sort', which can be the name
            of a single analyte (str), a list of several analyte names (array-like)
            or True (bool), to specify all analytes used to calcualte the cluster.
        min_data : int
            The minimum number of data points that should be considered by
            the filter. Default = 10.
        **kwargs
            Parameters passed to the clustering algorithm specified by
            `method`.

        Meanshift Parameters
        --------------------
        bandwidth : str or float
            The bandwith (float) or bandwidth method ('scott' or 'silverman')
            used to estimate the data bandwidth.
        bin_seeding : bool
            Modifies the behaviour of the meanshift algorithm. Refer to
            sklearn.cluster.meanshift documentation.

        K - Means Parameters
        ------------------
        n_clusters : int
            The number of clusters expected in the data.

        DBSCAN Parameters
        -----------------
        eps : float
            The minimum 'distance' points must be apart for them to be in the
            same cluster. Defaults to 0.3. Note: If the data are normalised
            (they should be for DBSCAN) this is in terms of total sample
            variance. Normalised data have a mean of 0 and a variance of 1.
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
        None
        """
        params = locals()
        del(params['self'])

        # convert string to list, if single analyte
        if isinstance(analytes, str):
            analytes = [analytes]

        setn = self.filt.maxset + 1

        # generate filter
        vals = np.vstack(nominal_values(list(self.focus.values())))
        if filt is not None:
            ind = (self.filt.grab_filt(filt, analytes) &
                   np.apply_along_axis(all, 0, ~np.isnan(vals)))
        else:
            ind = np.apply_along_axis(all, 0, ~np.isnan(vals))

        if sum(ind) > min_data:

            # get indices for data passed to clustering
            sampled = np.arange(self.Time.size)[ind]

            # generate data for clustering
            if include_time:
                extra = self.Time
            else:
                extra = None
            # get data as array
            ds = stack_keys(self.focus, analytes, extra)
            # apply filter, and get nominal values
            ds = nominal_values(ds[ind, :])

            if normalise | (len(analytes) > 1):
                ds = preprocessing.scale(ds)

            method_key = {'kmeans': clustering.cluster_kmeans,
                        #   'DBSCAN': clustering.cluster_DBSCAN,
                          'meanshift': clustering.cluster_meanshift}

            cfun = method_key[method]

            labels, core_samples_mask = cfun(ds, **kwargs)
            # return labels, and if DBSCAN core_sample_mask

            labels_unique = np.unique(labels)

            # label the clusters according to their contents
            if (sort is not None) & (sort is not False):

                if isinstance(sort, str):
                    sort = [sort]

                sanalytes = analytes

                # make boolean filter to select analytes
                if sort is True:
                    sortk = np.array([True] * len(sanalytes))
                else:
                    sortk = np.array([s in sort for s in sanalytes])

                # create per-point mean based on selected analytes.
                sd = np.apply_along_axis(sum, 1, ds[:, sortk])
                # calculate per-cluster means
                avs = [np.nanmean(sd[labels == lab]) for lab in labels_unique]
                # re-order the cluster labels based on their means
                order = [x[0] for x in sorted(enumerate(avs), key=lambda x:x[1])]
                sdict = dict(zip(order, labels_unique))
            else:
                sdict = dict(zip(labels_unique, labels_unique))

            filts = {}
            for ind, lab in sdict.items():
                filts[lab] = labels == ind

            # only applies to DBSCAN results.
            if not all(np.isnan(core_samples_mask)):
                filts['core'] = core_samples_mask

            resized = {}
            for k, v in filts.items():
                resized[k] = np.zeros(self.Time.size, dtype=bool)
                resized[k][sampled] = v

            namebase = '-'.join(analytes) + '_cluster-' + method
            info = '-'.join(analytes) + ' cluster filter.'

            if method == 'DBSCAN':
                for k, v in resized.items():
                    if isinstance(k, str):
                        name = namebase + '_core'
                    elif k < 0:
                        name = namebase + '_noise'
                    else:
                        name = namebase + '_{:.0f}'.format(k)
                    self.filt.add(name, v, info=info, params=params, setn=setn)
            else:
                for k, v in resized.items():
                    name = namebase + '_{:.0f}'.format(k)
                    self.filt.add(name, v, info=info, params=params, setn=setn)
        else:
            # if there are no data
            name = '-'.join(analytes) + '_cluster-' + method + '_0'
            info = '-'.join(analytes) + ' cluster filter failed.'

            self.filt.add(name, np.zeros(self.Time.size, dtype=bool),
                          info=info, params=params, setn=setn)

        return

    @_log
    def calc_correlation(self, x_analyte, y_analyte, window=15, filt=True, recalc=True):
        """
        Calculate local correlation between two analytes.

        Parameters
        ----------
        x_analyte, y_analyte : str
            The names of the x and y analytes to correlate.
        window : int, None
            The rolling window used when calculating the correlation.
        filt : bool
            Whether or not to apply existing filters to the data before
            calculating this filter.
        recalc : bool
            If True, the correlation is re-calculated, even if it is already present.

        Returns
        -------
        None
        """
        label = '{:}_{:}_{:.0f}'.format(x_analyte, y_analyte, window)

        if label in self.correlations and not recalc:
            return

        # make window odd
        if window % 2 != 1:
            window += 1
        
        # get filter
        ind = self.filt.grab_filt(filt, [x_analyte, y_analyte])

        x = nominal_values(self.focus[x_analyte])
        x[~ind] = np.nan
        xr = rolling_window(x, window, pad=np.nan)

        y = nominal_values(self.focus[y_analyte])
        y[~ind] = np.nan
        yr = rolling_window(y, window, pad=np.nan)

        r, p = zip(*map(nan_pearsonr, xr, yr))

        r = np.array(r)
        p = np.array(p)

        # save correlation info
        
        self.correlations[label] = r, p
        return

    @_log
    def filter_correlation(self, x_analyte, y_analyte, window=15,
                           r_threshold=0.9, p_threshold=0.05, filt=True, recalc=False):
        """
        Calculate correlation filter.

        Parameters
        ----------
        x_analyte, y_analyte : str
            The names of the x and y analytes to correlate.
        window : int, None
            The rolling window used when calculating the correlation.
        r_threshold : float
            The correlation index above which to exclude data.
            Note: the absolute pearson R value is considered, so
            negative correlations below -`r_threshold` will also
            be excluded.
        p_threshold : float
            The significant level below which data are excluded.
        filt : bool
            Whether or not to apply existing filters to the data before
            calculating this filter.
        recalc : bool
            If True, the correlation is re-calculated, even if it is already present.

        Returns
        -------
        None
        """
        # make window odd
        if window % 2 != 1:
            window += 1

        params = locals()
        del(params['self'])

        setn = self.filt.maxset + 1

        label = '{:}_{:}_{:.0f}'.format(x_analyte, y_analyte, window)
        
        self.calc_correlation(x_analyte, y_analyte, window, filt, recalc)
        r, p = self.correlations[label]

        cfilt = (abs(r) > r_threshold) & (p < p_threshold)
        cfilt = ~cfilt

        name = x_analyte + '_' + y_analyte + '_corr'

        self.filt.add(name=name,
                      filt=cfilt,
                      info=(x_analyte + ' vs. ' + y_analyte +
                            ' correlation filter.'),
                      params=params, setn=setn)
        self.filt.off(filt=name)
        self.filt.on(analyte=y_analyte, filt=name)

        return

    @_log
    def correlation_plot(self, x_analyte, y_analyte, window=15, filt=True, recalc=False):
        """
        Plot the local correlation between two analytes.

        Parameters
        ----------
        x_analyte, y_analyte : str
            The names of the x and y analytes to correlate.
        window : int, None
            The rolling window used when calculating the correlation.
        filt : bool
            Whether or not to apply existing filters to the data before
            calculating this filter.
        recalc : bool
            If True, the correlation is re-calculated, even if it is already present.

        Returns
        -------
        fig, axs : figure and axes objects
        """
        label = '{:}_{:}_{:.0f}'.format(x_analyte, y_analyte, window)

        self.calc_correlation(x_analyte, y_analyte, window, filt, recalc)
        r, p = self.correlations[label]

        fig, axs = plt.subplots(3, 1, figsize=[7, 5], sharex=True)
        
        # plot analytes
        ax = axs[0]
            
        ax.plot(self.Time, nominal_values(self.focus[x_analyte]), color=self.cmap[x_analyte], label=x_analyte)
        ax.plot(self.Time, nominal_values(self.focus[y_analyte]), color=self.cmap[y_analyte], label=y_analyte)
        
        ax.set_yscale('log')
        ax.legend()
        ax.set_ylabel('Signals')
        
        # plot r
        ax = axs[1]
        ax.plot(self.Time, r)
        ax.set_ylabel('Pearson R')
        
        # plot p
        ax = axs[2]
        ax.plot(self.Time, p)
        ax.set_ylabel('pignificance Level (p)')
        
        fig.tight_layout()
        
        return fig, axs
    
    @_log
    def filter_new(self, name, filt_str):
        """
        Make new filter from combination of other filters.

        Parameters
        ----------
        name : str
            The name of the new filter. Should be unique.
        filt_str : str
            A logical combination of partial strings which will create
            the new filter. For example, 'Albelow & Mnbelow' will combine
            all filters that partially match 'Albelow' with those that
            partially match 'Mnbelow' using the 'AND' logical operator.

        Returns
        -------
        None
        """
        filt = self.filt.grab_filt(filt=filt_str)
        self.filt.add(name, filt, info=filt_str)
        return
    
    @_log
    def filter_trim(self, start=1, end=1, filt=True):
        """
        Remove points from the start and end of filter regions.
        
        Parameters
        ----------
        start, end : int
            The number of points to remove from the start and end of
            the specified filter.
        filt : valid filter string or bool
            Which filter to trim. If True, applies to currently active
            filters.
        """
        params = locals()
        del(params['self'])
            
        f = self.filt.grab_filt(filt)
        nf = filters.trim(f, start, end)
        
        self.filt.add('trimmed_filter',
                    nf,
                    'Trimmed Filter ({:.0f} start, {:.0f} end)'.format(start, end),
                    params, setn=self.filt.maxset + 1)

    @_log
    def filter_exclude_downhole(self, threshold, filt=True):
        """
        Exclude all points down-hole (after) the first excluded data.

        Parameters
        ----------
        threhold : int
            The minimum number of contiguous excluded data points
            that must exist before downhole exclusion occurs.
        file : valid filter string or bool
            Which filter to consider. If True, applies to currently active
            filters.
        """
        f = self.filt.grab_filt(filt)

        if self.n == 1:
            nfilt = filters.exclude_downhole(f, threshold)

        else:
            nfilt = []
            for i in range(self.n):
                nf = self.ns == i + 1
                nfilt.append(filters.exclude_downhole(f & nf, threshold))
            nfilt = np.apply_along_axis(any, 0, nfilt)

        self.filt.add(name='downhole_excl_{:.0f}'.format(threshold),
                      filt=nfilt,
                      info='Exclude data downhole of {:.0f} consecutive filtered points.'.format(threshold),
                      params=(threshold, filt))

    # Signal optimiser
    @_log
    def signal_optimiser(self, analytes, min_points=5,
                         threshold_mode='kde_first_max',
                         threshold_mult=1., x_bias=0,
                         weights=None, filt=True, mode='minimise'):
        """
        Optimise data selection based on specified analytes.

        Identifies the longest possible contiguous data region in
        the signal where the relative standard deviation (std) and 
        concentration of all analytes is minimised.

        Optimisation is performed via a grid search of all possible
        contiguous data regions. For each region, the mean std and
        mean scaled analyte concentration ('amplitude') are calculated. 
        
        The size and position of the optimal data region are identified 
        using threshold std and amplitude values. Thresholds are derived
        from all calculated stds and amplitudes using the method specified
        by `threshold_mode`. For example, using the 'kde_max' method, a
        probability density function (PDF) is calculated for std and
        amplitude values, and the threshold is set as the maximum of the
        PDF. These thresholds are then used to identify the size and position
        of the longest contiguous region where the std is below the threshold, 
        and the amplitude is either below the threshold.

        All possible regions of the data that have at least
        `min_points` are considered.

        For a graphical demonstration of the action of signal_optimiser, 
        use `optimisation_plot`. 

        Parameters
        ----------
        d : latools.D object
            An latools data object.
        analytes : str or array-like
            Which analytes to consider.
        min_points : int
            The minimum number of contiguous points to
            consider.
        threshold_mode : str
            The method used to calculate the optimisation
            thresholds. Can be 'mean', 'median', 'kde_max'
            or 'bayes_mvs', or a custom function. If a
            function, must take a 1D array, and return a
            single, real number.
        weights : array-like of length len(analytes)
            An array of numbers specifying the importance of
            each analyte considered. Larger number makes the
            analyte have a greater effect on the optimisation.
            Default is None.
        """
        params = locals()
        del(params['self'])
        setn = self.filt.maxset + 1

        if isinstance(analytes, str):
            analytes = [analytes]
        
        # get filter
        if filt is not False:
            ind = (self.filt.grab_filt(filt, analytes))
        else:
            ind = np.full(self.Time.shape, True)
        
        errmsg = []
        ofilt = []
        self.opt = {}
        for i in range(self.n):
            nind = ind & (self.ns == i + 1)

            self.opt[i + 1], err = signal_optimiser(self, analytes=analytes,
                                                    min_points=min_points, 
                                                    threshold_mode=threshold_mode,
                                                    threshold_mult=threshold_mult,
                                                    weights=weights,
                                                    ind=nind, x_bias=x_bias,
                                                    mode=mode)

            if err == '':
                ofilt.append(self.opt[i + 1].filt)
            else:
                errmsg.append(self.sample + '_{:.0f}: '.format(i + 1) + err)

        if len(ofilt) > 0:
            ofilt = np.apply_along_axis(any, 0, ofilt)

            name = 'optimise_' + '_'.join(analytes)
            self.filt.add(name=name,
                        filt=ofilt,
                        info="Optimisation filter to minimise " + ', '.join(analytes),
                        params=params, setn=setn)            
        
        if len(errmsg) > 0:
            return '\n'.join(errmsg)
        else:
            return ''
    
    @_log
    def optimisation_plot(self, overlay_alpha=0.5, **kwargs):
        """
        Plot the result of signal_optimise.

        `signal_optimiser` must be run first, and the output
        stored in the `opt` attribute of the latools.D object.

        Parameters
        ----------
        d : latools.D object
            A latools data object.
        overlay_alpha : float
            The opacity of the threshold overlays. Between 0 and 1.
        **kwargs
            Passed to `tplot`
        """
        return optimisation_plot(self, overlay_alpha=0.5, **kwargs)

    @_log
    def tplot(self, analytes=None, figsize=[10, 4], scale='log', filt=None,
              ranges=False, stats=False, stat='nanmean', err='nanstd',
              focus_stage=None, err_envelope=False, ax=None):
        """
        Plot analytes as a function of Time.

        Parameters
        ----------
        analytes : array_like
            list of strings containing names of analytes to plot.
            None = all analytes.
        figsize : tuple
            size of final figure.
        scale : str or None
           'log' = plot data on log scale
        filt : bool, str or dict
            False: plot unfiltered data.
            True: plot filtered data over unfiltered data.
            str: apply filter key to all analytes
            dict: apply key to each analyte in dict. Must contain all
            analytes plotted. Can use self.filt.keydict.
        ranges : bool
            show signal/background regions.
        stats : bool
            plot average and error of each trace, as specified by `stat` and
            `err`.
        stat : str
            average statistic to plot.
        err : str
            error statistic to plot.

        Returns
        -------
        figure, axis
        """

        return plot.tplot(self=self, analytes=analytes, figsize=figsize, scale=scale, filt=filt,
                          ranges=ranges, stats=stats, stat=stat, err=err,
                          focus_stage=focus_stage, err_envelope=err_envelope, ax=ax)

    @_log
    def gplot(self, analytes=None, win=5, figsize=[10, 4],
              ranges=False, focus_stage=None, ax=None):
        """
        Plot analytes gradients as a function of Time.

        Parameters
        ----------
        analytes : array_like
            list of strings containing names of analytes to plot.
            None = all analytes.
        win : int
            The window over which to calculate the rolling gradient.
        figsize : tuple
            size of final figure.
        ranges : bool
            show signal/background regions.

        Returns
        -------
        figure, axis
        """

        return plot.gplot(self=self, analytes=analytes, win=win, figsize=figsize,
                          ranges=ranges, focus_stage=focus_stage, ax=ax)

        # if type(analytes) is str:
        #     analytes = [analytes]
        # if analytes is None:
        #     analytes = self.analytes

        # if focus_stage is None:
        #     focus_stage = self.focus_stage

        # fig = plt.figure(figsize=figsize)
        # ax = fig.add_axes([.1, .12, .77, .8])

        # x = self.Time
        # grads = calc_grads(x, self.data[focus_stage], analytes, win)

        # for a in analytes:
        #     ax.plot(x, grads[a], color=self.cmap[a], label=a)

        # if ranges:
        #     for lims in self.bkgrng:
        #         ax.axvspan(*lims, color='k', alpha=0.1, zorder=-1)
        #     for lims in self.sigrng:
        #         ax.axvspan(*lims, color='r', alpha=0.1, zorder=-1)

        # ax.text(0.01, 0.99, self.sample + ' : ' + self.focus_stage + ' : gradient',
        #         transform=ax.transAxes,
        #         ha='left', va='top')

        # ax.set_xlabel('Time (s)')
        # ax.set_xlim(np.nanmin(x), np.nanmax(x))

        # # y label
        # ud = {'rawdata': 'counts/s',
        #       'despiked': 'counts/s',
        #       'bkgsub': 'background corrected counts/s',
        #       'ratios': 'counts/{:s} count/s',
        #       'calibrated': 'mol/mol {:s}/s'}
        # if focus_stage in ['ratios', 'calibrated']:
        #     ud[focus_stage] = ud[focus_stage].format(self.internal_standard)
        # ax.set_ylabel(ud[focus_stage])
        # # y tick format

        # def yfmt(x, p):
        #     return '{:.0e}'.format(x)
        # ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(yfmt))

        # ax.legend(bbox_to_anchor=(1.15, 1))

        # ax.axhline(0, c='k', lw=1, ls='dashed', alpha=0.5)

        # return fig, ax

    @_log
    def crossplot(self, analytes=None, bins=25, lognorm=True, filt=True, colourful=True, figsize=(12, 12)):
        """
        Plot analytes against each other.

        Parameters
        ----------
        analytes : optional, array_like or str
            The analyte(s) to plot. Defaults to all analytes.
        lognorm : bool
            Whether or not to log normalise the colour scale
            of the 2D histogram.
        bins : int
            The number of bins in the 2D histogram.
        filt : str, dict or bool
            Either logical filter expression contained in a str,
            a dict of expressions specifying the filter string to
            use for each analyte or a boolean. Passed to `grab_filt`.

        Returns
        -------
        (fig, axes)
        """
        if analytes is None:
            analytes = self.analytes
        if self.focus_stage in ['ratio', 'calibrated']:
            analytes = [a for a in analytes if self.internal_standard not in a]

        if figsize[0] < 1.5 * len(analytes):
            figsize = [1.5 * len(analytes)] * 2

        numvars = len(analytes)
        fig, axes = plt.subplots(nrows=numvars, ncols=numvars,
                                 figsize=(12, 12))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        for ax in axes.flat:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            if ax.is_first_col():
                ax.yaxis.set_ticks_position('left')
            if ax.is_last_col():
                ax.yaxis.set_ticks_position('right')
            if ax.is_first_row():
                ax.xaxis.set_ticks_position('top')
            if ax.is_last_row():
                ax.xaxis.set_ticks_position('bottom')

        # set up colour scales
        if colourful:
            cmlist = ['Blues', 'BuGn', 'BuPu', 'GnBu',
                      'Greens', 'Greys', 'Oranges', 'OrRd',
                      'PuBu', 'PuBuGn', 'PuRd', 'Purples',
                      'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']
        else:
            cmlist = ['Greys']

        while len(cmlist) < len(analytes):
            cmlist *= 2

        udict = {}
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i, j), (j, i)]:
                # set unit multipliers
                mx, ux = unitpicker(np.nanmean(self.focus[analytes[x]]),
                                    denominator=self.internal_standard,
                                    focus_stage=self.focus_stage)
                my, uy = unitpicker(np.nanmean(self.focus[analytes[y]]),
                                    denominator=self.internal_standard,
                                    focus_stage=self.focus_stage)
                udict[analytes[x]] = (x, ux)

                # get filter
                xd = nominal_values(self.focus[analytes[x]])
                yd = nominal_values(self.focus[analytes[y]])

                ind = (self.filt.grab_filt(filt, analytes[x]) &
                       self.filt.grab_filt(filt, analytes[y]) &
                       ~np.isnan(xd) &
                       ~np.isnan(yd))

                # make plot
                pi = xd[ind] * mx
                pj = yd[ind] * my

                # determine normalisation shceme
                if lognorm:
                    norm = mpl.colors.LogNorm()
                else:
                    norm = None

                # draw plots
                axes[i, j].hist2d(pj, pi, bins,
                                  norm=norm,
                                  cmap=plt.get_cmap(cmlist[i]))
                axes[j, i].hist2d(pi, pj, bins,
                                  norm=norm,
                                  cmap=plt.get_cmap(cmlist[j]))

                axes[x, y].set_ylim([pi.min(), pi.max()])
                axes[x, y].set_xlim([pj.min(), pj.max()])
        # diagonal labels
        for a, (i, u) in udict.items():
            axes[i, i].annotate(a + '\n' + u, (0.5, 0.5),
                                xycoords='axes fraction',
                                ha='center', va='center')
        # switch on alternating axes
        for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
            axes[j, i].xaxis.set_visible(True)
            for label in axes[j, i].get_xticklabels():
                label.set_rotation(90)
            axes[i, j].yaxis.set_visible(True)

        axes[0, 0].set_title(self.sample, weight='bold', x=0.05, ha='left')

        return fig, axes

    def crossplot_filters(self, filter_string, analytes=None):
        """
        Plot the results of a group of filters in a crossplot.

        Parameters
        ----------
        filter_string : str
            A string that identifies a group of filters.
            e.g. 'test' would plot all filters with 'test' in the
            name.
        analytes : optional, array_like or str
            The analyte(s) to plot. Defaults to all analytes.

        Returns
        -------
        fig, axes objects
        """

        if analytes is None:
            analytes = [a for a in self.analytes if 'Ca' not in a]

        # isolate relevant filters
        filts = self.filt.components.keys()
        cfilts = [f for f in filts if filter_string in f]
        flab = re.compile('.*_(.*)$')  # regex to get cluster number

        # set up axes
        numvars = len(analytes)
        fig, axes = plt.subplots(nrows=numvars, ncols=numvars,
                                 figsize=(12, 12))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        for ax in axes.flat:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            if ax.is_first_col():
                ax.yaxis.set_ticks_position('left')
            if ax.is_last_col():
                ax.yaxis.set_ticks_position('right')
            if ax.is_first_row():
                ax.xaxis.set_ticks_position('top')
            if ax.is_last_row():
                ax.xaxis.set_ticks_position('bottom')

        # isolate nominal_values for all analytes
        focus = {k: nominal_values(v) for k, v in self.focus.items()}
        # determine units for all analytes
        udict = {a: unitpicker(np.nanmean(focus[a]),
                               denominator=self.internal_standard,
                               focus_stage=self.focus_stage) for a in analytes}
        # determine ranges for all analytes
        rdict = {a: (np.nanmin(focus[a] * udict[a][0]),
                     np.nanmax(focus[a] * udict[a][0])) for a in analytes}

        for f in cfilts:
            ind = self.filt.grab_filt(f)
            focus = {k: nominal_values(v[ind]) for k, v in self.focus.items()}
            lab = flab.match(f).groups()[0]
            axes[0, 0].scatter([], [], s=10, label=lab)

            for i, j in zip(*np.triu_indices_from(axes, k=1)):
                # get analytes
                ai = analytes[i]
                aj = analytes[j]

                # remove nan, apply multipliers
                pi = focus[ai][~np.isnan(focus[ai])] * udict[ai][0]
                pj = focus[aj][~np.isnan(focus[aj])] * udict[aj][0]

                # make plot
                axes[i, j].scatter(pj, pi, alpha=0.4, s=10, lw=0)
                axes[j, i].scatter(pi, pj, alpha=0.4, s=10, lw=0)

                axes[i, j].set_ylim(*rdict[ai])
                axes[i, j].set_xlim(*rdict[aj])

                axes[j, i].set_ylim(*rdict[aj])
                axes[j, i].set_xlim(*rdict[ai])

        # diagonal labels
        for a, n in zip(analytes, np.arange(len(analytes))):
            axes[n, n].annotate(a + '\n' + udict[a][1], (0.5, 0.5),
                                xycoords='axes fraction',
                                ha='center', va='center')
            axes[n, n].set_xlim(*rdict[a])
            axes[n, n].set_ylim(*rdict[a])

        axes[0, 0].legend(loc='upper left', title=filter_string, fontsize=8)

        # switch on alternating axes
        for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
            axes[j, i].xaxis.set_visible(True)
            for label in axes[j, i].get_xticklabels():
                label.set_rotation(90)
            axes[i, j].yaxis.set_visible(True)

        return fig, axes

    def filt_nremoved(self, filt=True):
        ntot = sum(self.sig)
        nfilt = sum(self.filt.grab_filt(filt) & self.sig)
        pcrm = 100. * (ntot - nfilt) / ntot
        return (ntot, nfilt, pcrm)

    @_log
    def filter_report(self, filt=None, analytes=None, savedir=None, nbin=5):
        """
        Visualise effect of data filters.

        Parameters
        ----------
        filt : str
            Exact or partial name of filter to plot. Supports
            partial matching. i.e. if 'cluster' is specified, all
            filters with 'cluster' in the name will be plotted.
            Defaults to all filters.
        analyte : str
            Name of analyte to plot.
        save : str
            file path to save the plot

        Returns
        -------
        (fig, axes)
        """
        return plot.filter_report(self, filt, analytes, savedir, nbin)

    # reporting
    def get_params(self):
        """
        Returns paramters used to process data.

        Returns
        -------
        dict
            dict of analysis parameters
        """
        outputs = ['sample',
                   'ratio_params',
                   'despike_params',
                   'autorange_params',
                   'bkgcorrect_params']

        out = {}
        for o in outputs:
            out[o] = getattr(self, o)

        out['filter_params'] = self.filt.params
        out['filter_sequence'] = self.filt.sequence
        out['filter_used'] = self.filt.make_keydict()

        return out
