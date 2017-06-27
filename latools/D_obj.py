import os
import re
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import brewer2mpl as cb  # for colours
import warnings
import sklearn.cluster as cl
import scipy.interpolate as interp
import uncertainties.unumpy as un

from io import BytesIO
from IPython import display
from scipy.stats import gaussian_kde, pearsonr
from scipy.optimize import curve_fit
from sklearn import preprocessing
from functools import wraps
from mpld3 import enable_notebook, disable_notebook, plugins

import latools.process_fns as proc
from .filt_obj import filt
from .helpers import bool_2_indices, fastgrad, rolling_window, fastsmooth
from .helpers import unitpicker, pretty_element
from .stat_fns import nominal_values, std_devs, unpack_uncertainties, gauss


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
            rawdata: created during initialisation.
            despiked: created by `despike`
            signal: created by `autorange`
            background: created by `autorange`
            bkgsub: created by `bkg_correct`
            ratios: created by `ratio`
            calibrated: created by `calibrate`
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
        analysys spots when calculating sample statistics.
    filt : filt object
        An object for storing, selecting and creating data filters.

    Methods
    -------
    ablation_times
    autorange
    bkg_subtract
    calibrate
    cluster_DBSCAN
    cluster_kmeans
    cluster_meanshift
    crossplot
    despike
    drift_params
    expdecay_despiker
    filt_report
    filter_clustering
    filter_correlation
    filter_distribution
    filter_threshold
    findmins
    get_params
    mkrngs
    noise_despiker
    ratio
    sample_stats
    setfocus
    tplot

    """

    def __init__(self, data_file, dataformat=None, errorhunt=False, cmap=None, internal_standard='Ca43', name='file_names'):
        if errorhunt:
            # errorhunt prints each csv file name before it tries to load it,
            # so you can tell which file is failing to load.
            print(data_file)
        params = locals()
        del(params['self'])
        self.log = ['__init__ :: args=() kwargs={}'.format(str(params))]

        self.file = data_file
        self.internal_standard = internal_standard

        with open(data_file) as f:
            lines = f.readlines()

        # read the metadata, using key, regex pairs in the line - numbered
        # dataformat['meta_regex'] dict.
        # metadata
        if 'meta_regex' in dataformat.keys():
            self.meta = {}
            for k, v in dataformat['meta_regex'].items():
                out = re.search(v[-1], lines[int(k)]).groups()
                for i in np.arange(len(v[0])):
                    self.meta[v[0][i]] = out[i]

        # sample name
        if name == 'file_names':
            self.sample = os.path.basename(self.file).split('.')[0]
        elif name == 'metadata_names':
            self.sample = self.meta['name']
        else:
            self.sample = 0

        # column names
        columns = np.array(lines[dataformat['column_id']['name_row']].strip().split(dataformat['column_id']['delimiter']))
        if 'pattern' in dataformat['column_id'].keys():
            pr = re.compile(dataformat['column_id']['pattern'])
            columns = [pr.match(c).groups()[0] for c in columns if pr.match(c)]
        self.analytes = np.array(columns)

        columns = np.insert(columns, dataformat['column_id']['timecolumn'], 'Time')

        # do any required pre-formatting
        if 'preformat_replace' in dataformat.keys():
            clean = True
            with open(data_file) as f:
                    fbuffer = f.read()
            for k, v in dataformat['preformat_replace'].items():
                fbuffer = re.sub(k, v, fbuffer)

            read_data = np.genfromtxt(BytesIO(fbuffer.encode()),
                                      **dataformat['genfromtext_args']).T

        else:
            read_data = np.genfromtxt(data_file,
                                      **dataformat['genfromtext_args']).T

        # create data dict
        self.data = {}
        self.data['rawdata'] = dict(zip(columns, read_data))
        self.data['total_counts'] = read_data.sum(0)

        # set focus to rawdata
        self.setfocus('rawdata')

        # make a colourmap for plotting
        try:
            self.cmap = dict(zip(self.analytes,
                                 cb.get_map('Paired', 'qualitative',
                                            len(columns)).hex_colors))
        except:
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

    def _log(fn):
        """
        Function for logging method calls and parameters
        """
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            self.log.append(fn.__name__ + ' :: args={} kwargs={}'.format(args, kwargs))
            return fn(self, *args, **kwargs)
        return wrapper

    def setfocus(self, focus):
        """
        Set the 'focus' attribute of the data file.

        The 'focus' attribute of the object points towards data from a
        particular stage of analysis. It is used to identify the 'working
        stage' of the data. Processing functions operate on the 'focus'
        stage, so if steps are done out of sequence, things will break.

        Parameters
        ----------
        focus : str
            The name of the analysis stage desired:
                'rawdata': raw data, loaded from csv file when object
                    is initialised.
                'despiked': despiked data.
                'signal'/'background': isolated signal and background data,
                    padded with np.nan. Created by self.separate, after
                    signal and background regions have been identified by
                    self.autorange.
                'bkgsub': background subtracted data, created by
                    self.bkg_correct
                'ratios': element ratio data, created by self.ratio.
                'calibrated': ratio data calibrated to standards, created by
                    self.calibrate.

        Returns
        -------
        None
        """
        self.focus = self.data[focus]
        self.focus_stage = focus
        for k in self.focus.keys():
            setattr(self, k, self.focus[k])

    @_log
    def despike(self, expdecay_despiker=True, exponent=None, tstep=None,
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
        tstep : None or float
            The timeinterval between measurements. If None, it is
            determined automatically from the Time variable.
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
            self.data['despiked'] = {}

        out = {}
        for a, v in self.focus.items():
            if 'time' not in a.lower():
                sig = v.copy()  # copy data
                if noise_despiker:
                    sig = proc.noise_despike(sig, int(win), nlim, maxiter)
                if expdecay_despiker:
                    warnings.warn('expdecay_spiker is broken... not run.')
                    # sig = proc.expdecay_despike(v, exponent, tstep, maxiter)
                out[a] = sig

        self.data['despiked'].update(out)
        self.setfocus('despiked')
        return

    # helper functions for data selection
    def findmins(self, x, y):
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

    @_log
    def autorange(self, analyte=None, gwin=11, win=40, smwin=5,
                  conf=0.01, on_mult=[1., 1.], off_mult=None, d_mult=1.2,
                  transform='log', bkg_thresh=None, ploterrs=True):
        """
        Automatically separates signal and background data regions.

        Automatically detect signal and background regions in the laser
        data, based on the behaviour of a single analyte. The analyte used
        should be abundant and homogenous in the sample.

        Step 1: Thresholding
        The background signal is determined using a gaussian kernel density
        estimator (kde) of all the data. Under normal circumstances, this
        kde should find two distinct data distributions, corresponding to
        'signal' and 'background'. The minima between these two distributions
        is taken as a rough threshold to identify signal and background
        regions. Any point where the trace crosses this thrshold is identified
        as a 'transition'.

        Step 2: Transition Removal
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
        smwin : int
            The smoothing window used for calculating the second derivative.
            Must be odd.
        on_mult and off_mult : tuple, len=2
            Factors to control the width of the excluded transition regions.
            A region n times the full - width - half - maximum of the transition
            gradient will be removed either side of the transition center.
            `on_mult` and `off_mult` refer to the laser - on and laser - off
            transitions, respectively. See manual for full explanation.
            Defaults to (1.5, 1) and (1, 1.5).


        Adds
        ----
        bkg, sig, trn : bool, array_like
            Boolean arrays the same length as the data, identifying
            'background', 'signal' and 'transition' data regions.
        bkgrng, sigrng, trnrng: array_like
            Pairs of values specifying the edges of the 'background', 'signal'
            and 'transition' data regions in the same units as the Time axis.

        Returns
        -------
        None
        """

        if analyte is None:
            analyte = self.internal_standard

        if off_mult is None:
            off_mult = on_mult[::-1]

        # define transformation and back-transformation functions
        if transform is None:
            trans = btrans = lambda x: x
        elif transform == 'log':
            trans = lambda x: np.log10(x[x > 1])  # forward transform
            btrans = lambda x: 10**x  # back transform
        # add more transformation functions here, if required

        bins = 50  # determine automatically? As a function of bkg rms noise?

        if analyte == 'total_counts':
            v = self.data['total_counts']
        else:
            v = self.focus[analyte]  # get trace data
        vl = trans(v)  # apply transformation
        x = np.linspace(vl.min(), vl.max(), bins)  # define bin limits

        n, _ = np.histogram(vl, x)  # make histogram of sample
        kde = gaussian_kde(vl)
        yd = kde.pdf(x)  # calculate gaussian_kde of sample

        mins = self.findmins(x, yd)  # find minima in kde

        vs = fastsmooth(v, gwin)
        bkg = vs < btrans(d_mult * mins[0])  # set background as lowest distribution
        if not bkg[0]:
            bkg[0] = True

        # assign rough background and signal regions based on kde minima
        self.bkg = bkg
        self.sig = ~bkg

        # remove transitions by fitting a gaussian to the gradients of
        # each transition
        # 1. calculate the absolute gradient of the target trace.
        g = abs(fastgrad(v, gwin))
        # 2. determine the approximate index of each transition
        zeros = bool_2_indices(bkg).flatten()
        if zeros[0] == 0:
            zeros = zeros[1:]
        if zeros[-1] == bkg.size:
            zeros = zeros[:-1]
        tran = []  # initialise empty list for transition pairs

        makeplot = False
        plotlines = []

        for z in zeros:  # for each approximate transition
            # isolate the data around the transition
            if z - win > 0:
                xs = self.Time[z - win:z + win]
                ys = g[z - win:z + win]
                # determine type of transition (on/off)
                # checkes whether first - last value in window is
                # positive ('on') or negative ('off')
                tp = np.diff(v[z - win:z + win][[0, -1]]) > 0

            else:
                xs = self.Time[:z + win]
                ys = g[:z + win]
                # determine type of transition (on/off)
                tp = np.diff(v[:z + win][[0, -1]]) > 0
            # determine location of maximum gradient
            c = self.Time[z]  # xs[ys == np.nanmax(ys)]
            try:  # in case some of them don't work...
                # fit a gaussian to the first derivative of each
                # transition. Initial guess parameters are determined
                # by:
                #   - A: maximum gradient in data
                #   - mu: c
                #   - sigma: half the exponential decay coefficient used
                #       for despiking OR 1., if there is no exponent.
                try:
                    width = 0.5 * abs(self.despike_params['exponent'])
                except:
                    width = 1.
                # The 'sigma' parameter of curve_fit:
                # This weights the fit by distance from c - i.e. data closer
                # to c are more important in the fit than data further away
                # from c. This allows the function to fit the correct curve,
                # even if the data window has captured two independent
                # transitions (i.e. end of one ablation and start of next)
                # ablation are < win time steps apart).
                pg, _ = curve_fit(gauss, xs, ys,
                                  p0=(np.nanmax(ys),
                                      c,
                                      width),
                                  sigma=abs(xs - c) + .1)
                # get the x positions when the fitted gaussian is at 'conf' of
                # maximum
                # determine transition FWHM
                fwhm = 2 * pg[-1] * np.sqrt(2 * np.log(2))
                # apply on_mult or off_mult, as appropriate.
                if tp:
                    lim = np.array([-fwhm, fwhm]) * np.array(on_mult) + pg[1]
                else:
                    lim = np.array([-fwhm, fwhm]) * np.array(off_mult) + pg[1]

                tran.append(lim)
            except:
                if ploterrs:
                    makeplot = True
                    plotlines.append(self.Time[z])
                warnings.warn(("\n\nSample {:s}: ".format(self.sample) +
                               "Transition identification at " +
                               "{:.1f} failed.".format(self.Time[z]) +
                               "\n  **This is not necessarily a problem**"
                               "\nBut please check the data plots and make sure " +
                               "everything is OK.\n"))
                pass  # if it fails for any reason, warn and skip it!
        # remove the transition regions from the signal and background ids.
        for t in tran:
            self.bkg[(self.Time > t[0]) & (self.Time < t[1])] = False
            self.sig[(self.Time > t[0]) & (self.Time < t[1])] = False

        self.trn = ~self.bkg & ~self.sig

        self.mkrngs()

        # final check to catch missed transitions
        # calculate average transition width
        tr = self.Time[self.trn ^ np.roll(self.trn, 1)]
        tr = np.reshape(tr, [tr.size // 2, 2])
        self.trnrng = tr
        trw = np.mean(np.diff(tr, axis=1))

        corr = False
        for b in self.bkgrng.flat:
            if (self.sigrng - b < 0.3 * trw).any():
                self.bkg[(self.Time >= b - trw / 2) &
                         (self.Time <= b + trw / 2)] = False
                self.sig[(self.Time >= b - trw / 2) &
                         (self.Time <= b + trw / 2)] = False
                corr = True

        if corr:
            self.mkrngs()

        # remove any background regions that contain internal_standard concs above bkg_thresh
        if bkg_thresh is not None:
            self.bkg[self.focus[self.internal_standard] > bkg_thresh] = False
            self.mkrngs()

        # number the signal regions (used for statistics and standard matching)
        # self.ns = np.zeros(self.Time.size)
        # n = 1
        # for i in range(len(self.sig) - 1):
        #     if self.sig[i]:
        #         self.ns[i] = n
        #     if self.sig[i] and ~self.sig[i + 1]:
        #         n += 1
        # self.n = int(max(self.ns))  # record number of traces

        if makeplot:
            f, ax = self.tplot(ranges=True)
            for pl in plotlines:
                ax.axvline(pl, c='r', alpha=0.6, lw=3, ls='dashed')

        return

    def mkrngs(self):
        """
        Transform boolean arrays into list of limit pairs.

        Gets Time limits of signal/background boolean arrays and stores them as
        sigrng and bkgrng arrays. These arrays can be saved by 'save_ranges' in
        the analyse object.
        """
        self.bkg[[0, -1]] = False
        bkgr = self.Time[self.bkg ^ np.roll(self.bkg, -1)]
        self.bkgrng = np.reshape(bkgr, [bkgr.size // 2, 2])

        self.sig[[0, -1]] = False
        sigr = self.Time[self.sig ^ np.roll(self.sig, 1)]
        self.sigrng = np.reshape(sigr, [sigr.size // 2, 2])

        self.trn[[0, -1]] = False
        trnr = self.Time[self.trn ^ np.roll(self.trn, 1)]
        self.trnrng = np.reshape(trnr, [trnr.size // 2, 2])

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
    def bkg_subtract(self, analyte, bkg, ind=None):
        """
        Subtract provided background from signal (focus stage).

        Results is saved in new 'bkgsub' focus stage


        Returns
        -------
        None
        """

        if 'bkgsub' not in self.data.keys():
            self.data['bkgsub'] = {}

        self.data['bkgsub'][analyte] = self.focus[analyte] - bkg

        if ind is not None:
            self.data['bkgsub'][analyte][ind] = np.nan

        return

    @_log
    def ratio(self, internal_standard=None, focus='bkgsub'):
        """
        Divide all analytes by a specified internal_standard analyte.

        Parameters
        ----------
        internal_standard : str
            The analyte used as the internal_standard.
        focus : str
            The analysis stage to perform the ratio calculation on.
            Defaults to 'signal'.

        Returns
        -------
        None
        """
        if internal_standard is not None:
            self.internal_standard = internal_standard

        self.setfocus(focus)
        self.data['ratios'] = {}
        for a in self.analytes:
            self.data['ratios'][a] = \
                self.focus[a] / self.focus[self.internal_standard]
        self.setfocus('ratios')
        return

    def drift_params(self, pout, a):
        p_nom = list(zip(*pout.loc[:, a].apply(nominal_values)))
        p_err = list(zip(*pout.loc[:, a].apply(std_devs)))

        if len(p_nom) > 1:
            npar = len(p_nom)

            ps = []
            for i in range(npar):
                p_est = interp.interp1d(pout.index.values, p_nom[i])
                e_est = interp.interp1d(pout.index.values, p_err[i])
                ps.append(un.uarray(p_est(self.uTime), e_est(self.uTime)))

            return ps
        else:
            return pout[a]

    @_log
    def calibrate(self, calib_ms, analytes=None):
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
            self.data['calibrated'] = {}

        for a in analytes:
            P = calib_ms[a].new(self.uTime)

            self.data['calibrated'][a] = self.data['ratios'][a] * P

        if self.internal_standard not in analytes:
            self.data['calibrated'][self.internal_standard] = \
                np.empty(len(self.data['ratios'][self.internal_standard]))

            # coefs = calib_params[a]
            # if len(coefs) == 1:
            #     self.data['calibrated'][a] = \
            #         self.data['ratios'][a] * coefs
            # else:
            #     self.data['calibrated'][a] = \
            #         np.polyval(coefs, self.data['ratios'][a])
            #         self.data['ratios'][a] * coefs[0] + coefs[1]
        self.setfocus('calibrated')
        return

    # Function for calculating sample statistics
    @_log
    def sample_stats(self, analytes=None, filt=True,
                     stat_fns={},
                     eachtrace=True):
        """
        TODO: WORK OUT HOW TO HANDLE ERRORS PROPERLY!

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

        self.stats = {}
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

        # try:
            # self.unstats = un.uarray(self.stats['nanmean'],
            #                          self.stats['nanstd'])
        # except:
            # pass

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
    def filter_threshold(self, analyte, threshold, filt=False):
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
        filt : TYPE
            Description of `filt`.

        Returns
        -------
        None
        """
        params = locals()
        del(params['self'])

        # generate filter
        vals = trace = nominal_values(self.focus[analyte])
        if not isinstance(filt, bool):
            ind = (self.filt.grab_filt(filt, analyte) & ~np.isnan(vals))
        else:
            ind = ~np.isnan(vals)

        setn = self.filt.maxset + 1

        if any(ind):
            self.filt.add(analyte + '_thresh_below',
                          trace <= threshold,
                          'Keep below {:.3e} '.format(threshold) + analyte,
                          params, setn=setn)
            self.filt.add(analyte + '_thresh_above',
                          trace >= threshold,
                          'Keep above {:.3e} '.format(threshold) + analyte,
                          params, setn=setn)
        else:
            # if there are no data
            name = analyte + '_thresh_nodata'
            info = analyte + ' threshold filter (no data)'

            self.filt.add(name, np.zeros(self.Time.size, dtype=bool),
                          info=info, params=params, setn=setn)

    @_log
    def filter_distribution(self, analytes, binwidth='scott', filt=False,
                            transform=None, min_data=10):
        """
        Apply a Distribution Filter

        Parameters
        ----------
        analytes : str or list
            The analyte that the filter applies to.
        binwidth : str of float
            Specify the bin width of the kernel density estimator.
            Passed to `scipy.stats.gaussian_kde`.
            str: The method used to automatically estimate bin width.
                 Can be 'scott' or 'silverman'.
            float: Manually specify the binwidth of the data.
        filt : bool
            Whether or not to apply existing filters to the data before
            calculating this filter.
        transform : str
            If 'log', applies a log transform to the data before calculating
            the distribution.
        min_data : int
            The minimum number of data points that should be considered by
            the filter. Default = 10.

        Returns
        -------
        None
        """
        params = locals()
        del(params['self'])

        if isinstance(analytes, str):
            analytes = [analytes]

        for analyte in analytes:
            # generate filter
            vals = np.vstack(nominal_values(list(self.focus.values())))
            if filt is not None:
                ind = (self.filt.grab_filt(filt, analyte) &
                       np.apply_along_axis(all, 0, ~np.isnan(vals)))
            else:
                ind = np.apply_along_axis(all, 0, ~np.isnan(vals))

            if sum(ind) > min_data:
                # isolate data
                d = nominal_values(self.focus[analyte][ind])
                setn = self.filt.maxset + 1

                if transform == 'log':
                    d = np.log10(d)

                # gaussian kde of data
                kde = gaussian_kde(d, bw_method=binwidth)
                x = np.linspace(np.nanmin(d), np.nanmax(d),
                                kde.dataset.size // 3)
                yd = kde.pdf(x)
                limits = np.concatenate([self.findmins(x, yd), [x.max()]])

                if transform == 'log':
                    limits = 10**limits

                if limits.size > 1:
                    first = True
                    for i in np.arange(limits.size):
                        if first:
                            filt = self.focus[analyte] < limits[i]
                            info = analyte + ' distribution filter, 0 <i> {:.2e}'.format(limits[i])
                            first = False
                        else:
                            filt = (self.focus[analyte] < limits[i]) & (self.focus[analyte] > limits[i - 1])
                            info = analyte + ' distribution filter, {:.2e} <i> {:.2e}'.format(limits[i - 1], limits[i])

                        self.filt.add(name=analyte + '_distribution_{:.0f}'.format(i),
                                      filt=filt,
                                      info=info,
                                      params=params, setn=setn)
                else:
                    self.filt.add(name=analyte + '_distribution_failed',
                                  filt=~np.isnan(nominal_values(self.focus[analyte])),
                                  info=analyte + ' is within a single distribution. No data removed.',
                                  params=params, setn=setn)
            else:
                # if there are no data
                name = analyte + '_distribution_0'
                info = analyte + ' distribution filter (< {:.0f} data points)'.format(min_data)

                self.filt.add(name, np.zeros(self.Time.size, dtype=bool),
                              info=info, params=params, setn=setn)
        return

    @_log
    def filter_clustering(self, analytes, filt=False, normalise=True,
                          method='meanshift', include_time=False,
                          sort=None, min_data=10, **kwargs):
        """
        Applies an n - dimensional clustering filter to the data.

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
            Which clustering algorithm to use. Can be:
                'meanshift': The `sklearn.cluster.MeanShift` algorithm.
                             Automatically determines number of clusters
                             in data based on the `bandwidth` of expected
                             variation.
                'kmeans': The `sklearn.cluster.KMeans` algorithm. Determines
                          the characteristics of a known number of clusters
                          within the data. Must provide `n_clusters` to specify
                          the expected number of clusters.
                'DBSCAN': The `sklearn.cluster.DBSCAN` algorithm. Automatically
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
        include_time : bool
            Whether or not to include the Time variable in the clustering
            analysis. Useful if you're looking for spatially continuous
            clusters in your data, i.e. this will identify each spot in your
            analysis as an individual cluster.
        sort : bool, str or array-like
            Whether or not to label the resulting clusters according to their
            contents. If used, the cluster with the lowest values will be
            labelled from 0, in order of increasing cluster mean value.analytes
                True: Sort by all analytes used to generate the cluster.
                str: Sort by a single specified analyte
                array-like: Sort by a number of specified analytes.
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
            if len(analytes) == 1:
                # if single analyte
                d = nominal_values(self.focus[analytes[0]][ind])
                if include_time:
                    t = self.Time[ind]
                    ds = np.vstack([d, t]).T
                else:
                    ds = np.array(list(zip(d, np.zeros(len(d)))))
            else:
                # package multiple analytes
                d = [nominal_values(self.focus[a][ind]) for a in analytes]
                if include_time:
                    d.append(self.Time[ind])
                ds = np.vstack(d).T

            if normalise | (len(analytes) > 1):
                ds = preprocessing.scale(ds)

            method_key = {'kmeans': self.cluster_kmeans,
                          'DBSCAN': self.cluster_DBSCAN,
                          'meanshift': self.cluster_meanshift}

            cfun = method_key[method]

            labels, core_samples_mask = cfun(ds, **kwargs)
            # return labels, and if DBSCAN core_sample_mask

            labels_unique = np.unique(labels)

            # label the clusters according to their contents
            if (sort is not None) & (sort is not False):
                if isinstance(sort, str):
                    sort = [sort]

                if len(analytes) == 1:
                    sanalytes = analytes + [False]
                else:
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

    def cluster_meanshift(self, data, bandwidth=None, bin_seeding=False):
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

        ms = cl.MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
        ms.fit(data)

        labels = ms.labels_

        return labels, [np.nan]

    def cluster_kmeans(self, data, n_clusters):
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
        km = cl.KMeans(n_clusters)
        kmf = km.fit(data)

        labels = kmf.labels_

        return labels, [np.nan]

    def cluster_DBSCAN(self, data, eps=None, min_samples=None,
                       n_clusters=None, maxiter=200):
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
        if min_samples is None:
            min_samples = self.Time.size // 20

        if n_clusters is None:
            if eps is None:
                eps = 0.3
            db = cl.DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        else:
            clusters = 0
            eps_temp = 1 / .95
            niter = 0
            while clusters < n_clusters:
                clusters_last = clusters
                eps_temp *= 0.95
                db = cl.DBSCAN(eps=eps_temp, min_samples=15).fit(data)
                clusters = (len(set(db.labels_)) -
                            (1 if -1 in db.labels_ else 0))
                if clusters < clusters_last:
                    eps_temp *= 1 / 0.95
                    db = cl.DBSCAN(eps=eps_temp, min_samples=15).fit(data)
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

    @_log
    def filter_correlation(self, x_analyte, y_analyte, window=None,
                           r_threshold=0.9, p_threshold=0.05, filt=True):
        """
        Apply correlation filter.

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

        Returns
        -------
        None
        """

        # automatically determine appripriate window?

        # make window odd
        if window is None:
            window = 11
        elif window % 2 != 1:
            window += 1

        params = locals()
        del(params['self'])

        setn = self.filt.maxset + 1

        # get filter
        ind = self.filt.grab_filt(filt, [x_analyte, y_analyte])

        x = nominal_values(self.focus[x_analyte])
        x[~ind] = np.nan
        xr = rolling_window(x, window, pad=np.nan)

        y = nominal_values(self.focus[y_analyte])
        y[~ind] = np.nan
        yr = rolling_window(y, window, pad=np.nan)

        r, p = zip(*map(pearsonr, xr, yr))

        r = np.array(r)
        p = np.array(p)

        cfilt = (abs(r) > r_threshold) & (p < p_threshold)
        cfilt = ~cfilt

        name = x_analyte + ' - ' + y_analyte + '_corr'

        self.filt.add(name=name,
                      filt=cfilt,
                      info=(x_analyte + ' vs. ' + y_analyte +
                            ' correlation filter.'),
                      params=params, setn=setn)
        self.filt.off(filt=name)
        self.filt.on(analyte=y_analyte, filt=name)

        return

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

    # Plotting Functions
    # def genaxes(self, n, ncol=4, panelsize=[3, 3], tight_layout=True,
    #             **kwargs):
    #     """
    #     Function to generate a grid of subplots for a given set of plots.
    #     """
    #     if n % ncol is 0:
    #         nrow = int(n/ncol)
    #     else:
    #         nrow = int(n//ncol + 1)

    #     fig, axes = plt.subplots(nrow, ncol, figsize=[panelsize[0] * ncol,
    #                              panelsize[1] * nrow],
    #                              tight_layout=tight_layout,
    #                              **kwargs)
    #     for ax in axes.flat[n:]:
    #         fig.delaxes(ax)

    #     return fig, axes

    @_log
    def tplot(self, analytes=None, figsize=[10, 4], scale='log', filt=None,
              ranges=False, stats=False, stat='nanmean', err='nanstd',
              interactive=False, focus_stage=None, err_envelope=False):
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
        interactive : bool
            Make the plot interactive.

        Returns
        -------
        figure, axis
        """

        if interactive:
            enable_notebook()  # make the plot interactive

        if type(analytes) is str:
            analytes = [analytes]
        if analytes is None:
            analytes = self.analytes

        if focus_stage is None:
            focus_stage = self.focus_stage

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([.1, .12, .77, .8])

        for a in analytes:
            x = self.Time
            y, yerr = unpack_uncertainties(self.data[focus_stage][a])

            if scale is 'log':
                ax.set_yscale('log')
                y[y == 0] = np.nan

            if filt:
                ind = self.filt.grab_filt(filt, a)
                xf = x.copy()
                yf = y.copy()
                yerrf = yerr.copy()
                if any(~ind):
                    xf[~ind] = np.nan
                    yf[~ind] = np.nan
                    yerrf[~ind] = np.nan
                if any(~ind):
                    ax.plot(x, y, color=self.cmap[a], alpha=.4, lw=0.6)
                ax.plot(xf, yf, color=self.cmap[a], label=a)
                if err_envelope:
                    ax.fill_between(xf, yf - yerrf, yf + yerrf, color=self.cmap[a],
                                    alpha=0.2, zorder=-1)
            else:
                ax.plot(x, y, color=self.cmap[a], label=a)
                if err_envelope:
                    ax.fill_between(x, y - yerr, y + yerr, color=self.cmap[a],
                                    alpha=0.2, zorder=-1)

            # Plot averages and error envelopes
            if stats and hasattr(self, 'stats'):
                warnings.warn('Stat plot is broken.')
                pass
                # sts = self.stats[sig][0].size
                # if sts > 1:
                #     for n in np.arange(self.n):
                #         n_ind = ind & (self.ns == n + 1)
                #         if sum(n_ind) > 2:
                #             x = [self.Time[n_ind][0], self.Time[n_ind][-1]]
                #             y = [self.stats[sig][self.stats['analytes'] == a][0][n]] * 2

                #             yp = ([self.stats[sig][self.stats['analytes'] == a][0][n] +
                #                   self.stats[err][self.stats['analytes'] == a][0][n]] * 2)
                #             yn = ([self.stats[sig][self.stats['analytes'] == a][0][n] -
                #                   self.stats[err][self.stats['analytes'] == a][0][n]] * 2)

                #             ax.plot(x, y, color=self.cmap[a], lw=2)
                #             ax.fill_between(x + x[::-1], yp + yn,
                #                             color=self.cmap[a], alpha=0.4,
                #                             linewidth=0)
                # else:
                #     x = [self.Time[0], self.Time[-1]]
                #     y = [self.stats[sig][self.stats['analytes'] == a][0]] * 2
                #     yp = ([self.stats[sig][self.stats['analytes'] == a][0] +
                #           self.stats[err][self.stats['analytes'] == a][0]] * 2)
                #     yn = ([self.stats[sig][self.stats['analytes'] == a][0] -
                #           self.stats[err][self.stats['analytes'] == a][0]] * 2)

                #     ax.plot(x, y, color=self.cmap[a], lw=2)
                #     ax.fill_between(x + x[::-1], yp + yn, color=self.cmap[a],
                #                     alpha=0.4, linewidth=0)

        if ranges:
            for lims in self.bkgrng:
                ax.axvspan(*lims, color='k', alpha=0.1, zorder=-1)
            for lims in self.sigrng:
                ax.axvspan(*lims, color='r', alpha=0.1, zorder=-1)

        if filt is not None:
            ind = self.filt.grab_filt(filt)
            lims = bool_2_indices(~ind)
            for l, u in lims:
                if abs(u) >= len(self.Time):
                    u = -1
                if l < 0:
                    l = 0
                ax.axvspan(self.Time[l], self.Time[u], color='k',
                           alpha=0.05, lw=0)

            # drawn = []
            # for k, v in self.filt.switches.items():
            #     for f, s in v.items():
            #         if s & (f not in drawn):
            #             lims = bool_2_indices(~self.filt.components[f])
            #             for u, l in lims:
            #                 ax.axvspan(self.Time[u-1], self.Time[l], color='k',
            #                            alpha=0.05, lw=0)
            #             drawn.append(f)

        ax.text(0.01, 0.99, self.sample + ' : ' + self.focus_stage,
                transform=ax.transAxes,
                ha='left', va='top')

        ax.set_xlabel('Time (s)')
        ax.set_xlim(np.nanmin(x), np.nanmax(x))

        # y label
        ud = {'rawdata': 'counts',
              'despiked': 'counts',
              'bkgsub': 'background corrected counts',
              'ratios': 'counts/{:s} count',
              'calibrated': 'mol/mol {:s}'}
        if focus_stage in ['ratios', 'calibrated']:
            ud[focus_stage] = ud[focus_stage].format(self.internal_standard)
        ax.set_ylabel(ud[focus_stage])

        if interactive:
            ax.legend()
            plugins.connect(fig, plugins.MousePosition(fontsize=14))
            display.clear_output(wait=True)
            display.display(fig)
            input('Press [Return] when finished.')
            disable_notebook()  # stop the interactivity
        else:
            ax.legend(bbox_to_anchor=(1.15, 1))

        return fig, ax

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
                ind = (self.filt.grab_filt(filt, analytes[x]) &
                       self.filt.grab_filt(filt, analytes[y]) &
                       ~np.isnan(self.focus[analytes[x]]) &
                       ~np.isnan(self.focus[analytes[y]]))

                # make plot
                pi = self.focus[analytes[x]][ind] * mx
                pj = self.focus[analytes[y]][ind] * my

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

    @_log
    def filt_report(self, filt=None, analytes=None, savedir=None):
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
        if filt is None or filt == 'all':
            sets = self.filt.sets
        else:
            sets = {k: v for k, v in self.filt.sets.items() if any(filt in f for f in self.filt.components.keys())}

        regex = re.compile('^([0-9]+)_([A-Za-z0-9-]+)_'
                           '([A-Za-z0-9-]+)[_$]?'
                           '([a-z0-9]+)?')

        cm = plt.cm.get_cmap('Spectral')
        ngrps = len(sets)

        if analytes is None:
            analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        for analyte in analytes:
            if analyte != self.internal_standard:
                fig = plt.figure()

                for i in sorted(sets.keys()):
                    filts = sets[i]
                    nfilts = np.array([re.match(regex, f).groups() for f in filts])
                    fgnames = np.array(['_'.join(a) for a in nfilts[:, 1:3]])
                    fgrp = np.unique(fgnames)[0]

                    fig.set_size_inches(10, 3.5 * ngrps)
                    h = .8 / ngrps

                    y = nominal_values(self.focus[analyte])
                    yh = y[~np.isnan(y)]

                    m, u = unitpicker(np.nanmax(y),
                                      denominator=self.internal_standard,
                                      focus_stage=self.focus_stage)

                    axs = tax, hax = (fig.add_axes([.1, .9 - (i + 1) * h, .6, h * .98]),
                                      fig.add_axes([.7, .9 - (i + 1) * h, .2, h * .98]))

                    # get variables
                    fg = sets[i]
                    cs = cm(np.linspace(0, 1, len(fg)))
                    fn = ['_'.join(x) for x in nfilts[:, (0, 3)]]
                    an = nfilts[:, 0]
                    bins = np.linspace(np.nanmin(y), np.nanmax(y), 50) * m

                    if 'DBSCAN' in fgrp:
                        # determine data filters
                        core_ind = self.filt.components[[f for f in fg
                                                         if 'core' in f][0]]
                        other = np.array([('noise' not in f) & ('core' not in f)
                                          for f in fg])
                        tfg = fg[other]
                        tfn = fn[other]
                        tcs = cm(np.linspace(0, 1, len(tfg)))

                        # plot all data
                        hax.hist(m * yh, bins, alpha=0.5, orientation='horizontal',
                                 color='k', lw=0)
                        # legend markers for core/member
                        tax.scatter([], [], s=15, label='core', c='w', lw=0.5, edgecolor='k')
                        tax.scatter([], [], s=5, label='member', c='w', lw=0.5, edgecolor='k')
                        # plot noise
                        try:
                            noise_ind = self.filt.components[[f for f in fg
                                                              if 'noise' in f][0]]
                            tax.scatter(self.Time[noise_ind], m * y[noise_ind],
                                        lw=1, c='k', s=10, marker='x',
                                        label='noise', alpha=0.6)
                        except:
                            pass

                        # plot filtered data
                        for f, c, lab in zip(tfg, tcs, tfn):
                            ind = self.filt.components[f]
                            tax.scatter(self.Time[~core_ind & ind],
                                        m * y[~core_ind & ind], lw=.5, c=c, s=5, edgecolor='k')
                            tax.scatter(self.Time[core_ind & ind],
                                        m * y[core_ind & ind], lw=.5, c=c, s=15, edgecolor='k',
                                        label=lab)
                            hax.hist(m * y[ind][~np.isnan(y[ind])], bins, color=c, lw=0.1,
                                     orientation='horizontal', alpha=0.6)

                    else:
                        # plot all data
                        tax.scatter(self.Time, m * y, c='k', alpha=0.5, lw=0.1,
                                    s=15, label='excl')
                        hax.hist(m * yh, bins, alpha=0.5, orientation='horizontal',
                                 color='k', lw=0)

                        # plot filtered data
                        for f, c, lab in zip(fg, cs, fn):
                            ind = self.filt.components[f]
                            tax.scatter(self.Time[ind], m * y[ind], lw=.5,
                                        edgecolor='k', c=c, s=15, label=lab)
                            hax.hist(m * y[ind][~np.isnan(y[ind])], bins, color=c, lw=0.1,
                                     orientation='horizontal', alpha=0.6)

                    if 'thresh' in fgrp and analyte in fgrp:
                        tax.axhline(self.filt.params[fg[0]]['threshold'] * m,
                                    ls='dashed', zorder=-2, alpha=0.5, c='k')
                        hax.axhline(self.filt.params[fg[0]]['threshold'] * m,
                                    ls='dashed', zorder=-2, alpha=0.5, c='k')

                    # formatting
                    for ax in axs:
                        mn = np.nanmin(y) * m
                        mx = np.nanmax(y) * m
                        rn = mx - mn
                        ax.set_ylim(mn - .05 * rn, mx + 0.05 * rn)

                    # legend
                    hn, la = tax.get_legend_handles_labels()
                    hax.legend(hn, la, loc='upper right', scatterpoints=1)

                    tax.text(.02, .98, self.sample + ': ' + fgrp, size=12,
                             weight='bold', ha='left', va='top',
                             transform=tax.transAxes)
                    tax.set_ylabel(pretty_element(analyte) + ' (' + u + ')')
                    tax.set_xticks(tax.get_xticks()[:-1])
                    hax.set_yticklabels([])

                    if i < ngrps - 1:
                        tax.set_xticklabels([])
                        hax.set_xticklabels([])
                    else:
                        tax.set_xlabel('Time (s)')
                        hax.set_xlabel('n')

            if isinstance(savedir, str):
                fig.savefig(savedir + '/' + self.sample + '_' +
                            analyte + '.pdf')
                plt.close(fig)

        return
        # return fig, axes

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
