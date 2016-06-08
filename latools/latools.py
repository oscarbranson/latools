import os
import re
import itertools
import warnings
import configparser
import pkg_resources
import time
import json
import numpy as np
import pandas as pd
import brewer2mpl as cb  # for colours
import matplotlib.pyplot as plt
import matplotlib as mpl
import uncertainties.unumpy as un
import sklearn.cluster as cl
from sklearn import preprocessing
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from mpld3 import plugins
from mpld3 import enable_notebook, disable_notebook
from IPython import display

class analyse(object):
    """
    For processing and analysing whole LA-ICPMS datasets.

    Parameters
    ----------
    csv_folder : str
        The path to a directory containing multiple data files.
    errorhunt : bool
        Whether or not to print each data file name before
        import. This is useful for tracing which data file
        is causing the import to fail.
    config : str
        The name of the configuration to use for the analysis.
        This determines which configuration set from the
        latools.cfg file is used, and overrides the default
        configuration setup. You might sepcify this if your lab
        routinely uses two different instruments.
    dataformat : str or dict
        Either a path to a data format .json file, or a
        dataformat dict. See documentation for more details.

    Attributes
    ----------
    folder : str
        Path to the directory containing the data files, as
        specified by `csv_folder`.
    dirname : str
        The name of the directory containing the data files,
        without the entire path.
    files : array_like
        A list of all files in `folder`.
    param_dir : str
        The directory where parameters are stored.
    report_dir : str
        The directory where plots are saved.
    data : array_like
        A list of `latools.D` data objects.
    data_dict : dict
        A dict of `latools.D` data objects, labelled by sample
        name.
    samples : array_like
        A list of samples.
    analytes : array_like
        A list of analytes measured.
    stds : array_like
        A list of the `latools.D` objects containing hte SRM
        data. These must contain 'STD' in the file name.
    cmaps : dict
        An analyte-specific colour map, used for plotting.

    Methods
    -------
    autorange
    bkgcorrect
    calibrate
    calibration_plot
    crossplot
    despike
    filter_clear
    filter_clustering
    filter_correlation
    filter_distribution
    filter_off
    filter_on
    filter_threshold
    find_expcoef
    get_focus
    getstats
    load_calibration
    load_params
    load_ranges
    ratio
    save_params
    save_ranges
    srm_id
    stat_boostrap
    stat_samples
    trace_plots
    """
    def __init__(self, csv_folder, errorhunt=False, config=None, dataformat=None):
        """
        For processing and analysing whole LA-ICPMS datasets.
        """
        self.folder = csv_folder
        self.dirname = [n for n in self.folder.split('/') if n is not ''][-1]
        self.files = np.array(os.listdir(self.folder))

        # make output directories
        self.param_dir = self.folder + '/params/'
        if not os.path.isdir(self.param_dir):
            os.mkdir(self.param_dir)
        self.report_dir = self.folder + '/reports/'
        if not os.path.isdir(self.report_dir):
            os.mkdir(self.report_dir)

        # load configuration parameters
        # read in config file
        conf = configparser.ConfigParser()
        conf.read(pkg_resources.resource_filename('latools','latools.cfg'))
        # load defaults into dict
        pconf = dict(conf.defaults())
        # if no config is given, check to see what the default setting is
        if (config is None) & (pconf['config'] != 'DEFAULT'):
            config = pconf['config']
        else:
            config = 'DEFAULT'
        # if ther eare any non-default parameters, replace defaults in the pconf dict
        if config != 'DEFAULT':
            for o in conf.options(config):
                pconf[o] = conf.get(config, o)

        # check srmfile exists, and store it in a class attribute.
        if os.path.exists(pconf['srmfile']):
            self.srmfile = pconf['srmfile']
        elif os.path.exists(pkg_resources.resource_filename('latools',pconf['srmfile'])):
            self.srmfile = pkg_resources.resource_filename('latools',pconf['srmfile'])
        else:
            raise ValueError('The SRM file specified in the' + config + ' configuration cannot be found.\nPlease make sure the file exists, and that the path in the config file is correct.\nTo locate the config file, run `latools.config_locator()`.')

        # load in dataformat information.
        # if dataformat is not provided during initialisation, assign it from configuration file
        if dataformat is None:
            dataformat = pconf['dataformat']
        # if it's a string, check the file exists and import it.
        if isinstance(dataformat, str):
            if os.path.exists(dataformat):
                self.dataformat = json.load(open(dataformat))
            elif os.path.exists(pkg_resources.resource_filename('latools',dataformat)):
                self.dataformat = json.load(open(pkg_resources.resource_filename('latools',dataformat)))
            else:
                raise ValueError('The dataformat file specified in the' + config + ' configuration cannot be found.\nPlease make sure the file exists, and that the path in the config file is correct.\nTo locate the config file, run `latools.config_locator()`.')
        # if it's a dict, just assign it straight away.
        elif isinstance(dataformat, dict):
            self.dataformat = dataformat

        # load data (initialise D objects)
        self.data = np.array([D(self.folder + '/' + f, dataformat=self.dataformat, errorhunt=errorhunt) for f in self.files if 'csv' in f])
        self.samples = np.array([s.sample for s in self.data])
        self.analytes = np.array(self.data[0].analytes)

        self.data_dict = {}
        for s, d in zip(self.samples, self.data):
            self.data_dict[s] = d

        self.stds = []
        _ = [self.stds.append(s) for s in self.data if 'STD' in s.sample]
        self.srms_ided = False

        self.cmaps = self.data[0].cmap

        f = open('errors.log', 'a')
        f.write('Errors and warnings during LATOOLS analysis are stored here.\n\n')
        f.close()

        # report
        print('{:.0f} Analysis Files Loaded:'.format(len(self.data)))
        print('{:.0f} standards, {:.0f} samples'.format(len(self.stds),
              len(self.data) - len(self.stds)))
        print('Analytes: ' + ' '.join(self.analytes))

    def autorange(self, analyte='Ca43', gwin=11, win=40, smwin=5,
                  conf=0.01, trans_mult=[0., 0.]):
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
        is taken as a rough threshold to identify signal and background regions.
        Any point where the trace crosses this thrshold is identified as a
        'transition'.

        Step 2: Transition Removal
        The width of the transition regions between signal and background are
        then determined, and the transitions are excluded from analysis. The
        width of the transitions is determined by fitting a gaussian to the
        smoothed first derivative of the analyte trace, and determining its
        width at a point where the gaussian intensity is at at `conf` time the
        gaussian maximum. These gaussians are fit to subsets of the data centered
        around the transitions regions determined in Step 1, +/- `win` data points.
        The peak is further isolated by finding the minima and maxima of a second
        derivative within this window, and the gaussian is fit to the isolated peak.

        Parameters
        ----------
        analyte : str
            The analyte that autorange should consider. For best results,
            choose an analyte that is present homogeneously in high concentrations.
        gwin : int
            The smoothing window used for calculating the first derivative.
            Must be odd.
        win : int
            Determines the width (c +/- win) of the transition data subsets.
        smwin : int
            The smoothing window used for calculating the second derivative.
            Must be odd.
        conf : float
            The proportional intensity of the fitted gaussian tails that
            determines the transition width cutoff (lower = wider transition
            regions excluded).
        trans_mult : array_like, len=2
            Multiples of the peak FWHM to add to the transition cutoffs, e.g.
            if the transitions consistently leave some bad data proceeding the
            transition, set trans_mult to [0, 0.5] to ad 0.5 * the FWHM to the
            right hand side of the limit.

        Adds
        ----
        bkg, sig, trn : bool, array_like
            Boolean arrays the same length as the data, identifying 'background',
            'signal' and 'transition' data regions.
        bkgrng, sigrng, trnrng: array_like
            Pairs of values specifying the edges of the 'background', 'signal'
            and 'transition' data regions in the same units as the Time axis.

        Returns
        -------
        None
        """
        for d in self.data:
            d.autorange(analyte, gwin, win, smwin,
                        conf, trans_mult)

    def find_expcoef(self, nsd_below=12., analyte='Ca43', plot=False, trimlim=None):
        """
        Determines exponential decay coefficient for despike filter.

        Fits an exponential decay function to the washout phase of standards
        to determine the washout time of your laser cell. The exponential coefficient
        reported is `nsd_below` standard deviations below the fitted exponent, to ensure that
        no real data is removed.

        Parameters
        ----------
        nsd_below : float
            The number of standard deviations to subtract from the fitted
            coefficient when calculating the filter exponent.
        analyte : str
            The analyte to consider when determining the coefficient.
            Use high-concentration analyte for best estimates.
        plot : bool or str
            bool: Creates a plot of the fit if True.
            str: Creates a plot, and saves it to the location
                 specified in the str.
        trimlim : float
            A threshold limit used in determining the start of the
            exponential decay region of the washout. Defaults to half
            the increase in signal over background. If the data in
            the plot don't fall on an exponential decay line, change
            this number. Normally you'll need to increase it.

        Returns
        -------
        None
        """

        if isinstance(analyte, str):
            analyte = [analyte]

        def findtrim(tr, lim=None):
            trr = np.roll(tr, -1)
            trr[-1] = 0
            if lim is None:
                lim = 0.5 * np.nanmax(tr - trr)
            ind = (tr - trr) >= lim
            return np.arange(len(ind))[ind ^ np.roll(ind, -1)][0]

        def normalise(a):
            """
            Returns array scaled between 0 and 1.
            """
            return (a - np.nanmin(a)) / np.nanmax(a - np.nanmin(a))

        if not hasattr(self.stds[0], 'trnrng'):
            for s in self.stds:
                s.autorange()

        trans = []
        times = []
        for analyte in analyte:
            for v in self.stds:
                for trnrng in v.trnrng[1::2]:
                    tr = normalise(v.focus[analyte][(v.Time > trnrng[0]) & (v.Time < trnrng[1])])
                    sm = np.apply_along_axis(np.nanmean, 1, v.rolling_window(tr, 3, pad=0))
                    sm[0] = sm[1]
                    trim = findtrim(sm, trimlim) + 2
                    trans.append(normalise(tr[trim:]))
                    times.append(np.arange(tr[trim:].size) * np.diff(v.Time[:2]))

        times = np.concatenate(times)
        trans = np.concatenate(trans)

        ti = []
        tr = []
        for t in np.unique(times):
            ti.append(t)
            tr.append(np.nanmin(trans[times == t]))

        def expfit(x, e):
            """
            Exponential decay function.
            """
            return np.exp(e * x)

        ep, ecov = curve_fit(expfit, ti, tr, p0=(-1.))

        def R2calc(x, y, yp):
            """
            Calculate fit R2.
            """
            SStot = np.sum((y - np.nanmean(y))**2)
            SSfit = np.sum((y - yp)**2)
            return 1 - (SSfit / SStot)

        eeR2 = R2calc(times, trans, expfit(times, ep))

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=[6, 4])

            ax.scatter(times, trans, alpha=0.2, color='k', marker='x')
            ax.scatter(ti, tr, alpha=0.6, color='k', marker='o')
            fitx = np.linspace(0, max(ti))
            ax.plot(fitx, expfit(fitx, ep), color='r', label='Fit')
            ax.plot(fitx, expfit(fitx, ep - nsd_below * np.diag(ecov)**.5,),
                    color='b', label='Used')
            ax.text(0.95, 0.75, 'y = $e^{%.2f \pm %.2f * x}$\n$R^2$= %.2f \nCoefficient: %.2f' % (ep, np.diag(ecov)**.5, eeR2, ep - nsd_below * np.diag(ecov)**.5),
                    transform=ax.transAxes, ha='right', va='top', size=12)
            ax.set_xlim(0, ax.get_xlim()[-1])
            ax.set_xlabel('Time (s)')
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Proportion of Signal')
            plt.legend()
            if isinstance(plot, str):
                fig.savefig(plot)

        self.expdecay_coef = ep - nsd_below * np.diag(ecov)**.5

        print('-------------------------------------')
        print('Exponential Decay Coefficient: {:0.2f}'.format(self.expdecay_coef[0]))
        print('-------------------------------------')

        return

    def despike(self, expdecay_filter=True, exponent=None, tstep=None, spike_filter=True, win=3, nlim=12., exponentplot=False):
        """
        Despikes data with exponential decay and noise filters.

        Parameters
        ----------
        expdecay_filter : bool
            Whether or not to apply the exponential decay filter.
        exponent : None or float
            The exponent for the exponential decay filter. If None,
            it is determined automatically using `find_expocoef`.
        tstep : None or float
            The timeinterval between measurements. If None, it is
            determined automatically from the Time variable.
        spike_filter : bool
            Whether or not to apply the standard deviation spike filter.
        win : int
            The rolling window over which the spike filter calculates
            the trace statistics.
        nlim : float
            The number of standard deviations above the rolling mean
            that data are excluded.
        exponentplot : bool
            Whether or not to show a plot of the automatically determined
            exponential decay exponent.

        Returns
        -------
        None
        """
        if exponent is None:
            if ~hasattr(self, 'expdecay_coef'):
                print('Exponential Decay Coefficient not provided.')
                print('Coefficient will be determined from the washout\ntimes of the standards (takes a while...).')
                self.find_expcoef(plot=exponentplot)
            exponent = self.expdecay_coef
        for d in self.data:
            d.despike(expdecay_filter, exponent, tstep, spike_filter, win, nlim)
        return

    def save_ranges(self):
        """
        Saves signal/background/transition data ranges for each sample.
        """
        if os.path.isfile(self.param_dir + 'bkg.rng'):
            f = input('Range files already exist. Do you want to overwrite them (old files will be lost)? [Y/n]: ')
            if 'n' in f or 'N' in f:
                print('Ranges not saved. Run self.save_ranges() to try again.')
                return
        bkgrngs = []
        sigrngs = []
        for d in self.data:
            bkgrngs.append(d.sample + ':' + str(d.bkgrng.tolist()))
            sigrngs.append(d.sample + ':' + str(d.sigrng.tolist()))
        bkgrngs = '\n'.join(bkgrngs)
        sigrngs = '\n'.join(sigrngs)

        fb = open(self.param_dir + 'bkg.rng', 'w')
        fb.write(bkgrngs)
        fb.close()
        fs = open(self.param_dir + 'sig.rng', 'w')
        fs.write(sigrngs)
        fs.close()
        return

    def load_ranges(self, bkgrngs=None, sigrngs=None):
        """
        Loads signal/background/transition data ranges for each sample.

        Parameters
        ----------
        bkgrngs : str or None
            A array of size (2, n) specifying time intervals that are
            background regions.
        sigrngs : str or None
            A array of size (2, n) specifying time intervals that are
            signal regions.

        Returns
        -------
        None
        """
        if bkgrngs is None:
            bkgrngs = self.param_dir + 'bkg.rng'
        bkgs = open(bkgrngs).readlines()
        samples = []
        bkgrngs = []
        for b in bkgs:
            samples.append(re.match('(.*):{1}(.*)',
                           b.strip()).groups()[0])
            bkgrngs.append(eval(re.match('(.*):{1}(.*)',
                           b.strip()).groups()[1]))
        for s, rngs in zip(samples, bkgrngs):
            self.data_dict[s].bkgrng = np.array(rngs)

        if sigrngs is None:
            sigrngs = self.param_dir + 'sig.rng'
        sigs = open(sigrngs).readlines()
        samples = []
        sigrngs = []
        for s in sigs:
            samples.append(re.match('(.*):{1}(.*)',
                           s.strip()).groups()[0])
            sigrngs.append(eval(re.match('(.*):{1}(.*)',
                           s.strip()).groups()[1]))
        for s, rngs in zip(samples, sigrngs):
            self.data_dict[s].sigrng = np.array(rngs)

        # number the signal regions (used for statistics and standard matching)
        for s in self.data:
            # re-create booleans
            s.makerangebools()

            # make trnrng
            s.trn[[0, -1]] = False
            s.trnrng = s.Time[s.trn ^ np.roll(s.trn, 1)]

            # number traces
            n = 1
            for i in range(len(s.sig)-1):
                if s.sig[i]:
                    s.ns[i] = n
                if s.sig[i] and ~s.sig[i+1]:
                    n += 1
            s.n = int(max(s.ns))  # record number of traces

        return

    # functions for background correction and ratios
    def bkgcorrect(self, mode='constant'):
        """
        Subtracts background from signal.

        Parameters
        ----------
        mode : str or int
            str: 'constant' subtracts the mean of all background
            regions from signal.
            int: fits an nth order polynomial to the background
            data, and subtracts the predicted background values
            from the signal regions. The integer values of `mode`
            specifies the order of the polynomial. Useful if you
            have significant drift in your background.

        Returns
        -------
        None
        """
        for s in self.data:
            s.bkg_correct(mode=mode)
        return

    def ratio(self,  denominator='Ca43', focus='signal'):
        """
        Calculates the ratio of all analytes to a single analyte.

        Parameters
        ----------
        denominator : str
            The name of the analyte to divide all other analytes
            by.
        focus : str
            The `focus` stage of the data used to calculating the
            ratios.

        Returns
        -------
        None
        """
        for s in self.data:
            s.ratio(denominator=denominator, focus=focus)
        return

    # functions for identifying SRMs
    def srm_id(self):
        """
        Asks the user to name the SRMs measured.
        """
        s = self.stds[0]
        fig = s.tplot(scale='log')
        display.clear_output(wait=True)
        display.display(fig)

        n0 = s.n

        def id(self, s):
            stdnms = []
            s.srm_rngs = {}
            for n in np.arange(s.n) + 1:
                fig = s.tplot(scale='log')
                lims = s.Time[s.ns == n][[0, -1]]
                fig.axes[0].axvspan(lims[0], lims[1],
                                    color='r', alpha=0.2, lw=0)
                display.clear_output(wait=True)
                display.display(fig)
                stdnm = input('Name this standard: ')
                stdnms.append(stdnm)
                s.srm_rngs[stdnm] = lims
                plt.close(fig)
            return stdnms

        nms0 = id(self, s)

        if len(self.stds) > 1:
            ans = input('Were all other SRMs measured in the same sequence? [Y/n]')
            if ans.lower() == 'n':
                for s in self.stds[1:]:
                    id(self, s)
            else:
                for s in self.stds[1:]:
                    if s.n == n0:
                        s.srm_rngs = {}
                        for n in np.arange(s.n) + 1:
                            s.srm_rngs[nms0[n-1]] = s.Time[s.ns == n][[0, -1]]
                    else:
                        _ = id(self, s)

        display.clear_output()

        # record srm_rng in self
        self.srm_rng = {}
        for s in self.stds:
            self.srm_rng[s.sample] = s.srm_rngs

        # make boolean identifiers in standard D
        for sn, rs in self.srm_rng.items():
            s = self.data_dict[sn]
            s.std_labels = {}
            for srm, rng in rs.items():
                s.std_labels[srm] = tuples_2_bool(rng, s.Time)

        self.srms_ided = True

        return

    def load_calibration(self, params=None):
        """
        Loads calibration from global .calib file.

        Parameters
        ----------
        params : str
            Specify the parameter filt to load the calibration from.
            If None, it assumes that the parameters are already loaded
            (using `load_params`).

        Returns
        -------
        None
        """
        if isinstance(params, str):
            self.load_params(params)

        # load srm_rng and expand to standards
        self.srm_rng = self.params['calib']['srm_rng']

        # make boolean identifiers in standard D
        for s in self.stds:
            s.srm_rngs = self.srm_rng[s.sample]
            s.std_labels = {}
            for srm, rng in s.srm_rngs.items():
                s.std_labels[srm] = tuples_2_bool(rng, s.Time)
        self.srms_ided = True

        # load calib dict
        self.calib_dict = self.params['calib']['calib_dict']

        return

    # apply calibration to data
    def calibrate(self, poly_n=0, focus='ratios',
                  srmfile=None):
        """
        Calibrates the data to measured SRM values.

        Parameters
        ----------
        poly_n : int
            Specifies the type of function used to map
            known SRM values to SRM measurements.
            0: A linear function, forced through 0.
            1 or more: An nth order polynomial.
        focus : str
            The `focus` stage of the data used to calculating the
            ratios.
        srmfile : str or None
            Path the the file containing the known SRM values.
            If None, the default file specified in the `latools.cfg`
            is used. Refer to the documentation for more information
            on the srmfile format.

        Returns
        -------
        None
        """
        # MAKE CALIBRATION CLEVERER!?
        #   USE ALL DATA, NOT AVERAGES?
        #   IF POLY_N > 0, STILL FORCE THROUGH ZERO IF ALL
        #   STDS ARE WITHIN ERROR OF EACH OTHER (E.G. AL/CA)
        # can store calibration function in self and use *coefs?
        # check for identified srms
        params = locals()
        del(params['self'])
        self.calibration_params = params

        if srmfile is not None:
            self.srmfile = srmfile

        if not self.srms_ided:
            self.srm_id()
        # get SRM values
        f = open(self.srmfile).readlines()
        self.srm_vals = {}
        for srm in self.stds[0].std_labels.keys():
            self.srm_vals[srm] = {}
            for a in self.analytes:
                self.srm_vals[srm][a] = [l.split(',')[1] for l in f if re.match(re.sub("[^A-Za-z]", "", a) + '.*' + srm, l.strip()) is not None][0]

        # make calibration
        self.calib_dict = {}
        self.calib_data = {}
        for a in self.analytes:
            self.calib_data[a] = {}
            self.calib_data[a]['srm'] = []
            self.calib_data[a]['counts'] = []
            x = []
            y = []
            for s in self.stds:
                s.setfocus(focus)
                for srm in s.std_labels.keys():
                    y = s.focus[a][s.std_labels[srm]]
                    y = y[~np.isnan(y)]
                    x = [float(self.srm_vals[srm][a])] * len(y)

                    self.calib_data[a]['counts'].append(y)
                    self.calib_data[a]['srm'].append(x)

            self.calib_data[a]['counts'] = np.concatenate(self.calib_data[a]['counts']).astype(float)
            self.calib_data[a]['srm'] = np.concatenate(self.calib_data[a]['srm']).astype(float)

            if poly_n == 0:
                self.calib_dict[a], _, _, _ = np.linalg.lstsq(self.calib_data[a]['counts'][:, np.newaxis],
                                                              self.calib_data[a]['srm'])
            else:
                self.calib_dict[a] = np.polyfit(self.calib_data[a]['counts'],
                                                self.calib_data[a]['srm'],
                                                poly_n)

        # apply calibration
        for d in self.data:
            d.calibrate(self.calib_dict)

        # save calibration parameters
        # self.save_calibration()
        return

    # data filtering

    def filter_threshold(self, analyte, threshold, filt=False, samples=None):
        """
        Applies a threshold filter to the data.

        Generates two filters above and below the threshold value for a given analyte.

        Parameters
        ----------
        analyte : str
            The analyte that the filter applies to.
        threshold : float
            The threshold value.
        filt : bool
            Whether or not to apply existing filters to the data before
            calculating this filter.
        samples : array_like or None
            Which samples to apply this filter to. If None, applies to all
            samples.

        Returns
        -------
        None
        """
        if samples is None:
            samples = self.samples
        if isinstance(samples, str):
            samples = []

        for s in samples:
            self.data_dict[s].filter_threshold(analyte, threshold, filt=False)

    def filter_distribution(self, analyte, binwidth='scott', filt=False, transform=None,
                            samples=None):
        """
        Applies a distribution filter to the data.

        Parameters
        ----------
        analyte : str
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
        samples : array_like or None
            Which samples to apply this filter to. If None, applies to all
            samples.

        Returns
        -------
        None
        """
        if samples is None:
            samples = self.samples
        if isinstance(samples, str):
            samples = []

        for s in samples:
            self.data_dict[s].filter_distribution(analyte, binwidth='scott', filt=False, transform=None, output=False)

    def filter_clustering(self, analytes, filt=False, normalise=True,
                          method='meanshift', include_time=False, samples=None, **kwargs):
        """
        Applies an n-dimensional clustering filter to the data.

        Parameters
        ----------
        analytes : str
            The analyte(s) that the filter applies to.
        filt : bool
            Whether or not to apply existing filters to the data before
            calculating this filter.
        normalise : bool
            Whether or not to normalise the data to zero mean and unit variance.
            Reccomended if clustering based on more than 1 analyte.
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
                          within the data based on the 'connectivity' of the data
                          (i.e. how far apart each data point is in a
                          multi-dimensional parameter space). Requires you to set
                          `eps`, the minimum distance point must be from another
                          point to be considered in the same cluster, and
                          `min_samples`, the minimum number of points that must be
                          within the minimum distance for it to be considered a
                          cluster. It may also be run in automatic mode by specifying
                          `n_clusters` alongside `min_samples`, where eps is
                          decreased until the desired number of clusters is obtained.
                For more information on these algorithms, refer to the documentation.
        include_time : bool
            Whether or not to include the Time variable in the clustering analysis.
            Useful if you're looking for spatially continuous clusters in your data,
            i.e. this will identify each spot in your analysis as an individual
            cluster.
        samples : optional, array_like or None
            Which samples to apply this filter to. If None, applies to all
            samples.
        **kwargs
            Parameters passed to the clustering algorithm specified by `method`.

        Meanshift Parameters
        --------------------
        bandwidth : str or float
            The bandwith (float) or bandwidth method ('scott' or 'silverman')
            used to estimate the data bandwidth.
        bin_seeding : bool
            Modifies the behaviour of the meanshift algorithm. Refer to
            sklearn.cluster.meanshift documentation.

        K-Means Parameters
        ------------------
        n_clusters : int
            The number of clusters expected in the data.

        DBSCAN Parameters
        -----------------
        eps : float
            The minimum 'distance' points must be apart for them to be in the
            same cluster. Defaults to 0.3. Note: If the data are normalised
            (they should be for DBSCAN) this is in terms of total sample variance.
            Normalised data have a mean of 0 and a variance of 1.
        min_samples : int
            The minimum number of samples within distance `eps` required
            to be considered as an independent cluster.
        n_clusters : int
            The number of clusters expected. If specified, `eps` will be
            incrementally reduced until the expected number of clusters is found.
        maxiter : int
            The maximum number of iterations DBSCAN will run.

        Returns
        -------
        None
        """
        if samples is None:
            samples = self.samples
        if isinstance(samples, str):
            samples = []

        for s in samples:
            self.data_dict[s].filter_clustering(analytes, filt=filt, normalise=normalise, method=method,
                                                include_time=include_time, **kwargs)

    def filter_correlation(self, x_analyte, y_analyte, window=None, r_threshold=0.9,
                           p_threshold=0.05, filt=True):
        """
        Applies a correlation filter to the data.

        Calculates a rolling correlation between every `window` points of
        two analytes, and excludes data where their Pearson's R value is
        above `r_threshold` and statistically significant.

        Data will be excluded where their absolute R value is greater than
        `r_threshold` AND the p-value associated with the correlation is
        less than `p_threshold`. i.e. only correlations that are statistically
        significant are considered.

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
        if samples is None:
            samples = self.samples
        if isinstance(samples, str):
            samples = []

        for s in samples:
            self.data_dict[s].filter_correlation(x_analyte, y_analyte, window=None, r_threshold=0.9, p_threshold=0.05, filt=True)

    def filter_on(self, filt=None, analyte=None, samples=None):
        """
        Turns data filters on for particular analytes and samples.

        Parameters
        ----------
        filt : optional, str or array_like
            Name, partial name or list of names of filters. Supports
            partial matching. i.e. if 'cluster' is specified, all
            filters with 'cluster' in the name are activated.
            Defaults to all filters.
        analyte : optional, str or array_like
            Name or list of names of analytes. Defaults to all analytes.
        samples : optional, array_like or None
            Which samples to apply this filter to. If None, applies to all
            samples.

        Returns
        -------
        None
        """
        if samples is None:
            samples = self.samples
        if isinstance(samples, str):
            samples = [samples]

        for s in samples:
            self.data_dict[s].filt.on(analyte, filt)

    def filter_off(self, filt=None, analyte=None, samples=None):
        """
        Turns data filters off for particular analytes and samples.

        Parameters
        ----------
        filt : optional, str or array_like
            Name, partial name or list of names of filters. Supports
            partial matching. i.e. if 'cluster' is specified, all
            filters with 'cluster' in the name are activated.
            Defaults to all filters.
        analyte : optional, str or array_like
            Name or list of names of analytes. Defaults to all analytes.
        samples : optional, array_like or None
            Which samples to apply this filter to. If None, applies to all
            samples.

        Returns
        -------
        None
        """
        if samples is None:
            samples = self.samples
        if isinstance(samples, str):
            samples = [samples]

        for s in samples:
            self.data_dict[s].filt.off(analyte, filt)

    def filter_clear(self):
        """
        Clears (deletes) all data filters.
        """
        for d in self.data:
            d.filt.clear()

    # def filter_status(self, sample=None):
    #     if sample is not None:
    #         print(self.data_dict[sample].filt)
    #     else:

    # plot calibrations
    def calibration_plot(self, analytes=None, plot='errbar'):
        """
        Plot the calibration lines between measured and known SRM values.

        Parameters
        ----------
        analytes : optional, array_like or str
            The analyte(s) to plot. Defaults to all analytes.
        plot : str
            Type of plot to produce.
                'errbar': plots the mean and standard deviation
                          of SRM measurements.
                'scatter': plots all SRM data as individual points.

        Returns
        -------
        (fig, axes)
            matplotlib objects
        """
        if analytes is None:
            analytes = [a for a in self.analytes if 'Ca' not in a]

        def rangecalc(xs, ys, pad=0.05):
            xd = max(xs)
            yd = max(ys)
            return ([0 - pad * xd, max(xs) + pad * xd],
                    [0 - pad * yd, max(ys) + pad * yd])

        n = len(analytes)
        if n % 4 is 0:
            nrow = n/4
        else:
            nrow = n//4 + 1

        fig, axes = plt.subplots(int(nrow), 4, figsize=[12, 3 * nrow], tight_layout=True)

        for ax, a in zip(axes.flat, analytes):
            if plot is 'errbar':
                srms = []
                means = []
                stds = []
                for s in np.unique(self.calib_data[a]['srm']):
                    srms.append(s)
                    means.append(np.nanmean(self.calib_data[a]
                                 ['counts'][self.calib_data[a]
                                 ['srm'] == s]))
                    stds.append(np.nanstd(self.calib_data[a]
                                ['counts'][self.calib_data[a]
                                ['srm'] == s]))
                ax.errorbar(means, srms, xerr=stds, lw=0, elinewidth=2,
                            ecolor=self.cmaps[a])
            if plot is 'scatter':
                ax.scatter(self.calib_data[a]['counts'],
                           self.calib_data[a]['srm'],
                           color=self.cmaps[a], alpha=0.2)
            xlim, ylim = rangecalc(self.calib_data[a]['counts'],
                                   self.calib_data[a]['srm'])
            xlim[0] = ylim[0] = 0
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # calculate line
            x = np.array(xlim)
            coefs = self.calib_dict[a]
            if len(coefs) == 1:
                line = x * coefs[0]
                label = 'y = {:0.3e}x'.format(coefs[0])
            else:
                line = x * coefs[0] + coefs[1]
                label = 'y = {:0.3e}x + {:0.3e}'.format(coefs[0], coefs[1])
            ax.plot(x, line, color=(0, 0, 0, 0.5), ls='dashed')
            ax.text(.05, .95, a, transform=ax.transAxes,
                    weight='bold', va='top', ha='left', size=12)
            ax.set_xlabel('counts/counts Ca')
            ax.set_ylabel('mol/mol Ca')

            # write calibration equation on graph
            ax.text(0.98, 0.04, label, transform=ax.transAxes,
                    va='bottom', ha='right')

        for ax in axes.flat[n:]:
            fig.delaxes(ax)

        return fig, axes

    # fetch all the data from the data objects
    def get_focus(self, filt=False):
        """
        Collect all data from all samples into a single array.

        Parameters
        ----------
        filt : str, dict or bool
            Either logical filter expression contained in a str,
            a dict of expressions specifying the filter string to
            use for each analyte or a boolean. Passed to `grab_filt`.

        Returns
        -------
        None
        """
        t = 0
        self.focus = {'Time': []}
        for a in self.analytes:
            self.focus[a] = []

        for s in self.data:
            if 'STD' not in s.sample:
                self.focus['Time'].append(s.Time + t)
                t += max(s.Time)
                ind = s.filt.grab_filt(filt)
                for a in self.analytes:
                    tmp = s.focus[a].copy()
                    tmp[ind] = np.nan
                    self.focus[a].append(tmp)

        for k, v in self.focus.items():
            self.focus[k] = np.concatenate(v)

    # crossplot of all data
    def crossplot(self, analytes=None, lognorm=True,
                  bins=25, filt=False, **kwargs):
        """
        Plot analytes against each other as 2D histograms.

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
        None
        """
        if analytes is None:
            analytes = [a for a in self.analytes if 'Ca' not in a]
        if not hasattr(self, 'focus'):
            self.get_focus(filt)

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

        cmlist = ['Blues', 'BuGn', 'BuPu', 'GnBu',
                  'Greens', 'Greys', 'Oranges', 'OrRd',
                  'PuBu', 'PuBuGn', 'PuRd', 'Purples',
                  'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']
        udict = {}
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i, j), (j, i)]:
                # set unit multipliers
                mx, ux = unitpicker(np.nanmean(self.focus[analytes[x]]))
                my, uy = unitpicker(np.nanmean(self.focus[analytes[y]]))
                udict[analytes[x]] = (x, ux)

                # make plot
                px = self.focus[analytes[x]][~np.isnan(self.focus[analytes[x]])] * mx
                py = self.focus[analytes[y]][~np.isnan(self.focus[analytes[y]])] * my
                if lognorm:
                    axes[x, y].hist2d(py, px, bins,
                                      norm=mpl.colors.LogNorm(),
                                      cmap=plt.get_cmap(cmlist[x]))
                else:
                    axes[x, y].hist2d(py, px, bins,
                                      cmap=plt.get_cmap(cmlist[x]))
                axes[x, y].set_ylim([px.min(), px.max()])
                axes[x, y].set_xlim([py.min(), py.max()])
        # diagonal labels
        for a, (i, u) in udict.items():
            axes[i, i].annotate(a+'\n'+u, (0.5, 0.5),
                                xycoords='axes fraction',
                                ha='center', va='center')
        # switch on alternating axes
        for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
            axes[j, i].xaxis.set_visible(True)
            for label in axes[j, i].get_xticklabels():
                label.set_rotation(90)
            axes[i, j].yaxis.set_visible(True)

        return fig, axes

    # Plot traces
    def trace_plots(self, analytes=None, outdir=None, ranges=False, focus=None, filt=False,
                    scale='log', figsize=[10, 4], stats=True, stat='nanmean', err='nanstd',):
        """
        Plot analytes as a function of time.

        Parameters
        ----------
        analytes : optional, array_like or str
            The analyte(s) to plot. Defaults to all analytes.
        outdir : TYPE
            Description of `outdir`.
        ranges : bool
            Description of `ranges`.
        focus : str
            The focus 'stage' of the analysis to plot. Can be
            'rawdata', 'despiked':, 'signal', 'background',
            'bkgsub', 'ratios' or 'calibrated'.
        filt : str, dict or bool
            Either logical filter expression contained in a str,
            a dict of expressions specifying the filter string to
            use for each analyte or a boolean. Passed to `grab_filt`.
        scale : str
            If 'log', plots the data on a log scale.
        figsize : array_like
            Array of length 2 specifying figure [width, height] in
            inches.
        stats : bool
            Whether or not to overlay the mean and standard deviations
            for each trace.
        stat, err: str
            The names of the statistic and error components to plot.
            Deafaults to 'nanmean' and 'nanstd'.


        Returns
        -------
        None
        """
        if outdir is None:
            outdir = self.report_dir
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        for s in self.data:
            if focus is None:
                focus = stg = s.focus_stage
            else:
                stg = s.focus_stage
            s.setfocus(focus)
            fig = s.tplot(analytes=analytes, figsize=figsize, scale=scale, filt=filt,
                          ranges=ranges, stats=stats, stat=stat, err=err)
            # ax = fig.axes[0]
            # for l, u in s.sigrng:
            #     ax.axvspan(l, u, color='r', alpha=0.1)
            # for l, u in s.bkgrng:
            #     ax.axvspan(l, u, color='k', alpha=0.1)
            fig.savefig(outdir + '/' + s.sample + '_traces.pdf')
            plt.close(fig)
            s.setfocus(stg)


    def stat_boostrap(self, analytes=None, filt=True,
                      stat_fn=np.nanmean, ci=95):
        """
        Calculate sample statistics with bootstrapped confidence intervals.

        Parameters
        ----------
        analytes : optional, array_like or str
            The analyte(s) to calculate statistics for. Defaults to
            all analytes.
        filt : str, dict or bool
            Either logical filter expression contained in a str,
            a dict of expressions specifying the filter string to
            use for each analyte or a boolean. Passed to `grab_filt`.
        stat_fns : array_like
            list of functions that take a single array_like input,
            and return a single statistic. Function should be able
            to cope with numpy NaN values.
        ci : float
            Confidence interval to calculate.

        Returns
        -------
        None
        """

        return

    def stat_samples(self, analytes=None, filt=True,
                     stat_fns=[np.nanmean, np.nanstd],
                     eachtrace=True):
        """
        Calculate sample statistics.

        Returns samples, analytes, and arrays of statistics
        of shape (samples, analytes). Statistics are calculated
        from the 'focus' data variable, so output depends on how
        the data have been processed.

        Parameters
        ----------
        analytes : optional, array_like or str
            The analyte(s) to calculate statistics for. Defaults to
            all analytes.
        filt : str, dict or bool
            Either logical filter expression contained in a str,
            a dict of expressions specifying the filter string to
            use for each analyte or a boolean. Passed to `grab_filt`.
        stat_fns : array_like
            list of functions that take a single array_like input,
            and return a single statistic. Function should be able
            to cope with numpy NaN values.
        eachtrace : bool
            Whether to calculate the statistics for each analysis
            spot individually, or to produce per-sample means.
            Default is True.

        Returns
        -------
        None
            Adds dict to analyse object containing samples, analytes and
            functions and data.
        """
        if analytes is None:
            analytes = self.analytes
        self.stats = {}
        self.stats_calced = [f.__name__ for f in stat_fns]

        # calculate stats for each sample
        for s in self.data:
            if 'STD' not in s.sample:
                s.sample_stats(analytes, filt=filt, stat_fns=stat_fns,
                               eachtrace=eachtrace)

                self.stats[s.sample] = s.stats

        # for f in stat_fns:
        #     setattr(self, f.__name__, [])
        #     for s in self.data:
        #         setattr(s, f.__name__, [])
        #         if analytes is None:
        #             analytes = self.analytes
        #         for a in analytes:
        #             if filt and hasattr(s, filts):
        #                 if a in s.filts.keys():
        #                     ind = s.filts[a]
        #             else:
        #                 ind = np.array([True] * s.focus[a].size)

        #             getattr(s, f.__name__).append(f(s.focus[a][ind]))
        #         getattr(self, f.__name__).append(getattr(s, f.__name__))
        #     setattr(self, f.__name__, np.array(getattr(self, f.__name__)))
        # return (np.array([f.__name__ for f in stat_fns]), np.array(self.samples), np.array(analytes)), np.array([getattr(self, f.__name__) for f in stat_fns])

    def getstats(self):
        """
        Return pandas dataframe of all sample statistics.
        """
        slst = []

        for s in self.stats_calced:
            for nm in [n for n in self.samples if 'STD' not in n.upper()]:
                # make multi-index
                reps = np.arange(self.stats[nm][s].shape[1])
                ss = np.array([s] * reps.size)
                nms = np.array([nm] * reps.size)
                # make sub-dataframe
                stdf = pd.DataFrame(self.stats[nm][s].T,
                                    columns=self.stats[nm]['analytes'],
                                    index=[ss, nms, reps])
                stdf.index.set_names(['statistic', 'sample', 'rep'], inplace=True)
                slst.append(stdf)

        return pd.concat(slst)

    # parameter input/output
    def save_params(self, output_file=None):
        """
        Save analysis parameters.

        Parameters
        ----------
        output_file : str
            Where to save the output file. Defaults to
            './params/YYMMDD-HHMM.param'.

        Returns
        -------
        None
        """
        # get all parameters from all samples as a dict
        dparams = {}
        plist = []
        for d in self.data:
            dparams[d.sample] = d.get_params()
            plist.append(list(dparams[d.sample].keys()))
        # get all parameter keys
        plist = np.unique(plist)
        plist = plist[plist != 'sample']

        # convert dict into array
        params = []
        for s in self.samples:
            row = []
            for p in plist:
                row.append(dparams[s][p])
            params.append(row)
        params = np.array(params)

        # calculate parameter 'sets'
        sets = np.zeros(params.shape)
        for c in np.arange(plist.size):
            col = params[:,c]
            i = 0
            for r in np.arange(1, col.size):
                if isinstance(col[r], (str, float, dict, int)):
                    if col[r] != col[r-1]:
                        i += 1
                else:
                    if any(col[r] != col[r-1]):
                        i += 1

                sets[r,c] = i

        ssets = np.apply_along_axis(sum,1,sets)
        nsets = np.unique(ssets, return_counts=True)
        setorder = np.argsort(nsets[1])[::-1]

        out = {}
        out['exceptions'] = {}
        first = True
        for so in setorder:
            setn = nsets[0][so]
            setn_samples = self.samples[ssets == setn]
            if first:
                out['general'] = dparams[setn_samples[0]]
                del out['general']['sample']
                general_key = sets[self.samples == setn_samples[0],:][0,:]
                first = False
            else:
                setn_key = sets[self.samples == setn_samples[0],:][0,:]
                exception_param = plist[general_key != setn_key]
                for s in setn_samples:
                    out['exceptions'][s] = {}
                    for ep in exception_param:
                        out['exceptions'][s][ep] = dparams[s][ep]

        out['calib'] = {}
        out['calib']['calib_dict'] = self.calib_dict
        out['calib']['srm_rng'] = self.srm_rng
        out['calib']['calibration_params'] = self.calibration_params

        self.params = out

        if output_file is None:
            outputfile = './params/' + time.strftime('%y%m%d-%H%M') + '.param'
        f = open(output_file, 'w')
        f.write(str(self.params))
        f.close()

        return

    def load_params(self, params):
        """
        Load analysis parameters.

        Parameters
        ----------
        params : str or dict
            str: path to latools .param file.
            dict: param dict in .param format
                  (refer to manual)

        Returns
        -------
        None
        """
        if isinstance(params, str):
            s = open(params, 'r').read()
            # make it numpy-friendly for eval
            s = re.sub('array', 'np.array', s)
            params = eval(s)
        self.params = params
        return


analyze = analyse  # for the yanks

class D(object):
    """
    Container for data from a single laser ablation analysis.

    Parameters
    ----------
    csv_folder : str
        The path to a directory containing multiple data files.
    errorhunt : bool
        Whether or not to print each data file name before
        import. This is useful for tracing which data file
        is causing the import to fail.
    dataformat : str or dict
        Either a path to a data format .json file, or a
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
    autorange
    bkg_correct
    bkgrange
    calibrate
    cluster_DBSCAN
    cluster_kmeans
    cluster_meanshift
    crossplot
    despike
    expdecay_filter
    fastgrad
    filt_report
    filter_clustering
    filter_correlation
    filter_distribution
    filter_threshold
    findlower
    findmins
    findupper
    gauss
    gauss_inv
    get_params
    makerangebools
    mkrngs
    ratio
    rolling_window
    sample_stats
    separate
    setfocus
    sigrange
    spike_filter
    tplot
    """
    def __init__(self, csv_file, dataformat=None, errorhunt=False):
        if errorhunt:
            print(csv_file)  # errorhunt prints each csv file name before it tries to load it, so you can tell which file is failing to load.
        self.file = csv_file
        self.sample = os.path.basename(self.file).split('.')[0]

        with open(csv_file) as f:
            lines = f.readlines()
            # read the metadata, using key, regex pairs in the line-numbered
            # dataformat['regex'] dict.
            if 'regex' in dataformat.keys():
                self.meta = {}
                for k, v in dataformat['regex'].items():
                    if v is not None:
                        out = re.search(v[-1], lines[int(k)]).groups()
                        for i in np.arange(len(v[0])):
                            self.meta[v[0][i]] = out[i]
            # identify column names
            if dataformat['column_id']['name_row'] is not None:
                columns = np.array(lines[dataformat['column_id']['name_row']].strip().split(','))
                timecol = np.array([dataformat['column_id']['timekey'] in c.lower() for c in columns])
                columns[timecol] = 'Time'
                self.analytes = columns[~timecol]

        read_data = np.genfromtxt(csv_file, **dataformat['genfromtext_args']).T

        # create data dict
        self.data = {}
        self.data['rawdata'] = dict(zip(columns,read_data))

        # set focus to rawdata
        self.setfocus('rawdata')

        # make a colourmap for plotting
        self.cmap = dict(zip(self.analytes,
                             cb.get_map('Paired', 'qualitative',
                                        len(columns)).hex_colors))

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

        # set up corrections dict
        # self.corrections = {}

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
                'bkgsub': background subtracted data, created by self.bkg_correct
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

    # despiking functions
    def rolling_window(self, a, window, pad=None):
        """
        Returns (win, len(a)) rolling-window array of data.

        Parameters
        ----------
        a : array_like
            Array to calculate the rolling window of
        window : int
            Description of `window`.
        pad : same as dtype(a)
            Description of `pad`.

        Returns
        -------
        array_like
            An array of shape (n, window), where n is either len(a) - window
            if pad is None, or len(a) if pad is not None.
        """
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        out = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        if pad is not None:
            blankpad = np.empty((window//2, window, ))
            blankpad[:] = pad
            return np.concatenate([blankpad, out, blankpad])
        else:
            return out

    def expdecay_filter(self, exponent=None, tstep=None):
        """
        Apply exponential decay filter to remove unrealistically low values.

        Parameters
        ----------
        exponent : float
            Expinent used in filter
        tstep : float
            The time increment between data points.
            Calculated from Time variable if None.

        Returns
        -------
        None
        """
        # if exponent is None:
        #     if ~hasattr(self, 'expdecay_coef'):
        #         self.find_expcoef()
        #     exponent = self.expdecay_coef
        if tstep is None:
            tstep = np.diff(self.Time[:2])
        if ~hasattr(self, 'despiked'):
            self.data['despiked'] = {}
        for a, vo in self.focus.items():
            v = vo.copy()
            if 'time' not in a.lower():
                lowlim = np.roll(v * np.exp(tstep * exponent), 1)
                over = np.roll(lowlim > v, -1)

                if sum(over) > 0:
                    # get adjacent values to over-limit values
                    neighbours = np.hstack([v[np.roll(over, -1)][:, np.newaxis],
                                            v[np.roll(over, 1)][:, np.newaxis]])
                    # calculate the mean of the neighbours
                    replacements = np.apply_along_axis(np.nanmean, 1, neighbours)
                    # and subsitite them in
                    v[over] = replacements
                self.data['despiked'][a] = v
        self.setfocus('despiked')
        return

    # spike filter
    def spike_filter(self, win=3, nlim=12.):
        """
        Apply standard deviation filter to remove anomalous high values.

        Parameters
        ----------
        win : int
            The window used to calculate rolling statistics.
        nlim : float
            The number of standard deviations above the rolling
            mean above which data are considered outliers.

        Returns
        -------
        None
        """
        if ~isinstance(win, int):
            win = int(win)
        if ~hasattr(self, 'despiked'):
            self.data['despiked'] = {}
        for a, vo in self.focus.items():
            v = vo.copy()
            if 'time' not in a.lower():
                # calculate rolling mean
                with warnings.catch_warnings():  # to catch 'empty slice' warnings
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    rmean = np.apply_along_axis(np.nanmean, 1, self.rolling_window(v, win, pad=np.nan))
                    rmean = np.apply_along_axis(np.nanmean, 1, self.rolling_window(v, win, pad=np.nan))
                # calculate rolling standard deviation (count statistics, so **0.5)
                rstd = rmean**0.5

                # find which values are over the threshold (v > rmean + nlim * rstd)
                over = v > rmean + nlim * rstd
                if sum(over) > 0:
                    # get adjacent values to over-limit values
                    neighbours = np.hstack([v[np.roll(over, -1)][:, np.newaxis],
                                            v[np.roll(over, 1)][:, np.newaxis]])
                    # calculate the mean of the neighbours
                    replacements = np.apply_along_axis(np.nanmean, 1, neighbours)
                    # and subsitite them in
                    v[over] = replacements
                self.data['despiked'][a] = v
        self.setfocus('despiked')
        return

    def despike(self, expdecay_filter=True, exponent=None, tstep=None, spike_filter=True, win=3, nlim=12.):
        """
        Applies expdecay_filter and spike_filter to data.

        Parameters
        ----------
        expdecay_filter : bool
            Whether or not to apply the exponential decay filter.
        exponent : None or float
            The exponent for the exponential decay filter. If None,
            it is determined automatically using `find_expocoef`.
        tstep : None or float
            The timeinterval between measurements. If None, it is
            determined automatically from the Time variable.
        spike_filter : bool
            Whether or not to apply the standard deviation spike filter.
        win : int
            The rolling window over which the spike filter calculates
            the trace statistics.
        nlim : float
            The number of standard deviations above the rolling mean
            that data are excluded.

        Returns
        -------
        None
        """
        if spike_filter:
            self.spike_filter(win, nlim)
        if expdecay_filter:
            self.expdecay_filter(exponent, tstep)

        params = locals()
        del(params['self'])
        self.despike_params = params
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

    def gauss(self, x, *p):
        """ Gaussian function.

        Parameters
        ----------
        x : array_like
            Independent variable.
        *p : parameters unpacked to A, mu, sigma
            A: area
            mu: centre
            sigma: width

        Return
        ------
        array_like
            gaussian descriped by *p.
        """
        A, mu, sigma = p
        return A * np.exp(-0.5*(-mu + x)**2/sigma**2)

    def gauss_inv(self, y, *p):
        """
        Inverse Gaussian function.

        For determining the x coordinates
        for a given y intensity (i.e. width at a given height).

        Parameters
        ----------
        y : float
            The height at which to calculate peak width.
        *p : parameters unpacked to mu, sigma
            mu: peak center
            sigma: peak width

        Return
        ------
        array_like
            x positions either side of mu where gauss(x) == y.
        """
        mu, sigma = p
        return np.array([mu - 1.4142135623731 * np.sqrt(sigma**2*np.log(1/y)),
                         mu + 1.4142135623731 * np.sqrt(sigma**2*np.log(1/y))])

    def findlower(self, x, y, c, win=3):
        """
        Returns the first local minima in y below c.

        Finds the first local minima below a specified point. Used for
        defining the lower limit of the data window used for transition
        fitting.

        Parameters
        ----------
        x, y : array_like
            1D Arrays of independent and dependent variables.
        c : float
            The threshold below which the first minimum should
            be returned.
        win : int
            The window used to calculate rolling statistics.

        Returns
        -------
        float
            x position of minima

        """
        yd = self.fastgrad(y[::-1], win)
        mins = self.findmins(x[::-1], yd)
        clos = abs(mins - c)
        return mins[clos == min(clos)] - min(clos)

    def findupper(self, x, y, c, win=3):
        """
        Returns the first local minima in y above c.

        Finds the first local minima above a specified point. Used for
        defining the lower limit of the data window used for transition
        fitting.

        Parameters
        ----------
        x, y : array_like
            1D Arrays of independent and dependent variables.
        c : float
            The threshold above which the first minimum should
            be returned.
        win : int
            The window used to calculate rolling statistics.

        Returns
        -------
        float
            x position of minima
        """
        yd = self.fastgrad(y, win)
        mins = self.findmins(x, yd)
        clos = abs(mins - c)
        return mins[clos == min(abs(clos))] + min(clos)

    def fastgrad(self, a, win=11):
        """
        Returns rolling-window gradient of a.

        Function to efficiently calculate the rolling gradient of a numpy
        array using 'stride_tricks' to split up a 1D array into an ndarray of
        sub-sections of the original array, of dimensions [len(a)-win, win].

        Parameters
        ----------
        a : array_like
            The 1D array to calculate the rolling gradient of.
        win : int
            The width of the rolling window.

        Returns
        -------
        array_like
            Gradient of a, assuming as constant integer x-scale.
        """
        # check to see if 'window' is odd (even does not work)
        if win % 2 == 0:
            win -= 1  # subtract 1 from window if it is even.
        # trick for efficient 'rolling' computation in numpy
        # shape = a.shape[:-1] + (a.shape[-1] - win + 1, win)
        # strides = a.strides + (a.strides[-1],)
        # wins = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        wins = self.rolling_window(a, win)
        # apply rolling gradient to data
        a = map(lambda x: np.polyfit(np.arange(win), x, 1)[0], wins)

        return np.concatenate([np.zeros(int(win/2)), list(a),
                              np.zeros(int(win / 2))])

    def autorange(self, analyte='Ca43', gwin=11, win=40, smwin=5,conf=0.01, trans_mult=[0., 0.]):
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
        is taken as a rough threshold to identify signal and background regions.
        Any point where the trace crosses this thrshold is identified as a
        'transition'.

        Step 2: Transition Removal
        The width of the transition regions between signal and background are
        then determined, and the transitions are excluded from analysis. The
        width of the transitions is determined by fitting a gaussian to the
        smoothed first derivative of the analyte trace, and determining its
        width at a point where the gaussian intensity is at at `conf` time the
        gaussian maximum. These gaussians are fit to subsets of the data centered
        around the transitions regions determined in Step 1, +/- `win` data points.
        The peak is further isolated by finding the minima and maxima of a second
        derivative within this window, and the gaussian is fit to the isolated peak.

        Parameters
        ----------
        analyte : str
            The analyte that autorange should consider. For best results,
            choose an analyte that is present homogeneously in high concentrations.
        gwin : int
            The smoothing window used for calculating the first derivative.
            Must be odd.
        win : int
            Determines the width (c +/- win) of the transition data subsets.
        smwin : int
            The smoothing window used for calculating the second derivative.
            Must be odd.
        conf : float
            The proportional intensity of the fitted gaussian tails that
            determines the transition width cutoff (lower = wider transition
            regions excluded).
        trans_mult : array_like, len=2
            Multiples of the peak FWHM to add to the transition cutoffs, e.g.
            if the transitions consistently leave some bad data proceeding the
            transition, set trans_mult to [0, 0.5] to ad 0.5 * the FWHM to the
            right hand side of the limit.


        Adds
        ----
        bkg, sig, trn : bool, array_like
            Boolean arrays the same length as the data, identifying 'background',
            'signal' and 'transition' data regions.
        bkgrng, sigrng, trnrng: array_like
            Pairs of values specifying the edges of the 'background', 'signal'
            and 'transition' data regions in the same units as the Time axis.

        Returns
        -------
        None
        """
        params = locals()
        del(params['self'])
        self.autorange_params = params

        bins = 50  # determine automatically? As a function of bkg rms noise?
        # bkg = np.array([True] * self.Time.size)  # initialise background array

        v = self.focus[analyte]  # get trace data
        vl = np.log10(v[v > 1])  # remove zeros from value
        x = np.linspace(vl.min(), vl.max(), bins)  # define bin limits

        n, _ = np.histogram(vl, x)  # make histogram of sample
        kde = gaussian_kde(vl)
        yd = kde.pdf(x)  # calculate gaussian_kde of sample

        mins = self.findmins(x, yd)  # find minima in kde

        bkg = v < 1.2 * 10**mins[0]  # set background as lowest distribution

        # assign rough background and signal regions based on kde minima
        self.bkg = bkg
        self.sig = ~bkg

        # remove transitions by fitting a gaussian to the gradients of
        # each transition
        # 1. calculate the absolute gradient of the target trace.
        g = abs(self.fastgrad(v, gwin))
        # 2. determine the approximate index of each transition
        zeros = np.arange(len(self.bkg))[self.bkg ^ np.roll(self.bkg, 1)] - 1
        tran = []  # initialise empty list for transition pairs
        for z in zeros:  # for each approximate transition
            # isolate the data around the transition
            if z - win > 0:
                xs = self.Time[z-win:z+win]
                ys = g[z-win:z+win]
            else:
                xs = self.Time[:z+win]
                ys = g[:z+win]
            # determine location of maximum gradient
            c = xs[ys == np.nanmax(ys)]
            try:  # in case some of them don't work...
                # locate the limits of the main peak (find turning point either side of
                # peak centre using a second derivative)
                lower = self.findlower(xs, ys, c, smwin)
                upper = self.findupper(xs, ys, c, smwin)
                # isolate transition peak for fit
                x = self.Time[(self.Time >= lower) & (self.Time <= upper)]
                y = g[(self.Time >= lower) & (self.Time <= upper)]
                # fit a gaussian to the transition gradient
                pg, _ = curve_fit(self.gauss, x, y, p0=(np.nanmax(y),
                                                        x[y == np.nanmax(y)],
                                                        (upper - lower) / 2))
                # get the x positions when the fitted gaussian is at 'conf' of
                # maximum
                tran.append(self.gauss_inv(conf, *pg[1:]) +
                            pg[-1] * np.array(trans_mult))
            except:
                try:
                    # fit a gaussian to the transition gradient
                    pg, _ = curve_fit(self.gauss, x, y, p0=(np.nanmax(y),
                                                            x[y == np.nanmax(y)],
                                                            (upper - lower) / 2))
                    # get the x positions when the fitted gaussian is at 'conf' of
                    # maximum
                    tran.append(self.gauss_inv(conf, *pg[1:]) +
                                pg[-1] * np.array(trans_mult))
                except:
                    pass
        # remove the transition regions from the signal and background ids.
        for t in tran:
            self.bkg[(self.Time > t[0]) & (self.Time < t[1])] = False
            self.sig[(self.Time > t[0]) & (self.Time < t[1])] = False

        self.trn = ~self.bkg & ~self.sig

        self.mkrngs()

        # final check to catch missed transitions
        # calculate average transition width
        tr = self.Time[self.trn ^ np.roll(self.trn, 1)]
        tr = np.reshape(tr, [tr.size//2, 2])
        self.trnrng = tr
        trw = np.mean(np.diff(tr, axis=1))

        corr = False
        for b in self.bkgrng.flat:
            if (self.sigrng - b < 0.3 * trw).any():
                self.bkg[(self.Time >= b - trw/2) & (self.Time <= b + trw/2)] = False
                self.sig[(self.Time >= b - trw/2) & (self.Time <= b + trw/2)] = False
                corr = True

        if corr:
            self.mkrngs()

        # number the signal regions (used for statistics and standard matching)
        n = 1
        for i in range(len(self.sig)-1):
            if self.sig[i]:
                self.ns[i] = n
            if self.sig[i] and ~self.sig[i+1]:
                n += 1
        self.n = int(max(self.ns))  # record number of traces

        return

    def mkrngs(self):
        """
        Transform boolean arrays into list of limit pairs.

        Gets Time limits of signal/background boolean arrays and stores them as
        sigrng and bkgrng arrays. These arrays can be saved by 'save_ranges' in
        the analyse object.
        """
        self.bkg[[0,-1]] = False
        bkgr = self.Time[self.bkg ^ np.roll(self.bkg, -1)]
        self.bkgrng = np.reshape(bkgr, [bkgr.size//2, 2])

        self.sig[[0, -1]] = False
        sigr = self.Time[self.sig ^ np.roll(self.sig, 1)]
        self.sigrng = np.reshape(sigr, [sigr.size//2, 2])

        self.trn[[0, -1]] = False
        trnr = self.Time[self.trn ^ np.roll(self.trn, 1)]
        self.trnrng = np.reshape(trnr, [trnr.size//2, 2])

    def bkgrange(self, rng=None):
        """
        Calculate background boolean array from list of limit pairs.

        Generate a background boolean string based on a list of [min,max] value
        pairs stored in self.bkgrng.

        If `rng` is supplied, these will be added to the bkgrng list before
        the boolean arrays are calculated.

        Parameters
        ----------
        rng : array_like
            [min,max] pairs defining the upper and lowe limits of background regions.

        Returns
        -------
        None
        """
        if rng is not None:
            if np.array(rng).ndim is 1:
                self.bkgrng = np.append(self.bkgrng, np.array([rng]), 0)
            else:
                self.bkgrng = np.append(self.bkgrng, np.array(rng), 0)

        self.bkg = tuples_2_bool(self.bkgrng, self.Time)
        # self.bkg = np.array([False] * self.Time.size)
        # for lb, ub in self.bkgrng:
        #     self.bkg[(self.Time > lb) & (self.Time < ub)] = True

        self.trn = ~self.bkg & ~self.sig  # redefine transition regions
        return

    def sigrange(self, rng=None):
        """
        Calculate signal boolean array from list of limit pairs.

        Generate a background boolean string based on a list of [min,max] value
        pairs stored in self.bkgrng.

        If `rng` is supplied, these will be added to the sigrng list before
        the boolean arrays are calculated.

        Parameters
        ----------
        rng : array_like
            [min,max] pairs defining the upper and lowe limits of signal regions.

        Returns
        -------
        None
        """
        if rng is not None:
            if np.array(rng).ndim is 1:
                self.sigrng = np.append(self.sigrng, np.array([rng]), 0)
            else:
                self.sigrng = np.append(self.sigrng, np.array(rng), 0)

        self.sig = tuples_2_bool(self.sigrng, self.Time)
        # self.sig = np.array([False] * self.Time.size)
        # for ls, us in self.sigrng:
        #     self.sig[(self.Time > ls) & (self.Time < us)] = True

        self.trn = ~self.bkg & ~self.sig  # redefine transition regions
        return

    def makerangebools(self):
        """
        Calculate signal and background boolean arrays from lists of limit pairs.
        """
        self.sig = tuples_2_bool(self.sigrng, self.Time)
        self.bkg = tuples_2_bool(self.bkgrng, self.Time)
        self.trn = ~self.bkg & ~self.sig
        return

    def separate(self, analytes=None):
        """
        Extract signal and backround data into separate arrays.

        Isolates signal and background signals from raw data for specified
        elements.

        Parameters
        ----------
        analytes : array_like
            list of analyte names (default = all analytes)

        Returns
        -------
        None
        """
        if analytes is None:
            analytes = self.analytes
        self.data['background'] = {}
        self.data['signal'] = {}
        for a in analytes:
            self.data['background'][a] = self.focus[a].copy()
            self.data['background'][a][~self.bkg] = np.nan
            self.data['signal'][a] = self.focus[a].copy()
            self.data['signal'][a][~self.sig] = np.nan

    def bkg_correct(self, mode='constant'):
        """
        Subtract background from signal.

        Subtract constant or polynomial background from all analytes.

        Parameters
        ----------
        mode : str or int
            'constant' or an int describing the degree of polynomial background.

        Returns
        -------
        None
        """
        params = locals()
        del(params['self'])
        self.bkgcorrect_params = params

        self.bkgrange()
        self.sigrange()
        self.separate()

        self.data['bkgsub'] = {}
        if mode == 'constant':
            for c in self.analytes:
                self.data['bkgsub'][c] = self.data['signal'][c] - np.nanmean(self.data['background'][c])
        if (mode != 'constant'):
            for c in self.analytes:
                p = np.polyfit(self.Time[self.bkg], self.focus[c][self.bkg], mode)
                self.data['bkgsub'][c] = self.data['signal'][c] - np.polyval(p, self.Time)
        self.setfocus('bkgsub')
        return

    def ratio(self, denominator='Ca43', focus='signal'):
        """
        Divide all analytes by a specified denominator analyte.

        Parameters
        ----------
        denominator : str
            The analyte used as the denominator.
        focus : str
            The analysis stage to perform the ratio calculation on.
            Defaults to 'signal'.

        Returns
        -------
        None
        """
        params = locals()
        del(params['self'])
        self.ratio_params = params

        self.setfocus(focus)
        self.data['ratios'] = {}
        for a in self.analytes:
            self.data['ratios'][a] = \
                self.focus[a] / self.focus[denominator]
        self.setfocus('ratios')
        return

    def calibrate(self, calib_dict):
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
        self.data['calibrated'] = {}
        for a in self.analytes:
            coefs = calib_dict[a]
            if len(coefs) == 1:
                self.data['calibrated'][a] = \
                    self.data['ratios'][a] * coefs
            else:
                self.data['calibrated'][a] = \
                    np.polyval(coefs, self.data['ratios'][a])
                    # self.data['ratios'][a] * coefs[0] + coefs[1]
        self.setfocus('calibrated')
        return

    # Function for calculating sample statistics
    def sample_stats(self, analytes=None, filt=True,
                     stat_fns=[np.nanmean, np.nanstd],
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
        stat_fns : array_like
            List of functions that take a single array_like input,
            and return a single statistic. Function should be able
            to cope with numpy NaN values.
        eachtrace : bool
            True: per-ablation statistics
            False: whole sample statistics

        Returns
        -------
        None
        """
        if analytes is None:
                analytes = self.analytes

        if isinstance(analytes, str):
            analytes = [analytes]

        self.stats = {}
        self.stats['analytes'] = analytes

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for f in stat_fns:
                self.stats[f.__name__] = []
                for a in analytes:
                    ind = self.filt.grab_filt(filt, a)
                    if eachtrace:
                        sts = []
                        for t in np.arange(self.n) + 1:
                            sts.append(f(self.focus[a][ind & (self.ns==t)]))
                        self.stats[f.__name__].append(sts)
                    else:
                        self.stats[f.__name__].append(f(self.focus[a][ind]))
                self.stats[f.__name__] = np.array(self.stats[f.__name__])

        try:
            self.unstats = un.uarray(self.stats['nanmean'], self.stats['nanstd'])
        except:
            pass

        return


    # Data Selections Tools

    def filter_threshold(self, analyte, threshold, filt=False):
        """
        Apply threshold filter.

        Generates threshold filters for analytes, when provided with analyte,
        threshold, and mode. Mode specifies whether data 'below'
        or 'above' the threshold are kept.

        Parameters
        ----------
        analyte : TYPE
            Description of `analyte`.
        threshold : TYPE
            Description of `threshold`.
        filt : TYPE
            Description of `filt`.
        mode : TYPE
            Description of `mode`.

        Returns
        -------
        None
        """
        params = locals()
        del(params['self'])

        # generate filter
        ind = self.filt.grab_filt(filt, analyte) & np.apply_along_axis(all, 0,~np.isnan(np.vstack(self.focus.values())))

        self.filt.add(analyte + '_thresh_below',
                           self.focus[analyte] <= threshold,
                           'Keep below {:.3e} '.format(threshold) + analyte,
                           params)
        self.filt.add(analyte + '_thresh_above',
                           self.focus[analyte] >= threshold,
                           'Keep above {:.3e} '.format(threshold) + analyte,
                           params)


    def filter_distribution(self, analyte, binwidth='scott', filt=False, transform=None):
        """
        Parameters
        ----------
        analyte : str
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

        Returns
        -------
        None
        """
        params = locals()
        del(params['self'])

        # generate filter
        ind = self.filt.grab_filt(filt, analyte) & np.apply_along_axis(all, 0,~np.isnan(np.vstack(self.focus.values())))

        # isolate data
        d = self.focus[analyte][ind]

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
                                   params=params)
        else:
            self.filt.add(name=analyte + '_distribution_failed',
                               filt=~np.isnan(self.focus[analyte]),
                               info=analyte + ' is within a single distribution. No data removed.',
                               params=params)
        return

    def filter_clustering(self, analytes, filt=False, normalise=True, method='meanshift', include_time=False, **kwargs):
        """
        Applies an n-dimensional clustering filter to the data.

        Parameters
        ----------
        analytes : str
            The analyte(s) that the filter applies to.
        filt : bool
            Whether or not to apply existing filters to the data before
            calculating this filter.
        normalise : bool
            Whether or not to normalise the data to zero mean and unit variance.
            Reccomended if clustering based on more than 1 analyte.
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
                          within the data based on the 'connectivity' of the data
                          (i.e. how far apart each data point is in a
                          multi-dimensional parameter space). Requires you to set
                          `eps`, the minimum distance point must be from another
                          point to be considered in the same cluster, and
                          `min_samples`, the minimum number of points that must be
                          within the minimum distance for it to be considered a
                          cluster. It may also be run in automatic mode by specifying
                          `n_clusters` alongside `min_samples`, where eps is
                          decreased until the desired number of clusters is obtained.
                For more information on these algorithms, refer to the documentation.
        include_time : bool
            Whether or not to include the Time variable in the clustering analysis.
            Useful if you're looking for spatially continuous clusters in your data,
            i.e. this will identify each spot in your analysis as an individual
            cluster.
        **kwargs
            Parameters passed to the clustering algorithm specified by `method`.

        Meanshift Parameters
        --------------------
        bandwidth : str or float
            The bandwith (float) or bandwidth method ('scott' or 'silverman')
            used to estimate the data bandwidth.
        bin_seeding : bool
            Modifies the behaviour of the meanshift algorithm. Refer to
            sklearn.cluster.meanshift documentation.

        K-Means Parameters
        ------------------
        n_clusters : int
            The number of clusters expected in the data.

        DBSCAN Parameters
        -----------------
        eps : float
            The minimum 'distance' points must be apart for them to be in the
            same cluster. Defaults to 0.3. Note: If the data are normalised
            (they should be for DBSCAN) this is in terms of total sample variance.
            Normalised data have a mean of 0 and a variance of 1.
        min_samples : int
            The minimum number of samples within distance `eps` required
            to be considered as an independent cluster.
        n_clusters : int
            The number of clusters expected. If specified, `eps` will be
            incrementally reduced until the expected number of clusters is found.
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

        # generate filter
        ind = self.filt.grab_filt(filt, analytes) & np.apply_along_axis(all, 0,~np.isnan(np.vstack(self.focus.values())))

        # get indices for data passed to clustering
        sampled = np.arange(self.Time.size)[ind]

        # generate data for clustering
        if len(analytes) == 1:
            # if single analyte
            d = self.focus[analytes[0]][ind]
            if include_time:
                t = self.Time[ind]
                ds = np.vstack([d,t]).T
            else:
                ds = np.array(list(zip(d,np.zeros(len(d)))))
        else:
            # package multiple analytes
            d = [self.focus[a][ind] for a in analytes]
            if include_time:
                d.append(self.Time[ind])
            ds = np.vstack(d).T

        if normalise | (len(analytes) > 1):
            ds = preprocessing.scale(ds)

        method_key = {'kmeans': self.cluster_kmeans,
                      'DBSCAN': self.cluster_DBSCAN,
                      'meanshift': self.cluster_meanshift}

        cfun = method_key[method]

        filts = cfun(ds, **kwargs)  # return dict of cluster_no: (filt, params)

        resized = {}
        for k, v in filts.items():
            resized[k] = np.zeros(self.Time.size, dtype=bool)
            resized[k][sampled] = v

        namebase = '-'.join(analytes) + '_cluster-' + method
        info = '-'.join(analytes) + ' cluster filter.'

        if method == 'DBSCAN':
            for k,v in resized.items():
                if isinstance(k, str):
                    name = namebase + '_core'
                elif k < 0:
                    name = namebase + '_noise'
                else:
                    name = namebase + '_{:.0f}'.format(k)
                self.filt.add(name, v, info=info, params=params)
        else:
            for k,v in resized.items():
                name = namebase + '_{:.0f}'.format(k)
                self.filt.add(name, v, info=info, params=params)

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
        labels_unique = np.unique(labels)

        out = {}
        for lab in labels_unique:
            out[lab] = labels == lab

        return out

    def cluster_kmeans(self, data, n_clusters):
        """
        Identify clusters using K-Means algorithm.

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
        labels_unique = np.unique(labels)

        out = {}
        for lab in labels_unique:
            out[lab] = labels == lab

        return out

    def cluster_DBSCAN(self, data, eps=None, min_samples=None, n_clusters=None, maxiter=200):
        """
        Identify clusters using DBSCAN algorithm.

        Parameters
        ----------
        data : array_like
            array of size [n_samples, n_features].
        eps : float
            The minimum 'distance' points must be apart for them to be in the
            same cluster. Defaults to 0.3. Note: If the data are normalised
            (they should be for DBSCAN) this is in terms of total sample variance.
            Normalised data have a mean of 0 and a variance of 1.
        min_samples : int
            The minimum number of samples within distance `eps` required
            to be considered as an independent cluster.
        n_clusters : int
            The number of clusters expected. If specified, `eps` will be
            incrementally reduced until the expected number of clusters is found.
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
                clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                if clusters < clusters_last:
                    eps_temp *= 1/0.95
                    db = cl.DBSCAN(eps=eps_temp, min_samples=15).fit(data)
                    clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                    warnings.warn('\n\n***Unable to find {:.0f} clusters in data. Found {:.0f} with an eps of {:.2e}'.format(n_clusters, clusters, eps_temp))
                    break
                niter += 1
                if niter == maxiter:
                    warnings.warn('\n\n***Maximum iterations ({:.0f}) reached, {:.0f} clusters not found.\nDeacrease min_samples or n_clusters (or increase maxiter).'.format(maxiter, n_clusters))
                    break

        labels = db.labels_
        labels_unique = np.unique(labels)

        core_samples_mask = np.zeros_like(labels)
        core_samples_mask[db.core_sample_indices_] = True

        out = {}
        for lab in labels_unique:
            out[lab] = labels == lab

        out['core'] = core_samples_mask

        return out

    def filter_correlation(self, x_analyte, y_analyte, window=None, r_threshold=0.9, p_threshold=0.05, filt=True):
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
        if window % 2 != 1:
            window += 1

        params = locals()
        del(params['self'])

        # get filter
        ind = self.filt.grab_filt(filt, [x_analyte, y_analyte])

        x = self.focus[x_analyte]
        x[~ind] = np.nan
        xr = self.rolling_window(x, window, pad=np.nan)

        y = self.focus[y_analyte]
        y[~ind] = np.nan
        yr = self.rolling_window(y, window, pad=np.nan)

        r, p = zip(*map(pearsonr, xr,yr))

        r = np.array(r)
        p = np.array(p)

        cfilt = (abs(r) > r_threshold) & (p < p_threshold)
        cfilt = ~cfilt

        name = x_analyte + '-' + y_analyte + '_corr'

        self.filt.add(name=name,
                           filt=cfilt,
                           info=x_analyte + ' vs. ' + y_analyte + ' correlation filter.',
                           params=params)
        self.filt.off(filt=name)
        self.filt.on(analyte=y_analyte, filt=name)

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

    def tplot(self, analytes=None, figsize=[10, 4], scale=None, filt=False,
              ranges=False, stats=True, stat='nanmean', err='nanstd', interactive=False):
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
            plot average and error of each trace, as specified by `stat` and `err`.
        stat : str
            average statistic to plot.
        err : str
            error statistic to plot.
        interactive : bool
            Make the plot interactive.

        Returns
        -------
        figure
        """

        if interactive:
            enable_notebook()  # make the plot interactive

        if type(analytes) is str:
            analytes = [analytes]
        if analytes is None:
            analytes = self.analytes

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        for a in analytes:
            x = self.Time
            y = self.focus[a]

            if scale is 'log':
                ax.set_yscale('log')
                y[y == 0] = 1

            ind = self.filt.grab_filt(filt, a)
            xf = x.copy()
            yf = y.copy()
            if any(~ind):
                xf[~ind] = np.nan
                yf[~ind] = np.nan

            if any(~ind):
                ax.plot(x, y, color=self.cmap[a], alpha=.4, lw=0.6)
            ax.plot(xf, yf, color=self.cmap[a], label=a)

            # Plot averages and error envelopes
            if stats and hasattr(self, 'stats'):
                sts = self.stats[sig][0].size
                if sts > 1:
                    for n in np.arange(self.n):
                        n_ind = ind & (self.ns==n+1)
                        if sum(n_ind) > 2:
                            x = [self.Time[n_ind][0], self.Time[n_ind][-1]]
                            y = [self.stats[sig][self.stats['analytes']==a][0][n]] * 2

                            yp = [self.stats[sig][self.stats['analytes']==a][0][n] + self.stats[err][self.stats['analytes']==a][0][n]] * 2
                            yn = [self.stats[sig][self.stats['analytes']==a][0][n] - self.stats[err][self.stats['analytes']==a][0][n]] * 2

                            ax.plot(x, y, color=self.cmap[a], lw=2)
                            ax.fill_between(x + x[::-1], yp + yn, color=self.cmap[a], alpha=0.4, linewidth=0)
                else:
                    x = [self.Time[0], self.Time[-1]]
                    y = [self.stats[sig][self.stats['analytes']==a][0]] * 2
                    yp = [self.stats[sig][self.stats['analytes']==a][0] + self.stats[err][self.stats['analytes']==a][0]] * 2
                    yn = [self.stats[sig][self.stats['analytes']==a][0] - self.stats[err][self.stats['analytes']==a][0]] * 2

                    ax.plot(x, y, color=self.cmap[a], lw=2)
                    ax.fill_between(x + x[::-1], yp + yn, color=self.cmap[a], alpha=0.4, linewidth=0)

        if ranges:
            for lims in self.bkgrng:
                ax.axvspan(*lims, color='k', alpha=0.1, zorder=-1)
            for lims in self.sigrng:
                ax.axvspan(*lims, color='r', alpha=0.1, zorder=-1)

        ax.text(0.01, 0.99, self.sample + ' : ' + self.focus_stage, transform=ax.transAxes,
                ha='left', va='top')

        ax.set_xlabel('Time (s)')

        if interactive:
            ax.legend()
            plugins.connect(fig, plugins.MousePosition(fontsize=14))
            display.clear_output(wait=True)
            display.display(fig)
            input('Press [Return] when finished.')
            disable_notebook()  # stop the interactivity
        else:
            ax.legend(bbox_to_anchor=(1.12, 1))

        return fig

    def crossplot(self, analytes=None, bins=25, lognorm=True, filt=True):
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
            analytes = [a for a in self.analytes if a != self.ratio_params['denominator']]

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

        cmlist = ['Blues', 'BuGn', 'BuPu', 'GnBu',
                  'Greens', 'Greys', 'Oranges', 'OrRd',
                  'PuBu', 'PuBuGn', 'PuRd', 'Purples',
                  'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']
        udict = {}
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            for x, y in [(i, j), (j, i)]:
                # set unit multipliers
                mx, ux = unitpicker(np.nanmean(self.focus[analytes[x]]))
                my, uy = unitpicker(np.nanmean(self.focus[analytes[y]]))
                udict[analytes[x]] = (x, ux)

                # get filter
                ind = (self.filt.grab_filt(filt, analytes[x]) &
                       self.filt.grab_filt(filt, analytes[y]) &
                       ~np.isnan(self.focus[analytes[x]]) &
                       ~np.isnan(self.focus[analytes[y]]))

                # make plot
                px = self.focus[analytes[x]][ind] * mx
                py = self.focus[analytes[y]][ind] * my

                if lognorm:
                    axes[x, y].hist2d(py, px, bins,
                                      norm=mpl.colors.LogNorm(),
                                      cmap=plt.get_cmap(cmlist[x]))
                else:
                    axes[x, y].hist2d(py, px, bins,
                                      cmap=plt.get_cmap(cmlist[x]))
                axes[x, y].set_ylim([px.min(), px.max()])
                axes[x, y].set_xlim([py.min(), py.max()])
        # diagonal labels
        for a, (i, u) in udict.items():
            axes[i, i].annotate(a+'\n'+u, (0.5, 0.5),
                                xycoords='axes fraction',
                                ha='center', va='center')
        # switch on alternating axes
        for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
            axes[j, i].xaxis.set_visible(True)
            for label in axes[j, i].get_xticklabels():
                label.set_rotation(90)
            axes[i, j].yaxis.set_visible(True)

        axes[0,0].set_title(self.sample, weight='bold', x=0.05, ha='left')

        return fig, axes

    def filt_report(self, filt=None, analyte=None, save=None):
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
        if filt is None:
            filts = list(self.filt.components.keys())
        else:
            filts = np.array(sorted([f for f in self.filt.components.keys() if filt in f]))
        nfilts = np.array([re.match('^([A-Za-z0-9-]+)_([A-Za-z0-9-]+)[_$]?([a-z0-9]+)?', f).groups() for f in filts])
        fgnames = np.array(['_'.join(a) for a in nfilts[:,:2]])
        fgrps = np.unique(fgnames) #np.unique(nfilts[:,1])

        ngrps = fgrps.size

        plots = {}

        m, u = unitpicker(np.nanmax(self.focus[analyte]))

        fig = plt.figure(figsize=(10, 3.5 * ngrps))
        axes = []

        h = .8 / ngrps

        cm = plt.cm.get_cmap('Spectral')

        for i in np.arange(ngrps):
            axs = tax, hax = fig.add_axes([.1,.9-(i+1)*h,.6,h*.98]), fig.add_axes([.7,.9-(i+1)*h,.2,h*.98])

            # get variables
            fg = filts[fgnames == fgrps[i]]
            cs = cm(np.linspace(0,1,len(fg)))
            fn = nfilts[:,2][fgnames == fgrps[i]]
            an = nfilts[:,0][fgnames == fgrps[i]]
            bins = np.linspace(np.nanmin(self.focus[analyte]), np.nanmax(self.focus[analyte]), 50) * m

            if 'DBSCAN' in fgrps[i]:
                # determine data filters
                core_ind = self.filt.components[[f for f in fg if 'core' in f][0]]
                noise_ind = self.filt.components[[f for f in fg if 'noise' in f][0]]
                other = np.array([('noise' not in f) & ('core' not in f) for f in fg])
                tfg = fg[other]
                tfn = fn[other]
                tcs = cm(np.linspace(0,1,len(tfg)))

                # plot all data
                hax.hist(m * self.focus[analyte], bins, alpha=0.5, orientation='horizontal', color='k', lw=0)
                # legend markers for core/member
                tax.scatter([],[],s=25,label='core',c='w')
                tax.scatter([],[],s=10,label='member',c='w')
                # plot noise
                tax.scatter(self.Time[noise_ind], m * self.focus[analyte][noise_ind], lw=1, c='k', s=15, marker='x', label='noise')

                # plot filtered data
                for f, c, lab in zip(tfg, tcs, tfn):
                    ind = self.filt.components[f]
                    tax.scatter(self.Time[~core_ind & ind], m * self.focus[analyte][~core_ind & ind], lw=.1, c=c, s=10)
                    tax.scatter(self.Time[core_ind & ind], m * self.focus[analyte][core_ind & ind], lw=.1, c=c, s=25, label=lab)
                    hax.hist(m * self.focus[analyte][ind], bins, color=c, lw=0.1, orientation='horizontal', alpha=0.6)

            else:
                # plot all data
                tax.scatter(self.Time, m * self.focus[analyte], c='k', alpha=0.5, lw=0.1, s=25, label='excl')
                hax.hist(m * self.focus[analyte], bins, alpha=0.5, orientation='horizontal', color='k', lw=0)

                # plot filtered data
                for f, c, lab in zip(fg, cs, fn):
                    ind = self.filt.components[f]
                    tax.scatter(self.Time[ind], m * self.focus[analyte][ind], lw=.1, c=c, s=25, label=lab)
                    hax.hist(m * self.focus[analyte][ind], bins, color=c, lw=0.1, orientation='horizontal', alpha=0.6)

            # formatting
            for ax in axs:
                ax.set_ylim(np.nanmin(self.focus[analyte]) * m, np.nanmax(self.focus[analyte]) * m)

            tax.legend(scatterpoints=1, framealpha=0.5)
            tax.text(.02, .98, fgrps[i], size=12, weight='bold', ha='left', va='top', transform=tax.transAxes)
            tax.set_ylabel(pretty_element(analyte) + ' (' + u + ')')
            tax.set_xticks(tax.get_xticks()[:-1])
            hax.set_yticklabels([])

            if i < ngrps - 1:
                tax.set_xticklabels([])
                hax.set_xticklabels([])
            else:
                tax.set_xlabel('Time (s)')
                hax.set_xlabel('n')

            axes.append(axs)

        return fig, axes

    # reporting
    def get_params(self):
        """
        Returns paramters used to process data.

        Returns
        -------
        dict
            dict of analysis parameters
        """
        outputs = ['sample', 'method',
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

class filt(object):
    """
    Container for storing, selecting and creating data filters.

    Parameters
    ----------
    size : int
        The length that the filters need to be (should be
        the same as your data).
    analytes : array_like
        A list of the analytes measured in your data.

    Attributes
    ----------
    size : int
        The length that the filters need to be (should be
        the same as your data).
    analytes : array_like
        A list of the analytes measured in your data.
    components : dict
        A dict containing each individual filter that has been
        created.
    info : dict
        A dict containing descriptive information about each
        filter in `components`.
    params : dict
        A dict containing the parameters used to create
        each filter, which can be passed directly to the
        corresponding filter function to recreate the filter.
    switches : dict
        A dict of boolean switches specifying which filters
        are active for each analyte.
    keys : dict
        A dict of logical strings specifying which filters are
        applied to each analyte.
    sequence : dict
        A numbered dict specifying what order the filters were
        applied in (for some filters, order matters).
    n : int
        The number of filters applied to the data.

    Methods
    -------
    add
    remove
    clear
    clean
    on
    off
    make
    make_fromkey
    make_keydict
    grab_filt
    get_components
    get_info
    """
    def __init__(self, size, analytes):
        self.size = size
        self.analytes = analytes
        self.components = {}
        self.info = {}
        self.params = {}
        self.keys = {}
        self.sequence = {}
        self.n = 0
        self.switches = {}
        for a in self.analytes:
            self.switches[a] = {}

    def __repr__(self):
        leftpad = max([len(s) for s in self.switches[self.analytes[0]].keys()] + [11]) + 2
        out = '{string:{number}s}'.format(string='Filter Name', number=leftpad)
        for a in self.analytes:
            out += '{:7s}'.format(a)
        out += '\n'

        for t in sorted(self.switches[self.analytes[0]].keys()):
            out += '{string:{number}s}'.format(string=str(t), number=leftpad)
            for a in self.analytes:
                out += '{:7s}'.format(str(self.switches[a][t]))
            out += '\n'
        return(out)

    def add(self, name, filt, info='', params=()):
        """
        Add filter.

        Parameters
        ----------
        name : str
            filter name
        filt : array_like
            boolean filter array
        info : str
            informative description of the filter
        params : tuple
            parameters used to make the filter

        Returns
        -------
        None
        """
        self.components[name] = filt
        self.info[name] = info
        self.params[name] = params
        self.sequence[self.n] = name
        self.n += 1
        for a in self.analytes:
            self.switches[a][name] = True
        return

    def remove(self, name):
        """
        Remove filter.

        Parameters
        ----------
        name : str
            name of the filter to remove

        Returns
        -------
        None
        """
        del self.components[name]
        del self.info[name]
        del self.params[name]
        del self.keys[name]
        del self.sequence[name]
        for a in self.analytes:
            del self.switches[a][name]
        return

    def clear(self):
        """
        Clear all filters.
        """
        self.components = {}
        self.info = {}
        self.params = {}
        self.switches = {}
        self.keys = {}
        self.sequence = {}
        self.n = 0
        for a in self.analytes:
            self.switches[a] = {}
        return

    def clean(self):
        """
        Remove unused filters.
        """
        for f in sorted(self.components.keys()):
            unused = not any(self.switches[a][f] for a in self.analytes)
            if unused:
                self.remove(f)

    def on(self, analyte=None, filt=None):
        """
        Turn on specified filter(s) for specified analyte(s).

        Parameters
        ----------
        analyte : optional, str or array_like
            Name or list of names of analytes.
            Defaults to all analytes.
        filt : optional, str or array_like
            Name or list of names of filters.

        Returns
        -------
        None
        """
        if isinstance(analyte, str):
            analyte = [analyte]
        if isinstance(filt, str):
            filt = [filt]

        if analyte is None:
            analyte = self.analytes
        if filt is None:
            filt = self.switches[analyte[0]].keys()

        for a in analyte:
            for f in filt:
                for k in self.switches[a].keys():
                    if f in k:
                        self.switches[a][k] = True
        return

    def off(self, analyte=None, filt=None):
        """
        Turn off specified filter(s) for specified analyte(s).

        Parameters
        ----------
        analyte : optional, str or array_like
            Name or list of names of analytes.
            Defaults to all analytes.
        filt : optional, str or array_like
            Name or list of names of filters.

        Returns
        -------
        None
        """
        if isinstance(analyte, str):
            analyte = [analyte]
        if isinstance(filt, str):
            filt = [filt]

        if analyte is None:
            analyte = self.analytes
        if filt is None:
            filt = self.switches[analyte[0]].keys()

        for a in analyte:
            for f in filt:
                for k in self.switches[a].keys():
                    if f in k:
                        self.switches[a][k] = False
        return

    def make(self, analyte):
        """
        Make filter for specified analyte(s).

        Filter specified in filt.switches.

        Parameters
        ----------
        analyte : str or array_like
            Name or list of names of analytes.

        Returns
        -------
        array_like
            boolean filter
        """
        if isinstance(analyte, str):
            analyte = [analyte]

        out = []
        for f in self.components.keys():
            for a in analyte:
                if self.switches[a][f]:
                    out.append(f)
        key = ' & '.join(sorted(out))
        for a in analyte:
            self.keys[a] = key
        return self.make_fromkey(key)

    def make_fromkey(self, key):
        """
        Make filter from logical expression.

        Takes a logical expression as an input, and returns a filter. Used for advanced
        filtering, where combinations of nested and/or filters are desired. Filter names must
        exactly match the names listed by print(filt).

        Example:
            key = '(Filter_1 | Filter_2) & Filter_3'
        is equivalent to:
            (Filter_1 OR Filter_2) AND Filter_3
        statements in parentheses are evaluated first.

        Parameters
        ----------
        key : str
            logical expression describing filter construction.

        Returns
        -------
        array_like
            boolean filter

        """
        if key != '':
            def make_runable(match):
                return "self.components['" + match.group(0) + "']"

            runable = re.sub('[^\(\)|& ]+', make_runable, key)
            return eval(runable)
        else:
            return ~np.zeros(self.size, dtype=bool)

    def make_keydict(self, analyte=None):
        """
        Make logical expressions describing the filter(s) for specified analyte(s).

        Parameters
        ----------
        analyte : optional, str or array_like
            Name or list of names of analytes.
            Defaults to all analytes.

        Returns
        -------
        dict
            containing the logical filter expression for each analyte.
        """
        if analyte is None:
            analyte = self.analytes
        elif isinstance(analyte, str):
            analyte = [analyte]

        out = {}
        for a in analyte:
            key = []
            for f in self.components.keys():
                if self.switches[a][f]:
                    key.append(f)
            out[a] = ' & '.join(sorted(key))
        self.keydict = out
        return out

    def grab_filt(self,filt,analyte=None):
        """
        Flexible access to specific filter using any key format.

        Parameters
        ----------
        f : str, dict or bool
            either logical filter expression, dict of expressions,
            or a boolean
        analyte : str
            name of analyte the filter is for.

        Returns
        -------
        array_like
            boolean filter
        """
        if isinstance(filt, str):
            try:
                ind = self.make_fromkey(filt)
            except ValueError:
                print("\n\n***Filter key invalid. Please consult manual and try again.")
        elif isinstance(filt, dict):
            try:
                ind = self.make_fromkey(filt[analyte])
            except ValueError:
                print("\n\n***Filter key invalid. Please consult manual and try again.\nOR\nAnalyte missing from filter key dict.")
        elif filt:
            ind = self.make(analyte)
        else:
            ind = ~np.zeros(self.size, dtype=bool)
        return ind

    def get_components(self, key, analyte=None):
        """
        Extract filter components for specific analyte(s).

        Parameters
        ----------
        key : str
            string present in one or more filter names.
            e.g. 'Al27' will return all filters with
            'Al27' in their names.
        analyte : str
            name of analyte the filter is for

        Returns
        -------
        array_like
            boolean filter
        """
        out = {}
        for k, v in self.components.items():
            if key in k:
                if analyte is None:
                    out[k] = v
                elif self.switches[analyte][k]:
                    out[k] = v
        return out

    def get_info(self):
        """
        Get info for all filters.
        """
        out = ''
        for k in sorted(self.components.keys()):
            out += '{:s}: {:s}'.format(k, self.info[k]) + '\n'
        return(out)

    # def plot(self, ax=None, analyte=None):
    #     if ax is None:
    #         fig, ax = plt.subplots(1,1)
    #     else:
    #         ax = ax.twinx()
    #         ax.set_yscale('linear')
    #         ax.set_yticks([])

    #     if analyte is not None:
    #         filts = []
    #         for k, v in self.switches[analyte].items():
    #             if v:
    #                 filts.append(k)
    #         filts = sorted(filts)
    #     else:
    #         filts = sorted(self.switches[self.analytes[0]].keys())

    #     n = len(filts)

    #     ylim = ax.get_ylim()
    #     yrange = max(ylim) - min(ylim)
    #     yd = yrange / (n * 1.2)

    #     yl = min(ylim) + 0.1 * yd
    #     for i in np.arange(n):
    #         f = filts[i]
    #         xlims = bool_2_indices(self.components[f])

    #         yu = yl + yd

    #         for xl, xu in zip(xlims[0::2], xlims[1::2]):
    #             xl /= self.size
    #             xu /= self.size
    #             ax.axhspan(yl, yu, xl, xu, color='k', alpha=0.3)

    #         ym = np.mean([yu,yl])

    #         ax.text(ax.get_xlim()[1] * 1.01, ym, f, ha='left')

    #         yl += yd * 1.2

    #     return(ax)


# other useful functions
def unitpicker(a, llim=0.1):
    """
    Determines the most appropriate plotting unit for data.

    Parameters
    ----------
    a : array_like
        raw data array
    llim : float
        minimum allowable value in scaled data.

    Returns
    -------
    (float, str)
        (multiplier, unit)
    """
    udict = {0: 'mol/mol',
             1: 'mmol/mol',
             2: '$\mu$mol/mol',
             3: 'nmol/mol',
             4: 'pmol/mol',
             5: 'fmol/mol'}
    a = abs(a)
    n = 0
    if a < llim:
        while a < llim:
            a *= 1000
            n += 1
    return float(1000**n), udict[n]

def pretty_element(s):
    """
    Returns formatted element name.

    Parameters
    ----------
    s : str
        of format [A-Z][a-z]?[0-9]+

    Returns
    -------
    str
        LaTeX formatted string with superscript numbers.
    """
    g = re.match('([A-Z][a-z]?)([0-9]+)', s).groups()
    return '$^{' + g[1] + '}$' + g[0]


def collate_csvs(in_dir,out_dir='./csvs'):
    """
    Copy all csvs in nested directroy to single directory.

    Function to copy all csvs from a directory, and place
    them in a new directory.

    Parameters
    ----------
    in_dir : str
        input directory containing csv files in subfolders
    out_dir : str
        destination directory

    Returns
    -------
    None
    """
    import os
    import shutil

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for p, d, fs in os.walk(in_dir):
        for f in fs:
            if '.csv' in f:
                shutil.copy(p + '/' + f, out_dir + '/' + f)
    return

def bool_2_indices(bool_array):
    """
    Get list of limit tuples from boolean array.

    Parameters
    ----------
    bool_array : array_like
        boolean array

    Returns
    -------
    array_like
        [2,n] array of (start, end) values describing True parts
        of bool_array
    """
    if ~isinstance(bool_array, np.ndarray):
        bool_array = np.array(bool_array)
    return np.arange(len(bool_array))[bool_array ^ np.roll(bool_array, 1)]

def tuples_2_bool(tuples, x):
    """
    Generate boolean array from list of limit tuples.

    Parameters
    ----------
    tuples : array_like
        [2,n] array of (start, end) values
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

def config_locator():
    """
    Prints the location of the latools.cfg file.
    """
    print(pkg_resources.resource_filename('latools', 'latools.cfg'))
    return

def add_config(config_name, params, config_file=None, make_default=True):
    """
    Adds a new configuration to latools.cfg.

    Parameters
    ----------
    config_name : str
        The name of the new configuration. This should be descriptive
        (e.g. UC Davis Foram Group)
    params : dict
        A (parameter, value) dict defining non-default parameters
        associated with the new configuration.
        Possible parameters include:
        srmfile : str
            Path to srm file used in calibration. Defaults to GeoRem
            values for NIST610, NIST612 and NIST614 provided with latools.
        dataformat : dict (as str)
            See dataformat documentation.
    config_file : str
        Path to the configuration file that will be modified. Defaults to
        latools.cfg in package install location.
    make_default : bool
        Whether or not to make the new configuration the default
        for future analyses. Default = True.

    Returns
    -------
    None
    """

    if config_file is None:
        config_file = pkg_resources.resource_filename('latools', 'latools.cfg')
    cf = configparser.ConfigParser()
    cf.read(config_file)

    # if config doesn't already exist, create it.
    if config_name not in cf.sections():
        cf.add_section(config_name)
    # iterate through parameter dict and set values
    for k,v in params.items():
        cf.set(config_name, k, v)
    # make the parameter set default, if requested
    if make_default:
        cf.set('DEFAULT', 'default_config', config_name)

    cf.write(open(config_file, 'w'))

    return

def intial_configuration():
    """
    Convenience function for configuring latools.
    """
    print('You will be asked a few questions to configure latools\nfor your specific laboratory needs.')
    lab_name = input('What is the name of your lab? : ')

    params = {}
    params['srmfile'] = input('Where is your SRM.csv file? [blank = default] : ')

    make_default = input('Do you want this to be your default? [Y/n] : ').lower() != 'n'

    add_config(lab_name, params, make_default=make_default)

    print("\nConfiguration set. You're good to go!")

    return
