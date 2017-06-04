import configparser
import itertools
import inspect
import os
import re
import shutil
import time
import warnings

import brewer2mpl as cb  # for colours
import matplotlib.pyplot as plt
import matplotlib as mpl
# import multiprocessing as mp
import numpy as np
import pandas as pd
import pkg_resources as pkgrs
import sklearn.cluster as cl
import uncertainties as unc
import uncertainties.unumpy as un

from io import BytesIO
from functools import wraps
from fuzzywuzzy import fuzz
from mpld3 import plugins
from mpld3 import enable_notebook, disable_notebook
from scipy import odr
from scipy.stats import gaussian_kde, pearsonr
from scipy.optimize import curve_fit
import scipy.interpolate as interp
from sklearn import preprocessing

from IPython import display
from tqdm import tqdm  # status bars!

# deactivate IPython deprecations warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
# deactivate numpy invalid comparison warnings
np.seterr(invalid='ignore')


class analyse(object):
    """
    For processing and analysing whole LA - ICPMS datasets.

    Parameters
    ----------
    data_folder : str
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
        Either a path to a data format file, or a
        dataformat dict. See documentation for more details.
    extension : str
        The file extension of your data files. Defaults to
        '.csv'.
    srm_identifier : str
        A string used to separate samples and standards. srm_identifier
        must be present in all standard measurements. Defaults to
        'STD'.
    cmap : dict
        A dictionary of {analyte: colour} pairs. Colour can be any valid
        matplotlib colour string, RGB or RGBA sequence, or hex string.
    time_format : str
        A regex string identifying the time format, used by pandas when
        created a universal time scale. If unspecified (None), pandas
        attempts to infer the time format, but in some cases this might
        not work.
    internal_standard : str
        The name of the analyte used as an internal standard throughout
        analysis.
    names : str
        'file_names' : use the file names as labels
        'metadata_names' : used the 'names' attribute of metadata as the name
        anything else : use numbers.

    Attributes
    ----------
    folder : str
        Path to the directory containing the data files, as
        specified by `data_folder`.
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
        data. These must contain srm_identifier in the file name.
    srm_identifier : str
        A string present in the file names of all standards.
    cmaps : dict
        An analyte - specific colour map, used for plotting.


    Methods
    -------
    ablation_times
    autorange
    bkg_calc_interp1d
    bkg_calc_weightedmean
    bkg_plot
    bkg_subtract
    calibrate
    calibration_plot
    crossplot
    despike
    export_traces
    filter_clear
    filter_clustering
    filter_correlation
    filter_distribution
    filter_off
    filter_on
    filter_reports
    filter_status
    filter_threshold
    find_expcoef
    get_background
    get_focus
    get_starttimes
    getstats
    make_subset
    minimal_export
    ratio
    sample_stats
    set_focus
    srm_id_auto
    statplot
    trace_plots
    zeroscreen
    """
    def __init__(self, data_folder, errorhunt=False, config='DEFAULT',
                 dataformat=None, extension='.csv', srm_identifier='STD',
                 cmap=None, time_format=None, internal_standard='Ca43',
                 names='file_names'):
        """
        For processing and analysing whole LA - ICPMS datasets.
        """
        # initialise log
        params = locals()
        del(params['self'])
        self.log = ['__init__ :: args=() kwargs={}'.format(str(params))]

        self.folder = os.path.realpath(data_folder)
        self.parent_folder = os.path.dirname(self.folder)
        self.files = np.array([f for f in os.listdir(self.folder)
                               if extension in f])

        # make output directories
        self.report_dir = re.sub('//', '/',
                                 self.parent_folder + '/' +
                                 os.path.basename(self.folder) + '_reports/')
        if not os.path.isdir(self.report_dir):
            os.mkdir(self.report_dir)
        self.export_dir = re.sub('//', '/',
                                 self.parent_folder + '/' +
                                 os.path.basename(self.folder) + '_export/')
        if not os.path.isdir(self.export_dir):
            os.mkdir(self.export_dir)

        self.focus_stage = 'rawdata'

        # load configuration parameters
        # read in config file
        conf = configparser.ConfigParser()
        conf.read(pkgrs.resource_filename('latools', 'latools.cfg'))
        # load defaults into dict
        pconf = dict(conf.defaults())
        # if no config is given, check to see what the default setting is
        # if (config is None) & (pconf['config'] != 'DEFAULT'):
        #     config = pconf['config']
        # else:
        #     config = 'DEFAULT'

        # if there are any non - default parameters, replace defaults in
        # the pconf dict
        if config != 'DEFAULT':
            for o in conf.options(config):
                pconf[o] = conf.get(config, o)
        self.config = config
        print('latools analysis using "' + self.config + '" configuration:')

        # check srmfile exists, and store it in a class attribute.
        if os.path.exists(pconf['srmfile']):
            self.srmfile = pconf['srmfile']
        elif os.path.exists(pkgrs.resource_filename('latools',
                                                    pconf['srmfile'])):
            self.srmfile = pkgrs.resource_filename('latools',
                                                   pconf['srmfile'])
        else:
            raise ValueError(('The SRM file specified in the ' + config +
                              ' configuration cannot be found.\n'
                              'Please make sure the file exists, and that the '
                              'path in the config file is correct.\n'
                              'To locate the config file, run '
                              '`latools.config_locator()`.\n\n'
                              '' + config + ' file: ' + pconf['srmfile']))

        # load in dataformat information.
        # check dataformat file exists, and store it in a class attribute.
        # if dataformat is not provided during initialisation, assign it
        # from configuration file
        if dataformat is None:
            if os.path.exists(pconf['dataformat']):
                dataformat = pconf['dataformat']
            elif os.path.exists(pkgrs.resource_filename('latools',
                                                        pconf['dataformat'])):
                dataformat = pkgrs.resource_filename('latools',
                                                     pconf['dataformat'])
            else:
                raise ValueError(('The dataformat file specified in the ' +
                                  config + ' configuration cannot be found.\n'
                                  'Please make sure the file exists, and that '
                                  'the path in the config file is correct.\n'
                                  'To locate the config file, run '
                                  '`latools.config_locator()`.\n\n' +
                                  config + ' file: ' + dataformat))
            self.dataformat_file = dataformat
        else:
            self.dataformat_file = 'None: dict provided'

        # if it's a string, check the file exists and import it.
        if isinstance(dataformat, str):
            if os.path.exists(dataformat):
                self.dataformat = eval(open(dataformat).read())
                # self.dataformat = json.load(open(dataformat))
            else:
                warnings.warn(("The dataformat file (" + dataformat +
                               ") cannot be found.\nPlease make sure the file "
                               "exists, and that the path is correct.\n\nFile "
                               "Path: " + dataformat))

        # if it's a dict, just assign it straight away.
        elif isinstance(dataformat, dict):
            self.dataformat = dataformat

        # load data (initialise D objects)
        data = np.array([D(self.folder + '/' + f,
                           dataformat=self.dataformat,
                           errorhunt=errorhunt,
                           cmap=cmap, 
                           internal_standard=internal_standard,
                           name=names) for f in self.files])

        # sort by time
        self.data = sorted(data, key=lambda d: d.meta['date'])

        # assign sample names
        if (names == 'file_names') | (names == 'metadata_names'):
            self.samples = np.array([s.sample for s in self.data])
            # rename duplicates sample names
            for u in np.unique(self.samples):
                ind = self.samples == u
                if sum(ind) > 1:
                    self.samples[ind] = [s + '_{}'.format(n) for s, n in zip(self.samples[ind], np.arange(sum(ind)))]
                    for i, sn in zip(np.arange(len(self.samples))[ind], self.samples[ind]):
                        self.data[i].sample = sn
        else:
            self.samples = np.arange(len(self.data))
            for i, s in enumerate(self.samples):
                self.data[i].sampleseq1 = s

        self.analytes = np.array(self.data[0].analytes)
        if internal_standard in self.analytes:
            self.internal_standard = internal_standard
        else:
            ValueError('The internal standard ({}) is not amongst the'.format(internal_standard) +
                       'analytes in\nyour data files. Please make sure it is specified correctly.')
        self.minimal_analytes = [internal_standard]

        self.data_dict = {}
        for s, d in zip(self.samples, self.data):
            self.data_dict[s] = d

        self.srm_identifier = srm_identifier
        self.stds = []
        _ = [self.stds.append(s) for s in self.data
             if self.srm_identifier in s.sample]
        self.srms_ided = False

        # set up subsets
        self._has_subsets = False
        self._subset_names = []
        self.subsets = {}
        self.subsets['All_Analyses'] = self.samples
        self.subsets[self.srm_identifier] = [s for s in self.samples if self.srm_identifier in s]
        self.subsets['All_Samples'] = [s for s in self.samples if self.srm_identifier not in s]
        self.subsets['not_in_set'] = self.subsets['All_Samples'].copy()

        # create universal time scale
        if 'date' in self.data[0].meta.keys():
            if (time_format is None) and ('time_format' in self.dataformat.keys()):
                time_format = self.dataformat['time_format']

            self.starttimes = self.get_starttimes(time_format)

            for d in self.data_dict.values():
                d.uTime = d.Time + self.starttimes.loc[d.sample, 'Dseconds']

        else:
            ts = 0
            for d in self.data_dict.values():
                d.uTime = d.Time + ts
                ts += d.Time[-1]
            warnings.warn("Time not found in data file. Universal time scale\n" +
                          "approximated as continuously measured samples.\n" +
                          "Background correction and calibration may not behave\n" +
                          "as expected.")

        # copy colour map to top level
        self.cmaps = self.data[0].cmap

        # initialise classifiers
        self.classifiers = {}

        # f = open('errors.log', 'a')
        # f.write(('Errors and warnings during LATOOLS analysis '
        #          'are stored here.\n\n'))
        # f.close()

        # report
        print(('  {:.0f} Data Files Loaded: {:.0f} standards, {:.0f} '
               'samples').format(len(self.data),
                                 len(self.stds),
                                 len(self.data) - len(self.stds)))
        print('  Analytes: ' + ' '.join(self.analytes))
        print('  Internal Standard: {}'.format(self.internal_standard))

    # Helper Functions
    def _log(fn):
        """
        Function for logging method calls and parameters
        """
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            a = fn(self, *args, **kwargs)
            self.log.append(fn.__name__ + ' :: args={} kwargs={}'.format(args, kwargs))
            return a
        return wrapper

    def _get_samples(self, subset=None):
        """
        Helper function to get sample names from subset.

        Parameters
        ----------
        subset : str
            Subset name. If None, returns all samples.

        Returns
        -------
        List of sample names
        """
        if subset is None:
            samples = self.subsets['All_Samples']
        else:
            try:
                samples = self.subsets[subset]
            except KeyError:
                raise KeyError(("Subset '{:s}' does not ".format(subset) +
                                "exist.\nUse 'make_subset' to create a" +
                                "subset."))
        return samples

    def _add_minimal_analte(self, analyte):
        """
        Check for existence of analyte(s) in minimal_list, add if not present.
        """
        if isinstance(analyte, str):
            analyte = [analyte]
        for a in analyte:
            if a not in self.minimal_analytes:
                self.minimal_analytes.append(a)

    # Work functions
    def get_starttimes(self, time_format=None):
        try:
            sd = {}
            for k, v in self.data_dict.items():
                sd[k] = v.meta['date']

            sd = pd.DataFrame.from_dict(sd, orient='index')
            sd.columns = ['date']

            sd.loc[:, 'date'] = pd.to_datetime(sd['date'], format=time_format)

            sd['Ddate'] = sd.date - sd.date.min()
            sd['Dseconds'] = sd.Ddate / np.timedelta64(1, 's')
            sd.sort_values(by='Dseconds', inplace=True)
            sd['sequence'] = range(sd.shape[0])
            return sd
        except:
            ValueError(("Cannot determine data file start times.\n" +
                        "This could be because:\n  1) 'date' " +
                        "not specified in 'meta_regex' section of \n" +
                        "     file format. Consult 'data format' documentation\n  " +
                        "   and modify appropriately.\n  2) time_format cannot be" +
                        " automatically determined.\n     Consult 'strptime'" +
                        " documentation, and provide a\n     valid 'time_format'."))

    @_log
    def autorange(self, analyte=None, gwin=11, win=40, smwin=5,
                  conf=0.01, on_mult=[1., 1.], off_mult=None, d_mult=1.2,
                  transform='log', thresh_n=None):
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
            Boolean arrays the same length as the data, identifying
            'background', 'signal' and 'transition' data regions.
        bkgrng, sigrng, trnrng: array_like
            Pairs of values specifying the edges of the 'background', 'signal'
            and 'transition' data regions in the same units as the Time axis.

        Returns
        -------
        None
        """

        if thresh_n is not None:
            # calculate maximum background composition of internal standard
            srms = self.subsets[self.srm_identifier]

            if not hasattr(self.data_dict[srms[0]], 'bkg'):
                for s in srms:
                    self.data_dict[s].autorange()

            srm_bkg_dat = []

            for s in srms:
                sd = self.data_dict[s]

                ind = (sd.Time >= sd.bkgrng[0][0]) & (sd.Time <= sd.bkgrng[0][1])
                srm_bkg_dat.append(sd.focus[self.internal_standard][ind])

            srm_bkg_dat = np.concatenate(srm_bkg_dat)

            bkg_mean = H15_mean(srm_bkg_dat)
            bkg_std = H15_std(srm_bkg_dat)
            bkg_thresh = bkg_mean + thresh_n * bkg_std
        else:
            bkg_thresh = None

        if analyte is None:
            analyte = self.internal_standard
        elif analyte not in self.minimal_analytes:
                self.minimal_analytes.append(analyte)

        for d in tqdm(self.data, desc='AutoRange'):
            d.autorange(analyte, gwin, win, smwin,
                        conf, on_mult, off_mult,
                        d_mult, transform, bkg_thresh)

        return

    def find_expcoef(self, nsd_below=0., analyte=None, plot=False,
                     trimlim=None, autorange_kwargs={}):
        """
        Determines exponential decay coefficient for despike filter.

        Fits an exponential decay function to the washout phase of standards
        to determine the washout time of your laser cell. The exponential
        coefficient reported is `nsd_below` standard deviations below the
        fitted exponent, to ensure that no real data is removed.

        Parameters
        ----------
        nsd_below : float
            The number of standard deviations to subtract from the fitted
            coefficient when calculating the filter exponent.
        analyte : str
            The analyte to consider when determining the coefficient.
            Use high - concentration analyte for best estimates.
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
        if analyte is None:
            analyte = self.internal_standard
        elif analyte not in self.minimal_analytes:
                self.minimal_analytes.append(analyte)

        print('Calculating exponential decay coefficient\nfrom SRM {} washouts...'.format(analyte))

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
                s.autorange(**autorange_kwargs)

        trans = []
        times = []
        for v in self.stds:
            for trnrng in v.trnrng[-1::-2]:
                tr = normalise(v.focus[analyte][(v.Time > trnrng[0]) &
                               (v.Time < trnrng[1])])
                sm = np.apply_along_axis(np.nanmean, 1,
                                         rolling_window(tr, 3, pad=0))
                sm[0] = sm[1]
                trim = findtrim(sm, trimlim) + 2
                trans.append(normalise(tr[trim:]))
                times.append(np.arange(tr[trim:].size) *
                             np.diff(v.Time[1:3]))

        times = np.concatenate(times)
        times = np.round(times, 2)
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

        eeR2 = R2calc(trans, expfit(times, ep))

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=[6, 4])

            ax.scatter(times, trans, alpha=0.2, color='k', marker='x', zorder=-2)
            ax.scatter(ti, tr, alpha=1, color='k', marker='o')
            fitx = np.linspace(0, max(ti))
            ax.plot(fitx, expfit(fitx, ep), color='r', label='Fit')
            ax.plot(fitx, expfit(fitx, ep - nsd_below * np.diag(ecov)**.5, ),
                    color='b', label='Used')
            ax.text(0.95, 0.75,
                    ('y = $e^{%.2f \pm %.2f * x}$\n$R^2$= %.2f \nCoefficient: '
                     '%.2f') % (ep,
                                np.diag(ecov)**.5,
                                eeR2,
                                ep - nsd_below * np.diag(ecov)**.5),
                    transform=ax.transAxes, ha='right', va='top', size=12)
            ax.set_xlim(0, ax.get_xlim()[-1])
            ax.set_xlabel('Time (s)')
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Proportion of Signal')
            plt.legend()
            if isinstance(plot, str):
                fig.savefig(plot)

        self.expdecay_coef = ep - nsd_below * np.diag(ecov)**.5

        print('  {:0.2f}'.format(self.expdecay_coef[0]))

        return

    @_log
    def despike(self, expdecay_despiker=True, exponent=None, tstep=None,
                noise_despiker=True, win=3, nlim=12., exponentplot=False,
                autorange_kwargs={}):
        """
        Despikes data with exponential decay and noise filters.

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
        exponentplot : bool
            Whether or not to show a plot of the automatically determined
            exponential decay exponent.

        Returns
        -------
        None
        """
        if exponent is None:
            if not hasattr(self, 'expdecay_coef'):
                self.find_expcoef(plot=exponentplot,
                                  autorange_kwargs=autorange_kwargs)
            exponent = self.expdecay_coef
            time.sleep(0.2)

        for d in tqdm(self.data, desc='Despiking'):
            d.despike(expdecay_despiker, exponent, tstep,
                      noise_despiker, win, nlim)

        self.focus_stage = 'despiked'
        return

    # functions for background correction
    def get_background(self, n_min=10, n_max=None, focus_stage='despiked', filter=False, f_win=5, f_n_lim=3):
        """
        Extract all background data from all samples on universal time scale.
        Used by both 'polynomial' and 'weightedmean' methods.

        Parameters
        ----------
        n_min : int
            The minimum number of points a background region must
            have to be included in calculation.
        n_max : int
            The maximum number of points a background region must
            have to be included in calculation.
        filter : bool
            If true, apply a rolling filter to the isolated background regions
            to exclude regions with anomalously high values. If True, two parameters
            alter the filter's behaviour:
                f_win : int
                    The size of the rolling window
                f_n_lim : float
                    The number of standard deviations above the rolling mean
                    to set the threshold.
        Returns
        -------
        pandas.DataFrame object containing background data.
        """
        idx = pd.IndexSlice

        allbkgs = {'uTime': [],
                   'ns': []}

        for a in self.analytes:
            allbkgs[a] = []

        n0 = 0
        for s in self.data_dict.values():
            if sum(s.bkg) > 0:
                allbkgs['uTime'].append(s.uTime[s.bkg])
                allbkgs['ns'].append(enumerate_bool(s.bkg, n0)[s.bkg])
                n0 = allbkgs['ns'][-1][-1]
                for a in self.analytes:
                    allbkgs[a].append(s.data[focus_stage][a][s.bkg])

        allbkgs.update((k, np.concatenate(v)) for k, v in allbkgs.items())
        bkgs = pd.DataFrame(allbkgs)

        self.bkg = {}
        # extract background data from whole dataset
        if n_max is None:
            self.bkg['raw'] = bkgs.groupby('ns').filter(lambda x: len(x) > n_min)
        else:
            self.bkg['raw'] = bkgs.groupby('ns').filter(lambda x: (len(x) > n_min) & (len(x) < n_max))
        # calculate per - background region stats
        self.bkg['summary'] = self.bkg['raw'].groupby('ns').aggregate([np.mean, np.std, stderr])

        if filter:
            # calculate rolling mean and std from summary
            t = self.bkg['summary'].loc[:, idx[:, 'mean']]
            r = t.rolling(f_win).aggregate([np.nanmean, np.nanstd])
            # calculate upper threshold
            upper = r.loc[:, idx[:, :, 'nanmean']] + f_n_lim * r.loc[:, idx[:, :, 'nanstd']].values
            # calculate which are over upper threshold
            over = r.loc[:, idx[:, :, 'nanmean']] > np.roll(upper.values, 1, 0)
            # identify them
            ns_drop = over.loc[over.apply(any, 1),:].index.values
            # drop them from summary
            self.bkg['summary'].drop(ns_drop, inplace=True)
            # remove them from raw
            ind = np.ones(self.bkg['raw'].shape[0], dtype=bool)
            for ns in ns_drop:
                ind = ind & (self.bkg['raw'].loc[:, 'ns'] != ns)
            self.bkg['raw'] = self.bkg['raw'].loc[ind, :]
        return

    @_log
    def bkg_calc_weightedmean(self, analytes=None, weight_fwhm=300.,
                              n_min=20, n_max=None, cstep=None,
                              filter=False, f_win=7, f_n_lim=3):
        """
        Background calculation using a gaussian weighted mean.

        Parameters
        ----------
        analytes : str or array - like
        weight_fwhm : float
            The full - width - at - half - maximum of the gaussian used
            to calculate the weighted average.
        n_min : int
            Background regions with fewer than n_min points
            will not be included in the fit.
        cstep : float or None
            The interval between calculated background points.
        filter : bool
            If true, apply a rolling filter to the isolated background regions
            to exclude regions with anomalously high values. If True, two parameters
            alter the filter's behaviour:
                f_win : int
                    The size of the rolling window
                f_n_lim : float
                    The number of standard deviations above the rolling mean
                    to set the threshold.

        """
        if analytes is None:
            analytes = self.analytes
            self.bkg = {}
        elif isinstance(analytes, str):
            analytes = [analytes]

        self.get_background(n_min=n_min, n_max=n_max,
                            filter=filter,
                            f_win=f_win, f_n_lim=f_n_lim)

        # Gaussian - weighted average
        if 'calc' not in self.bkg.keys():
            # create time points to calculate background
            if cstep is None:
                cstep = weight_fwhm / 20
            # TODO: Modify  bkg_t to make sure none of the calculated
            # bkg points are during a sample collection.
            bkg_t = np.arange(self.bkg['raw']['uTime'].min(),
                              self.bkg['raw']['uTime'].max(),
                              cstep)
            self.bkg['calc'] = {}
            self.bkg['calc']['uTime'] = bkg_t

        for a in tqdm(analytes, desc='Calculating Analyte Backgrounds'):
            self.bkg['calc'][a] = weighted_average(self.bkg['raw'].uTime,
                                                   self.bkg['raw'].loc[:, a],
                                                   self.bkg['calc']['uTime'],
                                                   weight_fwhm)
        return

    @_log
    def bkg_calc_interp1d(self, analytes=None, kind=1, n_min=10, n_max=None, cstep=None,
                          filter=False, f_win=7, f_n_lim=3):
        """
        Background calculation using a 1D interpolation.

        scipy.interpolate.interp1D is used for interpolation.

        Parameters
        ----------
        analytes : str or array - like
        kind : str or int
            Integer specifying the order of the spline interpolation
            used, or string specifying a type of interpolation.
            Passed to `scipy.interpolate.interp1D`
        n_min : int
            Background regions with fewer than n_min points
            will not be included in the fit.
        cstep : float or None
            The interval between calculated background points.
        filter : bool
            If true, apply a rolling filter to the isolated background regions
            to exclude regions with anomalously high values. If True, two parameters
            alter the filter's behaviour:
                f_win : int
                    The size of the rolling window
                f_n_lim : float
                    The number of standard deviations above the rolling mean
                    to set the threshold.

        """
        if analytes is None:
            analytes = self.analytes
            self.bkg = {}
        elif isinstance(analytes, str):
            analytes = [analytes]

        self.get_background(n_min=n_min, n_max=n_max,
                            filter=filter,
                            f_win=f_win, f_n_lim=f_n_lim)

        if 'calc' not in self.bkg.keys():
            # create time points to calculate background
            if cstep is None:
                cstep = self.bkg['raw']['uTime'].ptp() / 100
            bkg_t = np.arange(self.bkg['summary']['uTime']['mean'].min(),
                              self.bkg['summary']['uTime']['mean'].max(),
                              cstep)

            self.bkg['calc'] = {}
            self.bkg['calc']['uTime'] = bkg_t

        d = self.bkg['summary']
        for a in tqdm(analytes, desc='Calculating Analyte Backgrounds'):
            imean = interp.interp1d(d.loc[:, ('uTime', 'mean')],
                                    d.loc[:, (a, 'mean')],
                                    kind=kind)
            istd = interp.interp1d(d.loc[:, ('uTime', 'mean')],
                                   d.loc[:, (a, 'std')],
                                   kind=kind)
            ise = interp.interp1d(d.loc[:, ('uTime', 'mean')],
                                  d.loc[:, (a, 'stderr')],
                                  kind=kind)
            self.bkg['calc'][a] = {'mean': imean(self.bkg['calc']['uTime']),
                                   'std': istd(self.bkg['calc']['uTime']),
                                   'stderr': ise(self.bkg['calc']['uTime'])}
        return

    @_log
    def bkg_subtract(self, analytes=None, errtype='stderr'):
        """
        Subtract calculated background from data.

        Must run bkg_calc first!
        """

        if analytes is None:
            analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        for d in tqdm(self.data_dict.values(), desc='Background Subtraction'):
            [d.bkg_subtract(a,
                            un.uarray(np.interp(d.uTime, self.bkg['calc']['uTime'], self.bkg['calc'][a]['mean']),
                                      np.interp(d.uTime, self.bkg['calc']['uTime'], self.bkg['calc'][a][errtype])),
                            ~d.sig) for a in self.analytes]
            d.setfocus('bkgsub')

        self.focus_stage = 'bkgsub'
        return

    @_log
    def bkg_plot(self, analytes=None, figsize=None, yscale='log', ylim=None, err='stderr', save=True):
        if not hasattr(self, 'bkg'):
            raise ValueError("Please run bkg_calc before attempting to\n" +
                             "plot the background.")

        if analytes is None:
            analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        if figsize is None:
            if len(self.samples) > 50:
                figsize = (len(self.samples) * 0.15, 5)
            else:
                figsize = (7.5, 5)

        fig = plt.figure(figsize=figsize)

        ax = fig.add_axes([.07, .1, .84, .8])

        for a in tqdm(analytes, desc='Plotting backgrounds:',
                      leave=False, total=len(analytes)):
            ax.scatter(self.bkg['raw'].uTime, self.bkg['raw'].loc[:, a],
                       alpha=0.2, s=3, c=self.cmaps[a],
                       lw=0.5)

            for i, r in self.bkg['summary'].iterrows():
                x = (r.loc['uTime', 'mean'] - r.loc['uTime', 'std'] * 2,
                     r.loc['uTime', 'mean'] + r.loc['uTime', 'std'] * 2)
                yl = [r.loc[a, 'mean'] - r.loc[a, err]] * 2
                yu = [r.loc[a, 'mean'] + r.loc[a, err]] * 2

                l_se = plt.fill_between(x, yl, yu, alpha=0.5, lw=0.5, color=self.cmaps[a], zorder=1)

            ax.plot(self.bkg['calc']['uTime'],
                    self.bkg['calc'][a]['mean'],
                    c=self.cmaps[a], zorder=2, label=a)
            ax.fill_between(self.bkg['calc']['uTime'],
                            self.bkg['calc'][a]['mean'] + self.bkg['calc'][a][err],
                            self.bkg['calc'][a]['mean'] - self.bkg['calc'][a][err],
                            color=self.cmaps[a], alpha=0.3, zorder=-1)

        if yscale == 'log':
            ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Background Counts')

        ax.set_title('Points = raw data; Bars = {:s}; Lines = Calculated Background; Envelope = Background {:s}'.format(err, err),
                     fontsize=10)

        ha, la = ax.get_legend_handles_labels()

        ax.legend(labels=la[:len(analytes)], handles=ha[:len(analytes)], bbox_to_anchor=(1, 1))

        # scale x axis to range ± 2.5%
        ax.set_xlim(self.bkg['raw']['uTime'].min(),
                    self.bkg['raw']['uTime'].max())

        for s, r in self.starttimes.iterrows():
            x = r.Dseconds
            ax.axvline(x, alpha=0.2, color='k', zorder=-1)
            ax.text(x, ax.get_ylim()[1], s, rotation=90, va='top', ha='left', zorder=-1)

        if save:
            fig.savefig(self.report_dir + '/background.png', dpi=200)

        return fig, ax

    # functions for calculating ratios
    @_log
    def ratio(self, internal_standard=None, focus='bkgsub'):
        """
        Calculates the ratio of all analytes to a single analyte.

        Parameters
        ----------
        internal_standard : str
            The name of the analyte to divide all other analytes
            by.
        focus : str
            The `focus` stage of the data used to calculating the
            ratios.

        Returns
        -------
        None
        """

        if internal_standard is not None:
            self.internal_standard = internal_standard
            if internal_standard not in self.minimal_analytes:
                self.minimal_analytes.append(internal_standard)

        for s in tqdm(self.data, desc='Ratio Calculation'):
            s.ratio(internal_standard=self.internal_standard, focus=focus)

        self.focus_stage = 'ratio'
        return

    # functions for identifying SRMs
    # def srm_id(self):
    #     """
    #     Asks the user to name the SRMs measured.
    #     """
    #     s = self.stds[0]
    #     fig = s.tplot(scale='log')
    #     display.clear_output(wait=True)
    #     display.display(fig)

    #     n0 = s.n

    #     def id(self, s):
    #         stdnms = []
    #         s.srm_rngs = {}
    #         for n in np.arange(s.n) + 1:
    #             fig, ax = s.tplot(scale='log')
    #             lims = s.Time[s.ns == n][[0, -1]]
    #             ax.axvspan(lims[0], lims[1],
    #                        color='r', alpha=0.2, lw=0)
    #             display.clear_output(wait=True)
    #             display.display(fig)
    #             stdnm = input('Name this standard: ')
    #             stdnms.append(stdnm)
    #             s.srm_rngs[stdnm] = lims
    #             plt.close(fig)
    #         return stdnms

    #     nms0 = id(self, s)

    #     if len(self.stds) > 1:
    #         ans = input(('Were all other SRMs measured in '
    #                      'the same sequence? [Y/n]'))
    #         if ans.lower() == 'n':
    #             for s in self.stds[1:]:
    #                 id(self, s)
    #         else:
    #             for s in self.stds[1:]:
    #                 if s.n == n0:
    #                     s.srm_rngs = {}
    #                     for n in np.arange(s.n) + 1:
    #                         s.srm_rngs[nms0[n-1]] = s.Time[s.ns == n][[0, -1]]
    #                 else:
    #                     _ = id(self, s)

    #     display.clear_output()

    #     # record srm_rng in self
    #     self.srm_rng = {}
    #     for s in self.stds:
    #         self.srm_rng[s.sample] = s.srm_rngs

    #     # make boolean identifiers in standard D
    #     for sn, rs in self.srm_rng.items():
    #         s = self.data_dict[sn]
    #         s.std_labels = {}
    #         for srm, rng in rs.items():
    #             s.std_labels[srm] = tuples_2_bool(rng, s.Time)

    #     self.srms_ided = True

    #     return

    def srm_id_auto(self, srms_used=['NIST610', 'NIST612', 'NIST614'], n_min=10):
        # compile mean and standard errors of samples
        for s in self.stds:
            stdtab = pd.DataFrame(columns=pd.MultiIndex.from_product([s.analytes, ['err', 'mean']]))
            stdtab.index.name = 'uTime'

            for n in range(1, s.n + 1):
                ind = s.ns == n
                if sum(ind) >= n_min:
                    for a in s.analytes:
                        aind = ind & ~np.isnan(nominal_values(s.focus[a]))
                        stdtab.loc[np.nanmean(s.uTime[s.ns == n]),
                                   (a, 'mean')] = np.nanmean(s.focus[a][aind])
                        stdtab.loc[np.nanmean(s.uTime[s.ns == n]),
                                   (a, 'err')] = np.nanstd(nominal_values(s.focus[a][aind])) / np.sqrt(sum(aind))

            # sort column multiindex
            stdtab = stdtab.loc[:, stdtab.columns.sort_values()]
            # sort row index
            stdtab.sort_index(inplace=True)

            # create 'SRM' column for naming SRM
            stdtab.loc[:, 'SRM'] = ''
            stdtab.loc[:, 'STD'] = s.sample

            s.stdtab = stdtab

        stdtab = pd.concat([s.stdtab for s in self.stds]).apply(pd.to_numeric, 1, errors='ignore')

        if not hasattr(self, 'srmdat'):
            # load SRM info
            srmdat = pd.read_csv(self.srmfile)
            srmdat.set_index('SRM', inplace=True)
            srmdat = srmdat.loc[srms_used]

            # isolate measured elements
            elements = np.unique([re.findall('[A-Z][a-z]{0,}', a)[0] for a in self.analytes])
            srmdat = srmdat.loc[srmdat.Item.apply(lambda x: any([a in x for a in elements]))]
            # label elements
            srmdat.loc[:, 'element'] = np.nan

            elnames = re.compile('([A-Z][a-z]{0,})')  # regex to ID element names
            for e in elements:
                ind = [e in elnames.findall(i) for i in srmdat.Item]
                srmdat.loc[ind, 'element'] = str(e)

            # convert to table in same format as stdtab
            self.srmdat = srmdat.dropna()

        srm_tab = self.srmdat.loc[:, ['Value', 'element']].reset_index().pivot(index='SRM', columns='element', values='Value')

        # Auto - ID STDs
        # 1. identify elements in measured SRMS with biggest range of values
        meas_tab = stdtab.loc[:, (slice(None), 'mean')]  # isolate means of standards
        meas_tab.columns = meas_tab.columns.droplevel(1)  # drop 'mean' column names
        meas_tab.columns = [re.findall('[A-Za-z]+', a)[0] for a in meas_tab.columns]  # rename to element names
        meas_tab = meas_tab.T.groupby(level=0).first().T  # remove duplicate columns

        ranges = nominal_values(meas_tab.apply(lambda a: np.ptp(a) / np.nanmean(a), 0))  # calculate relative ranges of all elements
        # (used as weights later)

        # 2. Work out which standard is which
        # normalise all elements between 0-1
        def normalise(a):
            a = nominal_values(a)
            if np.nanmin(a) < np.nanmax(a):
                return (a - np.nanmin(a)) / np.nanmax(a - np.nanmin(a))
            else:
                return np.ones(a.shape)

        nmeas = meas_tab.apply(normalise, 0)
        nmeas.replace(np.nan, 1, inplace=True)
        nsrm_tab = srm_tab.apply(normalise, 0)
        nsrm_tab.replace(np.nan, 1, inplace=True)

        for uT, r in nmeas.iterrows():  # for each standard...
            idx = abs((nsrm_tab - r) * ranges).sum(1)
            # calculate the absolute difference between the normalised elemental
            # values for each measured SRM and the SRM table. Each element is
            # multiplied by the relative range seen in that element (i.e. range / mean
            # measuerd value), so that elements with a large difference are given
            # more importance in identifying the SRM.
            # This produces a table, where wach row contains the difference between
            # a known vs. measured SRM. The measured SRM is identified as the SRM that
            # has the smallest weighted sum value.
            stdtab.loc[uT, 'SRM'] = srm_tab.index[idx == min(idx)].values[0]
        stdtab = stdtab.reset_index().set_index(['STD', 'SRM', 'uTime'])

        # combine to make SRM reference tables
        srmtabs = {}
        for a in self.analytes:
            el = re.findall('[A-Za-z]+', a)[0]

            sub = stdtab.loc[:, a]

            srmsub = self.srmdat.loc[self.srmdat.element == el, ['Value', 'Uncertainty']]

            srmtab = sub.join(srmsub)
            srmtab.columns = ['meas_err', 'meas_mean', 'srm_mean', 'srm_err']

            srmtabs[a] = srmtab

        self.srmtabs = pd.concat(srmtabs).apply(nominal_values).sort_index()
        return

    # def load_calibration(self, params=None):
    #     """
    #     Loads calibration from global .calib file.

    #     Parameters
    #     ----------
    #     params : str
    #         Specify the parameter file to load the calibration from.
    #         If None, it assumes that the parameters are already loaded
    #         (using `load_params`).

    #     Returns
    #     -------
    #     None
    #     """
    #     if isinstance(params, str):
    #         self.load_params(params)

    #     # load srm_rng and expand to standards
    #     self.srm_rng = self.params['calib']['srm_rng']

    #     # make boolean identifiers in standard D
    #     for s in self.stds:
    #         s.srm_rngs = self.srm_rng[s.sample]
    #         s.std_labels = {}
    #         for srm, rng in s.srm_rngs.items():
    #             s.std_labels[srm] = tuples_2_bool(rng, s.Time)
    #     self.srms_ided = True

    #     # load calib dict
    #     self.calib_dict = self.params['calib']['calib_dict']

    #     return

    # apply calibration to data
    @_log
    def calibrate(self, poly_n=0, analytes=None, drift_correct=False,
                  srm_errors=False, srms_used=['NIST610', 'NIST612', 'NIST614'],
                  n_min=10):
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
        #   USE ALL DATA OR AVERAGES?
        #   IF POLY_N > 0, STILL FORCE THROUGH ZERO IF ALL
        #   STDS ARE WITHIN ERROR OF EACH OTHER (E.G. AL/CA)
        # can store calibration function in self and use *coefs?
        # check for identified srms

        if analytes is None:
            analytes = self.analytes[self.analytes != self.internal_standard]
        elif isinstance(analytes, str):
            analytes = [analytes]

        if not hasattr(self, 'srmtabs'):
            self.srm_id_auto(srms_used, n_min)

        # calibration functions
        def calib_0(P, x):
            return x * P[0]

        def calib_n(P, x):
            # where p is a list of polynomial coefficients n items long,
            # corresponding to [..., 2nd, 1st, 0th] order coefficients
            return np.polyval(P, x)

        # wrapper for ODR fitting
        def odrfit(x, y, fn, coef0, sx=None, sy=None):
            dat = odr.RealData(x=x, y=y,
                               sx=sx, sy=sy)
            m = odr.Model(fn)
            mod = odr.ODR(dat, m, coef0)
            mod.run()
            return un.uarray(mod.output.beta, mod.output.sd_beta)

        # make container for calibration params
        if not hasattr(self, 'calib_params'):
            self.calib_params = pd.DataFrame(columns=self.analytes)

        # set up calibration functions
        if not hasattr(self, 'calib_fns'):
            self.calib_fns = {}

        for a in analytes:
            if poly_n == 0:
                self.calib_fns[a] = calib_0
                p0 = [1]
            else:
                self.calib_fns[a] = calib_n
                p0 = [1] * (poly_n - 1) + [0]

            # calculate calibrations
            if drift_correct:
                for n, g in self.srmtabs.loc[a, :].groupby(level=0):
                    if srm_errors:
                        p = odrfit(x=self.srmtabs.loc[a, 'meas_mean'].values,
                                   y=self.srmtabs.loc[a, 'srm_mean'].values,
                                   sx=self.srmtabs.loc[a, 'meas_err'].values,
                                   sy=self.srmtabs.loc[a, 'srm_err'].values,
                                   fn=self.calib_fns[a],
                                   coef0=p0)
                    else:
                        p = odrfit(x=self.srmtabs.loc[a, 'meas_mean'].values,
                                   y=self.srmtabs.loc[a, 'srm_mean'].values,
                                   sx=self.srmtabs.loc[a, 'meas_err'].values,
                                   fn=self.calib_fns[a],
                                   coef0=p0)
                    uTime = g.index.get_level_values('uTime').values.mean()
                    self.calib_params.loc[uTime, a] = p
            else:
                if srm_errors:
                    p = odrfit(x=self.srmtabs.loc[a, 'meas_mean'].values,
                               y=self.srmtabs.loc[a, 'srm_mean'].values,
                               sx=self.srmtabs.loc[a, 'meas_err'].values,
                               sy=self.srmtabs.loc[a, 'srm_err'].values,
                               fn=self.calib_fns[a],
                               coef0=p0)
                else:
                    p = odrfit(x=self.srmtabs.loc[a, 'meas_mean'].values,
                               y=self.srmtabs.loc[a, 'srm_mean'].values,
                               sx=self.srmtabs.loc[a, 'meas_err'].values,
                               fn=self.calib_fns[a],
                               coef0=p0)
                self.calib_params.loc[0, a] = p

        # apply calibration
        for d in tqdm(self.data, desc='Calibration'):
            try:
                d.calibrate(self.calib_fns, self.calib_params, analytes, drift_correct=drift_correct)
            except:
                print(d.sample + ' failed - probably first or last SRM\nwhich is outside interpolated time range.')

        self.focus_stage = 'calibrated'
    #     # save calibration parameters
    #     # self.save_calibration()
        return

    # data filtering
    # TODO:
    #   - implement 'filter sets'. Subsets dicts of samples at the 'analyse'
    #       level that are all filtered in the same way. Should be able to:
    #           a) Name the set
    #           b) Apply and on/off filters independently for each set.
    #           c) Each set should have a 'filter_status' function, listing
    #               the state of each filter for each analyte within the set.

    @_log
    def make_subset(self, samples=None, name=None):
        """
        Creates a subset of samples, which can be treated independently.

        Parameters
        ----------
        samples : str or array - like
            Name of sample, or list of sample names.
        name : (optional) str or number
            The name of the sample group. Defaults to n + 1, where n is
            the highest existing group number
        """
        if isinstance(samples, str):
            samples = [samples]

        not_exists = [s for s in samples if s not in self.subsets['All_Analyses']]
        if len(not_exists) > 0:
            raise ValueError(', '.join(not_exists) + ' not in the list of sample names.\nPlease check your sample names.\nNote: Sample names are stored in the .samples attribute of your analysis.')

        if name is None:
            name = max([-1] + [x for x in self.subsets.keys() if isinstance(x, int)]) + 1

        self._subset_names.append(name)

        if samples is not None:
            self.subsets[name] = samples
            for s in samples:
                try:
                    self.subsets['not_in_set'].remove(s)
                except ValueError:
                    pass

        self._has_subsets = True

        # for subset in np.unique(list(self.subsets.values())):
        #     self.subsets[subset] = sorted([k for k, v in self.subsets.items() if str(v) == subset])

        return name

    @_log
    def zeroscreen(self, focus_stage=None):
        """
        Remove all points containing data below zero (impossible)
        """
        if focus_stage is None:
            focus_stage = self.focus_stage

        for s in self.data_dict.values():
            ind = np.ones(len(s.Time), dtype=bool)
            for v in s.data[focus_stage].values():
                ind = ind & (nominal_values(v) > 0)

            for k in s.data[focus_stage].keys():
                s.data[focus_stage][k][~ind] = unc.ufloat(np.nan, np.nan)

        self.set_focus(focus_stage)

        return

    @_log
    def filter_threshold(self, analyte, threshold, filt=False,
                         samples=None, subset=None):
        """
        Applies a threshold filter to the data.

        Generates two filters above and below the threshold value for a
        given analyte.

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
        subset : str or number
            The subset of samples (defined by make_subset) you want to apply
            the filter to.

        Returns
        -------
        None
        """
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        self._add_minimal_analte(analyte)

        for s in tqdm(samples, desc='Threshold Filter'):
            self.data_dict[s].filter_threshold(analyte, threshold, filt=False)

    @_log
    def filter_distribution(self, analyte, binwidth='scott', filt=False,
                            transform=None, samples=None, subset=None,
                            min_data=10):
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
        min_data : int
            The minimum number of data points that should be considered by
            the filter. Default = 10.

        Returns
        -------
        None
        """
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        self._add_minimal_analte(analyte)

        for s in tqdm(samples, desc='Distribution Filter'):
            self.data_dict[s].filter_distribution(analyte, binwidth='scott',
                                                  filt=filt, transform=None,
                                                  min_data=min_data)

    @_log
    def filter_clustering(self, analytes, filt=False, normalise=True,
                          method='meanshift', include_time=False, samples=None,
                          sort=True, subset=None, min_data=10, **kwargs):
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
        samples : optional, array_like or None
            Which samples to apply this filter to. If None, applies to all
            samples.
        sort : bool
            Whether or not you want the cluster labels to
            be sorted by the mean magnitude of the signals
            they are based on (0 = lowest)
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


        TODO
        ----
        Make cluster sorting element specific.
        """
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        self._add_minimal_analte(analytes)

        for s in tqdm(samples, desc='Clustering Filter'):
            self.data_dict[s].filter_clustering(analytes=analytes, filt=filt,
                                                normalise=normalise,
                                                method=method,
                                                include_time=include_time,
                                                min_data=min_data,
                                                **kwargs)

    @_log
    def filter_correlation(self, x_analyte, y_analyte, window=None,
                           r_threshold=0.9, p_threshold=0.05, filt=True,
                           samples=None, subset=None):
        """
        Applies a correlation filter to the data.

        Calculates a rolling correlation between every `window` points of
        two analytes, and excludes data where their Pearson's R value is
        above `r_threshold` and statistically significant.

        Data will be excluded where their absolute R value is greater than
        `r_threshold` AND the p - value associated with the correlation is
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
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        self._add_minimal_analte([x_analyte, y_analyte])

        for s in tqdm(samples, desc='Correlation Filter'):
            self.data_dict[s].filter_correlation(x_analyte, y_analyte,
                                                 window=window,
                                                 r_threshold=r_threshold,
                                                 p_threshold=p_threshold,
                                                 filt=filt)

    @_log
    def filter_on(self, filt=None, analyte=None, samples=None, subset=None):
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
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        for s in samples:
            try:
                self.data_dict[s].filt.on(analyte, filt)
            except:
                warnings.warn("filt.on failure in sample " + s)

        self.filter_status(subset=subset)
        return

    @_log
    def filter_off(self, filt=None, analyte=None, samples=None, subset=None):
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
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        for s in samples:
            try:
                self.data_dict[s].filt.off(analyte, filt)
            except:
                warnings.warn("filt.off failure in sample " + s)

        self.filter_status(subset=subset)
        return

    @_log
    def filter_combine(self, name, filt_str, samples=None, subset=None):
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
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        for s in tqdm(samples, desc='Threshold Filter'):
            self.data_dict[s].filt.filter_new(name, filter_string)

    def filter_status(self, sample=None, subset=None, stds=False):
        s = ''
        if sample is None and subset is None:
            if not self._has_subsets:
                s += 'Subset: All Samples\n\n'
                s += self.data_dict[self.subsets['All_Samples'][0]].filt.__repr__()
            else:
                for n in sorted(self._subset_names):
                    s += 'Subset: ' + str(n) + '\n'
                    s += 'Samples: ' + ', '.join(self.subsets[n]) + '\n\n'
                    s += self.data_dict[self.subsets[n][0]].filt.__repr__()
                if len(self.subsets['not_in_set']) > 0:
                    s += '\nNot in Subset:\n'
                    s += 'Samples: ' + ', '.join(self.subsets['not_in_set']) + '\n\n'
                    s += self.data_dict[self.subsets['not_in_set'][0]].filt.__repr__()
            print(s)
            return

        elif sample is not None:
            s += 'Sample: ' + sample + '\n'
            s += self.data_dict[sample].filt.__repr__()
            print(s)
            return

        elif subset is not None:
            if isinstance(subset, str):
                subset = [subset]
            for n in subset:
                s += 'Subset: ' + str(n) + '\n'
                s += 'Samples: ' + ', '.join(self.subsets[n]) + '\n\n'
                s += self.data_dict[self.subsets[n][0]].filt.__repr__()
            print(s)
            return

    @_log
    def filter_clear(self, samples=None, subset=None):
        """
        Clears (deletes) all data filters.
        """
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        for s in samples:
            self.data_dict[s].filt.clear()

    # def filter_status(self, sample=None):
    #     if sample is not None:
    #         print(self.data_dict[sample].filt)
    #     else:

    @_log
    def fit_classifier(self, name, analytes, method, samples=None,
                       subset=None, filt=True, sort_by=0, **kwargs):
        """
        Create a clustering classifier based on all samples, or a subset.

        Parameters
        ----------
        name : str
            The name of the classifier.
        analytes : str or array-like
            Which analytes the clustering algorithm should consider.
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
        samples : array-like
            list of samples to consider. Overrides 'subset'.
        subset : str
            The subset of samples used to fit the classifier. Ignored if
            'samples' is specified.
        sort_by : int
            Which analyte the resulting clusters should be sorted
            by - defaults to 0, which is the first analyte.
        **kwargs :
            method-specific keyword parameters - see below.

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

        Returns
        -------
        name : str
        """
        # isolate data
        if samples is not None:
            subset = self.make_subset(samples)

        self.get_focus(subset=subset, filt=filt)

        # create classifer
        c = classifier(analytes,
                       sort_by)
        # fit classifier
        c.fit(data=self.focus,
              method=method,
              **kwargs)

        self.classifiers[name] = c

        return name

    @_log
    def apply_classifier(self, name, samples=None, subset=None):
        """
        Apply a clustering classifier based on all samples, or a subset.

        Parameters
        ----------
        name : str
            The name of the classifier to apply.
        subset : str
            The subset of samples to apply the classifier to.
        Returns
        -------
        name : str
        """
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        c = self.classifiers[name]
        labs = c.classifier.ulabels_

        for s in tqdm(samples, desc='Applying ' + name + ' classifier'):
            d = self.data_dict[s]
            try:
                f = c.predict(d.focus)
            except ValueError:
                # in case there's no data
                f = np.array([-2] * len(d.Time))
            for l in labs:
                ind = f == l
                d.filt.add(name=name + '_{:.0f}'.format(l),
                           filt=ind,
                           info=name + ' ' + c.method + ' classifier',
                           params=(c.analytes, c.method))
        return name


    # plot calibrations
    @_log
    def calibration_plot(self, analytes=None, datarange=True, loglog=False, save=True):
        """
        Plot the calibration lines between measured and known SRM values.

        Parameters
        ----------
        analytes : optional, array_like or str
            The analyte(s) to plot. Defaults to all analytes.
        datarange : boolean
            Whether or not to show the distribution of the measured data
            alongside the calibration curve.
        loglog : boolean
            Whether or not to plot the data on a log - log scale. This is
            useful if you have two low standards very close together,
            and want to check whether your data are between them, or
            below them.

        Returns
        -------
        (fig, axes)
            matplotlib objects
        """

        if analytes is None:
            analytes = [a for a in self.analytes if self.internal_standard not in a]

        n = len(analytes)
        if n % 3 is 0:
            nrow = n / 3
        else:
            nrow = n // 3 + 1

        axes = []

        if not datarange:
            fig = plt.figure(figsize=[12, 3 * nrow])
        else:
            fig = plt.figure(figsize=[14, 3 * nrow])
            if not hasattr(self, 'focus'):
                self.get_focus()

        gs = mpl.gridspec.GridSpec(nrows=int(nrow), ncols=3,
                                   hspace=0.3, wspace=0.3)

        i = 0
        for a in analytes:
            if not datarange:
                ax = fig.add_axes(gs[i].get_position(fig))
                axes.append(ax)
                i += 1
            else:
                f = 0.8
                p0 = gs[i].get_position(fig)
                p1 = [p0.x0, p0.y0, p0.width * f, p0.height]
                p2 = [p0.x0 + p0.width * f, p0.y0, p0.width * (1 - f), p0.height]
                ax = fig.add_axes(p1)
                axh = fig.add_axes(p2)
                axes.append((ax, axh))
                i += 1

            # plot calibration data
            ax.errorbar(self.srmtabs.loc[a, 'meas_mean'].values,
                        self.srmtabs.loc[a, 'srm_mean'].values,
                        xerr=self.srmtabs.loc[a, 'meas_err'].values,
                        yerr=self.srmtabs.loc[a, 'srm_err'].values,
                        color=self.cmaps[a], alpha=0.6,
                        lw=0, elinewidth=1, marker='o',
                        capsize=0, markersize=5)

            # work out axis scaling
            if not loglog:
                xlim, ylim = rangecalc(nominal_values(self.srmtabs.loc[a, 'meas_mean'].values),
                                       nominal_values(self.srmtabs.loc[a, 'srm_mean'].values),
                                       pad=0.1)
                xlim[0] = 0
                ylim[0] = 0
            else:
                xd = self.srmtabs.loc[a, 'meas_mean'][self.srmtabs.loc[a, 'meas_mean'] > 0].values
                yd = self.srmtabs.loc[a, 'srm_mean'][self.srmtabs.loc[a, 'srm_mean'] > 0].values

                xlim = [10**np.floor(np.log10(np.nanmin(xd))),
                        10**np.ceil(np.log10(np.nanmax(xd)))]
                ylim = [10**np.floor(np.log10(np.nanmin(yd))),
                        10**np.ceil(np.log10(np.nanmax(yd)))]

                # scale sanity checks
                if xlim[0] == xlim[1]:
                    xlim[0] = ylim[0]
                if ylim[0] == ylim[1]:
                    ylim[0] = xlim[0]

                ax.set_xscale('log')
                ax.set_yscale('log')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            # calculate line and R2
            if loglog:
                x = np.logspace(*np.log10(xlim), 100)
            else:
                x = np.array(xlim)

            coefs = self.calib_params[a][0]

            # plot line of best fit
            line = nominal_values(self.calib_fns[a](coefs, x))
            ax.plot(x, line, color=(0, 0, 0, 0.5), ls='dashed')

            if len(coefs) == 1:
                force_zero = True
            else:
                force_zero = False

            R2 = R2calc(self.srmtabs.loc[a, 'srm_mean'],
                        nominal_values(self.calib_fns[a](coefs, self.srmtabs.loc[a, 'meas_mean'])),
                        force_zero=force_zero)

            # labels
            if len(coefs) == 1:
                label = 'y = {:.2e} x'.format(coefs[0])
            else:
                label = r''
                for n, p in enumerate(coefs):
                    if len(coefs) - n - 1 == 0:
                        label += '{:.1e}'.format(p)
                    elif len(coefs) - n - 1 == 1:
                        label += '{:.1e} x\n+ '.format(p)
                    else:
                        label += '{:.1e}$ x^'.format(p) + '{' + '{:.0f}'.format(len(coefs) - n - 1) + '}$\n+ '

            if '{:.3f}'.format(R2) == '1.000':
                label += '\n$R^2$: >0.999'
            else:
                label += '\n$R^2$: {:.3f}'.format(R2)

            ax.text(.05, .95, pretty_element(a), transform=ax.transAxes,
                    weight='bold', va='top', ha='left', size=12)
            ax.set_xlabel('counts/counts ' + self.internal_standard)
            ax.set_ylabel('mol/mol ' + self.internal_standard)
            # write calibration equation on graph
            ax.text(0.98, 0.04, label, transform=ax.transAxes,
                    va='bottom', ha='right')

            # plot data distribution historgram alongside calibration plot
            if datarange:
                # isolate data
                meas = nominal_values(self.focus[a])
                meas = meas[~np.isnan(meas)]

                # check and set y scale
                if np.nanmin(meas) < ylim[0]:
                    if loglog:
                        mmeas = meas[meas > 0]
                        ylim[0] = 10**np.floor(np.log10(np.nanmin(mmeas)))
                    else:
                        ylim[0] = 0
                    ax.set_ylim(ylim)

                # hist
                if loglog:
                    bins = np.logspace(*np.log10(ylim), 30)
                else:
                    bins = np.linspace(*ylim, 30)

                axh.hist(meas, bins=bins, orientation='horizontal',
                         color=self.cmaps[a], lw=0.5, alpha=0.5)

                if loglog:
                    axh.set_yscale('log')
                axh.set_ylim(ylim)
                axh.set_xticks([])
                axh.set_yticklabels([])

        if save:
            fig.savefig(self.report_dir + '/calibration.pdf')

        return fig, axes

    # set the focus attribute for specified samples
    @_log
    def set_focus(self, focus_stage=None, samples=None, subset=None):
        """
        """
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        if focus_stage is None:
            focus_stage = self.focus_stage
        else:
            self.focus_stage = focus_stage

        for s in samples:
            self.data_dict[s].setfocus(focus_stage)

    # fetch all the data from the data objects
    def get_focus(self, filt=False, samples=None, subset=None):
        """
        Collect all data from all samples into a single array.
        Data from standards is not collected.

        Parameters
        ----------
        filt : str, dict or bool
            Either logical filter expression contained in a str,
            a dict of expressions specifying the filter string to
            use for each analyte or a boolean. Passed to `grab_filt`.
        samples : str or list
        subset : str or int

        Returns
        -------
        None
        """

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        t = 0
        self.focus = {'uTime': []}
        for a in self.analytes:
            self.focus[a] = []

        for sa in samples:
            s = self.data_dict[sa]
            self.focus['uTime'].append(s.uTime)
            ind = s.filt.grab_filt(filt)
            for a in self.analytes:
                tmp = s.focus[a].copy()
                tmp[~ind] = np.nan
                self.focus[a].append(tmp)

        for k, v in self.focus.items():
            self.focus[k] = np.concatenate(v)
        return

    # crossplot of all data
    @_log
    def crossplot(self, analytes=None, lognorm=True,
                  bins=25, filt=False, samples=None, subset=None, **kwargs):
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

        if subset is None:
            subset = 'All_Samples'

        self.get_focus(filt, samples, subset)

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

        # isolate nominal_values for all analytes
        focus = {k: nominal_values(v) for k, v in self.focus.items()}
        # determine units for all analytes
        udict = {a: unitpicker(np.nanmean(focus[a])) for a in analytes}
        # determine ranges for all analytes
        rdict = {a: (np.nanmin(focus[a] * udict[a][0]),
                     np.nanmax(focus[a] * udict[a][0])) for a in analytes}

        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            # get analytes
            ai = analytes[i]
            aj = analytes[j]

            # remove nan, apply multipliers
            pi = focus[ai][~np.isnan(focus[ai])] * udict[ai][0]
            pj = focus[aj][~np.isnan(focus[aj])] * udict[aj][0]

            # make plot
            if lognorm:
                axes[i, j].hist2d(pj, pi, bins,
                                  norm=mpl.colors.LogNorm(),
                                  cmap=plt.get_cmap(cmlist[i]))
                axes[j, i].hist2d(pi, pj, bins,
                                  norm=mpl.colors.LogNorm(),
                                  cmap=plt.get_cmap(cmlist[j]))
            else:
                axes[i, j].hist2d(pj, pi, bins,
                                  cmap=plt.get_cmap(cmlist[i]))
                axes[j, i].hist2d(pi, pj, bins,
                                  cmap=plt.get_cmap(cmlist[j]))

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

        # switch on alternating axes
        for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
            axes[j, i].xaxis.set_visible(True)
            for label in axes[j, i].get_xticklabels():
                label.set_rotation(90)
            axes[i, j].yaxis.set_visible(True)

        return fig, axes

    def crossplot_filters(self, filter_string, analytes=None,
                          samples=None, subset=None, filt=None):
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

        if samples is None:
            samples = self._get_samples(subset)

        # isolate relevant filters
        filts = self.data_dict[samples[0]].filt.components.keys()
        cfilts = [f for f in filts if filter_string in f]
        flab = re.compile('.*_(.*)$')  # regex to get filter name

        # aggregate data
        self.get_focus(subset=subset, filt=filt)

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

        cmlist = ['Blues', 'BuGn', 'BuPu', 'GnBu',
                  'Greens', 'Greys', 'Oranges', 'OrRd',
                  'PuBu', 'PuBuGn', 'PuRd', 'Purples',
                  'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']

        # isolate nominal_values for all analytes
        focus = {k: nominal_values(v) for k, v in self.focus.items()}
        # determine units for all analytes
        udict = {a: unitpicker(np.nanmean(focus[a])) for a in analytes}
        # determine ranges for all analytes
        rdict = {a: (np.nanmin(focus[a] * udict[a][0]),
                     np.nanmax(focus[a] * udict[a][0])) for a in analytes}

        for f in cfilts:
            self.get_focus(f, subset=subset)
            focus = {k: nominal_values(v) for k, v in self.focus.items()}
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

        axes[0, 0].legend(loc='upper left', title=filter_string)

        # switch on alternating axes
        for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
            axes[j, i].xaxis.set_visible(True)
            for label in axes[j, i].get_xticklabels():
                label.set_rotation(90)
            axes[i, j].yaxis.set_visible(True)

        return fig, axes

    # Plot traces
    @_log
    def trace_plots(self, analytes=None, samples=None, ranges=False,
                    focus=None, outdir=None, filt=None, scale='log',
                    figsize=[10, 4], stats=False, stat='nanmean',
                    err='nanstd', subset='All_Analyses'):
        """
        Plot analytes as a function of time.

        Parameters
        ----------
        analytes : optional, array_like or str
            The analyte(s) to plot. Defaults to all analytes.
        samples: optional, array_like or str
            The sample(s) to plot. Defaults to all samples.
        ranges : bool
            Whether or not to show the signal/backgroudn regions
            identified by 'autorange'.
        focus : str
            The focus 'stage' of the analysis to plot. Can be
            'rawdata', 'despiked':, 'signal', 'background',
            'bkgsub', 'ratios' or 'calibrated'.
        outdir : str
            Path to a directory where you'd like the plots to be
            saved. Defaults to 'reports/[focus]' in your data directory.
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
        if focus is None:
            focus = self.data[0].focus_stage
        if outdir is None:
            outdir = self.report_dir + '/' + focus
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        for s in tqdm(samples, desc='Drawing Plots'):
            f, a = self.data_dict[s].tplot(analytes=analytes, figsize=figsize,
                                           scale=scale, filt=filt,
                                           ranges=ranges, stats=stats,
                                           stat=stat, err=err, focus_stage=focus)
            # ax = fig.axes[0]
            # for l, u in s.sigrng:
            #     ax.axvspan(l, u, color='r', alpha=0.1)
            # for l, u in s.bkgrng:
            #     ax.axvspan(l, u, color='k', alpha=0.1)
            f.savefig(outdir + '/' + s + '_traces.pdf')
            # TODO: on older(?) computers raises
            # 'OSError: [Errno 24] Too many open files'
            plt.close(f)
        return

    # filter reports
    @_log
    def filter_reports(self, analytes, filt_str='all', samples=None,
                       outdir=None, subset=None):
        """
        Plot filter reports for all filters that contain ``filt_str``
        in the name.
        """
        if outdir is None:
            outdir = self.report_dir + '/filters/' + filt_str
            if not os.path.isdir(self.report_dir + '/filters'):
                os.mkdir(self.report_dir + '/filters')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        for s in tqdm(samples, desc='Drawing Plots'):
            self.data_dict[s].filt_report(filt=filt_str,
                                          analytes=analytes,
                                          savedir=outdir)
            # plt.close(fig)
        return

    def _stat_boostrap(self, analytes=None, filt=True,
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

    @_log
    def sample_stats(self, analytes=None, filt=True,
                     stats=['mean', 'std'],
                     eachtrace=True, csf_dict={}):
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
        stats : array_like
            list of functions or names of functions that take a single
            array_like input, and return a single statistic. Function
            should be able to cope with NaN values. Built-in functions:
                'mean': arithmetic mean
                'std': arithmetic standard deviation
                'se': arithmetic standard error
                'H15_mean': Huber mean (outlier removal)
                'H15_std': Huber standard deviation (outlier removal)
                'H15_se': Huber standard error (outlier removal)
        eachtrace : bool
            Whether to calculate the statistics for each analysis
            spot individually, or to produce per - sample means.
            Default is True.

        Returns
        -------
        None
            Adds dict to analyse object containing samples, analytes and
            functions and data.
        """
        if analytes is None:
            analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        self.stats = {}

        self.stats_calced = []
        stat_fns = {}

        stat_dict = {'mean': np.nanmean,
                     'std': np.nanstd,
                     'nanmean': np.nanmean,
                     'nanstd': np.nanstd,
                     'se': stderr,
                     'H15_mean': H15_mean,
                     'H15_std': H15_std,
                     'H15_se': H15_se}

        for s in stats:
            if isinstance(s, str):
                if s in stat_dict.keys():
                    self.stats_calced.append(s)
                    stat_fns[s] = stat_dict[s]
                if s in csf_dict.keys():
                    self.stats_calced.append(s)
                    exec(csf_dict[s])
                    stat_fns[s] = eval(s)
            elif callable(s):
                self.stats_calced.append(s.__name__)
                stat_fns[s.__name__] = s
                if not hasattr(self, 'custom_stat_functions'):
                    self.custom_stat_functions = ''
                self.custom_stat_functions += inspect.getsource(s) + '\n\n\n\n'

        # calculate stats for each sample
        for s in tqdm(self.samples, desc='Calculating Stats'):
            if self.srm_identifier not in s:
                self.data_dict[s].sample_stats(analytes, filt=filt,
                                               stat_fns=stat_fns,
                                               eachtrace=eachtrace)

                self.stats[s] = self.data_dict[s].stats

    @_log
    def ablation_times(self, samples=None, subset=None):

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        ats = {}

        for s in samples:
            ats[s] = self.data_dict[s].ablation_times()

        frames = []
        for s in samples:
            d = ats[s]
            td = pd.DataFrame.from_dict(d, orient='index')
            td.columns = ['Time']
            frames.append(td)
        out = pd.concat(frames, keys=samples)
        out.index.names = ['sample', 'rep']
        return out

    # function for visualising sample statistics
    @_log
    def statplot(self, analytes=None, samples=None, figsize=None,
                 stat='nanmean', err='nanstd', subset=None):
        if not hasattr(self, 'stats'):
            self.sample_stats()

        if analytes is None:
                analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        analytes = [a for a in analytes if a !=
                    self.data[0].internal_standard]

        if figsize is None:
            figsize = (1.5 * len(self.stats), 3 * len(analytes))

        fig, axs = plt.subplots(len(analytes), 1, figsize=figsize)

        for ax, an in zip(axs, analytes):
            i = 0
            stab = self.getstats()
            m, u = unitpicker(np.percentile(stab.loc[:, an].dropna(), 25), 0.1)
            for s in samples:
                if self.srm_identifier not in s:
                    d = self.stats[s]
                    if d[stat].ndim == 2:
                        n = d[stat].shape[-1]
                        x = np.linspace(i - .1 * n / 2, i + .1 * n / 2, n)
                    else:
                        x = [i]
                    a_ind = d['analytes'] == an

                    # plot individual ablations with error bars
                    ax.errorbar(x, d[stat][a_ind][0] * m,
                                yerr=d[err][a_ind][0] * m,
                                marker='o', color=self.cmaps[an],
                                lw=0, elinewidth=1)

                    ax.set_ylabel('%s / %s (%s )' % (pretty_element(an),
                                                     pretty_element(self.data[0].internal_standard),
                                                     u))

                    # plot whole - sample mean
                    if len(x) > 1:
                        # mean calculation with error propagation?
                        # umean = un.uarray(d[stat][a_ind][0] * m, d[err][a_ind][0] * m).mean()
                        # std = un.std_devs(umean)
                        # mean = un.nominal_values(umean)
                        mean = np.nanmean(d[stat][a_ind][0] * m)
                        std = np.nanstd(d[stat][a_ind][0] * m)
                        ax.plot(x, [mean] * len(x), c=self.cmaps[an], lw=2)
                        ax.fill_between(x, [mean + std] * len(x),
                                        [mean - std] * len(x),
                                        lw=0, alpha=0.2, color=self.cmaps[an])

                    # highlight each sample
                    if i % 2 == 1:
                        ax.axvspan(i - .5, i + .5, color=(0, 0, 0, 0.05), lw=0)

                    i += 1

            ax.set_xticks(np.arange(0, len(self.stats)))
            ax.set_xlim(-0.5, len(self.stats) - .5)

            ax.set_xticklabels(samples)

        return fig, ax

    @_log
    def getstats(self, save=True, filename=None, samples=None, subset=None, ablation_time=False):
        """
        Return pandas dataframe of all sample statistics.
        """
        slst = []

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        for s in self.stats_calced:
            for nm in [n for n in samples if self.srm_identifier
                       not in n]:
                if self.stats[nm][s].ndim == 2:
                    # make multi - index
                    reps = np.arange(self.stats[nm][s].shape[-1])
                    ss = np.array([s] * reps.size)
                    nms = np.array([nm] * reps.size)
                    # make sub - dataframe
                    stdf = pd.DataFrame(self.stats[nm][s].T,
                                        columns=self.stats[nm]['analytes'],
                                        index=[ss, nms, reps])
                    stdf.index.set_names(['statistic', 'sample', 'rep'],
                                         inplace=True)
                else:
                    stdf = pd.DataFrame(self.stats[nm][s],
                                        index=self.stats[nm]['analytes'],
                                        columns=[[s], [nm]]).T

                    stdf.index.set_names(['statistic', 'sample'],
                                         inplace=True)
                slst.append(stdf)
        out = pd.concat(slst)

        if ablation_time:
            ats = self.ablation_times(samples=samples, subset=subset)
            ats['statistic'] = 'nanmean'
            ats.set_index('statistic', append=True, inplace=True)
            ats = ats.reorder_levels(['statistic', 'sample', 'rep'])

            out = out.join(ats)

        out.drop(self.internal_standard, 1, inplace=True)

        if save:
            if filename is None:
                filename = 'stat_export.csv'
            out.to_csv(self.export_dir + '/' + filename)

        self.stats_df = out

        return out

    # raw data export function
    def _minimal_export_traces(self, outdir=None, analytes=None,
                               samples=None, subset='All_Analyses'):
        """
        Used for exporting minimal dataset. DON'T USE.
        """
        if analytes is None:
            analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        focus_stage = 'rawdata'
        ud = 'counts'

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        for s in samples:
            d = self.data_dict[s].data[focus_stage]
            out = {}

            for a in analytes:
                out[a] = d[a]

            out = pd.DataFrame(out, index=self.data_dict[s].Time)
            out.index.name = 'Time'

            header = ['# Minimal Reproduction Dataset Exported from LATOOLS on %s' %
                      (time.strftime('%Y:%m:%d %H:%M:%S')),
                      "# Analysis described in '../analysis.log'",
                      '# Run latools.reproduce to import analysis.',
                      '#',
                      '# Sample: %s' % (s),
                      '# Analysis Time: ' + self.data_dict[s].meta['date']]

            header = '\n'.join(header) + '\n'

            csv = out.to_csv()

            with open('%s/%s.csv' % (outdir, s), 'w') as f:
                f.write(header)
                f.write(csv)
        return

    @_log
    def export_traces(self, outdir=None, focus_stage=None, analytes=None,
                      samples=None, subset='All_Analyses', filt=False):
        """
        Function to export raw data.

        Parameters
        ----------
        outdir : str
        focus_stage : str
            The name of the analysis stage desired:
                'rawdata': raw data, loaded from csv file.
                'despiked': despiked data.
                'signal'/'background': isolated signal and background data.
                    Created by self.separate, after signal and background
                    regions have been identified by self.autorange.
                'bkgsub': background subtracted data, created by
                    self.bkg_correct
                'ratios': element ratio data, created by self.ratio.
                'calibrated': ratio data calibrated to standards, created by
                    self.calibrate.
            Defaults to the most recent stage of analysis.
        analytes : str or array - like
            Either a single analyte, or list of analytes to export.
            Defaults to all analytes.
        samples : str or array - like
            Either a single sample name, or list of samples to export.
            Defaults to all samples.
        filt : str, dict or bool
            Either logical filter expression contained in a str,
            a dict of expressions specifying the filter string to
            use for each analyte or a boolean. Passed to `grab_filt`.
        """
        if analytes is None:
            analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        if focus_stage is None:
            focus_stage = self.data[0].focus_stage

        if outdir is None:
            outdir = self.export_dir

        ud = {'rawdata': 'counts',
              'despiked': 'counts',
              'bkgsub': 'background corrected counts',
              'ratios': 'counts/count {:s}',
              'calibrated': 'mol/mol {:s}'}
        if focus_stage in ['ratios', 'calibrated']:
            ud[focus_stage] = ud[focus_stage].format(self.internal_standard)

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        for s in samples:
            d = self.data_dict[s].data[focus_stage]
            ind = self.data_dict[s].filt.grab_filt(filt)
            out = {}

            for a in analytes:
                out[a] = nominal_values(d[a][ind])
                if focus_stage not in ['rawdata', 'despiked']:
                    out[a + '_std'] = std_devs(d[a][ind])
                    out[a + '_std'][out[a + '_std'] == 0] = np.nan

            out = pd.DataFrame(out, index=self.data_dict[s].Time[ind])
            out.index.name = 'Time'

            header = ['# Sample: %s' % (s),
                      '# Data Exported from LATOOLS on %s' %
                      (time.strftime('%Y:%m:%d %H:%M:%S')),
                      '# Processed using %s configuration' % (self.config),
                      '# Analysis Stage: %s' % (focus_stage),
                      '# Unit: %s' % ud[focus_stage]]

            header = '\n'.join(header) + '\n'

            csv = out.to_csv()

            with open('%s/%s_%s.csv' % (outdir, s, focus_stage), 'w') as f:
                f.write(header)
                f.write(csv)
        return

    def minimal_export(self, target_analytes=None, override=False, path=None):
        """
        Exports a analysis parameters, standard info and a minimal dataset,
        which can be imported by another user.
        """
        if target_analytes is None:
            target_analytes = self.analytes
        if isinstance(target_analytes, str):
            target_analytes = [target_analytes]

        if any([a not in target_analytes for a in self.minimal_analytes]) and override:
            excluded = [a for a in self.minimal_analytes if a not in target_analytes]
            warnings.warn('\n\nYou have chosen to specify particular analytes,\n' +
                          'and override the default minimal_export function.\n' +
                          'The following analytes are used in data\n' +
                          'processing, but not included in this export:\n' +
                          '  {}\n'.format(excluded) +
                          'Export will continue, but we cannot guarantee\n' +
                          'that the analysis will be reproducible. You MUST\n' +
                          'check to make sure the analysis can be duplicated\n' +
                          "by the `latools.reproduce` function before\n" +
                          'distributing this dataset.')
        else:
            target_analytes = np.unique(np.concatenate([target_analytes, self.minimal_analytes]))

        # set up data path
        if path is None:
            path = self.export_dir + '/minimal_export/'
        if not os.path.isdir(path):
            os.mkdir(path)

        # export data
        self._minimal_export_traces(path + '/data', analytes=target_analytes)

        # define analysis_log header
        log_header = ['# Minimal Reproduction Dataset Exported from LATOOLS on %s' %
                      (time.strftime('%Y:%m:%d %H:%M:%S')),
                      'data_folder :: ./data/',
                      'srm_table :: ./srm.table',
                      ]

        # save custom functions (of defined)
        if hasattr(self, 'custom_stat_functions'):
            with open(path + '/custom_stat_fns.py', 'w') as f:
                f.write(self.custom_stat_functions)
            log_header.append('custom_stat_functions :: ./custom_stat_fns.py')

        log_header.append('# Analysis Log Start: \n')

        # format sample_stats correctly
        lss = [(i, l) for i, l in enumerate(self.log) if 'sample_stats' in l]
        rep = re.compile("(.*'stats': )(\[.*?\])(.*)")
        for i, l in lss:
            self.log[i] = rep.sub(r'\1' + str(self.stats_calced) + r'\3', l)

        # save log
        with open(path + '/analysis.log', 'w') as f:
            f.write('\n'.join(log_header))
            f.write('\n'.join(self.log))

        # export srm table
        els = np.unique([re.sub('[0-9]', '', a) for a in target_analytes])
        srmdat = []
        for e in els:
            srmdat.append(self.srmdat.loc[self.srmdat.element == e, :])
        srmdat = pd.concat(srmdat)

        with open(path + '/srm.table', 'w') as f:
            f.write(srmdat.to_csv())


def reproduce(log_file, plotting=False, data_folder=None, srm_table=None, custom_stat_functions=None):
    """
    Reproduce a previous analysis exported with `latools.minimal_export`
    """
    dirname = os.path.dirname(log_file) + '/'

    with open(log_file, 'r') as f:
        rlog = f.readlines()

    hashind = [i for i, n in enumerate(rlog) if '#' in n]

    pathread = re.compile('(.*) :: (.*)\n')
    paths = dict([pathread.match(l).groups() for l in rlog[hashind[0] + 1:hashind[-1]] if pathread.match(l)])

    if data_folder is None:
        data_folder = dirname + paths['data_folder']
    if srm_table is None:
        srm_table = dirname + paths['srm_table']
    if custom_stat_functions is None and 'custom_stat_functions' in paths.keys():
        # load custom functions as a dict
        with open(dirname + paths['custom_stat_functions'], 'r') as f:
            csf = f.read()

        fname = re.compile('def (.*)\(.*')

        csfs = {}
        for c in csf.split('\n\n\n\n'):
            if fname.match(c):
                csfs[fname.match(c).groups()[0]] = c

    # reproduce analysis
    logread = re.compile('([a-z_]+) :: args=(\(.*\)) kwargs=(\{.*\})')

    init_kwargs = eval(logread.match(rlog[hashind[1] + 1]).groups()[-1])
    init_kwargs['config'] = 'REPRODUCE'
    init_kwargs['data_folder'] = data_folder

    dat = analyse(**init_kwargs)

    dat.srmdat = pd.read_csv(srm_table).set_index('SRM')
    print('SRM values loaded from: {}'.format(srm_table))

    # rest of commands
    for l in rlog[hashind[1] + 2:]:
        fname, args, kwargs = logread.match(l).groups()
        if 'sample_stats' in fname:
            dat.sample_stats(*eval(args), csf_dict=csfs, **eval(kwargs))
        elif 'plot' not in fname.lower():
            getattr(dat, fname)(*eval(args), **eval(kwargs))
        elif plotting:
            getattr(dat, fname)(*eval(args), **eval(kwargs))
        else:
            pass

    return dat


analyze = analyse  # for the yanks


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

    # despiking functions
    def expdecay_despiker(self, exponent=None, tstep=None):
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
        #     if not hasattr(self, 'expdecay_coef'):
        #         self.find_expcoef()
        #     exponent = self.expdecay_coef
        if tstep is None:
            tstep = np.diff(self.Time[:2])
        if not hasattr(self, 'despiked'):
            self.data['despiked'] = {}
        for a, vo in self.focus.items():
            v = vo.copy()
            if 'time' not in a.lower():
                lowlim = np.roll(v * np.exp(tstep * exponent), 1)
                over = np.roll(lowlim > v, -1)

                if sum(over) > 0:
                    # get adjacent values to over - limit values
                    # calculate replacement values
                    neighbours = []
                    fixend = False
                    oover = over.copy()
                    if oover[0]:
                        neighbours.append([v[1], np.nan])
                        oover[0] = False
                    if oover[-1]:
                        oover[-1] = False
                        fixend = True
                    neighbours.append(np.hstack([v[np.roll(oover, -1)][:, np.newaxis],
                                                 v[np.roll(oover, 1)][:, np.newaxis]]))
                    if fixend:
                        neighbours.append([v[-2], np.nan])

                    neighbours = np.vstack(neighbours)

                    replacements = np.apply_along_axis(np.nanmean, 1, neighbours)
                    # and subsitite them in
                    v[over] = replacements
                self.data['despiked'][a] = v
        self.setfocus('despiked')
        return

    # spike filter
    def noise_despiker(self, win=3, nlim=12.):
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
        if not hasattr(self, 'despiked'):
            self.data['despiked'] = {}
        for a, vo in self.focus.items():
            v = vo.copy()
            if 'time' not in a.lower():
                # calculate rolling mean using convolution
                kernel = np.ones(win) / win
                rmean = np.convolve(v, kernel, 'same')

            # with warnings.catch_warnings():
                # to catch 'empty slice' warnings
                # warnings.simplefilter("ignore", category=RuntimeWarning)
                # rmean = \
                #     np.apply_along_axis(np.nanmean, 1,
                #                         rolling_window(v, win,
                #                                             pad=np.nan))
                # rmean = \
                #     np.apply_along_axis(np.nanmean, 1,
                #                         rolling_window(v, win,
                #                                             pad=np.nan))
                # calculate rolling standard deviation
                # (count statistics, so **0.5)
                rstd = rmean**0.5

                # find which values are over the threshold
                # (v > rmean + nlim * rstd)
                over = v > rmean + nlim * rstd
                if sum(over) > 0:
                    # get adjacent values to over - limit values
                    neighbours = \
                        np.hstack([v[np.roll(over, -1)][:, np.newaxis],
                                   v[np.roll(over, 1)][:, np.newaxis]])
                    # calculate the mean of the neighbours
                    replacements = np.apply_along_axis(np.nanmean, 1,
                                                       neighbours)
                    # and subsitite them in
                    v[over] = replacements
                self.data['despiked'][a] = v
        self.setfocus('despiked')
        return

    @_log
    def despike(self, expdecay_despiker=True, exponent=None, tstep=None,
                noise_despiker=True, win=3, nlim=12.):
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

        Returns
        -------
        None
        """
        if noise_despiker:
            self.noise_despiker(win, nlim)
        if expdecay_despiker:
            self.expdecay_despiker(exponent, tstep)

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
                  transform='log', bkg_thresh=None):
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
                warnings.warn(("\nSample {:s}: ".format(self.sample) +
                               "Transition identification at " +
                               "{:.1f} failed.".format(self.Time[z]) +
                               "\nPlease check the data plots and make sure " +
                               "everything is OK.\n(Run " +
                               "'trace_plots(ranges=True)'\n\n"))
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
        self.ns = np.zeros(self.Time.size)
        n = 1
        for i in range(len(self.sig) - 1):
            if self.sig[i]:
                self.ns[i] = n
            if self.sig[i] and ~self.sig[i + 1]:
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
        self.bkg[[0, -1]] = False
        bkgr = self.Time[self.bkg ^ np.roll(self.bkg, -1)]
        self.bkgrng = np.reshape(bkgr, [bkgr.size // 2, 2])

        self.sig[[0, -1]] = False
        sigr = self.Time[self.sig ^ np.roll(self.sig, 1)]
        self.sigrng = np.reshape(sigr, [sigr.size // 2, 2])

        self.trn[[0, -1]] = False
        trnr = self.Time[self.trn ^ np.roll(self.trn, 1)]
        self.trnrng = np.reshape(trnr, [trnr.size // 2, 2])

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
    def calibrate(self, calib_fns, calib_params, analytes=None, drift_correct=False):
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
            if drift_correct:
                P = self.drift_params(calib_params, a)
            else:
                P = calib_params[a].values[0]

            self.data['calibrated'][a] = \
                calib_fns[a](P,
                             self.data['ratios'][a])

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
        figure
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
                sts = self.stats[sig][0].size
                if sts > 1:
                    for n in np.arange(self.n):
                        n_ind = ind & (self.ns == n + 1)
                        if sum(n_ind) > 2:
                            x = [self.Time[n_ind][0], self.Time[n_ind][-1]]
                            y = [self.stats[sig][self.stats['analytes'] == a][0][n]] * 2

                            yp = ([self.stats[sig][self.stats['analytes'] == a][0][n] +
                                  self.stats[err][self.stats['analytes'] == a][0][n]] * 2)
                            yn = ([self.stats[sig][self.stats['analytes'] == a][0][n] -
                                  self.stats[err][self.stats['analytes'] == a][0][n]] * 2)

                            ax.plot(x, y, color=self.cmap[a], lw=2)
                            ax.fill_between(x + x[::-1], yp + yn,
                                            color=self.cmap[a], alpha=0.4,
                                            linewidth=0)
                else:
                    x = [self.Time[0], self.Time[-1]]
                    y = [self.stats[sig][self.stats['analytes'] == a][0]] * 2
                    yp = ([self.stats[sig][self.stats['analytes'] == a][0] +
                          self.stats[err][self.stats['analytes'] == a][0]] * 2)
                    yn = ([self.stats[sig][self.stats['analytes'] == a][0] -
                          self.stats[err][self.stats['analytes'] == a][0]] * 2)

                    ax.plot(x, y, color=self.cmap[a], lw=2)
                    ax.fill_between(x + x[::-1], yp + yn, color=self.cmap[a],
                                    alpha=0.4, linewidth=0)

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
            analytes = [a for a in self.analytes
                        if a != self.internal_standard]

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

        cmlist = ['Blues', 'BuGn', 'BuPu', 'GnBu',
                  'Greens', 'Greys', 'Oranges', 'OrRd',
                  'PuBu', 'PuBuGn', 'PuRd', 'Purples',
                  'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']

        # isolate nominal_values for all analytes
        focus = {k: nominal_values(v) for k, v in self.focus.items()}
        # determine units for all analytes
        udict = {a: unitpicker(np.nanmean(focus[a])) for a in analytes}
        # determine ranges for all analytes
        rdict = {a: (np.nanmin(focus[a] * udict[a][0]),
                     np.nanmax(focus[a] * udict[a][0])) for a in analytes}

        for f in cfilts:
            ind = self.filt.grab_filt(f)
            focus = {k: nominal_values(v[ind]) for k, v in self.focus.items()}
            lab = flab.match(f).groups()[0]
            axes[0,0].scatter([],[],s=10,label=lab)

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

                    m, u = unitpicker(np.nanmax(y))

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


class filt(object):
    """
    Container for creating, storing and selecting data filters.

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
    clean
    clear
    get_components
    get_info
    grab_filt
    make
    make_fromkey
    make_keydict
    off
    on
    remove
    """
    def __init__(self, size, analytes):
        self.size = size
        self.analytes = analytes
        self.index = {}
        self.sets = {}
        self.maxset = -1
        self.components = {}
        self.info = {}
        self.params = {}
        self.keys = {}
        self.n = 0
        self.switches = {}
        for a in self.analytes:
            self.switches[a] = {}

    def __repr__(self):
        apad = max([len(a) for a in self.analytes] + [7])
        astr = '{:' + '{:.0f}'.format(apad) + 's}'
        leftpad = max([len(s) for s
                       in self.components.keys()] + [11]) + 2

        out = '{string:{number}s}'.format(string='n', number=3)
        out += '{string:{number}s}'.format(string='Filter Name', number=leftpad)
        for a in self.analytes:
            out += astr.format(a)
        out += '\n'

        reg = re.compile('[0-9]+_(.*)')
        for n, t in self.index.items():
            out += '{string:{number}s}'.format(string=str(n), number=3)
            tn = reg.match(t).groups()[0]
            out += '{string:{number}s}'.format(string=str(tn), number=leftpad)
            for a in self.analytes:
                out += astr.format(str(self.switches[a][t]))
            out += '\n'
        return(out)

    def add(self, name, filt, info='', params=(), setn=None):
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

        iname = '{:.0f}_'.format(self.n) + name
        self.index[self.n] = iname

        if setn is None:
            setn = self.maxset + 1
        self.maxset = setn

        if setn not in self.sets.keys():
            self.sets[setn] = [iname]
        else:
            self.sets[setn].append(iname)

        ## self.keys is not added to?

        self.components[iname] = filt
        self.info[iname] = info
        self.params[iname] = params
        for a in self.analytes:
            self.switches[a][iname] = False
        self.n += 1
        return

    def remove(self, name=None, setn=None):
        """
        Remove filter.

        Parameters
        ----------
        name : str
            name of the filter to remove
        setn : int or True
            int: number of set to remove
            True: remove all filters in set that 'name' belongs to

        Returns
        -------
        None
        """
        if isinstance(name, int):
            name = self.index[name]

        if setn is not None:
            name = self.sets[setn]
            del self.sets[setn]
        elif isinstance(name, (int, str)):
            name = [name]

        if setn is True:
            for n in name:
                for k, v in self.sets.items():
                    if n in v:
                        name.append([m for m in v if m != n])

        for n in name:
            for k, v in self.sets.items():
                if n in v:
                    self.sets[k] = [m for m in v if n != m]
            del self.components[n]
            del self.info[n]
            del self.params[n]
            del self.keys[n]
            for a in self.analytes:
                del self.switches[a][n]
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
        self.index = {}
        self.sets = {}
        self.maxset = -1
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
        filt : optional. int, str or array_like
            Name/number or iterable names/numbers of filters.

        Returns
        -------
        None
        """
        if isinstance(analyte, str):
            analyte = [analyte]
        if isinstance(filt, (int, float)):
            filt = [filt]
        elif isinstance(filt, str):
            filt = self.fuzzmatch(filt, multi=True)

        if analyte is None:
            analyte = self.analytes
        if filt is None:
            filt = list(self.index.values())

        for a in analyte:
            for f in filt:
                if isinstance(f, (int, float)):
                    f = self.index[int(f)]

                try:
                    self.switches[a][f] = True
                except KeyError:
                    f = self.fuzzmatch(f, multi=False)
                    self.switches[a][f] = True

                # for k in self.switches[a].keys():
                #     if f in k:
                #         self.switches[a][k] = True
        return

    def off(self, analyte=None, filt=None):
        """
        Turn off specified filter(s) for specified analyte(s).

        Parameters
        ----------
        analyte : optional, str or array_like
            Name or list of names of analytes.
            Defaults to all analytes.
        filt : optional. int, list of int or str
            Number(s) or partial string that corresponds to filter name(s).

        Returns
        -------
        None
        """
        if isinstance(analyte, str):
            analyte = [analyte]
        if isinstance(filt, (int, float)):
            filt = [filt]
        elif isinstance(filt, str):
            filt = self.fuzzmatch(filt, multi=True)

        if analyte is None:
            analyte = self.analytes
        if filt is None:
            filt = list(self.index.values())

        for a in analyte:
            for f in filt:
                if isinstance(f, int):
                    f = self.index[f]

                try:
                    self.switches[a][f] = False
                except KeyError:
                    f = self.fuzzmatch(f, multi=False)
                    self.switches[a][f] = False

                # for k in self.switches[a].keys():
                #     if f in k:
                #         self.switches[a][k] = False
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
        if analyte is None:
            analyte = self.analytes
        elif isinstance(analyte, str):
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

    def fuzzmatch(self, fuzzkey, multi=False):
        """
        Identify a filter by fuzzy string matching.

        Partial ('fuzzy') matching performed by `fuzzywuzzy.fuzzy.ratio`

        Parameters
        ----------
        fuzzkey : str
            A string that partially matches one filter name more than the others.

        Returns
        -------
        The name of the most closely matched filter.
        """

        keys, ratios = np.array([(f, fuzz.ratio(fuzzkey, f)) for f in self.components.keys()]).T
        ratios = ratios.astype(float)
        mratio = ratios.max()

        if multi:
            return keys[ratios == mratio]
        else:
            if sum(ratios == mratio) == 1:
                return keys[ratios == mratio][0]
            else:
                raise ValueError("\nThe filter key provided ('{:}') matches two or more filter names equally well:\n".format(fuzzkey) + ', '.join(keys[ratios == mratio]) + "\nPlease be more specific!")

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
                return "self.components['" + self.fuzzmatch(match.group(0)) + "']"

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

    def grab_filt(self, filt, analyte=None):
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
            except KeyError:
                print(("\n\n***Filter key invalid. Please consult "
                       "manual and try again."))
        elif isinstance(filt, dict):
            try:
                ind = self.make_fromkey(filt[analyte])
            except ValueError:
                print(("\n\n***Filter key invalid. Please consult manual "
                       "and try again.\nOR\nAnalyte missing from filter "
                       "key dict."))
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
    #         fig, ax = plt.subplots(1, 1)
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

    #         ym = np.mean([yu, yl])

    #         ax.text(ax.get_xlim()[1] * 1.01, ym, f, ha='left')

    #         yl += yd * 1.2

    #     return(ax)


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
            A string defining the clustering method used:
            kmeans : K-Means clustering algorithm
                n_clusters : int
                    the numebr of clusters to identify
            meanshift : Meanshift algorithm
                bandwidth : float
                    The bandwidth value used during clustering.
                    If none, determined automatically. Note:
                    the data are scaled before clutering, so
                    this is not in the same units as the data.
                bin_seeding : bool
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


# other useful functions
def gauss(x, *p):
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
    return A * np.exp(-0.5 * (-mu + x)**2 / sigma**2)


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
    el = re.match('.*?([A-z]{1,3}).*?', s).groups()[0]
    m = re.match('.*?([0-9]{1,3}).*?', s).groups()[0]

#     g = re.match('([A-Z][a-z]?)([0-9]+)', s).groups()
    return '$^{' + m + '}$' + el


def collate_data(in_dir, extension='.csv', out_dir=None):
    """
    Copy all csvs in nested directroy to single directory.

    Function to copy all csvs from a directory, and place
    them in a new directory.

    Parameters
    ----------
    in_dir : str
        Input directory containing csv files in subfolders
    extension : str
        The extension that identifies your data files.
        Defaults to '.csv'.
    out_dir : str
        Destination directory

    Returns
    -------
    None
    """
    if out_dir is None:
        out_dir = './' + re.search('^\.(.*)', extension).groups(0)[0]

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for p, d, fs in os.walk(in_dir):
        for f in fs:
            if extension in f:
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
        [2, n] array of (start, end) values describing True parts
        of bool_array
    """
    if ~isinstance(bool_array, np.ndarray):
        bool_array = np.array(bool_array)
    if bool_array[-1]:
        bool_array[-1] = False
    lims = np.arange(bool_array.size)[bool_array ^ np.roll(bool_array, 1)]
    if len(lims) > 0:
        if lims[-1] == bool_array.size - 1:
            lims[-1] = bool_array.size
        return np.reshape(lims, (len(lims) // 2, 2))
    else:
        return [[np.nan, np.nan]]


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
        ns[lims[0]:lims[-1]] = nstart + n + 1
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


def config_locator():
    """
    Prints the location of the latools.cfg file.
    """
    print(pkgrs.resource_filename('latools', 'latools.cfg'))
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
        A (parameter, value) dict defining non - default parameters
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
        config_file = pkgrs.resource_filename('latools', 'latools.cfg')
    cf = configparser.ConfigParser()
    cf.read(config_file)

    # if config doesn't already exist, create it.
    if config_name not in cf.sections():
        cf.add_section(config_name)
    # iterate through parameter dict and set values
    for k, v in params.items():
        cf.set(config_name, k, v)
    # make the parameter set default, if requested
    if make_default:
        cf.set('DEFAULT', 'default_config', config_name)

    cf.write(open(config_file, 'w'))

    return


def intial_configuration():
    """
    Convenience function for configuring latools.

    Run this function when you first use `latools` to specify the
    location of you SRM data file and your data format file.

    See documentation for full details.
    """
    print(('You will be asked a few questions to configure latools\n'
           'for your specific laboratory needs.'))
    lab_name = input('What is the name of your lab? : ')

    params = {}
    OK = False
    while ~OK:
        srmfile = input('Where is your SRM.csv file? [blank = default] : ')
        if srmfile != '':
            if os.path.exists(srmfile):
                params['srmfile'] = srmfile
                OK = True
            else:
                print(("You told us the SRM data file was at: " + semfile +
                       "\nlatools can't find that file. Please check it "
                       "exists, and \ncheck that the path was correct. "
                       "The file path must be complete, not relative."))
        else:
            print(("No path provided. Using default GeoRem values for "
                   "NIST610, NIST612 and NIST614."))
            OK = True

        OK = False

    while ~OK:
        dataformatfile = input(('Where is your dataformat.dict file? '
                                '[blank = default] : '))
        if dataformatfile != '':
            if os.path.exists(dataformatfile):
                params['srmfile'] = dataformatfile
                OK = True
            else:
                print(("You told us the dataformat file was at: " +
                       dataformatfile + "\nlatools can't find that file. "
                       "Please check it exists, and \ncheck that the path "
                       "was correct. The file path must be complete, not "
                       "relative."))
        else:
            print(("No path provided. Using default dataformat "
                   "for the UC Davis Agilent 7700."))
            OK = True

    make_default = input(('Do you want to use these files as your '
                          'default? [Y/n] : ')).lower() != 'n'

    add_config(lab_name, params, make_default=make_default)

    print("\nConfiguration set. You're good to go!")

    return


def get_example_data(destination_dir):
    if os.path.isdir(destination_dir):
        overwrite = input(destination_dir +
                          ' already exists. Overwrite? [N/y]: ').lower() == 'y'
        if overwrite:
            shutil.rmtree(destination_dir)
        else:
            print(destination_dir + ' was not overwritten.')

    shutil.copytree(pkgrs.resource_filename('latools', 'resources/test_data'),
                    destination_dir)

    return


def R2calc(meas, model, force_zero=False):
    if force_zero:
        SStot = np.sum(meas**2)
    else:
        SStot = np.sum((meas - np.nanmean(meas))**2)
    SSres = np.sum((meas - model)**2)
    return 1 - (SSres / SStot)


def rangecalc(xs, ys, pad=0.05):
    xd = max(xs)
    yd = max(ys)
    return ([0 - pad * xd, max(xs) + pad * xd],
            [0 - pad * yd, max(ys) + pad * yd])


# uncertainties unpackers
def unpack_uncertainties(uarray):
    """
    Convenience function to unpack nominal values and uncertainties from an
    ``uncertainties.uarray``.

    Returns:
        (nominal_values, std_devs)
    """
    try:
        return un.nominal_values(uarray), un.std_devs(uarray)
    except:
        return uarray


def nominal_values(a):
    try:
        return un.nominal_values(a)
    except:
        return a


def std_devs(a):
    try:
        return un.std_devs(a)
    except:
        return a


def rolling_window(a, window, pad=None):
    """
    Returns (win, len(a)) rolling - window array of data.

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
    strides = a.strides + (a.strides[-1], )
    out = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    if pad is not None:
        blankpad = np.empty((window // 2, window))
        blankpad[:] = pad
        return np.concatenate([blankpad, out, blankpad])
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
        win -= 1  # subtract 1 from window if it is even.
    # trick for efficient 'rolling' computation in numpy
    # shape = a.shape[:-1] + (a.shape[-1] - win + 1, win)
    # strides = a.strides + (a.strides[-1], )
    # wins = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    wins = rolling_window(a, win)
    # apply rolling gradient to data
    a = map(np.nanmean, wins)

    return np.concatenate([np.zeros(int(win / 2)), list(a),
                          np.zeros(int(win / 2))])


def fastgrad(a, win=11):
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

    Returns
    -------
    array_like
        Gradient of a, assuming as constant integer x - scale.
    """
    # check to see if 'window' is odd (even does not work)
    if win % 2 == 0:
        win -= 1  # subtract 1 from window if it is even.
    # trick for efficient 'rolling' computation in numpy
    # shape = a.shape[:-1] + (a.shape[-1] - win + 1, win)
    # strides = a.strides + (a.strides[-1], )
    # wins = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    wins = rolling_window(a, win)
    # apply rolling gradient to data
    a = map(lambda x: np.polyfit(np.arange(win), x, 1)[0], wins)

    return np.concatenate([np.zeros(int(win / 2)), list(a),
                          np.zeros(int(win / 2))])


# def gaus_deriv(x, *p):
#     A, mu, sigma = p
#     return A * ((np.exp((-(x - mu)**2)/(2*sigma**2)) * (x - mu)) /
#                 (np.sqrt(2 * np.pi) * sigma**3))

def weighted_average(x, y, x_new, fwhm=300):
    """
    Calculate gaussian weigted moving mean, SD and SE.

    Parameters
    ----------
    x, y : array - like
        The x and y data to smooth
    x_new : array - like
        The new x - scale to interpolate the data

    """
    bin_avg = np.zeros(len(x_new))
    bin_std = np.zeros(len(x_new))
    bin_se = np.zeros(len(x_new))

    # Gaussian function as weights
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    for index, xn in enumerate(x_new):
        weights = gauss(x, 1, xn, sigma)
        weights /= sum(weights)
        # weighted mean
        bin_avg[index] = np.average(y, weights=weights)
        # weighted standard deviation
        bin_std[index] = np.sqrt(np.average((y - bin_avg[index])**2, weights=weights))
        # weighted standard error (mean / sqrt(n_points_in_gaussian))
        bin_se[index] = np.sqrt(np.average((y - bin_avg[index])**2, weights=weights)) / \
            np.sqrt(sum((x > xn - 2 * sigma) & (x < xn + 2 * sigma)))

    return {'mean': bin_avg,
            'std': bin_std,
            'stderr': bin_se}

# Statistical Functions
def stderr(a):
    """
    Calculate the standard error of a.
    """
    return np.nanstd(a) / np.sqrt(sum(np.isfinite(a)))


# Robust Statistics. See:
#   - https://en.wikipedia.org/wiki/Robust_statistics
#   - http://www.cscjp.co.jp/fera/document/ANALYSTVol114Decpgs1693-97_1989.pdf
#   - http://www.rsc.org/images/robust-statistics-technical-brief-6_tcm18-214850.pdf
#   - http://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/h15.htm

def H15_mean(x):
    """
    Calculate the Huber (H15) Robust mean of x.

    For details, see:
        http://www.cscjp.co.jp/fera/document/ANALYSTVol114Decpgs1693-97_1989.pdf
        http://www.rsc.org/images/robust-statistics-technical-brief-6_tcm18-214850.pdf
    """
    mu = np.nanmean(x)
    sd = np.nanstd(x) * 1.134
    sig = 1.5

    hi = x > mu + sig * sd
    lo = x < mu - sig * sd

    if any(hi | lo):
        x[hi] = mu + sig * sd
        x[lo] = mu - sig * sd
        return H15_mean(x)
    else:
        return mu


def H15_std(x):
    """
    Calculate the Huber (H15) Robust standard deviation of x.

    For details, see:
        http://www.cscjp.co.jp/fera/document/ANALYSTVol114Decpgs1693-97_1989.pdf
        http://www.rsc.org/images/robust-statistics-technical-brief-6_tcm18-214850.pdf
    """
    mu = np.nanmean(x)
    sd = np.nanstd(x) * 1.134
    sig = 1.5

    hi = x > mu + sig * sd
    lo = x < mu - sig * sd

    if any(hi | lo):
        x[hi] = mu + sig * sd
        x[lo] = mu - sig * sd
        return H15_std(x)
    else:
        return sd


def H15_se(x):
    """
    Calculate the Huber (H15) Robust standard deviation of x.

    For details, see:
        http://www.cscjp.co.jp/fera/document/ANALYSTVol114Decpgs1693-97_1989.pdf
        http://www.rsc.org/images/robust-statistics-technical-brief-6_tcm18-214850.pdf
    """
    sd = H15_std(x)
    return sd / np.sqrt(sum(np.isfinite(x)))
