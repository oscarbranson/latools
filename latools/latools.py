"""
Main functions for interacting with LAtools.

(c) Oscar Branson : https://github.com/oscarbranson
"""

import configparser
import itertools
import inspect
import json
import os
import re
import time
import warnings
import dateutil
import textwrap

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pkg_resources as pkgrs

import uncertainties as unc
import uncertainties.unumpy as un

from sklearn.preprocessing import minmax_scale, scale
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

from .helpers import plot
from .filtering import filters
from .filtering.classifier_obj import classifier

from .D_obj import D
from .helpers.helpers import (rolling_window, enumerate_bool,
                      un_interp1d, get_date,
                      unitpicker, rangecalc, Bunch, calc_grads,
                      get_total_time_span)
from .helpers import logging
from .helpers.logging import _log
from .helpers.config import read_configuration, config_locator
from .helpers.stat_fns import *
from .helpers import utils
from .helpers import srm as srms
from .helpers.progressbars import progressbar
from .helpers.chemistry import analyte_mass, decompose_molecule
from .helpers.analyte_names import get_analyte_name, analyte_2_massname, pretty_element, analyte_sort_fn

idx = pd.IndexSlice  # multi-index slicing!

# deactivate IPython deprecations warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
# deactivate numpy invalid comparison warnings
np.seterr(invalid='ignore')


# TODO: Allow full sklearn integration by allowing sample-wise application of custom classifiers. i.e. Provide data collection (get_data) ajd filter addition API.
# Especially: PCA, Gaussian Mixture Models

class analyse(object):
    """
    For processing and analysing whole LA - ICPMS datasets.

    Parameters
    ----------
    data_folder : str
        The path to a directory containing multiple data files.
    errorhunt : bool
        If True, latools prints the name of each file before it
        imports the data. This is useful for working out which 
        data file is causing the import to fail.
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
        * 'file_names' : use the file names as labels (default)
        * 'metadata_names' : used the 'names' attribute of metadata as the name
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
    data : dict
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
    """

    def __init__(self, data_folder, errorhunt=False, config='DEFAULT',
                 dataformat=None, extension='.csv', srm_identifier='STD',
                 cmap=None, time_format=None, internal_standard='Ca43',
                 names='file_names', srm_file=None, pbar=None):
        """
        For processing and analysing whole LA - ICPMS datasets.
        """
        # initialise log
        params = {k: v for k, v in locals().items() if k not in ['self', 'pbar']}
        self.log = ['__init__ :: args=() kwargs={}'.format(str(params))]

        # assign file paths
        self.folder = os.path.realpath(data_folder)
        self.parent_folder = os.path.dirname(self.folder)
        self.files = np.array([f for f in os.listdir(self.folder)
                               if extension in f])

        # set line length for outputs
        self._line_width = 80

        # make output directories
        self.report_dir = re.sub('//', '/',
                                 os.path.join(self.parent_folder,
                                              os.path.basename(self.folder) + '_reports/'))
        if not os.path.isdir(self.report_dir):
            os.mkdir(self.report_dir)
        self.export_dir = re.sub('//', '/',
                                 os.path.join(self.parent_folder,
                                              os.path.basename(self.folder) + '_export/'))
        if not os.path.isdir(self.export_dir):
            os.mkdir(self.export_dir)

        # load configuration parameters
        self.config = read_configuration(config)

        # print some info about the analysis and setup.
        startmsg = self._fill_line('-') + 'Starting analysis:'
        if srm_file is None or dataformat is None:
            startmsg += '\n  Using {} configuration'.format(self.config['config'])
            if config == 'DEFAULT':
                startmsg += ' (default).'
            else:
                startmsg += '.'
            pretext = '  with'
        else:
            pretext = 'Using'
        
        if srm_file is not None:
            startmsg += '\n  ' + pretext + ' custom srm_file ({})'.format(srm_file)
        if isinstance(dataformat, str):
            startmsg += '\n  ' + pretext + ' custom dataformat file ({})'.format(dataformat)
        elif isinstance(dataformat, dict):
            startmsg += '\n  ' + pretext + ' custom dataformat dict'
        print(startmsg)

        self._load_srmfile(srm_file)

        self._load_dataformat(dataformat)

        # link up progress bars
        if pbar is None:
            self.pbar = progressbar()
        else:
            self.pbar = pbar

        # load data into list (initialise D objects)
        with self.pbar.set(total=len(self.files), desc='Loading Data') as prog:
            data = [None] * len(self.files)
            for i, f in enumerate(self.files):
                data[i] = (D(os.path.join(self.folder, f),
                           dataformat=self.dataformat,
                           errorhunt=errorhunt,
                           cmap=cmap,
                           internal_standard=internal_standard,
                           name=names))
                prog.update()

        # create universal time scale
        if 'date' in data[0].meta.keys():
            if (time_format is None) and ('time_format' in self.dataformat.keys()):
                time_format = self.dataformat['time_format']

            start_times = []
            for d in data:
                start_times.append(get_date(d.meta['date'], time_format))
            min_time = min(start_times)

            for d, st in zip(data, start_times):
                d.uTime = d.Time + (st - min_time).seconds
        else:
            ts = 0
            for d in data:
                d.uTime = d.Time + ts
                ts += d.Time[-1]
            msg = self._wrap_text( 
                          "Time not determined from dataformat. Universal time scale " +
                          "approximated as continuously measured samples. " +
                          "Samples might not be in the right order. "
                          "Background correction and calibration may not behave " +
                          "as expected.")
            warnings.warn(self._wrap_msg(msg, '*'))

        self.max_time = max([d.uTime.max() for d in data])

        # sort data by uTime
        data.sort(key=lambda d: d.uTime[0])

        # process sample names
        if (names == 'file_names') | (names == 'metadata_names'):
            samples = np.array([s.sample for s in data], dtype=object)  # get all sample names
            # if duplicates, rename them
            usamples, ucounts = np.unique(samples, return_counts=True)
            if usamples.size != samples.size:
                dups = usamples[ucounts > 1]  # identify duplicates
                nreps = ucounts[ucounts > 1]  # identify how many times they repeat
                for d, n in zip(dups, nreps):  # cycle through duplicates
                    new = [d + '_{}'.format(i) for i in range(n)]  # append number to duplicate names
                    ind = samples == d
                    samples[ind] = new  # rename in samples
                    for s, ns in zip([data[i] for i in np.where(ind)[0]], new):
                        s.sample = ns  # rename in D objects
        else:
            samples = np.arange(len(data))  # assign a range of numbers
            for i, s in enumerate(samples):
                data[i].sample = s
        self.samples = samples

        # copy colour map to top level
        self.cmaps = data[0].cmap

        # get analytes
        # TODO: does this preserve the *order* of the analytes?
        all_analytes = set()
        extras = set()
        for d in data:
            all_analytes.update(d.analytes)
            extras.update(all_analytes.symmetric_difference(d.analytes))
        self.analytes = all_analytes.difference(extras)
        mismatch = []
        if self.analytes != all_analytes:
            smax = 0
            for d in data:
                if d.analytes != self.analytes:
                    mismatch.append((d.sample, d.analytes.difference(self.analytes)))
                    if len(d.sample) > smax:
                        smax = len(d.sample)
            msg = (self._fill_line('*') +
                   'All data files do not contain the same analytes.\n' + 
                   'Only analytes present in all files will be processed.\n' + 
                   'In the following files, these analytes will be excluded:\n')
            for s, a in mismatch:
                msg += ('  {0: <' + '{:}'.format(smax + 2) + '}:  ').format(s) + str(a) + '\n'
            msg += self._fill_line('*')
            warnings.warn(msg)

        if len(self.analytes) == 0:
            raise ValueError('No analyte names identified. Please check the \ncolumn_id > pattern ReGeX in your dataformat file.')

        if internal_standard in self.analytes:
            self.internal_standard = internal_standard
        else:
            raise ValueError('The internal standard ({}) is not amongst the '.format(internal_standard) +
                             'analytes in\nyour data files. Please make sure it is specified correctly.')
        self.minimal_analytes = set([internal_standard])

        # keep record of which stages of processing have been performed
        self.stages_complete = set(['rawdata'])

        # From this point on, data stored in dicts
        self.data = Bunch(zip(self.samples, data))
        
        # remove mismatch analytes - QUICK-FIX - SHOULD BE DONE HIGHER UP?
        for s, a in mismatch:
            self.data[s].analytes = self.data[s].analytes.difference(a)
            
        # get SRM info
        self.srm_identifier = srm_identifier
        self.stds = []  # make this a dict
        _ = [self.stds.append(s) for s in self.data.values()
             if self.srm_identifier in s.sample]
        self.srms_ided = False

        # set up focus_stage recording
        self.focus_stage = 'rawdata'
        self.focus = Bunch()

        # set up subsets
        self._has_subsets = False
        self._subset_names = []
        self.subsets = Bunch()
        self.subsets['All_Analyses'] = self.samples
        self.subsets[self.srm_identifier] = [s for s in self.samples if self.srm_identifier in s]
        self.subsets['All_Samples'] = [s for s in self.samples if self.srm_identifier not in s]
        self.subsets['not_in_set'] = self.subsets['All_Samples'].copy()

        # remove any analytes for which all counts are zero
        # self.get_focus()
        # for a in self.analytes:
        #     if np.nanmean(self.focus[a] == 0):
        #         self.analytes.remove(a)
        #         warnings.warn('{} contains no data - removed from analytes')
        
        # initialise classifiers
        self.classifiers = Bunch()

        # report
        print(('Loading Data:\n  {:d} Data Files Loaded: {:d} standards, {:d} '
               'samples').format(len(self.data),
                                 len(self.stds),
                                 len(self.data) - len(self.stds)))
        astr = self._wrap_text('Analytes: ' + ' '.join(self.analytes_sorted()))
        print(astr)
        print('  Internal Standard: {}'.format(self.internal_standard))

    def _fill_line(self, char, newline=True):
        """Generate a full line of given character"""
        if newline:
            return char * self._line_width + '\n'
        else:
            return char * self._line_width

    def _wrap_text(self, text):
        """Splits text over multiple lines to fit within self._line_width"""
        return '\n'.join(textwrap.wrap(text, width=self._line_width, 
                                       break_long_words=False))

    
    def _wrap_msg(self, msg, char):
        return self._fill_line(char) + msg + '\n' + self._fill_line(char, False)

    def _load_dataformat(self, dataformat):
        """
        Load in dataformat.
        
        Check dataformat file exists, and store it in a class attribute.
        If dataformat is not provided during initialisation, assign it
        fom configuration file
        """
        if dataformat is None:
            if os.path.exists(self.config['dataformat']):
                dataformat = self.config['dataformat']
            elif os.path.exists(pkgrs.resource_filename('latools',
                                                        self.config['dataformat'])):
                dataformat = pkgrs.resource_filename('latools',
                                                     self.config['dataformat'])
            else:
                config_file = config_locator()
                raise ValueError(('The dataformat file specified in the ' +
                                  self.config['config'] + ' configuration cannot be found.\n'
                                  'Please make sure the file exists, and that'
                                  'the path in the config file is correct.\n'
                                  'Your configurations can be found here:'
                                  '    {}\n'.format(config_file)))
            self.dataformat_file = dataformat
        else:
            self.dataformat_file = 'None: dict provided'

        # if it's a string, check the file exists and import it.
        if isinstance(dataformat, str):
            if os.path.exists(dataformat):
                # self.dataformat = eval(open(dataformat).read())
                self.dataformat = json.load(open(dataformat))
            else:
                warnings.warn(("The dataformat file (" + dataformat +
                               ") cannot be found.\nPlease make sure the file "
                               "exists, and that the path is correct.\n\nFile "
                               "Path: " + dataformat))

        # if it's a dict, just assign it straight away.
        elif isinstance(dataformat, dict):
            self.dataformat = dataformat
    
    def _load_srmfile(self, srm_file):
        """
        Check srmfile exists, and store it in a class attribute.
        """
        if srm_file is not None:
            if os.path.exists(srm_file):
                self.srmfile = srm_file
            else:
                raise ValueError(('Cannot find the specified SRM file:\n   ' +
                                  srm_file +
                                  'Please check that the file location is correct.'))
        else:
            if os.path.exists(self.config['srmfile']):
                self.srmfile = self.config['srmfile']
            elif os.path.exists(pkgrs.resource_filename('latools',
                                                        self.config['srmfile'])):
                self.srmfile = pkgrs.resource_filename('latools',
                                                       self.config['srmfile'])
            else:
                config_file = config_locator()
                raise ValueError(('The SRM file specified in the ' + self.config['config'] +
                                  ' configuration cannot be found.\n'
                                  'Please make sure the file exists, and that the '
                                  'path in the config file is correct.\n'
                                  'Your configurations can be found here:'
                                  '    {}\n'.format(config_file)))

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
    
    def _log_header(self):
        return ['# LATOOLS analysis log saved at {}'.format(time.strftime('%Y:%m:%d %H:%M:%S')),
                'data_folder :: {}'.format(self.folder),
                '# Analysis Log Start: \n'
                ]

    def analytes_sorted(self, a=None):
        if a is None:
            a = self.analytes
        return sorted(a, key=analyte_sort_fn)

    @_log
    def basic_processing(self,
                         noise_despiker=True, despike_win=3, despike_nlim=12.,  # despike args
                         despike_maxiter=4,
                         autorange_analyte='total_counts', autorange_gwin=5, autorange_swin=3, autorange_win=20,  # autorange args
                         autorange_on_mult=[1., 1.5], autorange_off_mult=[1.5, 1],
                         autorange_transform='log',
                         bkg_weight_fwhm=300.,  # bkg_calc_weightedmean
                         bkg_n_min=20, bkg_n_max=None, bkg_cstep=None,
                         bkg_filter=False, bkg_f_win=7, bkg_f_n_lim=3,
                         bkg_errtype='stderr',  # bkg_sub
                         calib_drift_correct=True,  # calibrate
                         calib_srms_used=['NIST610', 'NIST612', 'NIST614'],
                         calib_zero_intercept=True, calib_n_min=10,
                         plots=True):
        
        self.despike(noise_despiker=noise_despiker,
                     win=despike_win, nlim=despike_nlim,
                     maxiter=despike_maxiter)
        self.autorange(analyte=autorange_analyte, gwin=autorange_gwin, swin=autorange_swin,
                       win=autorange_win, on_mult=autorange_on_mult,
                       off_mult=autorange_off_mult,
                       transform=autorange_transform)
        if plots:
            self.trace_plots(ranges=True)
        self.bkg_calc_weightedmean(weight_fwhm=bkg_weight_fwhm, n_min=bkg_n_min, n_max=bkg_n_max,
                                   cstep=bkg_cstep, bkg_filter=bkg_filter, f_win=bkg_f_win, f_n_lim=bkg_f_n_lim)
        if plots:
            self.bkg_plot()
        self.bkg_subtract(errtype=bkg_errtype)
        self.ratio()
        self.calibrate(drift_correct=calib_drift_correct, srms_used=calib_srms_used,
                       zero_intercept=calib_zero_intercept, n_min=calib_n_min)
        if plots:
            self.calibration_plot()

        return

    @_log
    def autorange(self, analyte='total_counts', gwin=5, swin=3, win=20,
                  on_mult=[1., 1.5], off_mult=[1.5, 1],
                  transform='log', ploterrs=True, focus_stage='despiked'):
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
            This can also be 'total_counts' to use the sum of all analytes.
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
        focus_stage : str
            Which stage of analysis to apply processing to. 
            Defaults to 'despiked', or rawdata' if not despiked. Can be one of:
            * 'rawdata': raw data, loaded from csv file.
            * 'despiked': despiked data.
            * 'signal'/'background': isolated signal and background data.
              Created by self.separate, after signal and background
              regions have been identified by self.autorange.
            * 'bkgsub': background subtracted data, created by 
              self.bkg_correct
            * 'ratios': element ratio data, created by self.ratio.
            * 'calibrated': ratio data calibrated to standards, created by self.calibrate.

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
        if focus_stage == 'despiked':
            if 'despiked' not in self.stages_complete:
                focus_stage = 'rawdata'

        if analyte is None:
            analyte = self.internal_standard
        elif analyte in self.analytes:
            self.minimal_analytes.update([analyte])

        fails = {}  # list for catching failures.
        with self.pbar.set(total=len(self.data), desc='AutoRange') as prog:
            for s, d in self.data.items():
                f = d.autorange(analyte=analyte, gwin=gwin, swin=swin, win=win,
                                on_mult=on_mult, off_mult=off_mult,
                                ploterrs=ploterrs, transform=transform)
                if f is not None:
                    fails[s] = f
                prog.update()  # advance progress bar
        # handle failures
        if len(fails) > 0:
            wstr = ('\n\n' + '*' * 41 + '\n' +
                    '                 WARNING\n' + '*' * 41 + '\n' +
                    'Autorange failed for some samples:\n')

            kwidth = max([len(k) for k in fails.keys()]) + 1
            fstr = '  {:' + '{}'.format(kwidth) + 's}: '
            for k in sorted(fails.keys()):
                wstr += fstr.format(k) + ', '.join(['{:.1f}'.format(f) for f in fails[k][-1]]) + '\n'

            wstr += ('\n*** THIS IS NOT NECESSARILY A PROBLEM ***\n' +
                     'But please check the plots below to make\n' +
                     'sure they look OK. Failures are marked by\n' +
                     'dashed vertical red lines.\n\n' +
                     'To examine an autorange failure in more\n' +
                     'detail, use the `autorange_plot` method\n' +
                     'of the failing data object, e.g.:\n' +
                     "dat.data['Sample'].autorange_plot(params)\n" +
                     '*' * 41 + '\n')
            warnings.warn(wstr)
        
        self.stages_complete.update(['autorange'])
        return

    def find_expcoef(self, nsd_below=0., plot=False,
                     trimlim=None, autorange_kwargs={}):
        """
        Determines exponential decay coefficient for despike filter.

        Fits an exponential decay function to the washout phase of standards
        to determine the washout time of your laser cell. The exponential
        coefficient reported is `nsd_below` standard deviations below the
        fitted exponent, to ensure that no real data is removed.

        Total counts are used in fitting, rather than a specific analyte.

        Parameters
        ----------
        nsd_below : float
            The number of standard deviations to subtract from the fitted
            coefficient when calculating the filter exponent.
        plot : bool or str
            If True, creates a plot of the fit, if str the plot is to the
            location specified in str.
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
        print('Calculating exponential decay coefficient\nfrom SRM washouts...')

        def findtrim(tr, lim=None):
            trr = np.roll(tr, -1)
            trr[-1] = 0
            if lim is None:
                lim = 0.5 * np.nanmax(tr - trr)
            ind = (tr - trr) >= lim
            return np.arange(len(ind))[ind ^ np.roll(ind, -1)][0]

        if not hasattr(self.stds[0], 'trnrng'):
            for s in self.stds:
                s.autorange(**autorange_kwargs, ploterrs=False)

        trans = []
        times = []
        for v in self.stds:
            for trnrng in v.trnrng[-1::-2]:
                tr = minmax_scale(v.data['total_counts'][(v.Time > trnrng[0]) & (v.Time < trnrng[1])])
                sm = np.apply_along_axis(np.nanmean, 1,
                                         rolling_window(tr, 3, pad=0))
                sm[0] = sm[1]
                trim = findtrim(sm, trimlim) + 2
                trans.append(minmax_scale(tr[trim:]))
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
    def despike(self, expdecay_despiker=False, exponent=None,
                noise_despiker=True, win=3, nlim=12., exponentplot=False,
                maxiter=4, autorange_kwargs={}, focus_stage='rawdata'):
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
        maxiter : int
            The max number of times that the fitler is applied.
        focus_stage : str
            Which stage of analysis to apply processing to. 
            Defaults to 'rawdata'. Can be one of:
            * 'rawdata': raw data, loaded from csv file.
            * 'despiked': despiked data.
            * 'signal'/'background': isolated signal and background data.
              Created by self.separate, after signal and background
              regions have been identified by self.autorange.
            * 'bkgsub': background subtracted data, created by 
              self.bkg_correct
            * 'ratios': element ratio data, created by self.ratio.
            * 'calibrated': ratio data calibrated to standards, created by self.calibrate.

        Returns
        -------
        None
        """
        if focus_stage != self.focus_stage:
            self.set_focus(focus_stage)

        if expdecay_despiker and exponent is None:
            if not hasattr(self, 'expdecay_coef'):
                self.find_expcoef(plot=exponentplot,
                                  autorange_kwargs=autorange_kwargs)
            exponent = self.expdecay_coef
            time.sleep(0.1)

        with self.pbar.set(total=len(self.data), desc='Despiking') as prog:
            for d in self.data.values():
                d.despike(expdecay_despiker, exponent,
                        noise_despiker, win, nlim, maxiter)
                prog.update()

        self.stages_complete.update(['despiked'])
        self.focus_stage = 'despiked'
        return

    # functions for background correction
    def get_background(self, n_min=10, n_max=None, focus_stage='despiked', bkg_filter=False, f_win=5, f_n_lim=3):
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
        focus_stage : str
            Which stage of analysis to apply processing to. 
            Defaults to 'despiked' if present, or 'rawdata' if not. 
            Can be one of:
            * 'rawdata': raw data, loaded from csv file.
            * 'despiked': despiked data.
            * 'signal'/'background': isolated signal and background data.
              Created by self.separate, after signal and background
              regions have been identified by self.autorange.
            * 'bkgsub': background subtracted data, created by 
              self.bkg_correct
            * 'ratios': element ratio data, created by self.ratio.
            * 'calibrated': ratio data calibrated to standards, created by self.calibrate.

        Returns
        -------
        pandas.DataFrame object containing background data.
        """
        allbkgs = {'uTime': [],
                   'ns': []}
                
        if focus_stage == 'despiked':
            if 'despiked' not in self.stages_complete:
                focus_stage = 'rawdata'

        for a in self.analytes:
            allbkgs[a] = []

        n0 = 0
        for s in self.data.values():
            if sum(s.bkg) > 0:
                allbkgs['uTime'].append(s.uTime[s.bkg])
                allbkgs['ns'].append(enumerate_bool(s.bkg, n0)[s.bkg])
                n0 = allbkgs['ns'][-1][-1]
                for a in self.analytes:
                    allbkgs[a].append(s.data[focus_stage][a][s.bkg])

        allbkgs.update((k, np.concatenate(v)) for k, v in allbkgs.items())
        bkgs = pd.DataFrame(allbkgs)  # using pandas here because it's much more efficient than loops.

        self.bkg = Bunch()
        # extract background data from whole dataset
        if n_max is None:
            self.bkg['raw'] = bkgs.groupby('ns').filter(lambda x: len(x) > n_min)
        else:
            self.bkg['raw'] = bkgs.groupby('ns').filter(lambda x: (len(x) > n_min) & (len(x) < n_max))
        # calculate per - background region stats
        self.bkg['summary'] = self.bkg['raw'].groupby('ns').aggregate([np.mean, np.std, stderr])
        # sort summary by uTime
        self.bkg['summary'].sort_values(('uTime', 'mean'), inplace=True)
        # self.bkg['summary'].index = np.arange(self.bkg['summary'].shape[0])
        # self.bkg['summary'].index.name = 'ns'

        if bkg_filter:
            # calculate rolling mean and std from summary
            t = self.bkg['summary'].loc[:, idx[:, 'mean']]
            r = t.rolling(f_win).aggregate([np.nanmean, np.nanstd])
            # calculate upper threshold
            upper = r.loc[:, idx[:, :, 'nanmean']] + f_n_lim * r.loc[:, idx[:, :, 'nanstd']].values
            # calculate which are over upper threshold
            over = r.loc[:, idx[:, :, 'nanmean']] > np.roll(upper.values, 1, 0)
            # identify them
            ns_drop = over.loc[over.apply(any, 1), :].index.values
            # drop them from summary
            self.bkg['summary'].drop(ns_drop, inplace=True)
            # remove them from raw
            ind = np.ones(self.bkg['raw'].shape[0], dtype=bool)
            for ns in ns_drop:
                ind = ind & (self.bkg['raw'].loc[:, 'ns'] != ns)
            self.bkg['raw'] = self.bkg['raw'].loc[ind, :]
        return

    @_log
    def bkg_calc_weightedmean(self, analytes=None, weight_fwhm=None,
                              n_min=20, n_max=None, cstep=None, errtype='stderr',
                              bkg_filter=False, f_win=7, f_n_lim=3, focus_stage='despiked'):
        """
        Background calculation using a gaussian weighted mean.

        Parameters
        ----------
        analytes : str or iterable
            Which analyte or analytes to calculate.
        weight_fwhm : float
            The full-width-at-half-maximum of the gaussian used
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
        focus_stage : str
            Which stage of analysis to apply processing to. 
            Defaults to 'despiked' if present, or 'rawdata' if not. 
            Can be one of:
            * 'rawdata': raw data, loaded from csv file.
            * 'despiked': despiked data.
            * 'signal'/'background': isolated signal and background data.
              Created by self.separate, after signal and background
              regions have been identified by self.autorange.
            * 'bkgsub': background subtracted data, created by 
              self.bkg_correct
            * 'ratios': element ratio data, created by self.ratio.
            * 'calibrated': ratio data calibrated to standards, created by self.calibrate.
        """
        if analytes is None:
            analytes = self.analytes
            self.bkg = Bunch()
        elif isinstance(analytes, str):
            analytes = [analytes]
        
        if weight_fwhm is None:
            weight_fwhm = 600  # 10 minute default window

        self.get_background(n_min=n_min, n_max=n_max,
                            bkg_filter=bkg_filter,
                            f_win=f_win, f_n_lim=f_n_lim, focus_stage=focus_stage)

        # Gaussian - weighted average
        if 'calc' not in self.bkg.keys():
            # create time points to calculate background
            if cstep is None:
                cstep = weight_fwhm / 20
            elif cstep > weight_fwhm:
                warnings.warn("\ncstep should be less than weight_fwhm. Your backgrounds\n" +
                              "might not behave as expected.\n")
            bkg_t = np.linspace(0,
                                self.max_time,
                                self.max_time // cstep)
            self.bkg['calc'] = Bunch()
            self.bkg['calc']['uTime'] = bkg_t

        # TODO : calculation then dict assignment is clumsy...
        mean, std, stderr = gauss_weighted_stats(self.bkg['raw'].uTime,
                                                 self.bkg['raw'].loc[:, analytes].values,
                                                 self.bkg['calc']['uTime'],
                                                 fwhm=weight_fwhm)
        self.bkg_interps = {}

        for i, a in enumerate(analytes):
            self.bkg['calc'][a] = {'mean': mean[i],
                                    'std': std[i],
                                    'stderr': stderr[i]}
            self.bkg_interps[a] = un_interp1d(x=self.bkg['calc']['uTime'],
                                              y=un.uarray(self.bkg['calc'][a]['mean'],
                                                          self.bkg['calc'][a][errtype]))

    @_log
    def bkg_calc_interp1d(self, analytes=None, kind=1, n_min=10, n_max=None, cstep=30,
                          bkg_filter=False, f_win=7, f_n_lim=3, errtype='stderr', focus_stage='despiked'):
        """
        Background calculation using a 1D interpolation.

        scipy.interpolate.interp1D is used for interpolation.

        Parameters
        ----------
        analytes : str or iterable
            Which analyte or analytes to calculate.
        kind : str or int
            Integer specifying the order of the spline interpolation
            used, or string specifying a type of interpolation.
            Passed to `scipy.interpolate.interp1D`.
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
        focus_stage : str
            Which stage of analysis to apply processing to. 
            Defaults to 'despiked' if present, or 'rawdata' if not. 
            Can be one of:
            * 'rawdata': raw data, loaded from csv file.
            * 'despiked': despiked data.
            * 'signal'/'background': isolated signal and background data.
              Created by self.separate, after signal and background
              regions have been identified by self.autorange.
            * 'bkgsub': background subtracted data, created by 
              self.bkg_correct
            * 'ratios': element ratio data, created by self.ratio.
            * 'calibrated': ratio data calibrated to standards, created by self.calibrate.            
        """
        if analytes is None:
            analytes = self.analytes
            self.bkg = Bunch()
        elif isinstance(analytes, str):
            analytes = [analytes]

        self.get_background(n_min=n_min, n_max=n_max,
                            bkg_filter=bkg_filter,
                            f_win=f_win, f_n_lim=f_n_lim, focus_stage=focus_stage)

        def pad(a, lo=None, hi=None):
            if lo is None:
                lo = [a[0]]
            if hi is None:
                hi = [a[-1]]
            return np.concatenate((lo, a, hi))

        if 'calc' not in self.bkg.keys():
            # create time points to calculate background
            
            bkg_t = pad(np.ravel(self.bkg.raw.loc[:, ['uTime', 'ns']].groupby('ns').aggregate([min, max])))
            bkg_t = np.unique(np.sort(np.concatenate([bkg_t, np.arange(0, self.max_time, cstep)])))

            self.bkg['calc'] = Bunch()
            self.bkg['calc']['uTime'] = bkg_t

        d = self.bkg['summary']
        self.bkg_interps = {}
        with self.pbar.set(total=len(analytes), desc='Calculating Analyte Backgrounds') as prog:
            for a in analytes:
                fill_vals = (un.uarray(d.loc[:, (a, 'mean')].iloc[0], d.loc[:, (a, errtype)].iloc[0]),
                             un.uarray(d.loc[:, (a, 'mean')].iloc[-1], d.loc[:, (a, errtype)].iloc[-1]))
                p = un_interp1d(x=d.loc[:, ('uTime', 'mean')],
                                y=un.uarray(d.loc[:, (a, 'mean')],
                                            d.loc[:, (a, errtype)]),
                                kind=kind, bounds_error=False, fill_value=fill_vals)
                self.bkg_interps[a] = p
                self.bkg['calc'][a] = {'mean': p.new_nom(self.bkg['calc']['uTime']),
                                       errtype: p.new_std(self.bkg['calc']['uTime'])}
                prog.update()

        # self.bkg['calc']

        return

    @_log
    def bkg_subtract(self, analytes=None, errtype='stderr', focus_stage='despiked'):
        """
        Subtract calculated background from data.

        Must run bkg_calc first!

        Parameters
        ----------
        analytes : str or iterable
            Which analyte(s) to subtract.
        errtype : str
            Which type of error to propagate. default is 'stderr'.
        focus_stage : str
            Which stage of analysis to apply processing to. 
            Defaults to 'despiked' if present, or 'rawdata' if not. 
            Can be one of:
            * 'rawdata': raw data, loaded from csv file.
            * 'despiked': despiked data.
            * 'signal'/'background': isolated signal and background data.
              Created by self.separate, after signal and background
              regions have been identified by self.autorange.
            * 'bkgsub': background subtracted data, created by 
              self.bkg_correct
            * 'ratios': element ratio data, created by self.ratio.
            * 'calibrated': ratio data calibrated to standards, created by self.calibrate.
        """
        if analytes is None:
            analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        if focus_stage == 'despiked':
            if 'despiked' not in self.stages_complete:
                focus_stage = 'rawdata'

        # make uncertainty-aware background interpolators
        # bkg_interps = {}
        # for a in analytes:
        #     bkg_interps[a] = un_interp1d(x=self.bkg['calc']['uTime'],
        #                                  y=un.uarray(self.bkg['calc'][a]['mean'],
        #                                              self.bkg['calc'][a][errtype]))
        # self.bkg_interps = bkg_interps

        # apply background corrections
        with self.pbar.set(total=len(self.data), desc='Background Subtraction') as prog:
            for d in self.data.values():
                # [d.bkg_subtract(a, bkg_interps[a].new(d.uTime), None, focus_stage=focus_stage) for a in analytes]
                [d.bkg_subtract(a, self.bkg_interps[a].new(d.uTime), ~d.sig, focus_stage=focus_stage) for a in analytes]
                d.setfocus('bkgsub')

                prog.update()

        self.stages_complete.update(['bkgsub'])
        self.focus_stage = 'bkgsub'
        return

    @_log
    def correct_spectral_interference(self, target_analyte, source_analyte, f):
        """
        Correct spectral interference.

        Subtract interference counts from target_analyte, based on the
        intensity of a source_analayte and a known fractional contribution (f).

        Correction takes the form:
        target_analyte -= source_analyte * f

        Only operates on background-corrected data ('bkgsub'). To undo a correction,
        rerun `self.bkg_subtract()`.

        Example
        -------
        To correct 44Ca+ for an 88Sr++ interference, where both 43.5 and 44 Da
        peaks are known:
        f = abundance(88Sr) / (abundance(87Sr) 

        counts(44Ca) = counts(44 Da) - counts(43.5 Da) * f


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

        with self.pbar.set(total=len(self.data), desc='Interference Correction') as prog:
            for d in self.data.values():
                d.correct_spectral_interference(target_analyte, source_analyte, f)

                prog.update()

    @_log
    def bkg_plot(self, analytes=None, figsize=None, yscale='log',
                 ylim=None, err='stderr', save=True):
        """
        Plot the calculated background.

        Parameters
        ----------
        analytes : str or iterable
            Which analyte(s) to plot.
        figsize : tuple
            The (width, height) of the figure, in inches.
            If None, calculated based on number of samples.
        yscale : str
            'log' (default) or 'linear'.
        ylim : tuple
            Manually specify the y scale.
        err : str
            What type of error to plot. Default is stderr.
        save : bool
            If True, figure is saved.

        Returns
        -------
        fig, ax : matplotlib.figure, matplotlib.axes
        """
        # if not hasattr(self, 'bkg'):
        #     raise ValueError("\nPlease calculate a background before attempting to\n" +
        #                      "plot it... either:\n" +
        #                      "   bkg_calc_interp1d\n" +
        #                      "   bkg_calc_weightedmean\n")
        if not hasattr(self, 'bkg'):
            self.get_background()

        if analytes is None:
            analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        if figsize is None:
            if len(self.samples) > 50:
                figsize = (len(self.samples) * 0.2, 5)
            else:
                figsize = (10, 5)

        fig = plt.figure(figsize=figsize)

        ax = fig.add_axes([.07, .1, .84, .8])
        
        with self.pbar.set(total=len(analytes), desc='Plotting backgrounds') as prog:
            for a in analytes:
                # draw data points
                ax.scatter(self.bkg['raw'].uTime, self.bkg['raw'].loc[:, a],
                        alpha=0.5, s=3, c=self.cmaps[a],
                        lw=0.5)
                
                # draw STD boxes
                for i, r in self.bkg['summary'].iterrows():
                    x = (r.loc['uTime', 'mean'] - r.loc['uTime', 'std'] * 2,
                        r.loc['uTime', 'mean'] + r.loc['uTime', 'std'] * 2)
                    yl = [r.loc[a, 'mean'] - r.loc[a, err]] * 2
                    yu = [r.loc[a, 'mean'] + r.loc[a, err]] * 2

                    ax.fill_between(x, yl, yu, alpha=0.8, lw=0.5, color=self.cmaps[a], zorder=1)
                prog.update()

        if yscale == 'log':
            ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(ax.get_ylim() * np.array([1, 10]))  # x10 to make sample names readable.

        if 'calc' in self.bkg:
            for a in analytes:
                # draw confidence intervals of calculated
                x = self.bkg['calc']['uTime']
                y = self.bkg['calc'][a]['mean']
                yl = self.bkg['calc'][a]['mean'] - self.bkg['calc'][a][err]
                yu = self.bkg['calc'][a]['mean'] + self.bkg['calc'][a][err]

                # trim values below zero if log scale=    
                if yscale == 'log':
                    yl[yl < ax.get_ylim()[0]] = ax.get_ylim()[0]

                ax.plot(x, y,
                        c=self.cmaps[a], zorder=2, label=a)
                ax.fill_between(x, yl, yu,
                                color=self.cmaps[a], alpha=0.3, zorder=-1)
        else:
            for a in analytes:
                ax.plot([], [], c=self.cmaps[a], label=a)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Background Counts')

        ax.set_title('Points = raw data; Bars = {:s}; Lines = Calculated Background; Envelope = Background {:s}'.format(err, err),
                     fontsize=10)

        ha, la = ax.get_legend_handles_labels()

        ax.legend(labels=la[:len(analytes)], handles=ha[:len(analytes)], bbox_to_anchor=(1, 1))

        # scale x axis to range ± 2.5%
        xlim = [0, max([d.uTime[-1] for d in self.data.values()])]
        ax.set_xlim(xlim)

        # add sample labels
        for s, d in self.data.items():
            ax.axvline(d.uTime[0], alpha=0.2, color='k', zorder=-1)
            ax.text(d.uTime[0], ax.get_ylim()[1], s, rotation=90,
                    va='top', ha='left', zorder=-1, fontsize=7)

        if save:
            fig.savefig(self.report_dir + '/background.png', dpi=200)

        return fig, ax

    # functions for calculating ratios
    @_log
    def ratio(self, internal_standard=None, analytes=None):
        """
        Calculates the ratio of all analytes to a single analyte.

        Parameters
        ----------
        internal_standard : str
            The name of the analyte to divide all other analytes
            by.

        Returns
        -------
        None
        """
        if 'bkgsub' not in self.stages_complete:
            raise RuntimeError('Cannot calculate ratios before background subtraction.')
        
        if analytes is None:
            analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        if internal_standard is not None:
            self.internal_standard = internal_standard
            self.minimal_analytes.update([internal_standard])

        with self.pbar.set(total=len(self.data), desc='Ratio Calculation') as prog:
            for s in self.data.values():
                s.ratio(internal_standard=self.internal_standard, analytes=analytes)
                prog.update()

        self.stages_complete.update(['ratios'])
        self.focus_stage = 'ratios'
        return

    def srm_load_database(self, srms_used=None, reload=False):
        if not hasattr(self, 'srmdat') or reload:
            # load SRM info
            srmdat = srms.read_table(self.srmfile)
            srmdat = srmdat.loc[srms_used]
            srmdat.reset_index(inplace=True)
            srmdat.set_index(['SRM', 'Item'], inplace=True)
            # empty columns for mol_ratio and mol_ratio_err
            srmdat.loc[:, 'mol_ratio'] = np.nan
            srmdat.loc[:, 'mol_ratio_err'] = np.nan

            # get element name
            internal_el = get_analyte_name(self.internal_standard)
            # calculate ratios to internal_standard for all elements

            analyte_srm_link = {}
            warns = []
            srmsubs = []

            for srm in srms_used:
                srmsub = srmdat.loc[srm]

                # determine analyte - Item pairs in table
                ad = {}
                for a in self.analytes:
                    # check ig there's an exact match of form [Mass][Element] in srmdat
                    mna = analyte_2_massname(a)
                    if mna in srmsub.index:
                        ad[a] = mna
                    else:
                        # if not, match by element name.
                        item = srmsub.index[srmsub.index.str.contains(get_analyte_name(a))].values
                        if len(item) > 1:
                            item = item[item == get_analyte_name(a)]
                        if len(item) == 1:
                            ad[a] = item[0]
                        else:
                            warns.append('   No {:} value for {:}.'.format(a, srm))

                analyte_srm_link[srm] = ad

                # find denominator
                denom = srmsub.loc[ad[self.internal_standard]]
                # calculate denominator composition (multiplier to account for stoichiometry,
                # e.g. if internal standard is Na, N will be 2 if measured in SRM as Na2O)
                N = float(decompose_molecule(ad[self.internal_standard])[internal_el])

                # calculate molar ratio
                ind = (srm, list(ad.values()))
                srmdat.loc[ind, 'mol_ratio'] = srmdat.loc[ind, 'mol/g'] / (denom['mol/g'] * N)
                srmdat.loc[ind, 'mol_ratio_err'] = (((srmdat.loc[ind, 'mol/g_err'] / srmdat.loc[ind, 'mol/g'])**2 +
                                                    (denom['mol/g_err'] / denom['mol/g']))**0.5 *
                                                    srmdat.loc[ind, 'mol_ratio'])  # propagate uncertainty

            srmdat.dropna(subset=['mol_ratio'], inplace=True)

            # compile stand-alone table of SRM values
            srmtab = pd.DataFrame(index=srms_used, columns=pd.MultiIndex.from_product([self.analytes, ['mean', 'err']]))
            for srm, ad in analyte_srm_link.items():
                for a, k in ad.items():
                    srmtab.loc[srm, (a, 'mean')] = srmdat.loc[(srm, k), 'mol_ratio']
                    srmtab.loc[srm, (a, 'err')] = srmdat.loc[(srm, k), 'mol_ratio_err']

            # record outputs
            self.srmdat = srmdat  # the full SRM table
            self._analyte_srmdat_link = analyte_srm_link  # dict linking analyte names to rows in srmdat
            self.srmtab = srmtab.reindex(self.analytes_sorted(), level=0, axis=1).astype(float)  # a summary of relevant mol/mol values only

            # Print any warnings
            if len(warns) > 0:
                print('WARNING: Some analytes are not present in the SRM database:')
                print('\n'.join(warns))
    
    def srm_compile_measured(self, n_min=10, focus_stage='ratios'):
        """
        Compile mean and standard errors of measured SRMs

        Parameters
        ----------
        n_min : int
            The minimum number of points to consider as a valid measurement.
            Default = 10.
        """
        warns = []
        # compile mean and standard errors of samples
        for s in self.stds:
            s_stdtab = pd.DataFrame(columns=pd.MultiIndex.from_product([s.analytes, ['err', 'mean']]))
            s_stdtab.index.name = 'uTime'

            if not s.n > 0:
                s.stdtab = s_stdtab
                continue

            for n in range(1, s.n + 1):
                ind = s.ns == n
                if sum(ind) >= n_min:
                    for a in s.analytes:
                        aind = ind & ~np.isnan(nominal_values(s.data[focus_stage][a]))
                        s_stdtab.loc[np.nanmean(s.uTime[s.ns == n]),
                                   (a, 'mean')] = np.nanmean(nominal_values(s.data[focus_stage][a][aind]))
                        s_stdtab.loc[np.nanmean(s.uTime[s.ns == n]),
                                   (a, 'err')] = np.nanstd(nominal_values(s.data[focus_stage][a][aind])) / np.sqrt(sum(aind))
                else:
                    warns.append('   Ablation {:} of SRM measurement {:} ({:} points)'.format(n, s.sample, sum(ind)))

            # sort column multiindex
            s_stdtab = s_stdtab.loc[:, s_stdtab.columns.sort_values()]
            # sort row index
            s_stdtab.sort_index(inplace=True)

            # create 'SRM' column for naming SRM
            s_stdtab.loc[:, 'STD'] = s.sample

            s.stdtab = s_stdtab

        if len(warns) > 0:
            print('WARNING: Some SRM ablations have been excluded because they do not contain enough data:')
            print('\n'.join(warns))
            print("To *include* these ablations, reduce the value of n_min (currently {:})".format(n_min))

        # compile them into a table
        stdtab = pd.concat([s.stdtab for s in self.stds]).apply(pd.to_numeric, 1, errors='ignore')
        stdtab = stdtab.reindex(self.analytes_sorted() + ['STD'], level=0, axis=1)

        # identify groups of consecutive SRMs
        ts = stdtab.index.values
        start_times = [s.uTime[0] for s in self.data.values()]

        lastpos = sum(ts[0] > start_times)
        group = [1]
        for t in ts[1:]:
            pos = sum(t > start_times)
            rpos = pos - lastpos
            if rpos <= 1:
                group.append(group[-1])
            else:
                group.append(group[-1] + 1)
            lastpos = pos

        stdtab.loc[:, 'group'] = group
        # calculate centre time for the groups
        stdtab.loc[:, 'gTime'] = np.nan

        for g, d in stdtab.groupby('group'):
            ind = stdtab.group == g
            stdtab.loc[ind, 'gTime'] = stdtab.loc[ind].index.values.mean()

        self.stdtab = stdtab

    def srm_id_auto(self, srms_used=['NIST610', 'NIST612', 'NIST614'], analytes=None, n_min=10, reload_srm_database=False):
        """
        Function for automarically identifying SRMs using KMeans clustering.

        KMeans is performed on the log of SRM composition, which aids separation
        of relatively similar SRMs within a large compositional range.

        Parameters
        ----------
        srms_used : iterable
            Which SRMs have been used. Must match SRM names
            in SRM database *exactly* (case sensitive!).
        analytes : array-like
            Which analytes to base the identification on. If None,
            all analytes are used (default).
        n_min : int
            The minimum number of data points a SRM measurement
            must contain to be included.
        reload_srm_database : bool
            Whether or not to re-load the SRM database before running the function.
        """
        if isinstance(srms_used, str):
            srms_used = [srms_used]
                
        if analytes is None:
            analytes = self.analytes
        
        # compile measured SRM data
        self.srm_compile_measured(n_min)

        # load SRM database
        self.srm_load_database(srms_used, reload_srm_database)
        
        # get and scale mean srm values for all analytes
        srmid = self.srmtab.loc[:, idx[:, 'mean']]
        _srmid = scale(np.log(srmid))
        srm_labels = srmid.index.values

        # get and scale measured srm values for all analytes
        stdid = self.stdtab.loc[:, idx[:, 'mean']]
        _stdid = scale(np.log(stdid))

        # fit KMeans classifier to srm database
        classifier = KMeans(len(srms_used)).fit(_srmid)
        # apply classifier to measured data
        std_classes = classifier.predict(_stdid)

        # get srm names from classes
        std_srm_labels = np.array([srm_labels[np.argwhere(classifier.labels_ == i)][0][0] for i in std_classes])

        self.stdtab.loc[:, 'SRM'] = std_srm_labels
        self.srms_ided = True

        self.srm_build_calib_table()

    def srm_build_calib_table(self):
        """
        Combine SRM database values and identified measured values into a calibration database.
        """
        caltab = self.stdtab.reset_index()
        caltab.set_index(['gTime', 'uTime'], inplace=True)
        levels = ['meas_' + c if c != '' else c for c in caltab.columns.levels[1]]
        caltab.columns.set_levels(levels, 1, inplace=True)

        for a in self.analytes:
            if a == self.internal_standard:
                continue

            caltab.loc[:, (a, 'srm_mean')] = self.srmtab.loc[caltab.SRM, (a, 'mean')].values
            caltab.loc[:, (a, 'srm_err')] = self.srmtab.loc[caltab.SRM, (a, 'err')].values
            
        self.caltab = caltab.reindex(self.stdtab.columns.levels[0], axis=1, level=0)

    
    # def srm_id_auto(self, srms_used=['NIST610', 'NIST612', 'NIST614'], n_min=10, reload_srm_database=False):
    #     """
    #     Function for automarically identifying SRMs
    
    #     Parameters
    #     ----------
    #     srms_used : iterable
    #         Which SRMs have been used. Must match SRM names
    #         in SRM database *exactly* (case sensitive!).
    #     n_min : int
    #         The minimum number of data points a SRM measurement
    #         must contain to be included.
    #     """
    #     if isinstance(srms_used, str):
    #         srms_used = [srms_used]
            
    #     # get mean and standard deviations of measured standards
    #     self.srm_compile_measured(n_min)
    #     stdtab = self.stdtab.copy()
    #     stdtab.loc[:, 'SRM'] = ''


    #     # load corresponding SRM database
    #     self.srm_load_database(srms_used, reload_srm_database)

    #     # create blank srm table
    #     srm_tab = self.srmdat.loc[:, ['mol_ratio', 'element']].reset_index().pivot(index='SRM', columns='element', values='mol_ratio')

    #     # Auto - ID STDs
    #     # 1. identify elements in measured SRMS with biggest range of values
    #     meas_tab = stdtab.loc[:, (slice(None), 'mean')]  # isolate means of standards
    #     meas_tab.columns = meas_tab.columns.droplevel(1)  # drop 'mean' column names
    #     meas_tab.columns = [re.findall('[A-Za-z]+', a)[0] for a in meas_tab.columns]  # rename to element names
    #     meas_tab = meas_tab.T.groupby(level=0).first().T  # remove duplicate columns

    #     ranges = nominal_values(meas_tab.apply(lambda a: np.ptp(a) / np.nanmean(a), 0))  # calculate relative ranges of all elements
    #     # (used as weights later)

    #     # 2. Work out which standard is which
    #     # normalise all elements between 0-1
    #     def normalise(a):
    #         a = nominal_values(a)
    #         if np.nanmin(a) < np.nanmax(a):
    #             return (a - np.nanmin(a)) / np.nanmax(a - np.nanmin(a))
    #         else:
    #             return np.ones(a.shape)

    #     nmeas = meas_tab.apply(normalise, 0)
    #     nmeas.dropna(1, inplace=True)  # remove elements with NaN values
    #     # nmeas.replace(np.nan, 1, inplace=True)
    #     nsrm_tab = srm_tab.apply(normalise, 0)
    #     nsrm_tab.dropna(1, inplace=True)
    #     # nsrm_tab.replace(np.nan, 1, inplace=True)

    #     for uT, r in nmeas.iterrows():  # for each standard...
    #         idx = np.nansum(((nsrm_tab - r) * ranges)**2, 1)
    #         idx = abs((nsrm_tab - r) * ranges).sum(1)
    #         # calculate the absolute difference between the normalised elemental
    #         # values for each measured SRM and the SRM table. Each element is
    #         # multiplied by the relative range seen in that element (i.e. range / mean
    #         # measuerd value), so that elements with a large difference are given
    #         # more importance in identifying the SRM.   
    #         # This produces a table, where wach row contains the difference between
    #         # a known vs. measured SRM. The measured SRM is identified as the SRM that
    #         # has the smallest weighted sum value.
    #         stdtab.loc[uT, 'SRM'] = srm_tab.index[idx == min(idx)].values[0]

    #     # calculate mean time for each SRM
    #     # reset index and sort
    #     stdtab.reset_index(inplace=True)
    #     stdtab.sort_index(1, inplace=True)
    #     # isolate STD and uTime
    #     uT = stdtab.loc[:, ['gTime', 'STD']].set_index('STD')
    #     uT.sort_index(inplace=True)
    #     uTm = uT.groupby(level=0).mean()  # mean uTime for each SRM
    #     # replace uTime values with means
    #     stdtab.set_index(['STD'], inplace=True)
    #     stdtab.loc[:, 'gTime'] = uTm
    #     # reset index
    #     stdtab.reset_index(inplace=True)
    #     stdtab.set_index(['STD', 'SRM', 'gTime'], inplace=True)

    #     # combine to make SRM reference tables
    #     srmtabs = Bunch()
    #     for a in self.analytes:
    #         el = re.findall('[A-Za-z]+', a)[0]

    #         sub = stdtab.loc[:, a]

    #         srmsub = self.srmdat.loc[self.srmdat.element == el, ['mol_ratio', 'mol_ratio_err']]

    #         srmtab = sub.join(srmsub)
    #         srmtab.columns = ['meas_err', 'meas_mean', 'srm_mean', 'srm_err']

    #         srmtabs[a] = srmtab

    #     self.srmtabs = pd.concat(srmtabs).apply(nominal_values).sort_index()
    #     self.srmtabs.dropna(subset=['srm_mean'], inplace=True)
    #     # replace any nan error values with zeros - nans cause problems later.
    #     self.srmtabs.loc[:, ['meas_err', 'srm_err']] = self.srmtabs.loc[:, ['meas_err', 'srm_err']].replace(np.nan, 0)

    #     # remove internal standard from calibration elements
    #     self.srmtabs.drop(self.internal_standard, level=0, inplace=True)

    #     self.srms_ided = True
    #     return

    def clear_calibration(self):
        if self.srms_ided:
            del self.stdtab
            del self.srmdat
            del self.srmtab

            self.srms_ided = False

        if 'calibrated' in self.stages_complete:
            del self.calib_params
            del self.calib_ps

            self.stages_complete.remove('calibrated')
            self.focus_stage = 'ratios'
            self.set_focus('ratios')

    # apply calibration to data
    @_log
    def calibrate(self, analytes=None, drift_correct=True,
                  srms_used=['NIST610', 'NIST612', 'NIST614'],
                  zero_intercept=True, n_min=10, reload_srm_database=False):
        """
        Calibrates the data to measured SRM values.

        Assumes that y intercept is zero.

        Parameters
        ----------  
        analytes : str or iterable
            Which analytes you'd like to calibrate. Defaults to all.
        drift_correct : bool
            Whether to pool all SRM measurements into a single calibration,
            or vary the calibration through the run, interpolating
            coefficients between measured SRMs.
        srms_used : str or iterable
            Which SRMs have been measured. Must match names given in
            SRM data file *exactly*.
        n_min : int
            The minimum number of data points an SRM measurement
            must have to be included.

        Returns
        -------
        None
        """
        if analytes is None:
            analytes = self.analytes_sorted(self.analytes.difference([self.internal_standard]))
        elif isinstance(analytes, str):
            analytes = [analytes]

        if isinstance(srms_used, str):
            srms_used = [srms_used]

        if not hasattr(self, 'srmtabs'):
            self.srm_id_auto(srms_used=srms_used, n_min=n_min, reload_srm_database=reload_srm_database)

        # make container for calibration params
        gTime = np.asanyarray(self.caltab.index.levels[0])
        if not hasattr(self, 'calib_params'):
            self.calib_params = pd.DataFrame(columns=pd.MultiIndex.from_product([analytes, ['m']]),
                                             index=gTime)
        
        if zero_intercept:
            fn  = lambda x, m: x * m
        else:
            fn = lambda x, m, c: x * m + c

        for a in analytes:
            if zero_intercept:
                if (a, 'c') in self.calib_params:
                    self.calib_params.drop((a, 'c'), 1, inplace=True)
            if drift_correct:
                for g in gTime:
                    if self.caltab.loc[g].size == 0:
                        continue
                    meas = self.caltab.loc[g, (a, 'meas_mean')].values
                    meas_err = self.caltab.loc[g, (a, 'meas_err')].values
                    srm = self.caltab.loc[g, (a, 'srm_mean')].values
                    srm_err = self.caltab.loc[g, (a, 'srm_err')].values
                    # TODO: replace curve_fit with Sambridge's 2D likelihood function for better uncertainty incorporation?
                    sigma = np.sqrt(meas_err**2 + srm_err**2)
                    if len(meas) > 1:
                        # multiple SRMs - do a regression
                        p, cov = curve_fit(fn, meas, srm, sigma=sigma)
                        pe = unc.correlated_values(p, cov)                
                        self.calib_params.loc[g, (a, 'm')] = pe[0]
                        if not zero_intercept:
                            self.calib_params.loc[g, (a, 'c')] = pe[1]
                    else:
                        # deal with case where there's only one datum
                        self.calib_params.loc[g, (a, 'm')] = (un.uarray(srm, srm_err) / 
                                                              un.uarray(meas, meas_err))[0]
                        if not zero_intercept:
                            self.calib_params.loc[g, (a, 'c')] = 0
            else:
                meas = self.caltab.loc[:, (a, 'meas_mean')].values
                meas_err = self.caltab.loc[:, (a, 'meas_err')].values
                srm = self.caltab.loc[:, (a, 'srm_mean')].values
                srm_err = self.caltab.loc[:, (a, 'srm_err')].values
                # TODO: replace curve_fit with Sambridge's 2D likelihood function for better uncertainty incorporation?
                sigma = np.sqrt(meas_err**2 + srm_err**2)
                
                if len(meas) > 1:
                    p, cov = curve_fit(fn, meas, srm, sigma=sigma)
                    pe = unc.correlated_values(p, cov)                
                    self.calib_params.loc[:, (a, 'm')] = pe[0]
                    if not zero_intercept:
                        self.calib_params.loc[:, (a, 'c')] = pe[1]
                else:
                    self.calib_params.loc[:, (a, 'm')] = (un.uarray(srm, srm_err) / 
                                                          un.uarray(meas, meas_err))[0]
                    if not zero_intercept:
                        self.calib_params.loc[:, (a, 'c')] = 0

        if self.calib_params.index.min() == 0:
            self.calib_params.drop(0, inplace=True)
            self.calib_params.drop(self.calib_params.index.max(), inplace=True)
        self.calib_params.loc[0, :] = self.calib_params.loc[self.calib_params.index.min(), :]
        maxuT = np.max([d.uTime.max() for d in self.data.values()])  # calculate max uTime
        self.calib_params.loc[maxuT, :] = self.calib_params.loc[self.calib_params.index.max(), :]
        # sort indices for slice access
        self.calib_params.sort_index(1, inplace=True)
        self.calib_params.sort_index(0, inplace=True)

        # calculcate interpolators for applying calibrations
        self.calib_ps = Bunch()
        for a in analytes:
            # TODO: revisit un_interp1d to see whether it plays well with correlated values. 
            # Possible re-write to deal with covariance matrices?
            self.calib_ps[a] = {'m': un_interp1d(self.calib_params.index.values,
                                                self.calib_params.loc[:, (a, 'm')].values)}
            if not zero_intercept:
                self.calib_ps[a]['c'] = un_interp1d(self.calib_params.index.values,
                                                    self.calib_params.loc[:, (a, 'c')].values)

        with self.pbar.set(total=len(self.data), desc='Applying Calibrations') as prog:
            for d in self.data.values():
                d.calibrate(self.calib_ps, analytes)
                prog.update()

        # record SRMs used for plotting
        markers = 'osDsv<>PX'  # for future implementation of SRM-specific markers.
        if not hasattr(self, 'srms_used'):
            self.srms_used = set(srms_used)
        else:
            self.srms_used.update(srms_used)
        self.srm_mdict = {k: markers[i] for i, k in enumerate(self.srms_used)}

        self.stages_complete.update(['calibrated'])
        self.focus_stage = 'calibrated'

        return

    # def calibrate(self, analytes=None, drift_correct=True,
    #               srms_used=['NIST610', 'NIST612', 'NIST614'],
    #               zero_intercept=True, n_min=10, reload_srm_database=False):
    #     """
    #     Calibrates the data to measured SRM values.

    #     Assumes that y intercept is zero.

    #     Parameters
    #     ----------  
    #     analytes : str or iterable
    #         Which analytes you'd like to calibrate. Defaults to all.
    #     drift_correct : bool
    #         Whether to pool all SRM measurements into a single calibration,
    #         or vary the calibration through the run, interpolating
    #         coefficients between measured SRMs.
    #     srms_used : str or iterable
    #         Which SRMs have been measured. Must match names given in
    #         SRM data file *exactly*.
    #     n_min : int
    #         The minimum number of data points an SRM measurement
    #         must have to be included.

    #     Returns
    #     -------
    #     None
    #     """
    #     if analytes is None:
    #         analytes = self.analytes.difference(self.internal_standard)
    #     elif isinstance(analytes, str):
    #         analytes = [analytes]

    #     if isinstance(srms_used, str):
    #         srms_used = [srms_used]

    #     if not hasattr(self, 'srmtabs'):
    #         self.srm_id_auto(srms_used=srms_used, n_min=n_min, reload_srm_database=reload_srm_database)

    #     # make container for calibration params
    #     if not hasattr(self, 'calib_params'):
    #         gTime = self.stdtab.gTime.unique()
    #         self.calib_params = pd.DataFrame(columns=pd.MultiIndex.from_product([analytes, ['m']]),
    #                                         index=gTime)

    #     calib_analytes = self.srmtabs.index.get_level_values(0).unique()

    #     if zero_intercept:
    #         fn  = lambda x, m: x * m
    #     else:
    #         fn = lambda x, m, c: x * m + c

    #     for a in calib_analytes:
    #         if zero_intercept:
    #             if (a, 'c') in self.calib_params:
    #                 self.calib_params.drop((a, 'c'), 1, inplace=True)
    #         if drift_correct:
    #             for g in self.stdtab.gTime.unique():
    #                 ind = idx[a, :, :, g]
    #                 if self.srmtabs.loc[ind].size == 0:
    #                     continue
    #                 # try:
    #                 meas = self.srmtabs.loc[ind, 'meas_mean']
    #                 srm = self.srmtabs.loc[ind, 'srm_mean']
    #                 # TODO: replace curve_fit with Sambridge's 2D likelihood function for better uncertainty incorporation.
    #                 merr = self.srmtabs.loc[ind, 'meas_err']
    #                 serr = self.srmtabs.loc[ind, 'srm_err']
    #                 sigma = np.sqrt(merr**2 + serr**2)

    #                 if len(meas) > 1:
    #                     # multiple SRMs - do a regression
    #                     p, cov = curve_fit(fn, meas, srm, sigma=sigma)
    #                     pe = unc.correlated_values(p, cov)                
    #                     self.calib_params.loc[g, (a, 'm')] = pe[0]
    #                     if not zero_intercept:
    #                         self.calib_params.loc[g, (a, 'c')] = pe[1]
    #                 else:
    #                     # deal with case where there's only one datum
    #                     self.calib_params.loc[g, (a, 'm')] = (un.uarray(srm, serr) / 
    #                                                           un.uarray(meas, merr))[0]
    #                     if not zero_intercept:
    #                         self.calib_params.loc[g, (a, 'c')] = 0

    #                 # This should be obsolete, because no-longer sourcing locator from calib_params index.
    #                 # except KeyError:
    #                 #     # If the calibration is being recalculated, calib_params
    #                 #     # will have t=0 and t=max(uTime) values that are outside
    #                 #     # the srmtabs index.
    #                 #     # If this happens, drop them, and re-fill them at the end.
    #                 #     self.calib_params.drop(g, inplace=True)
    #         else:
    #             ind = idx[a, :, :, :]
    #             meas = self.srmtabs.loc[ind, 'meas_mean']
    #             srm = self.srmtabs.loc[ind, 'srm_mean']
    #             merr = self.srmtabs.loc[ind, 'meas_err']
    #             serr = self.srmtabs.loc[ind, 'srm_err']
    #             sigma = np.sqrt(merr**2 + serr**2)
                
    #             if len(meas) > 1:
    #                 p, cov = curve_fit(fn, meas, srm, sigma=sigma)
    #                 pe = unc.correlated_values(p, cov)                
    #                 self.calib_params.loc[:, (a, 'm')] = pe[0]
    #                 if not zero_intercept:
    #                     self.calib_params.loc[:, (a, 'c')] = pe[1]
    #             else:
    #                 self.calib_params.loc[:, (a, 'm')] = (un.uarray(srm, serr) / 
    #                                                       un.uarray(meas, merr))[0]
    #                 if not zero_intercept:
    #                     self.calib_params.loc[:, (a, 'c')] = 0

    #     # if fill:
    #     # fill in uTime=0 and uTime = max cases for interpolation
    #     if self.calib_params.index.min() == 0:
    #         self.calib_params.drop(0, inplace=True)
    #         self.calib_params.drop(self.calib_params.index.max(), inplace=True)
    #     self.calib_params.loc[0, :] = self.calib_params.loc[self.calib_params.index.min(), :]
    #     maxuT = np.max([d.uTime.max() for d in self.data.values()])  # calculate max uTime
    #     self.calib_params.loc[maxuT, :] = self.calib_params.loc[self.calib_params.index.max(), :]
    #     # sort indices for slice access
    #     self.calib_params.sort_index(1, inplace=True)
    #     self.calib_params.sort_index(0, inplace=True)

    #     # calculcate interpolators for applying calibrations
    #     self.calib_ps = Bunch()
    #     for a in analytes:
    #         # TODO: revisit un_interp1d to see whether it plays well with correlated values. 
    #         # Possible re-write to deal with covariance matrices?
    #         self.calib_ps[a] = {'m': un_interp1d(self.calib_params.index.values,
    #                                             self.calib_params.loc[:, (a, 'm')].values)}
    #         if not zero_intercept:
    #             self.calib_ps[a]['c'] = un_interp1d(self.calib_params.index.values,
    #                                                 self.calib_params.loc[:, (a, 'c')].values)

    #     with self.pbar.set(total=len(self.data), desc='Applying Calibrations') as prog:
    #         for d in self.data.values():
    #             d.calibrate(self.calib_ps, analytes)
    #             prog.update()

    #     # record SRMs used for plotting
    #     markers = 'osDsv<>PX'  # for future implementation of SRM-specific markers.
    #     if not hasattr(self, 'srms_used'):
    #         self.srms_used = set(srms_used)
    #     else:
    #         self.srms_used.update(srms_used)
    #     self.srm_mdict = {k: markers[i] for i, k in enumerate(self.srms_used)}

    #     self.stages_complete.update(['calibrated'])
    #     self.focus_stage = 'calibrated'

    #     return


    # data filtering
    # TODO: Re-factor filtering to use 'classifier' objects?

    # functions for calculating mass fraction (ppm)
    def get_sample_list(self, save_as=None, overwrite=False):
        """
        Save a csv list of of all samples to be populated with internal standard concentrations.

        Parameters
        ----------
        save_as : str
            Location to save the file. Defaults to the export directory.
        """
        if save_as is None:
            save_as = os.path.join(self.export_dir, 'internal_standard_massfrac.csv')
        if os.path.exists(save_as):
            if not overwrite:
                raise IOError('File exists. Please change the save location or specify overwrite=True')

        empty = pd.DataFrame(index=self.samples, columns=['int_stand_massfrac'])
        empty.to_csv(save_as)
        print(self._wrap_text('Sample List saved to {} \nPlease modify and re-import using read_internal_standard_concs()'.format(save_as)))

    def read_internal_standard_concs(self, sample_concs=None):
        """
        Load in a per-sample list of internal sample concentrations.
        """
        if sample_concs is None:
            sample_concs = os.path.join(self.export_dir, 'internal_standard_massfrac.csv')
        
        return pd.read_csv(sample_concs, index_col=0)


    @_log
    def calculate_mass_fraction(self, internal_standard_conc=None, analytes=None, analyte_masses=None):
        """
        Convert calibrated molar ratios to mass fraction.

        Parameters
        ----------
        internal_standard_conc : float, pandas.DataFrame or str
            The concentration of the internal standard in your samples.
        """

        if analytes is None:
            analytes = self.analytes.difference(self.internal_standard)

        if analyte_masses is None:
            analyte_masses = analyte_mass(self.analytes, False)

        isc = internal_standard_conc

        if isinstance(isc, str) or isc is None:
            isc = self.read_internal_standard_concs(isc)

        if not isinstance(isc, pd.core.frame.DataFrame):
            with self.pbar.set(total=len(self.data), desc='Calculating Mass Fractions') as prog:        
                for d in self.data.values():
                    d.calc_mass_fraction(isc, analytes, analyte_masses)
                    prog.update() 
        else:
            with self.pbar.set(total=len(self.data), desc='Calculating Mass Fractions') as prog:        
                for k, d in self.data.items():
                    if k in isc.index:
                        d.calc_mass_fraction(isc.loc[k].values[0], analytes, analyte_masses)
                    else:
                        d.calc_mass_fraction(np.nan, analytes, analyte_masses)
                    prog.update()

        self.stages_complete.update(['mass_fraction'])
        self.focus_stage = 'mass_fraction'



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
        # Check if a subset containing the same samples already exists.
        for k, v in self.subsets.items():
            if set(v) == set(samples) and k != 'not_in_set':
                return k

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
        Remove all points containing data below zero (which are impossible!)
        """
        if focus_stage is None:
            focus_stage = self.focus_stage

        for s in self.data.values():
            ind = np.ones(len(s.Time), dtype=bool)
            for v in s.data[focus_stage].values():
                ind = ind & (nominal_values(v) > 0)

            for k in s.data[focus_stage].keys():
                s.data[focus_stage][k][~ind] = unc.ufloat(np.nan, np.nan)

        self.set_focus(focus_stage)

        return

    @_log
    def filter_threshold(self, analyte, threshold,
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

        self.minimal_analytes.update([analyte])

        with self.pbar.set(total=len(samples), desc='Threshold Filter') as prog:
            for s in samples:
                self.data[s].filter_threshold(analyte, threshold)
                prog.update()

    @_log
    def filter_threshold_percentile(self, analyte, percentiles, level='population', filt=False,
                                    samples=None, subset=None):
        """
        Applies a threshold filter to the data.

        Generates two filters above and below the threshold value for a
        given analyte.

        Parameters
        ----------
        analyte : str
            The analyte that the filter applies to.
        percentiles : float or iterable of len=2
            The percentile values.
        level : str
            Whether to calculate percentiles from the entire dataset
            ('population') or for each individual sample ('individual')
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
        params = locals()
        del(params['self'])

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        self.minimal_analytes.update([analyte])

        if isinstance(percentiles, (int, float)):
            percentiles = [percentiles]

        if level == 'population':
            # Get all samples
            self.get_focus(filt=filt, subset=subset, nominal=True)
            dat = self.focus[analyte][~np.isnan(self.focus[analyte])]

            # calculate filter limits
            lims = np.percentile(dat, percentiles)

        # Calculate filter for individual samples
        with self.pbar.set(total=len(samples), desc='Percentile theshold filter') as prog:
            for s in samples:
                d = self.data[s]
                setn = d.filt.maxset + 1
                g = d.focus[analyte]

                if level == 'individual':
                    gt = nominal_values(g)
                    lims = np.percentile(gt[~np.isnan(gt)], percentiles)

                if len(lims) == 1:
                    above = g >= lims[0]
                    below = g < lims[0]

                    d.filt.add(analyte + '_{:.1f}-pcnt_below'.format(percentiles[0]),
                            below,
                            'Values below {:.1f}th {:} percentile ({:.2e})'.format(percentiles[0], analyte, lims[0]),
                            params, setn=setn)
                    d.filt.add(analyte + '_{:.1f}-pcnt_above'.format(percentiles[0]),
                            above,
                            'Values above {:.1f}th {:} percentile ({:.2e})'.format(percentiles[0], analyte, lims[0]),
                            params, setn=setn)

                elif len(lims) == 2:
                    inside = (g >= min(lims)) & (g <= max(lims))
                    outside = (g < min(lims)) | (g > max(lims))

                    lpc = '-'.join(['{:.1f}'.format(p) for p in percentiles])
                    d.filt.add(analyte + '_' + lpc + '-pcnt_inside',
                            inside,
                            'Values between ' + lpc + ' ' + analyte + 'percentiles',
                            params, setn=setn)
                    d.filt.add(analyte + '_' + lpc + '-pcnt_outside',
                            outside,
                            'Values outside ' + lpc + ' ' + analyte + 'percentiles',
                            params, setn=setn)
                prog.update()
        return


    @_log
    def filter_gradient_threshold(self, analyte, threshold, win=15,
                                  samples=None, subset=None):
        """
        Calculate a gradient threshold filter to the data.

        Generates two filters above and below the threshold value for a
        given analyte.

        Parameters
        ----------
        analyte : str
            The analyte that the filter applies to.
        win : int
            The window over which to calculate the moving gradient
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

        self.minimal_analytes.update([analyte])
        
        with self.pbar.set(total=len(samples), desc='Gradient Threshold Filter') as prog:
            for s in samples:
                self.data[s].filter_gradient_threshold(analyte, win, threshold)
                prog.update()

    @_log
    def filter_gradient_threshold_percentile(self, analyte, percentiles, level='population', win=15, filt=False,
                                             samples=None, subset=None):
        """
        Calculate a gradient threshold filter to the data.

        Generates two filters above and below the threshold value for a
        given analyte.

        Parameters
        ----------
        analyte : str
            The analyte that the filter applies to.
        win : int
            The window over which to calculate the moving gradient
        percentiles : float or iterable of len=2
            The percentile values.
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
        params = locals()
        del(params['self'])

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        self.minimal_analytes.update([analyte])

        # Calculate gradients of all samples
        self.get_gradients(analytes=[analyte], win=win, filt=filt, subset=subset)
        grad = self.gradients[analyte][~np.isnan(self.gradients[analyte])]

        if isinstance(percentiles, (int, float)):
            percentiles = [percentiles]

        if level == 'population':
            # calculate filter limits
            lims = np.percentile(grad, percentiles)

        # Calculate filter for individual samples
        with self.pbar.set(total=len(samples), desc='Percentile Threshold Filter') as prog:
            for s in samples:
                d = self.data[s]
                setn = d.filt.maxset + 1
                g = calc_grads(d.Time, d.focus, [analyte], win)[analyte]

                if level == 'individual':
                    gt = nominal_values(g)
                    lims = np.percentile(gt[~np.isnan(gt)], percentiles)

                if len(lims) == 1:
                    above = g >= lims[0]
                    below = g < lims[0]

                    d.filt.add(analyte + '_{:.1f}-grd-pcnt_below'.format(percentiles[0]),
                            below,
                            'Gradients below {:.1f}th {:} percentile ({:.2e})'.format(percentiles[0], analyte, lims[0]),
                            params, setn=setn)
                    d.filt.add(analyte + '_{:.1f}-grd-pcnt_above'.format(percentiles[0]),
                            above,
                            'Gradients above {:.1f}th {:} percentile ({:.2e})'.format(percentiles[0], analyte, lims[0]),
                            params, setn=setn)

                elif len(lims) == 2:
                    inside = (g >= min(lims)) & (g <= max(lims))
                    outside = (g < min(lims)) | (g > max(lims))

                    lpc = '-'.join(['{:.1f}'.format(p) for p in percentiles])
                    d.filt.add(analyte + '_' + lpc + '-grd-pcnt_inside',
                            inside,
                            'Gradients between ' + lpc + ' ' + analyte + 'percentiles',
                            params, setn=setn)
                    d.filt.add(analyte + '_' + lpc + '-grd-pcnt_outside',
                            outside,
                            'Gradients outside ' + lpc + ' ' + analyte + 'percentiles',
                            params, setn=setn)
                prog.update()
        return

    @_log
    def filter_clustering(self, analytes, filt=False, normalise=True,
                          method='kmeans', include_time=False, samples=None,
                          sort=True, subset=None, level='sample', min_data=10, **kwargs):
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
            Which clustering algorithm to use:
            
            * 'meanshift': The `sklearn.cluster.MeanShift` algorithm.
              Automatically determines number of clusters
              in data based on the `bandwidth` of expected
              variation.
            * 'kmeans': The `sklearn.cluster.KMeans` algorithm. Determines
              the characteristics of a known number of clusters
              within the data. Must provide `n_clusters` to specify
              the expected number of clusters.
        level : str
            Whether to conduct the clustering analysis at the 'sample' or 
            'population' level.
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
            bandwidth : str or float
                The bandwith (float) or bandwidth method ('scott' or 'silverman')
                used to estimate the data bandwidth.
            bin_seeding : bool
                Modifies the behaviour of the meanshift algorithm. Refer to
                sklearn.cluster.meanshift documentation.
        K-Means Parameters
            n_clusters : int
                The number of clusters expected in the data.

        Returns
        -------
        None
        """
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        if isinstance(analytes, str):
            analytes = [analytes]

        self.minimal_analytes.update(analytes)

        if level == 'sample':
            with self.pbar.set(total=len(samples), desc='Clustering Filter') as prog:
                for s in samples:
                    self.data[s].filter_clustering(analytes=analytes, filt=filt,
                                                normalise=normalise,
                                                method=method,
                                                include_time=include_time,
                                                min_data=min_data,
                                                sort=sort,
                                                **kwargs)
                    prog.update()
        
        if level == 'population':
            if isinstance(sort, bool):
                sort_by = 0
            else:
                sort_by = sort
            
            name = '_'.join(analytes) + '_{}'.format(method)

            self.fit_classifier(name=name, analytes=analytes, method=method,
                                subset=subset, filt=filt, sort_by=sort_by, **kwargs)

            self.apply_classifier(name=name, subset=subset)

    @_log
    def fit_classifier(self, name, analytes, method, samples=None,
                       subset=None, filt=True, sort_by=0, **kwargs):
        """
        Create a clustering classifier based on all samples, or a subset.

        Parameters
        ----------
        name : str
            The name of the classifier.
        analytes : str or iterable
            Which analytes the clustering algorithm should consider.
        method : str
            Which clustering algorithm to use. Can be:

            'meanshift'
                The `sklearn.cluster.MeanShift` algorithm.
                Automatically determines number of clusters
                in data based on the `bandwidth` of expected
                variation.
            'kmeans'
                The `sklearn.cluster.KMeans` algorithm. Determines
                the characteristics of a known number of clusters
                within the data. Must provide `n_clusters` to specify
                the expected number of clusters.
        samples : iterable
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
            bandwidth : str or float
                The bandwith (float) or bandwidth method ('scott' or 'silverman')
                used to estimate the data bandwidth.
            bin_seeding : bool
                Modifies the behaviour of the meanshift algorithm. Refer to
                sklearn.cluster.meanshift documentation.
        K - Means Parameters
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

        with self.pbar.set(total=len(samples), desc='Applying ' + name + ' classifier') as prog:
            for s in samples:
                d = self.data[s]
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
                prog.update()

        return name

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

        self.minimal_analytes.update([x_analyte, y_analyte])

        with self.pbar.set(total=len(samples), desc='Correlation Filter') as prog:
            for s in samples:
                self.data[s].filter_correlation(x_analyte, y_analyte,
                                                window=window,
                                                r_threshold=r_threshold,
                                                p_threshold=p_threshold,
                                                filt=filt)
                prog.update()

    @_log
    def correlation_plots(self, x_analyte, y_analyte, window=15, filt=True, recalc=False, samples=None, subset=None, outdir=None):
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
        None
        """
        if outdir is None:
            outdir = self.report_dir + '/correlations/'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        
        if subset is not None:
            samples = self._get_samples(subset)
        elif samples is None:
            samples = self.subsets['All_Analyses']
        elif isinstance(samples, str):
            samples = [samples]

        with self.pbar.set(total=len(samples), desc='Drawing Plots') as prog:
            for s in samples:
                f, _ = self.data[s].correlation_plot(x_analyte=x_analyte, y_analyte=y_analyte,
                                                     window=window, filt=filt, recalc=recalc)
                f.savefig('{}/{}_{}-{}.pdf'.format(outdir, s, x_analyte, y_analyte))
                plt.close(f)
                prog.update()
        return
        


    @_log
    def filter_on(self, filt=None, analyte=None, samples=None, subset=None, show_status=False):
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
                self.data[s].filt.on(analyte, filt)
            except:
                warnings.warn("filt.on failure in sample " + s)

        if show_status:
            self.filter_status(subset=subset)
        return

    @_log
    def filter_off(self, filt=None, analyte=None, samples=None, subset=None, show_status=False):
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
                self.data[s].filt.off(analyte, filt)
            except:
                warnings.warn("filt.off failure in sample " + s)

        if show_status:
            self.filter_status(subset=subset)
        return

    def filter_status(self, sample=None, subset=None, stds=False):
        """
        Prints the current status of filters for specified samples.

        Parameters
        ----------
        sample : str
            Which sample to print.
        subset : str
            Specify a subset
        stds : bool
            Whether or not to include standards.
        """
        s = ''
        if sample is None and subset is None:
            if not self._has_subsets:
                s += 'Subset: All Samples\n\n'
                s += self.data[self.subsets['All_Samples'][0]].filt.__repr__()
            else:
                for n in sorted(str(sn) for sn in self._subset_names):
                    if n in self.subsets:
                        pass
                    elif int(n) in self.subsets:
                        n = int(n)
                        pass
                    s += 'Subset: ' + str(n) + '\n'
                    s += 'Samples: ' + ', '.join(self.subsets[n]) + '\n\n'
                    s += self.data[self.subsets[n][0]].filt.__repr__()
                if len(self.subsets['not_in_set']) > 0:
                    s += '\nNot in Subset:\n'
                    s += 'Samples: ' + ', '.join(self.subsets['not_in_set']) + '\n\n'
                    s += self.data[self.subsets['not_in_set'][0]].filt.__repr__()
            print(s)
            return

        elif sample is not None:
            s += 'Sample: ' + sample + '\n'
            s += self.data[sample].filt.__repr__()
            print(s)
            return

        elif subset is not None:
            if isinstance(subset, (str, int, float)):
                subset = [subset]
            for n in subset:
                s += 'Subset: ' + str(n) + '\n'
                s += 'Samples: ' + ', '.join(self.subsets[n]) + '\n\n'
                s += self.data[self.subsets[n][0]].filt.__repr__()
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
            self.data[s].filt.clear()
    
    @_log
    def filter_defragment(self, threshold, mode='include', filt=True, samples=None, subset=None):
        """
        Remove 'fragments' from the calculated filter

        Parameters
        ----------
        threshold : int
            Contiguous data regions that contain this number
            or fewer points are considered 'fragments'
        mode : str
            Specifies wither to 'include' or 'exclude' the identified
            fragments.
        filt : bool or filt string
            Which filter to apply the defragmenter to. Defaults to True
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

        for s in samples:
            f = self.data[s].filt.grab_filt(filt)
            self.data[s].filt.add(name='defrag_{:s}_{:.0f}'.format(mode, threshold),
                                  filt=filters.defrag(f, threshold, mode),
                                  info='Defrag {:s} filter with threshold {:.0f}'.format(mode, threshold),
                                  params=(threshold, mode, filt, samples, subset))
    
    @_log
    def filter_exclude_downhole(self, threshold, filt=True, samples=None, subset=None):
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
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        for s in samples:
            self.data[s].filter_exclude_downhole(threshold, filt)

    @_log
    def filter_trim(self, start=1, end=1, filt=True, samples=None, subset=None):
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
        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        for s in samples:
            self.data[s].filter_trim(start, end, filt)

    def filter_nremoved(self, filt=True, quiet=False):
        """
        Report how many data are removed by the active filters.
        """
        rminfo = {}
        for n in self.subsets['All_Samples']:
            s = self.data[n]
            rminfo[n] = s.filt_nremoved(filt)
        if not quiet:
            maxL = max([len(s) for s in rminfo.keys()])
            print('{string:{number}s}'.format(string='Sample ', number=maxL + 3) +
                  '{total:4s}'.format(total='tot') +
                  '{removed:4s}'.format(removed='flt') +
                  '{percent:4s}'.format(percent='%rm'))
            for k, (ntot, nfilt, pcrm) in rminfo.items():
                print('{string:{number}s}'.format(string=k, number=maxL + 3) +
                      '{total:4.0f}'.format(total=ntot) +
                      '{removed:4.0f}'.format(removed=nfilt) +
                      '{percent:4.0f}'.format(percent=pcrm))

        return rminfo
    
    @_log
    def optimise_signal(self, analytes, min_points=5,
                        threshold_mode='kde_first_max', 
                        threshold_mult=1., x_bias=0, filt=True,
                        weights=None, mode='minimise',
                        samples=None, subset=None):
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
        if samples is not None:
            subset = self.make_subset(samples)
        samples = self._get_samples(subset)

        if isinstance(analytes, str):
            analytes = [analytes]

        self.minimal_analytes.update(analytes)

        errs = []

        with self.pbar.set(total=len(samples), desc='Optimising Data selection') as prog:
            for s in samples:
                e = self.data[s].signal_optimiser(analytes=analytes, min_points=min_points,
                                                  threshold_mode=threshold_mode, threshold_mult=threshold_mult,
                                                  x_bias=x_bias, weights=weights, filt=filt, mode=mode)
                if e != '':
                    errs.append(e)
                prog.update()
        
        if len(errs) > 0:
            print('\nA Few Problems:\n' + '\n'.join(errs) + '\n\n  *** Check Optimisation Plots ***')
    
    @_log
    def optimisation_plots(self, overlay_alpha=0.5, samples=None, subset=None, **kwargs):
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
        if samples is not None:
            subset = self.make_subset(samples)
        samples = self._get_samples(subset)

        outdir=self.report_dir + '/optimisation_plots/'
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        
        with self.pbar.set(total=len(samples), desc='Drawing Plots') as prog:
            for s in samples:
                figs = self.data[s].optimisation_plot(overlay_alpha, **kwargs)
                
                n = 1
                for f, _ in figs:
                    if f is not None:
                        f.savefig(os.path.join(outdir, s + '_optim_{:.0f}.pdf'.format(n)))
                        plt.close(f)
                    n += 1
                prog.update()
        return

    # plot calibrations
    @_log
    def calibration_plot(self, analytes=None, datarange=True, loglog=False, ncol=3, save=True):
        return plot.calibration_plot(self=self, analytes=analytes, datarange=datarange, 
                                     loglog=loglog, ncol=ncol, save=save)

    # set the focus attribute for specified samples
    @_log
    def set_focus(self, focus_stage=None, samples=None, subset=None):
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
        if samples is not None:
            subset = self.make_subset(samples)
        
        if subset is None:
            subset = 'All_Analyses'

        samples = self._get_samples(subset)

        if focus_stage is None:
            focus_stage = self.focus_stage
        else:
            self.focus_stage = focus_stage

        for s in samples:
            self.data[s].setfocus(focus_stage)

    # fetch all the data from the data objects
    def get_focus(self, filt=False, samples=None, subset=None, nominal=False):
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
            which samples to get
        subset : str or int
            which subset to get

        Returns
        -------
        None
        """

        if samples is not None:
            subset = self.make_subset(samples)
        
        samples = self._get_samples(subset)

        # t = 0
        focus = {'uTime': []}
        focus.update({a: [] for a in self.analytes})

        for sa in samples:
            s = self.data[sa]
            focus['uTime'].append(s.uTime)
            ind = s.filt.grab_filt(filt)
            for a in self.analytes:
                tmp = s.focus[a].copy()
                tmp[~ind] = np.nan
                focus[a].append(tmp)

        if nominal:
            self.focus.update({k: nominal_values(np.concatenate(v)) for k, v, in focus.items()})
        else:
            self.focus.update({k: np.concatenate(v) for k, v, in focus.items()})

        return

    # fetch all the gradients from the data objects
    def get_gradients(self, analytes=None, win=15, filt=False, samples=None, subset=None, recalc=True):
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
            which samples to get
        subset : str or int
            which subset to get

        Returns
        -------
        None
        """
        if analytes is None:
            analytes = self.analytes

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        # check if gradients already calculated
        if all([self.data[s].grads_calced for s in samples]) and hasattr(self, 'gradients'):
            if not recalc:
                print("Using existing gradients. Set recalc=True to re-calculate.")
                return

        if not hasattr(self, 'gradients'):
            self.gradients = Bunch()

        # t = 0
        focus = {'uTime': []}
        focus.update({a: [] for a in analytes})

        with self.pbar.set(total=len(samples), desc='Calculating Gradients') as prog:
            for sa in samples:
                s = self.data[sa]
                focus['uTime'].append(s.uTime)
                ind = s.filt.grab_filt(filt)
                grads = calc_grads(s.uTime, s.focus, keys=analytes, win=win)
                for a in analytes:
                    tmp = grads[a]
                    tmp[~ind] = np.nan
                    focus[a].append(tmp)
                    s.grads = tmp
                s.grads_calced = True
                prog.update()

        self.gradients.update({k: np.concatenate(v) for k, v, in focus.items()})

        return

    def gradient_histogram(self, analytes=None, win=15, filt=False, bins=None, samples=None, subset=None, recalc=True, ncol=4):
        """
        Plot a histogram of the gradients in all samples.

        Parameters
        ----------
        filt : str, dict or bool
            Either logical filter expression contained in a str,
            a dict of expressions specifying the filter string to
            use for each analyte or a boolean. Passed to `grab_filt`.
        bins : None or array-like
            The bins to use in the histogram
        samples : str or list
            which samples to get
        subset : str or int
            which subset to get
        recalc : bool
            Whether to re-calculate the gradients, or use existing gradients.

        Returns
        -------
        fig, ax
        """
        if analytes is None:
            analytes = [a for a in self.analytes if self.internal_standard not in a]
        if not hasattr(self, 'gradients'):
            self.gradients = Bunch()

        ncol = int(ncol)
        n = len(analytes)
        nrow = plot.calc_nrow(n, ncol)

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        self.get_gradients(analytes=analytes, win=win, filt=filt, subset=subset, recalc=recalc)

        fig, axs = plt.subplots(nrow, ncol, figsize=[3. * ncol, 2.5 * nrow])

        if not isinstance(axs, np.ndarray):
            axs = [axs]

        i = 0
        for a, ax in zip(analytes, axs.flatten()):
            d = nominal_values(self.gradients[a])
            d = d[~np.isnan(d)]

            m, u = unitpicker(d, focus_stage=self.focus_stage, denominator=self.internal_standard)

            if bins is None:
                ibins = np.linspace(*np.percentile(d * m, [1, 99]), 50)
            else:
                ibins = bins

            ax.hist(d * m, bins=ibins, color=self.cmaps[a])
            ax.axvline(0, ls='dashed', lw=1, c=(0,0,0,0.7))

            ax.set_title(a, loc='left')
            if ax.is_first_col():
                ax.set_ylabel('N')
            ax.set_xlabel(u + '/s')

            i += 1

        if i < ncol * nrow:
            for ax in axs.flatten()[i:]:
                ax.set_visible(False)
        
        fig.tight_layout()

        return fig, axs

    # crossplot of all data
    @_log
    def crossplot(self, analytes=None, lognorm=True,
                  bins=25, filt=False, samples=None,
                  subset=None, figsize=(12, 12), save=False,
                  colourful=True, mode='hist2d', **kwargs):
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
        figsize : tuple
            Figure size (width, height) in inches.
        save : bool or str
            If True, plot is saves as 'crossplot.png', if str plot is
            saves as str.
        colourful : bool
            Whether or not the plot should be colourful :).
        mode : str
            'hist2d' (default) or 'scatter'

        Returns
        -------
        (fig, axes)
        """
        if analytes is None:
            analytes = self.analytes
        if self.focus_stage in ['ratio', 'calibrated']:
            analytes = [a for a in analytes if self.internal_standard not in a]

        # sort analytes
        try:
            analytes = sorted(analytes, key=lambda x: float(re.findall('[0-9.-]+', x)[0]))
        except IndexError:
            analytes = sorted(analytes)

        self.get_focus(filt=filt, samples=samples, subset=subset)

        fig, axes = plot.crossplot(dat=self.focus, keys=analytes, lognorm=lognorm,
                                   bins=bins, figsize=figsize, colourful=colourful,
                                   focus_stage=self.focus_stage, cmap=self.cmaps,
                                   denominator=self.internal_standard, mode=mode)

        if save or isinstance(save, str):
            if isinstance(save, str):
                fig.savefig(os.path.join(self.report_dir, save), dpi=200)            
            else:
                fig.savefig(os.path.join(self.report_dir, 'crossplot.png'), dpi=200)

        return fig, axes

    @_log
    def gradient_crossplot(self, analytes=None, win=15, lognorm=True,
                           bins=25, filt=False, samples=None,
                           subset=None, figsize=(12, 12), save=False,
                           colourful=True, mode='hist2d', recalc=True, **kwargs):
        """
        Plot analyte gradients against each other.

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
        figsize : tuple
            Figure size (width, height) in inches.
        save : bool or str
            If True, plot is saves as 'crossplot.png', if str plot is
            saves as str.
        colourful : bool
            Whether or not the plot should be colourful :).
        mode : str
            'hist2d' (default) or 'scatter'
        recalc : bool
            Whether to re-calculate the gradients, or use existing gradients.

        Returns
        -------
        (fig, axes)
        """

        if analytes is None:
            analytes = self.analytes
        if self.focus_stage in ['ratio', 'calibrated']:
            analytes = [a for a in analytes if self.internal_standard not in a]

        # sort analytes
        try:
            analytes = sorted(analytes, key=lambda x: float(re.findall('[0-9.-]+', x)[0]))
        except IndexError:
            analytes = sorted(analytes)

        samples = self._get_samples(subset)

        # calculate gradients
        self.get_gradients(analytes=analytes, win=win, filt=filt, subset=subset, recalc=recalc)

        # self.get_focus(filt=filt, samples=samples, subset=subset)
        # grads = calc_grads(self.focus.uTime, self.focus, analytes, win)

        fig, axes = plot.crossplot(dat=self.gradients, keys=analytes, lognorm=lognorm,
                                   bins=bins, figsize=figsize, colourful=colourful,
                                   focus_stage=self.focus_stage, cmap=self.cmaps,
                                   denominator=self.internal_standard, mode=mode)

        if save:
            fig.savefig(self.report_dir + '/g_crossplot.png', dpi=200)

        return fig, axes

    def histograms(self, analytes=None, bins=25, logy=False,
                   filt=False, colourful=True):
        """
        Plot histograms of analytes.

        Parameters
        ----------
        analytes : optional, array_like or str
            The analyte(s) to plot. Defaults to all analytes.
        bins : int
            The number of bins in each histogram (default = 25)
        logy : bool
            If true, y axis is a log scale.
        filt : str, dict or bool
            Either logical filter expression contained in a str,
            a dict of expressions specifying the filter string to
            use for each analyte or a boolean. Passed to `grab_filt`.
        colourful : bool
            If True, histograms are colourful :)

        Returns
        -------
        (fig, axes)
        """
        if analytes is None:
            analytes = self.analytes
        if self.focus_stage in ['ratio', 'calibrated']:
            analytes = [a for a in analytes if self.internal_standard not in a]
        if colourful:
            cmap = self.cmaps
        else:
            cmap = None

        self.get_focus(filt=filt)
        fig, axes = plot.histograms(self.focus, keys=analytes,
                                    bins=bins, logy=logy, cmap=cmap)

        return fig, axes
    
    def filter_effect(self, analytes=None, stats=['mean', 'std'], filt=True):
        """
        Quantify the effects of the active filters.
        
        Parameters
        ----------
        analytes : str or list
            Which analytes to consider.
        stats : list
            Which statistics to calculate.
        file : valid filter string or bool
            Which filter to consider. If True, applies all
            active filters.
        
        Returns
        -------
        pandas.DataFrame
            Contains statistics calculated for filtered and
            unfiltered data, and the filtered/unfiltered ratio.
        """
        if analytes is None:
            analytes = self.analytes
        if isinstance(analytes, str):
            analytes = [analytes]
        
        # calculate filtered and unfiltered stats
        self.sample_stats(analytes, stats=stats, filt=False)
        suf = self.stats.copy()
        self.sample_stats(analytes, stats=stats, filt=filt)
        sf = self.stats.copy()
        
        # create dataframe for results
        cols = []
        for s in self.stats_calced:
            cols += ['unfiltered_{:}'.format(s), 'filtered_{:}'.format(s)] 

        comp = pd.DataFrame(index=self.samples,
                            columns=pd.MultiIndex.from_arrays([cols, [None] * len(cols)]))

        # collate stats
        for k, v in suf.items():
            vf = sf[k]
            for i, a in enumerate(v['analytes']):
                for s in self.stats_calced:
                    comp.loc[k, ('unfiltered_{:}'.format(s), a)] = v[s][i,0]
                    comp.loc[k, ('filtered_{:}'.format(s), a)] = vf[s][i,0]
        comp.dropna(0, 'all', inplace=True)
        comp.dropna(1, 'all', inplace=True)
        comp.sort_index(1, inplace=True)

        # calculate filtered/unfiltered ratios
        rats = []
        for s in self.stats_calced:
            rat = comp.loc[:, 'filtered_{:}'.format(s)] / comp.loc[:, 'unfiltered_{:}'.format(s)]
            rat.columns = pd.MultiIndex.from_product([['{:}_ratio'.format(s)], rat.columns])
            rats.append(rat)
        
        # join it all up
        comp = comp.join(pd.concat(rats, 1))
        comp.sort_index(1, inplace=True)
        
        return comp.loc[:, (pd.IndexSlice[:], pd.IndexSlice[analytes])]

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
            analytes = [a for a in self.analytes if self.internal_standard not in a]

        if samples is None:
            samples = self._get_samples(subset)

        # isolate relevant filters
        filts = self.data[samples[0]].filt.components.keys()
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
        udict = {a: unitpicker(np.nanmean(focus[a]),
                               focus_stage=self.focus_stage,
                               denominator=self.internal_standard) for a in analytes}
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
                axes[i, j].scatter(pj, pi, alpha=0.4, s=10, lw=0.5, edgecolor='k')
                axes[j, i].scatter(pi, pj, alpha=0.4, s=10, lw=0.5, edgecolor='k')

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
                    err='nanstd', subset=None):
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
            focus = self.focus_stage
        if outdir is None:
            outdir = os.path.join(self.report_dir, focus)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        # if samples is not None:
        #     subset = self.make_subset(samples)

        if subset is not None:
            samples = self._get_samples(subset)
        elif samples is None:
            samples = self.subsets['All_Analyses']
        elif isinstance(samples, str):
            samples = [samples]
        
        with self.pbar.set(total=len(samples), desc='Drawing Plots') as prog:
            for s in samples:
                f, a = self.data[s].tplot(analytes=analytes, figsize=figsize,
                                        scale=scale, filt=filt,
                                        ranges=ranges, stats=stats,
                                        stat=stat, err=err, focus_stage=focus)
                # ax = fig.axes[0]
                # for l, u in s.sigrng:
                #     ax.axvspan(l, u, color='r', alpha=0.1)
                # for l, u in s.bkgrng:
                #     ax.axvspan(l, u, color='k', alpha=0.1)
                f.savefig(os.path.join(outdir, s + '_traces.pdf'))
                # TODO: on older(?) computers raises
                # 'OSError: [Errno 24] Too many open files'
                plt.close(f)
                prog.update()
        return

    # Plot gradients
    @_log
    def gradient_plots(self, analytes=None, win=15, samples=None, ranges=False,
                       focus=None, outdir=None,
                       figsize=[10, 4], subset='All_Analyses'):
        """
        Plot analyte gradients as a function of time.

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
            focus = self.focus_stage
        if outdir is None:
            outdir = os.path.join(self.report_dir, focus + '_gradient')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        # if samples is not None:
        #     subset = self.make_subset(samples)

        if subset is not None:
            samples = self._get_samples(subset)
        elif samples is None:
            samples = self.subsets['All_Analyses']
        elif isinstance(samples, str):
            samples = [samples]

        with self.pbar.set(total=len(samples), desc='Drawing Plots') as prog:
            for s in samples:
                f, a = self.data[s].gplot(analytes=analytes, win=win, figsize=figsize,
                                        ranges=ranges, focus_stage=focus)
                # ax = fig.axes[0]
                # for l, u in s.sigrng:
                #     ax.axvspan(l, u, color='r', alpha=0.1)
                # for l, u in s.bkgrng:
                #     ax.axvspan(l, u, color='k', alpha=0.1)
                f.savefig(os.path.join(outdir, s + '_gradients.pdf'))
                # TODO: on older(?) computers raises
                # 'OSError: [Errno 24] Too many open files'
                plt.close(f)
                prog.update()
        return

    # filter reports
    @_log
    def filter_reports(self, analytes, filt_str='all', nbin=5, samples=None,
                       outdir=None, subset='All_Samples'):
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

        with self.pbar.set(total=len(samples), desc='Drawing Plots') as prog:
            for s in samples:
                _ = self.data[s].filter_report(filt=filt_str,
                                            analytes=analytes,
                                            savedir=outdir,
                                            nbin=nbin)
                prog.update()
            # plt.close(fig)
        return

    # def _stat_boostrap(self, analytes=None, filt=True,
    #                    stat_fn=np.nanmean, ci=95):
    #     """
    #     Calculate sample statistics with bootstrapped confidence intervals.

    #     Parameters
    #     ----------
    #     analytes : optional, array_like or str
    #         The analyte(s) to calculate statistics for. Defaults to
    #         all analytes.
    #     filt : str, dict or bool
    #         Either logical filter expression contained in a str,
    #         a dict of expressions specifying the filter string to
    #         use for each analyte or a boolean. Passed to `grab_filt`.
    #     stat_fns : array_like
    #         list of functions that take a single array_like input,
    #         and return a single statistic. Function should be able
    #         to cope with numpy NaN values.
    #     ci : float
    #         Confidence interval to calculate.

    #     Returns
    #     -------
    #     None
    #     """

    #     return

    @_log
    def sample_stats(self, analytes=None, filt=True,
                     stats=['mean', 'std'], include_srms=False,
                     eachtrace=True, focus_stage=None, csf_dict={}):
        """
        Calculate sample statistics.

        Returns samples, analytes, and arrays of statistics
        of shape (samples, analytes). Statistics are calculated
        from the 'focus' data variable, so output depends on how
        the data have been processed.

        Included stat functions:

        * :func:`~latools.stat_fns.mean`: arithmetic mean
        * :func:`~latools.stat_fns.std`: arithmetic standard deviation
        * :func:`~latools.stat_fns.se`: arithmetic standard error
        * :func:`~latools.stat_fns.H15_mean`: Huber mean (outlier removal)
        * :func:`~latools.stat_fns.H15_std`: Huber standard deviation (outlier removal)
        * :func:`~latools.stat_fns.H15_se`: Huber standard error (outlier removal)

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
            take a single array_like input, and return a single statistic. 
            list of functions or names (see above) or functions that
            Function should be able to cope with NaN values.
        eachtrace : bool
            Whether to calculate the statistics for each analysis
            spot individually, or to produce per - sample means.
            Default is True.
        focus_stage : str
            Which stage of analysis to calculate stats for. 
            Defaults to current stage. 
            Can be one of:
            * 'rawdata': raw data, loaded from csv file.
            * 'despiked': despiked data.
            * 'signal'/'background': isolated signal and background data.
              Created by self.separate, after signal and background
              regions have been identified by self.autorange.
            * 'bkgsub': background subtracted data, created by 
              self.bkg_correct
            * 'ratios': element ratio data, created by self.ratio.
            * 'calibrated': ratio data calibrated to standards, created by self.calibrate.
            * 'massfrac': mass fraction of each element.

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

        if focus_stage is None:
            focus_stage = self.focus_stage

        self.stats = Bunch()

        self.stats_calced = []
        stat_fns = Bunch()

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
        if include_srms:
            samples = self.samples
        else:
            samples = [s for s in self.samples if self.srm_identifier not in s]

        with self.pbar.set(total=len(samples), desc='Calculating Stats') as prog:
            for s in samples:
                self.data[s].sample_stats(analytes, filt=filt,
                                          stat_fns=stat_fns,
                                          eachtrace=eachtrace,
                                          focus_stage=focus_stage)

                self.stats[s] = self.data[s].stats
                prog.update()

        return 

    @_log
    def ablation_times(self, samples=None, subset=None):

        if samples is not None:
            subset = self.make_subset(samples)

        samples = self._get_samples(subset)

        ats = Bunch()

        for s in samples:
            ats[s] = self.data[s].ablation_times()

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
                 stat='mean', err='std', subset=None):
        """
        Function for visualising per-ablation and per-sample means.

        Parameters
        ----------
        analytes : str or iterable
            Which analyte(s) to plot
        samples : str or iterable
            Which sample(s) to plot
        figsize : tuple
            Figure (width, height) in inches
        stat : str
            Which statistic to plot. Must match
            the name of the functions used in 
            'sample_stats'.
        err : str
            Which uncertainty to plot.
        subset : str
            Which subset of samples to plot.
        """
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
                    self.internal_standard]

        if figsize is None:
            figsize = (1.5 * len(self.stats), 3 * len(analytes))

        fig, axs = plt.subplots(len(analytes), 1, figsize=figsize)

        for ax, an in zip(axs, analytes):
            i = 0
            stab = self.getstats()
            m, u = unitpicker(np.percentile(stab.loc[:, an].dropna(), 25), 0.1,
                              focus_stage='calibrated',
                              denominator=self.internal_standard)
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
                                                     pretty_element(self.internal_standard),
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
            out.to_csv(os.path.join(self.export_dir, filename))

        self.stats_df = out

        return out.reindex(self.analytes_sorted(out.columns), axis=1)

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
        # ud = 'counts'

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        for s in samples:
            d = self.data[s].data[focus_stage]
            out = Bunch()

            for a in analytes:
                out[a] = d[a]

            out = pd.DataFrame(out, index=self.data[s].Time)
            out.index.name = 'Time'

            d = dateutil.parser.parse(self.data[s].meta['date'])
            header = ['# Minimal Reproduction Dataset Exported from LATOOLS on %s' %
                      (time.strftime('%Y:%m:%d %H:%M:%S')),
                      "# Analysis described in '../analysis.lalog'",
                      '# Run latools.reproduce to import analysis.',
                      '#',
                      '# Sample: %s' % (s),
                      '# Analysis Time: ' + d.strftime('%Y-%m-%d %H:%M:%S')]

            header = '\n'.join(header) + '\n'

            csv = out.to_csv()

            with open('%s/%s.csv' % (outdir, s), 'w') as f:
                f.write(header)
                f.write(csv)
        return

    @_log
    def export_traces(self, outdir=None, focus_stage=None, analytes=None,
                      samples=None, subset='All_Analyses', filt=False, zip_archive=False):
        """
        Function to export raw data.

        Parameters
        ----------
        outdir : str
            directory to save toe traces. Defaults to 'main-dir-name_export'.
        focus_stage : str
            The name of the analysis stage to export.

            * 'rawdata': raw data, loaded from csv file.
            * 'despiked': despiked data.
            * 'signal'/'background': isolated signal and background data.
              Created by self.separate, after signal and background
              regions have been identified by self.autorange.
            * 'bkgsub': background subtracted data, created by 
              self.bkg_correct
            * 'ratios': element ratio data, created by self.ratio.
            * 'calibrated': ratio data calibrated to standards, created by self.calibrate.

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
            focus_stage = self.focus_stage

        if focus_stage in ['ratios', 'calibrated']:
            analytes = [a for a in analytes if a != self.internal_standard]

        if outdir is None:
            outdir = os.path.join(self.export_dir, 'trace_export')

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
            d = self.data[s].data[focus_stage]
            ind = self.data[s].filt.grab_filt(filt)
            out = Bunch()

            for a in self.analytes_sorted(analytes):
                out[a] = nominal_values(d[a][ind])
                if focus_stage not in ['rawdata', 'despiked']:
                    out[a + '_std'] = std_devs(d[a][ind])
                    out[a + '_std'][out[a + '_std'] == 0] = np.nan

            out = pd.DataFrame(out, index=self.data[s].Time[ind])
            out.index.name = 'Time'

            header = ['# Sample: %s' % (s),
                      '# Data Exported from LATOOLS on %s' %
                      (time.strftime('%Y:%m:%d %H:%M:%S')),
                      '# Processed using %s configuration' % (self.config['config']),
                      '# Analysis Stage: %s' % (focus_stage),
                      '# Unit: %s' % ud[focus_stage]]

            header = '\n'.join(header) + '\n'

            csv = out.to_csv()

            with open('%s/%s_%s.csv' % (outdir, s, focus_stage), 'w') as f:
                f.write(header)
                f.write(csv)
        
        if zip_archive:
            utils.zipdir(outdir, delete=True)

        return

    def save_log(self, directory=None, logname=None, header=None):
        """
        Save analysis.lalog in specified location
        """
        if directory is None:
            directory = self.export_dir
        if not os.path.isdir(directory):
            directory = os.path.dirname(directory)

        if logname is None:
            logname = 'analysis.lalog'

        if header is None:
            header = self._log_header()

        loc = logging.write_logfile(self.log, header, 
                                    os.path.join(directory, logname))
        
        return loc

    def minimal_export(self, target_analytes=None, path=None):
        """
        Exports a analysis parameters, standard info and a minimal dataset,
        which can be imported by another user.

        Parameters
        ----------
        target_analytes : str or iterable
            Which analytes to include in the export. If specified, the export
            will contain these analytes, and all other analytes used during
            data processing (e.g. during filtering). If not specified, 
            all analytes are exported.
        path : str
            Where to save the minimal export. 
            If it ends with .zip, a zip file is created.
            If it's a folder, all data are exported to a folder.
        """
        if target_analytes is None:
            target_analytes = self.analytes
        if isinstance(target_analytes, str):
            target_analytes = [target_analytes]

        self.minimal_analytes.update(target_analytes)
        zip_archive = False

        # set up data path
        if path is None:
            path = self.export_dir + '/minimal_export.zip'
        if path.endswith('.zip'):
            path = path.replace('.zip', '')
            zip_archive = True
        if not os.path.isdir(path):
            os.mkdir(path)

        # export data
        self._minimal_export_traces(path + '/data', analytes=self.minimal_analytes)

         # define analysis_log header
        log_header = ['# Minimal Reproduction Dataset Exported from LATOOLS on %s' %
                      (time.strftime('%Y:%m:%d %H:%M:%S')),
                      'data_folder :: ./data/']
                      
        if hasattr(self, 'srmdat'):
            log_header.append('srm_table :: ./srm.table')

            # export srm table
            items = set()
            for a in self.minimal_analytes:
                for srm, ad in self._analyte_srmdat_link.items():
                    items.update([ad[a]])
            srmdat = self.srmdat.loc[idx[:, list(items)], :]
            with open(path + '/srm.table', 'w') as f:
                f.write(srmdat.to_csv())

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
        self.save_log(path, 'analysis.lalog', header=log_header)

        if zip_archive:
            utils.zipdir(directory=path, delete=True)
        
        return


def reproduce(past_analysis, plotting=False, data_folder=None,
              srm_table=None, custom_stat_functions=None):
    """
    Reproduce a previous analysis exported with :func:`latools.analyse.minimal_export`

    For normal use, supplying `log_file` and specifying a plotting option should be
    enough to reproduce an analysis. All requisites (raw data, SRM table and any
    custom stat functions) will then be imported from the minimal_export folder.

    You may also specify your own raw_data, srm_table and custom_stat_functions,
    if you wish.

    Parameters
    ----------
    log_file : str
        The path to the log file produced by :func:`~latools.analyse.minimal_export`.
    plotting : bool
        Whether or not to output plots.
    data_folder : str
        Optional. Specify a different data folder. Data folder
        should normally be in the same folder as the log file.
    srm_table : str
        Optional. Specify a different SRM table. SRM table
        should normally be in the same folder as the log file.
    custom_stat_functions : str
        Optional. Specify a python file containing custom
        stat functions for use by reproduce. Any custom
        stat functions should normally be included in the
        same folder as the log file.
    """
    if '.zip' in past_analysis:
        dirpath = utils.extract_zipdir(past_analysis)
        logpath = os.path.join(dirpath, 'analysis.lalog')
    elif os.path.isdir(past_analysis):
        if os.path.exists(os.path.join(past_analysis, 'analysis.lalog')):
            logpath = os.path.join(past_analysis, 'analysis.lalog')
    elif 'analysis.lalog' in past_analysis:
        logpath = past_analysis
    else:
        raise ValueError(('\n\n{} is not a valid input.\n\n' + 
                          'Must be one of:\n' +
                          '  - A .zip file exported by latools\n' + 
                          '  - An analysis.lalog file\n' +
                          '  - A directory containing an analysis.lalog files\n'))

    runargs, paths = logging.read_logfile(logpath)

    # parse custom stat functions
    csfs = Bunch()
    if custom_stat_functions is None and 'custom_stat_functions' in paths.keys():
        # load custom functions as a dict
        with open(paths['custom_stat_functions'], 'r') as f:
            csf = f.read()

        fname = re.compile('def (.*)\(.*')

        for c in csf.split('\n\n\n\n'):
            if fname.match(c):
                csfs[fname.match(c).groups()[0]] = c

    # create analysis object
    rep = analyse(*runargs[0][-1]['args'], **runargs[0][-1]['kwargs'])

    # rest of commands
    for fname, arg in runargs:
        if fname != '__init__':
            if 'plot' in fname.lower() and plotting:
                getattr(rep, fname)(*arg['args'], **arg['kwargs'])
            elif 'sample_stats' in fname.lower():
                rep.sample_stats(*arg['args'], csf_dict=csfs, **arg['kwargs'])
            else:
                getattr(rep, fname)(*arg['args'], **arg['kwargs'])

    return rep
