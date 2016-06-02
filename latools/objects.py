import os
import re
import itertools
import warnings
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
from IPython import display
from mpld3 import enable_notebook, disable_notebook

class analyse(object):
    def __init__(self, csv_folder, errorhunt=False):
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

        self.data = np.array([D(self.folder + '/' + f, errorhunt=errorhunt) for f in self.files if 'csv' in f])
        self.samples = np.array([s.sample for s in self.data])
        self.analytes = np.array(self.data[0].cols[1:])

        self.data_dict = {}
        for s, d in zip(self.samples, self.data):
            self.data_dict[s] = d

        self.sample_dict = {}
        for s, d in zip(self.samples, self.data):
            if 'STD' not in s:
                self.data_dict[s] = d

        self.stds = []
        _ = [self.stds.append(s) for s in self.data if 'STD' in s.sample]
        self.srms_ided = False

        self.cmaps = self.data[0].cmap

        self.bimodal_correction = False

        f = open('errors.log', 'a')
        f.write('Errors and warnings during LATOOLS analysis are stored here.\n\n')
        f.close()

        print('{:.0f} Analysis Files Loaded:'.format(len(self.data)))
        print('{:.0f} standards, {:.0f} samples'.format(len(self.stds),
              len(self.data) - len(self.stds)))
        print('Analytes: ' + ' '.join(self.analytes))


    # function for identifying sample and background regions
    def trace_id(self, analytes=['Ca44', 'Al27', 'Ba137', 'Ba138']):
        for s in self.data:
            fig = s.tplot(analytes, scale='log')
            plugins.connect(fig, plugins.MousePosition(fontsize=14))
            display.clear_output(wait=True)
            display.display(fig)

            OK = False
            while OK is False:
                try:
                    bkg = [float(f) for f in
                           input('Enter background limits (as: start, end, start, end):\n').split(',')]
                    bkg = np.array(bkg).reshape(len(bkg)//2, 2)
                    OK = True
                except:
                    print("Incorrect Values, try again:\n")

            OK = False
            while OK is False:
                try:
                    sig = [float(f) for f in
                           input('Enter sample limits (as: start, end, start, end):\n').split(',')]
                    sig = np.array(sig).reshape(len(sig)//2, 2)
                    OK = True
                except:
                    print("Incorrect Values, try again:\n")

            s.bkgrng = bkg
            s.sigrng = sig
        self.save_ranges()

        display.clear_output()
        return

    def autorange(self, analyte='Ca43', gwin=11, win=40, smwin=5,
                  conf=0.01, trans_mult=[0., 0.]):
        """
        Function to automatically detect signal and background regions in the
        laser data, based on the behaviour of a target analyte. An ideal target
        analyte should be abundant and homogenous in the sample.

        Step 1: Thresholding
        The background is initially determined using a gaussian kernel density
        estimator (kde) of all the data. The minima in the kde define the
        boundaries between distinct data distributions. All data below than the
        first (lowest) kde minima are labelled 'background', and all above this
        limit are labelled 'signal'.

        Step 2: Transition Removal
        The width of the transition regions between signal and background are
        then determined, and the transitions are removed from both signal and
        background. The width of the transitions is determined by fitting a
        gaussian to the smoothed first derivative of the analyte trace, and
        determining its width at a point where the gaussian intensity is at a
        set limit. These gaussians are fit to subsets of the data that contain
        the transitions, which are centered around the approximate transition
        locations determined in Step 1, ± win data points. The peak is isolated
        by finding the minima and maxima of a second derivative, and the
        gaussian is fit to the isolate peak.

        Parameters:
            win:    int
                Determines the width (c ± win) of the transition data subsets.
            gwin:   odd int
                The smoothing window used for calculating the first derivative.
            smwin:  odd int
                The smoothing window used for calculating the second derivative
            conf:   float
                The proportional intensity of the fitted gaussian tails that
                determines the transition width cutoff (lower = wider
                transition regions excluded).
            trans_mult: array-like of length 2
                Multiples of sigma to add to the transition cutoffs, e.g. if
                the transitions consistently leave some bad data proceeding
                the transition, set trans_mult to [0, 0.5] to ad 0.5 * the FWHM
                to the right hand side of the limit.

        Returns:
            self gains 'bkg', 'sig', 'bkgrng' and 'sigrng' properties, which
            contain bagkround & signal boolean arrays and limit arrays,
            respectively.
        """
        for d in self.data:
            d.autorange(analyte, gwin, win, smwin,
                        conf, trans_mult)

    def find_expcoef(self, nsd_below=12., analytes='Ca43', plot=False, trimlim=None):
        """
        Determines the exponential decay filter coefficient by
        looking at the washout time at the end of standards measurements

        Parameters:
            nsd_below: float
                The number of standard deviations to subtract
                from the fitted coefficient.
            analytes: str or array-lke
                The analytes to consider when determining the coefficient.
                Use high-concentration analytes for best estimates
            plot: bool or str
                bool: Creates a plot of the fit if True.
                str: Creates a plot, and saves it to the location
                     specified in the str.
            trimlim: float
                A threshold limit used in determining the start of the
                exponential decay region of the washout. If the data in
                the plot don't fall on an exponential decay line, change
                this number. Normally you'll need to increase it.
        """

        from scipy.optimize import curve_fit
        if isinstance(analytes, str):
            analytes = [analytes]

        def findtrim(tr, lim=None):
            trr = np.roll(tr, -1)
            trr[-1] = 0
            if lim is None:
                lim = 0.5 * np.nanmax(tr - trr)
            ind = (tr - trr) >= lim
            return np.arange(len(ind))[ind ^ np.roll(ind, -1)][0]

        def normalise(a):
            return (a - np.nanmin(a)) / np.nanmax(a - np.nanmin(a))

        if not hasattr(self.stds[0], 'trnrng'):
            for s in self.stds:
                s.autorange()

        trans = []
        times = []
        for analyte in analytes:
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
            return np.exp(e * x)

        ep, ecov = curve_fit(expfit, ti, tr, p0=(-1.))

        def R2calc(x, y, yp):
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
    def bkgcorrect(self, mode='constant', make_bools=True):
        for s in self.data:
            if make_bools:
                s.bkgrange()
                s.sigrange()
            s.separate()
            s.bkg_correct(mode=mode)
        return

    def ratio(self,  denominator='Ca43', stage='signal'):
        for s in self.data:
            s.ratio( denominator=denominator, stage=stage)
        return

    # functions for identifying SRMs
    def srm_id(self):
        s = self.stds[0]
        fig = s.tplot(scale='log')
        display.clear_output(wait=True)
        display.display(fig)

        n0 = s.n

        def id(self, s):
            stdnms = []
            s.std_rngs = {}
            for n in np.arange(s.n) + 1:
                fig = s.tplot(scale='log')
                lims = s.Time[s.ns == n][[0, -1]]
                fig.axes[0].axvspan(lims[0], lims[1],
                                    color='r', alpha=0.2, lw=0)
                display.clear_output(wait=True)
                display.display(fig)
                stdnm = input('Name this standard: ')
                stdnms.append(stdnm)
                s.std_rngs[stdnm] = lims
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
                        s.std_rngs = {}
                        for n in np.arange(s.n) + 1:
                            s.std_rngs[nms0[n-1]] = s.Time[s.ns == n][[0, -1]]
                    else:
                        _ = id(self, s)

        display.clear_output()

        self.save_srm_ids()

        for s in self.stds:
            s.std_labels = {}
            for srm in s.std_rngs.keys():
                s.std_labels[srm] = np.zeros(s.Time.size)
                s.std_labels[srm][(s.Time >= min(s.std_rngs[srm])) &
                                  (s.Time <= max(s.std_rngs[srm]))] = 1

        self.srms_ided = True

        return

    # def srm_id(self):
    #     enable_notebook()  # make the plot interactive
    #     for s in self.stds:
    #         fig = s.tplot(scale='log')

    #         plugins.connect(fig, plugins.MousePosition(fontsize=14))
    #         display.clear_output(wait=True)
    #         display.display(fig)

    #         s.std_rngs = {}

    #         n = int(input('How many standards? (int): '))

    #         for i in range(n):
    #             OK = False
    #             while OK is False:
    #                 try:
    #                     name = input('Enter Standard Name: ')
    #                     ans = [float(f) for f in input(name + ': Enter start and end points of data as: start, end)\n').split(',')]
    #                     OK = True
    #                 except:
    #                     print("Incorrect Values, try again:\n")
    #             s.std_rngs[name] = ans

    #     self.save_srm_ids()

    #     for s in self.stds:
    #         s.std_labels = {}
    #         for srm in s.std_rngs.keys():
    #             s.std_labels[srm] = np.zeros(s.Time.size)
    #             s.std_labels[srm][(s.Time >= min(s.std_rngs[srm])) &
    #                               (s.Time <= max(s.std_rngs[srm]))] = 1
    #     disable_notebook()  # stop the interactivity

    #     self.srms_ided = True
    #     return

    def save_srm_ids(self):
        if os.path.isfile(self.param_dir + 'srm.rng'):
            f = input('SRM range files already exist. Do you want to overwrite them (old files will be lost)? [Y/n]: ')
            if 'n' in f or 'N' in f:
                print('SRM ranges not saved. Run self.save_srm_ids() to try again.')
                return
        srm_ids = []
        for d in self.stds:
            srm_ids.append(d.sample + ' ' + str(d.std_rngs))
        srm_ids = '\n'.join(srm_ids)

        fb = open(self.param_dir + 'srm.rng', 'w')
        fb.write(srm_ids)
        fb.close()
        return

    def load_srm_ids(self, srm_ids):
        rng = open(srm_ids).readlines()
        samples = []
        ids = []
        for r in rng:
            samples.append(re.match('(.*) ({.*)', r.strip()).groups()[0])
            ids.append(eval(re.sub('array', 'np.array',
                       re.match('(.*) ({.*)', r.strip()).groups()[1])))
        samples = np.array(samples)
        ids = np.array(ids)
        for s in self.stds:
            s.std_rngs = ids[samples == s.sample][0]

        for s in self.stds:
            s.std_labels = {}
            for srm in s.std_rngs.keys():
                s.std_labels[srm] = np.zeros(s.Time.size)
                s.std_labels[srm][(s.Time >= min(s.std_rngs[srm])) &
                                  (s.Time <= max(s.std_rngs[srm]))] = 1

        self.srms_ided = True

        return

    # apply calibration to data
    def calibrate(self, poly_n=0, focus='ratios',
                  srmfile='/Users/oscarbranson/UCDrive/Projects/latools/latools/resources/GeoRem_150105_ratios.csv'):
        # MAKE CALIBRATION CLEVERER!
        #   USE ALL DATA, NOT AVERAGES?
        #   IF POLY_N > 0, STILL FORCE THROUGH ZERO IF ALL STDS ARE WITHIN ERROR OF EACH OTHER (E.G. AL/CA)
        # can store calibration function in self and use *coefs?
        # check for identified srms
        if not self.srms_ided:
            self.srm_id()
        # get SRM values
        f = open(srmfile).readlines()
        self.srm_vals = {}
        for srm in self.stds[0].std_rngs.keys():
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
                for srm in s.std_rngs.keys():
                    y = s.focus[a][s.std_labels[srm] == 1]
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
        self.save_calibration()
        return

    def save_calibration(self):
        fname = self.param_dir + self.dirname + '.calibdat'
        # if os.path.isfile(fname):
        #     f = input("SRM range files already exist in '" + fname + "'. Do you want to overwrite them (old files will be lost)? [Y/n]: ")
        #     if 'n' in f or 'N' in f:
        #         print('SRM ranges not saved. Run self.save_srm_ids() to try again.')
        #         return
        fb = open(fname, 'w')
        fb.write(str(self.calib_dict))
        fb.close
        return

    def load_calibration(self, fname=None):
        if fname is None:
            fname = self.param_dir + self.dirname + '.calibdat'

        try:
            strdict = re.sub('array', 'np.array', open(fname).read())
            self.calib_dict = eval(strdict)
        except:
            print("File '" + fname + "' does not exist.")

        self.srms_ided = True

        return

    # data filtering

    def filter_threshold(self, analyte, threshold, filt=False, mode='above', samples=None):
        """
        Generates threshold filters for analytes, when provided with analyte,
        threshold, and mode. Mode specifies whether data 'below'
        or 'above' the threshold are kept.
        """
        if samples is None:
            samples = self.samples
        if isinstance(samples, str):
            samples = []

        for s in samples:
            self.data_dict[s].filter_threshold(analyte, threshold, filt=False, mode='above')

    def filter_distribution(self, analyte, binwidth='scott', filt=False, transform=None, output=False, samples=None):
        if samples is None:
            samples = self.samples
        if isinstance(samples, str):
            samples = []

        for s in samples:
            self.data_dict[s].filter_distribution(analyte, binwidth='scott', filt=False, transform=None, output=False)

    def filter_clustering(self, analytes, filt=False, normalise=True, method='meanshift', include_time=False, samples=None, **kwargs):
        if samples is None:
            samples = self.samples
        if isinstance(samples, str):
            samples = []

        for s in samples:
            self.data_dict[s].filter_clustering(analytes, filt=False, normalise=True, method='meanshift', include_time=False, samples=None, **kwargs)

    def filter_on(self, filters=None, analytes=None, samples=None):
        if samples is None:
            samples = self.samples
        if isinstance(samples, str):
            samples = []

        for s in samples:
            self.data_dict[s].filt.on(analytes, filters)

    def filter_off(self, filters=None, analytes=None, samples=None):
        if samples is None:
            samples = self.samples
        if isinstance(samples, str):
            samples = []

        for s in samples:
            self.data_dict[s].filt.off(analytes, filters)

    def filter_clear(self):
        for d in self.data:
            d.filt.clear()

    # def filter_off(self):

    # def distribution_check(self, analytes=None, mode='lower', filt=False):
    #     """
    #     Checks the specified analytes for bimodality by looking for minima
    #     in a gaussian kde data density curve. If minima are found, data either
    #     above or below the threshold are excluded. Behaviour is determined by
    #     the 'mode' argument.

    #     mode:   str ('lower'/'upper')
    #         'lower': data below the cutoff are kept.
    #         'upper': data above the cutoff are kept.
    #     """
    #     if analytes is None:
    #         analytes = self.analytes
    #     analytes = np.array(analytes).flatten()
    #     for d in self.data:
    #         d.bimodality_fix(analytes, report=False, mode=mode, filt=filt)

    # def distribution_reports(self, analytes=['Ba138'], dirpath=None, filt=False):
    #     """
    #     Saves data distribution pdfs for all analytes specified,
    #     showing where they have been cut by a bimodality check
    #     (if it has been run).
    #     pdfs are saved in the specified directory (dirpath).
    #     """
    #     fails = []
    #     if dirpath is None:
    #         dirpath = self.report_dir
    #     if not os.path.isdir(dirpath):
    #         os.mkdir(dirpath)
    #     analytes = np.array([analytes]).flatten()
    #     for k, v in self.data_dict.items():
    #         try:
    #             fig = v.bimodality_report(filt=filt)
    #             fig.savefig(dirpath + '/' + k + '_distributions.pdf')
    #             plt.close(fig)
    #         except:
    #             fails.append(k)
    #     if len(fails) > 0:
    #         f = open("errors.log", 'w')
    #         f.write('\nDistribution Reports:\n')
    #         f.write('\n'.join(fails))
    #         f.close()
    #         print('Some reports failed. See log.')

    #     return

    # def clear_filters(self):
    #     for d in self.data:
    #         d.filt = {}
    #         d.filtrngs = {}

    # def threshold_filter(self, analytes, thresholds, modes):
    #     for d in self.data:
    #         params = zip(np.array(analytes, ndmin=1),
    #                      np.array(thresholds, ndmin=1),
    #                      np.array(modes, ndmin=1))
    #         for p in params:
    #             d.threshold_filter(p[0], p[1], p[2])

    # plot calibrations
    def calibration_plot(self, analytes=None, plot='errbar'):
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
        t = 0
        self.focus = {'Time': []}
        for a in self.analytes:
            self.focus[a] = []

        for s in self.data:
            if 'STD' not in s.sample:
                self.focus['Time'].append(s.Time + t)
                t += max(s.Time)
                if isinstance(filt, str):
                    ind = ~s.filt[filt]
                else:
                    ind = np.array([False] * len(s.Time))
                for a in self.analytes:
                    tmp = s.focus[a].copy()
                    tmp[ind] = np.nan
                    self.focus[a].append(tmp)

        for k, v in self.focus.items():
            self.focus[k] = np.concatenate(v)

    # crossplot of all data
    def crossplot(self, analytes=None, lognorm=True,
                  bins=25, filt=False, **kwargs):
        if analytes is None:
            analytes = [a for a in self.analytes if 'Ca' not in a]
        if not hasattr(self, 'focus'):
            self.get_focus()

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
    def trace_plots(self, analytes=None, dirpath=None, ranges=False, focus='despiked', plot_filt=None):
        if dirpath is None:
            dirpath = self.report_dir
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        for s in self.data:
            stg = s.focus_stage
            s.setfocus(focus)
            fig = s.tplot(scale='log', ranges=ranges, plot_filt=plot_filt)
            # ax = fig.axes[0]
            # for l, u in s.sigrng:
            #     ax.axvspan(l, u, color='r', alpha=0.1)
            # for l, u in s.bkgrng:
            #     ax.axvspan(l, u, color='k', alpha=0.1)
            fig.savefig(dirpath + '/' + s.sample + '_traces.pdf')
            plt.close(fig)
            s.setfocus(stg)


    def stat_boostrap(self, analytes=None, filt=True,
                      stat_fn=np.nanmean, ci=95):
        """
        Function to calculate the sample mean and bootstrap confidence
        intervals
        """

        return

    def stat_samples(self, analytes=None, filt=True,
                     stat_fns=[np.nanmean, np.nanstd],
                     eachtrace=True):
        """
        Returns samples, analytes, and arrays of statistics
        of shape (samples, analytes). Statistics are calculated
        from the 'focus' data variable, so output depends on how
        the data have been processed.

        analytes: array-like
            list of analytes to calculate the statistic on
        stat_fns: array-like
            list of functions that take a single array-like input,
            and return a single statistic. Function should be able
            to cope with numpy NaN values.
        filt: boolean
            Should the means take any active filters into account
            (in self.filt)?

        Returns:
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
        Returns pandas dataframe of all sample statistics
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

    def getstat(self, analyte=None, sample=None):
        if analyte is None:
            analyte = self.analytes
        if sample is None:
            sample = self.samples

        if isinstance(analyte, str):
            analyte = [analyte]
        if isinstance(sample, str):
            sample = [sample]

        ankey = [a in self.analytes for a in analyte]

        ss = [s for s in list(self.stats.values())[0].keys() if s is not 'analytes']
        out = {}
        for s in ss:
            out[s] = []

        for k, v in self.stats.items():
            if k in sample:
                for s in ss:
                    out[s].append(v[s][ankey])

        return out

    def stat_csvtable(self, stat='nanmean', file=None):
        """
        Generates a csv table of statistics for all samples and analytes.
        stat:   str
            a string that describes one of the statistics calculated for the
            samples, e.g. 'mean' or 'std'. String doesn't have to match the
            stat function name exactly, but must be contained within it.
            (i.e. 'mean' will return results for 'nanmean').
        file:   str
            if a file is specified, the csv will be saved directly, otherwise
            the raw csv string will be returned.

        """
        analytes = list(self.stats.values())[0]['analytes']
        head = '# Statistic: ' + stat + '\n' + 'Sample,'+','.join(analytes)
        outrows = []
        for k,v in self.stats.items():
            if stat not in v.keys():
                raise ValueError("Requested 'stat' has not been calculated yet. Re-run stat_samples and include 'stat' in calculations.")
            i = 1
            for l in v[stat].T:
                outrows.append(k + '-s{:.0f},'.format(i) + ','.join(l.astype(str)))
                i += 1
        out = head + '\n' + '\n'.join(outrows)

        if file is not None:
            f = open(file, 'w')
            f.write(out)
            f.close
            return
        else:
            return out

analyze = analyse  # for the yanks

class D(object):
    def __init__(self, csv_file, errorhunt=False):
        if errorhunt:
            print(csv_file)  # errorhunt prints each csv file name before it tries to load it, so you can tell which file is failing to load.
        self.file = csv_file
        self.sample = os.path.basename(self.file).split('.')[0]

        # open file
        f = open(self.file)
        lines = f.readlines()

        # determine header size
        def nskip(lines):
            for i, s in enumerate(lines):
                if 'time [sec]' in s.lower():
                    return i
            return -1
        dstart = nskip(lines) + 1

        # get run info
        self.Dfile = lines[0]
        try:
            info = re.search('.*([A-Z][a-z]{2} [0-9]+ [0-9]{4}[ ]+[0-9:]+) .*AcqMethod (.*)',lines[2]).groups()
            self.date = info[0]
            self.method = info[1]
            self.despiked = lines[3][:8] == 'Despiked'
        except:
            pass

        self.cols = np.array([l for l in lines[:dstart] if l.startswith('Time')][0].strip().split(','))
        self.cols[0] = 'Time'
        self.analytes = self.cols[1:]
        f.close()

        # load data
        raw = np.loadtxt(csv_file, delimiter=',', skiprows=dstart, comments='     ').T
        self.rawdata = {}
        for i in range(len(self.cols)):
            self.rawdata[self.cols[i]] = raw[i]

        # most recently worked on data step
        self.setfocus('rawdata')
        self.cmap = dict(zip(self.analytes,
                             cb.get_map('Paired', 'qualitative',
                                        len(self.cols)).hex_colors))

        # set up flags
        self.sig = np.array([False] * self.Time.size)
        self.bkg = np.array([False] * self.Time.size)
        self.trn = np.array([False] * self.Time.size)
        self.ns = np.zeros(self.Time.size)
        self.bkgrng = np.array([]).reshape(0, 2)
        self.sigrng = np.array([]).reshape(0, 2)

        # set up filtering environment
        self.filt = filt(self.Time.size, self.analytes)

        # set up corrections dict
        # self.corrections = {}

    def setfocus(self, stage):
        """
        Sets the 'focus' attribute of the onject, which points towards
        data from a particular stage of analysis.
        Used to update the 'working stage' of the data.
        Functions generally operate on the 'focus' dataset,
        so if steps are done out of sequence, things will break.

        Parameters:
            stage:  string describing analysis stage
                rawdata: raw data, loaded from csv file when object
                    is initialised
                signal/background: isolated signal and background data,
                    padded with np.nan. Created by self.separate, after
                    signal and background regions have been identified by
                    self.autorange.
                bkgsub: background subtracted data, created by self.bkg_correct
                ratios: element ratio data, created by self.ratio.
                calibrated: ratio data calibrated to standards, created by
                    self.calibrate.
        """
        self.focus = getattr(self, stage)
        self.focus_stage = stage
        for k in self.focus.keys():
            setattr(self, k, self.focus[k])

    # despiking functions
    def rolling_window(self, a, window, pad=None):
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
        Exponential decay filter for removing anomalous low values.
        """
        # if exponent is None:
        #     if ~hasattr(self, 'expdecay_coef'):
        #         self.find_expcoef()
        #     exponent = self.expdecay_coef
        if tstep is None:
            tstep = np.diff(self.Time[:2])
        if ~hasattr(self, 'despiked'):
            self.despiked = {}
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
                self.despiked[a] = v
        self.setfocus('despiked')
        return

    # spike filter
    def spike_filter(self, win=3, nlim=12.):
        """
        Spike filter for removing anomalous high values.
        """
        if ~isinstance(win, int):
            win = int(win)
        if ~hasattr(self, 'despiked'):
            self.despiked = {}
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
                self.despiked[a] = v
        self.setfocus('despiked')
        return

    def despike(self, expdecay_filter=True, exponent=None, tstep=None, spike_filter=True, win=3, nlim=12.):
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
        """
        Function to find local minima.
        Returns array of points in x where y has a local minimum.
        """
        return x[np.r_[False, y[1:] < y[:-1]] & np.r_[y[:-1] < y[1:], False]]

    def gauss(self, x, *p):
        """
        Gaussian function for transition fitting.

        Parameters:
            x:  array-like
            *p: parameters unpacked to A, mu, sigma
                A: area
                mu: centre
                sigma: width
        """
        A, mu, sigma = p
        return A * np.exp(-0.5*(-mu + x)**2/sigma**2)

    def gauss_inv(self, y, *p):
        """
        Inverse gaussian function for determining the x coordinates
        for a given y intensity (i.e. width at a given height).

        Parameters:
            y:  float
                The height at which to calculate peak width.
            *p: parameters unpacked to mu, sigma
                mu: peak center
                sigma: peak width
        """
        mu, sigma = p
        return np.array([mu - 1.4142135623731 * np.sqrt(sigma**2*np.log(1/y)),
                         mu + 1.4142135623731 * np.sqrt(sigma**2*np.log(1/y))])

    def findlower(self, x, y, c, win=3):
        """
        Finds the first local minima below a specified point. Used for
        defining the lower limit of the data window used for transition
        fitting.

        Parameters:
            x:  array-like
            y:  array-like
            c:  center point
        """
        yd = self.fastgrad(y[::-1], win)
        mins = self.findmins(x[::-1], yd)
        clos = abs(mins - c)
        return mins[clos == min(clos)] - min(clos)

    def findupper(self, x, y, c, win=3):
        """
        Finds the first local minima above a specified point. Used for
        defining the lower limit of the data window used for transition
        fitting.

        Parameters:
            x:  array-like
            y:  array-like
            c:  center point
        """
        yd = self.fastgrad(y, win)
        mins = self.findmins(x, yd)
        clos = abs(mins - c)
        return mins[clos == min(abs(clos))] + min(clos)

    def fastgrad(self, a, win=11):
        """
        Function to efficiently calculate the rolling gradient of a numpy
        array using 'stride_tricks' to split up a 1D array into an ndarray of
        sub-sections of the original array, of dimensions [len(a)-win, win].

        Parameters:
            a:   array-like
            win: int
                The width of the rolling window.
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
        Function to automatically detect signal and background regions in the
        laser data, based on the behaviour of a target analyte. An ideal target
        analyte should be abundant and homogenous in the sample.

        Step 1: Thresholding
        The background is initially determined using a gaussian kernel density
        estimator (kde) of all the data. The minima in the kde define the
        boundaries between distinct data distributions. All data below than the
        first (lowest) kde minima are labelled 'background', and all above this
        limit are labelled 'signal'.

        Step 2: Transition Removal
        The width of the transition regions between signal and background are
        then determined, and the transitions are removed from both signal and
        background. The width of the transitions is determined by fitting a
        gaussian to the smoothed first derivative of the analyte trace, and
        determining its width at a point where the gaussian intensity is at a
        set limit. These gaussians are fit to subsets of the data that contain
        the transitions, which are centered around the approximate transition
        locations determined in Step 1, ± win data points. The peak is isolated
        by finding the minima and maxima of a second derivative, and the
        gaussian is fit to the isolate peak.

        Parameters:
            win:    int
                Determines the width (c ± win) of the transition data subsets.
            gwin:   odd int
                The smoothing window used for calculating the first derivative.
            smwin:  odd int
                The smoothing window used for calculating the second derivative
            conf:   float
                The proportional intensity of the fitted gaussian tails that
                determines that determined the transition width cutoff (lower =
                wider transition cutoff).
            trans_mult: array-like of length 2
                Multiples of sigma to add to the transition cutoffs, e.g. if
                the transitions consistently leave some bad data proceeding
                the transition, set trans_mult to [0, 0.5] to ad 0.5 * the FWHM
                to the right hand side of the limit.

        Returns:
            self gains 'bkg', 'sig', 'bkgrng' and 'sigrng' properties, which
            contain bagkround & signal boolean arrays and limit arrays,
            respectively.
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

        # bkgr = np.concatenate([[0],
        #                       self.Time[self.bkg ^ np.roll(self.bkg, -1)],
        #                       [self.Time[-1]]])
        # self.bkgrng = np.reshape(bkgr, [bkgr.size//2, 2])

        # if self.sig[-1]:
        #     self.sig[-1] = False
        # sigr = self.Time[self.sig ^ np.roll(self.sig, 1)]
        # self.sigrng = np.reshape(sigr, [sigr.size//2, 2])

    def bkgrange(self, rng=None):
        """
        Generate a background boolean string based on a list of [min,max] value
        pairs stored in self.bkgrng.

        Parameters:
            rng:   2xn d numpy array
                [min,max] pairs defining the upper and lowe limits of
                background regions.
        """
        if rng is not None:
            if np.array(rng).ndim is 1:
                self.bkgrng = np.append(self.bkgrng, np.array([rng]), 0)
            else:
                self.bkgrng = np.append(self.bkgrng, np.array(rng), 0)

        self.bkg = np.array([False] * self.Time.size)
        for lb, ub in self.bkgrng:
            self.bkg[(self.Time > lb) & (self.Time < ub)] = True

        self.trn = ~self.bkg & ~self.sig  # redefine transition regions
        return

    def sigrange(self, rng=None):
        """
        Generate a background boolean string based on a list of [min,max] value
        pairs stored in self.bkgrng.

        Parameters:
            rng:   2xn d numpy array
                [min,max] pairs defining the upper and lowe limits of
                signal regions.
        """
        if rng is not None:
            if np.array(rng).ndim is 1:
                self.sigrng = np.append(self.sigrng, np.array([rng]), 0)
            else:
                self.sigrng = np.append(self.sigrng, np.array(rng), 0)

        self.sig = np.array([False] * self.Time.size)
        for ls, us in self.sigrng:
            self.sig[(self.Time > ls) & (self.Time < us)] = True

        self.trn = ~self.bkg & ~self.sig  # redefine transition regions
        return

    def makerangebools(self):
        self.sig = np.array([False] * self.Time.size)
        for ls, us in self.sigrng:
            self.sig[(self.Time > ls) & (self.Time < us)] = True
        self.bkg = np.array([False] * self.Time.size)
        for lb, ub in self.bkgrng:
            self.bkg[(self.Time > lb) & (self.Time < ub)] = True
        self.trn = ~self.bkg & ~self.sig
        return

    def bool_2_indices(self, bool_array):
        if ~isinstance(ool_array, np.ndarray):
            bool_array = np.array(bool_array)
        return np.arange(len(bool_array))[bool_array ^ np.roll(bool_array, 1)]

    def separate(self, analytes=None):
        """
        Isolates signal and background signals from raw data for specified
        elements.

        Parameters:
            analytes: list of analyte names (default = all analytes)
        """
        if analytes is None:
            analytes = self.analytes
        self.background = {}
        self.signal = {}
        for v in analytes:
            self.background[v] = self.focus[v].copy()
            self.background[v][~self.bkg] = np.nan
            self.signal[v] = self.focus[v].copy()
            self.signal[v][~self.sig] = np.nan

    def bkg_correct(self, mode='constant'):
        """
        Subtract constant or linear background from all analytes.
        mode may be 'constant' or an int describing the degree of polynomial background.
        """
        self.bkgsub = {}
        if mode == 'constant':
            for c in self.analytes:
                self.bkgsub[c] = self.signal[c] - np.nanmean(self.background[c])
        if (mode != 'constant'):
            for c in self.analytes:
                p = np.polyfit(self.Time[self.bkg], self.focus[c][self.bkg], mode)
                self.bkgsub[c] = self.signal[c] - np.polyval(p, self.Time)
        self.setfocus('bkgsub')
        return

    def ratio(self, denominator='Ca43', stage='signal'):
        """
        Divide all analytes by a specified denominator (default = 'Ca43').

        Parameters:
            denominator:    string
                The analyte used as the denominator
            stage:  string
                The analysis stage to perform the ratio calculation on.
                Defaults to 'signal', the isolates, background-corrected
                regions identified as good data.
        """
        self.setfocus(stage)
        self.ratios = {}
        for a in self.analytes:
            self.ratios[a] = \
                self.focus[a] / self.focus[denominator]
        self.setfocus('ratios')
        self.ratio_denominator = denominator
        return

    def calibrate(self, calib_dict):
        """

        """
        # can have calibration function stored in self and pass *coefs?
        self.calibrated = {}
        for a in self.analytes:
            coefs = calib_dict[a]
            if len(coefs) == 1:
                self.calibrated[a] = \
                    self.ratios[a] * coefs
            else:
                self.calibrated[a] = \
                    np.polyval(coefs, self.ratios[a])
                    # self.ratios[a] * coefs[0] + coefs[1]
        self.setfocus('calibrated')
        return

    # Function for calculating sample statistics
    def sample_stats(self, analytes=None, filt=True,
                     stat_fns=[np.nanmean, np.nanstd],
                     eachtrace=True):
        """
        Returns samples, analytes, and arrays of statistics
        of shape (samples, analytes). Statistics are calculated
        from the 'focus' data variable, so output depends on how
        the data have been processed.

        analytes: array-like
            list of analytes to calculate the statistic on
        stat_fns: array-like
            list of functions that take a single array-like input,
            and return a single statistic. Function should be able
            to cope with numpy NaN values.
        filt: bool or str
            filt specifies the filter to apply to the data when calculating
            sample statistics. It can either:
            bool:  True | False
                If True, applies filter created by bimodality_fix to each
                analyte individually.
            str: name of analyte specific filter
                applies a specific filter to all the data,
                or a filter resulting from the union of all analyte-specific
                filters.
        eachtrace: bool
            Return individual statistics for each analysis trace (True),
            or each sample (False)?
        """
        if analytes is None:
                analytes = self.analytes

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

    def filter_threshold(self, analyte, threshold, filt=False, mode='above'):
        """
        Generates threshold filters for analytes, when provided with analyte,
        threshold, and mode. Mode specifies whether data 'below'
        or 'above' the threshold are kept.
        """
        params = locals()
        del(params['self'])

        # generate filter
        ind = self.filt.grab_filt(filt, analyte) & np.apply_along_axis(all, 0,~np.isnan(np.vstack(self.focus.values())))

        if mode == 'below':
            self.filt.add(analyte + '_thresh_below',
                               self.focus[analyte] <= threshold,
                               'Keep ' + mode + ' {:.3e} '.format(threshold) + analyte,
                               params)
        if mode == 'above':
            self.filt.add(analyte + '_thresh_above',
                               self.focus[analyte] >= threshold,
                               'Keep ' + mode + ' {:.3e} '.format(threshold) + analyte,
                               params)


    def filter_distribution(self, analyte, binwidth='scott', filt=False, transform=None, output=False):
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
        if output:
            return x, yd, limits
        else:
            return

    def filter_clustering(self, analytes, filt=False, normalise=True, method='meanshift', include_time=False, **kwargs):
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


    def cluster_meanshift(self, data, bandwidth=None, bin_seeding=True):
        if bandwidth is None:
            bandwidth = cl.estimate_bandwidth(data)

        ms = cl.MeanShift(bandwidth=bandwidth)
        ms.fit(data)

        labels = ms.labels_
        labels_unique = np.unique(labels)

        out = {}
        for lab in labels_unique:
            out[lab] = labels == lab

        return out

    def cluster_kmeans(self, data, n_clusters):
        km = cl.KMeans(n_clusters)
        kmf = km.fit(data)

        labels = kmf.labels_
        labels_unique = np.unique(labels)

        out = {}
        for lab in labels_unique:
            out[lab] = labels == lab

        return out

    def cluster_DBSCAN(self, data, eps=None, min_samples=None, n_clusters=None, maxiter=200):
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
        # automatically determine appripriate window

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

        return #r, p


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
          ranges=False, stats=True, sig='nanmean', err='nanstd', interactive=False):
        """
        Convenience function for plotting traces.

        Parameters:
            analytes:   list of strings containing names of analytes to plot.
                        None = all analytes.
            figsize:    tuple-like
                        size of final figure.
            scale:      str ('log') or blank.
                        whether to plot data on a log scale.
            filt:       boolean, string or dict
                        False: plot unfiltered data.
                        True: plot filtered data over unfiltered data.
                        str: apply filter key to all analytes
                        dict: apply key to each analyte in dict. Must contain all
                        analytes plotted. Can use self.filt.keydict.
            stats:      boolean
                        Whether or not to plot the mean and standard deviation
                        for the traces.
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

        ax.text(0.01, 0.99, self.sample, transform=ax.transAxes,
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

    def crossplot(self, analytes=None, ptype='hist2D', bins=25, lognorm=True, filt=True, **kwargs):
        if analytes is None:
            analytes = [a for a in self.analytes if a != self.ratio_denominator]

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
        outputs = ['sample', 'method', 'analytes', 'ratio_denominator',
                   'despike_params', 'autorange_params']

        out = {}
        for o in outputs:
            out[o] = getattr(self, o)

        out['filter_params'] = self.filt.params
        out['filter_sequence'] = self.filt.sequence
        out['filter_used'] = self.filt.make_keydict()

        return out

class filt(object):
    def __init__(self, size, analytes):
        self.size = size
        self.analytes = analytes
        self.components = {}
        self.info = {}
        self.params = {}
        self.switches = {}
        self.keys = {}
        self.sequence = {}
        self.n = 0
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
        self.components[name] = filt
        self.info[name] = info
        self.params[name] = params
        self.sequence[self.n] = name
        self.n += 1
        for a in self.analytes:
            self.switches[a][name] = True

    def remove(self, name):
        del self.components[name]
        del self.info[name]
        del self.params[name]
        del self.keys[name]
        del self.sequence[name]
        for a in self.analytes:
            del self.switches[a][name]

    def clear(self):
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
        for f in sorted(self.components.keys()):
            unused = not any(self.switches[a][f] for a in self.analytes)
            if unused:
                self.remove(f)

    def on(self, analyte=None, filt=None):
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
        Takes a logical expression as an input, and returns a filter. Used for advanced
        filtering, where combinations of nested and/or filters are desired. Filter names must
        exactly match the names listed by print(filt).

        Example:
            key = '(Filter_1 | Filter_2) & Filter_3'
        is equivalent to:
            (Filter_1 OR Filter_2) AND Filter_3
        statements in parentheses are evaluated first.

        """
        if key != '':
            def make_runable(match):
                return "self.components['" + match.group(0) + "']"

            runable = re.sub('[^\(\)|& ]+', make_runable, key)
            return eval(runable)
        else:
            return ~np.zeros(self.size, dtype=bool)

    def make_keydict(self, analyte=None):
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

    def grab_filt(self,f,a=None):
        if isinstance(f, str):
            try:
                ind = self.make_fromkey(f)
            except ValueError:
                print("\n\n***Filter key invalid. Please consult manual and try again.")
        elif isinstance(f, dict):
            try:
                ind = self.make_fromkey(f[a])
            except ValueError:
                print("\n\n***Filter key invalid. Please consult manual and try again.\nOR\nAnalyte missing from filter key dict.")
        elif f:
            ind = self.make(a)
        else:
            ind = ~np.zeros(self.size, dtype=bool)
        return ind

    def get_components(self, key, analyte=None):
        out = {}
        for k, v in self.components.items():
            if key in k:
                if analyte is None:
                    out[k] = v
                elif self.switches[analyte][k]:
                    out[k] = v
        return out

    def info(self):
        out = ''
        for k in sorted(self.components.keys()):
            out += '{:s}: {:s}'.format(k, self.info[k]) + '\n'
        return(out)

    def bool_2_indices(self, bool_array):
        if ~isinstance(bool_array, np.ndarray):
            bool_array = np.array(bool_array)
        return np.arange(len(bool_array))[bool_array ^ np.roll(bool_array, 1)]

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
    #         xlims = self.bool_2_indices(self.components[f])

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
    Function to format element names nicely.
    """
    g = re.match('([A-Z][a-z]?)([0-9]+)', s).groups()
    return '$^{' + g[1] + '}$' + g[0]


def collate_csvs(in_dir,out_dir='./csvs'):
    """
    Function to grab all csvs from a directory, and place them in a new
    directory.
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

### more involved functions
#     def stridecalc(self, win, var=None):
#         if var is None:
#             var = self.cols[1:]
#         shape = self.Time.shape[:-1] + (self.Time.shape[-1] - win + 1, win)
#         strides = self.Time.strides + (self.Time.strides[-1], )
#         self.strides[win] = {'Time': np.lib.stride_tricks.as_strided(self.Time, shape=shape, strides=strides)}
#         for i in np.arange(var.size):
#             self.strides[win][var[i]] = np.lib.stride_tricks.as_strided(getattr(self, var[i]), shape=shape, strides=strides)

#     def gradientcalc(self, win, var=None, mode='mid'):
#         """
#         calculated moving gradient of line.
#            mode: start/end/mid
#               Determines how the gradient is shifted to match the timescale.
#         """
#         if win not in self.strides.keys():
#             self.stridecalc(win, var)
#         if var is None:
#             var = self.cols[1:]

#         if mode is 'mid':
#             pad = [win//2] * 2
#             while sum(pad) >= win:
#                 pad[1] -= 1
#         if mode is 'start':
#             pad = [win-1, 0]
#         if mode is 'end':
#             pad = [0, win-1]

#         self.gradient = {win: {}}
#         for i in np.arange(var.size):
#             self.gradient[win][var[i]] = np.pad(list(map(lambda x, y, o: np.polyfit(x, y, o)[0],
#                                                          self.strides[win]['Time'],
#                                                          self.strides[win][var[i]],
#                                                          [1] * self.strides[win]['Time'].shape[0])),
#                                                 pad, 'constant')

