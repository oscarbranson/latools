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
from scipy.stats import gaussian_kde
from mpld3 import plugins
from IPython import display
from mpld3 import enable_notebook, disable_notebook
from scipy.optimize import curve_fit


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
                           input('Enter background limits (as: \
                                 start, end, start, end):\n').split(',')]
                    bkg = np.array(bkg).reshape(len(bkg)//2, 2)
                    OK = True
                except:
                    print("Incorrect Values, try again:\n")

            OK = False
            while OK is False:
                try:
                    sig = [float(f) for f in
                           input('Enter sample limits (as: start, \
                                 end, start, end):\n').split(',')]
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
        if type(analytes) is str:
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
            if type(plot) is str:
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
        if os.path.isfile(fname):
            f = input("SRM range files already exist in '" + fname + "'. Do you want to overwrite them (old files will be lost)? [Y/n]: ")
            if 'n' in f or 'N' in f:
                print('SRM ranges not saved. Run self.save_srm_ids() to try again.')
                return
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

    def distribution_check(self, analytes=None, mode='lower', filt=False):
        """
        Checks the specified analytes for bimodality by looking for minima
        in a gaussian kde data density curve. If minima are found, data either
        above or below the threshold are excluded. Behaviour is determined by
        the 'mode' argument.

        mode:   str ('lower'/'upper')
            'lower': data below the cutoff are kept.
            'upper': data above the cutoff are kept.
        """
        if analytes is None:
            analytes = self.analytes
        analytes = np.array(analytes).flatten()
        for d in self.data:
            d.bimodality_fix(analytes, report=False, mode=mode, filt=filt)

    def distribution_reports(self, analytes=['Ba138'], dirpath=None, filt=False):
        """
        Saves data distribution pdfs for all analytes specified,
        showing where they have been cut by a bimodality check
        (if it has been run).
        pdfs are saved in the specified directory (dirpath).
        """
        fails = []
        if dirpath is None:
            dirpath = self.report_dir
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        analytes = np.array([analytes]).flatten()
        for k, v in self.data_dict.items():
            try:
                fig = v.bimodality_report(filt=filt)
                fig.savefig(dirpath + '/' + k + '_distributions.pdf')
                plt.close(fig)
            except:
                fails.append(k)
        if len(fails) > 0:
            f = open("errors.log", 'w')
            f.write('\nDistribution Reports:\n')
            f.write('\n'.join(fails))
            f.close()
            print('Some reports failed. See log.')

        return

    def clear_filters(self):
        for d in self.data:
            d.filt = {}
            d.filtrngs = {}

    def threshold_filter(self, analytes, thresholds, modes):
        for d in self.data:
            params = zip(np.array(analytes, ndmin=1),
                         np.array(thresholds, ndmin=1),
                         np.array(modes, ndmin=1))
            for p in params:
                d.threshold_filter(p[0], p[1], p[2])

    # plot helper functions
    def unitpicker(self, a, llim=0.1):
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
                if type(filt) is str:
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
                mx, ux = self.unitpicker(np.nanmean(self.focus[analytes[x]]))
                my, uy = self.unitpicker(np.nanmean(self.focus[analytes[y]]))
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

        if type(analyte) is str:
            analyte = [analyte]
        if type(sample) is str:
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
        self.filt_switches = {}
        for a in self.analytes:
            self.filt_switches[a] = {}
        self.filt = filt(self.Time.size)

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
        if type(win) is not int:
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
    #             print(a, sum(over))
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

    # automagically select signal and background regions
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

    # # Function for calculating sample statistics
    # def sample_stats(self, analytes=None, filt=True,
    #                  stat_fns=[np.nanmean, np.nanstd],
    #                  eachtrace=True):
    #     """
    #     Returns samples, analytes, and arrays of statistics
    #     of shape (samples, analytes). Statistics are calculated
    #     from the 'focus' data variable, so output depends on how
    #     the data have been processed.

    #     analytes: array-like
    #         list of analytes to calculate the statistic on
    #     stat_fns: array-like
    #         list of functions that take a single array-like input,
    #         and return a single statistic. Function should be able
    #         to cope with numpy NaN values.
    #     filt: bool or str
    #         filt specifies the filter to apply to the data when calculating
    #         sample statistics. It can either:
    #         bool:  True | False
    #             If True, applies filter created by bimodality_fix to each
    #             analyte individually.
    #         str: name of analyte specific filter
    #             applies a specific filter to all the data,
    #             or a filter resulting from the union of all analyte-specific
    #             filters.
    #     eachtrace: bool
    #         Return individual statistics for each analysis trace (True),
    #         or each sample (False)?
    #     """
    #     if analytes is None:
    #             analytes = self.analytes

    #     self.stats = {}
    #     self.stats['analytes'] = analytes

    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", category=RuntimeWarning)
    #         for f in stat_fns:
    #             self.stats[f.__name__] = []
    #             for a in analytes:
    #                 if type(filt) is bool:
    #                     if filt and a in self.filt.keys():
    #                         ind = self.filt[a]
    #                     else:
    #                         ind = np.array([True] * self.focus[a].size)
    #                 if type(filt) is str:
    #                     ind = self.filt[filt]
    #                 if eachtrace:
    #                     sts = []
    #                     for t in np.arange(self.n) + 1:
    #                         sts.append(f(self.focus[a][ind & (self.ns==t)]))
    #                     self.stats[f.__name__].append(sts)
    #                 else:
    #                     self.stats[f.__name__].append(f(self.focus[a][ind]))
    #             self.stats[f.__name__] = np.array(self.stats[f.__name__])

    #     try:
    #         self.unstats = un.uarray(self.stats['nanmean'], self.stats['nanstd'])
    #     except:
    #         pass

    #     return


    # Data Selections Tools

    # def clear_filters(self):
    #     self.filt = {}
    #     self.filtrngs = {}

    # Filter Operations

    def filter_on(self, analyte=None, filt=None):
        if type(analyte) is str:
            analyte = [analyte]
        if type(filt) is str:
            filt = [filt]

        if analyte is None:
            analyte = self.analytes
        if filt is None:
            filt = self.filt_switches[analyte[0]].keys()

        for a in analyte:
            for f in filt:
                self.filt_switches[a][f] = True

    def filter_off(self, analyte=None, filt=None):
        if type(analyte) is str:
            analyte = [analyte]
        if type(filt) is str:
            filt = [filt]

        if analyte is None:
            analyte = self.analytes
        if filt is None:
            filt = self.filt_switches[analyte[0]].keys()

        for a in analyte:
            for f in filt:
                self.filt_switches[a][f] = False

    def print_filt_switches(self):
        # also has to happen at analysis level.
        leftpad = max([len(s) for s in self.filt_switches[self.analytes[0]].keys()] + [11]) + 2
        out = '{string:{number}s}'.format(string='Filter Name', number=leftpad)
        for a in self.analytes:
            out += '{:7s}'.format(a)
        out += '\n'

        for t in self.filt_switches[self.analytes[0]].keys():
            out += '{string:{number}s}'.format(string=str(t), number=leftpad)
            for a in self.analytes:
                out += '{:7s}'.format(str(self.filt_switches[a][t]))
            out += '\n'

        print (out)


    def filter_threshold(self, analyte, threshold, mode='above'):
        """
        Generates threshold filters for analytes, when provided with analyte,
        threshold, and mode. Mode specifies whether data 'below'
        or 'above' the threshold are kept.
        """
        params = locals()
        del(params['self'])

        if mode == 'below':
            self.filt.add_filt(analyte + '_thresh', self.focus[analyte] <= threshold, analyte + '_thresh',
                               'Keep ' + mode + ' {:.3e} '.format(threshold) + analyte, params)
        if mode == 'above':
            self.filt.add_filt(analyte + '_thresh', self.focus[analyte] >= threshold,
                               'Keep ' + mode + ' {:.3e} '.format(threshold) + analyte, params)

        for a in self.analytes:
            self.filt_switches[a][analyte + '_thresh'] = True

        # self.filt_switches = {}
        # self.filt = filt(self.Time.size)

    # def threshold_filter(self, analyte, threshold, mode='above'):
    #     """
    #     Generates threshold filters for analytes, when provided with analyte,
    #     threshold, and mode. Mode specifies whether data 'below'
    #     or 'above' the threshold are kept.
    #     """
    #     if not hasattr(self, 'filt'):
    #         self.filt = {}

    #     if mode == 'below':
    #         self.filt[analyte + '_thresh'] = self.focus[analyte] <= threshold
    #     if mode == 'above':
    #         self.filt[analyte + '_thresh'] = self.focus[analyte] >= threshold

    #     # make 'master' filter
    #     combined = np.array([True] * self.Time.size)
    #     for k, v in self.filt.items():
    #         if k is not 'combined':
    #             combined = combined & v
    #     self.filt['combined'] = combined

    #     # update self.filtrngs
    #     for f, a in self.filt.items():
    #         if ~hasattr(self, 'filtrngs'):
    #             self.filtrngs = {}
    #         if f not in self.filtrngs.keys():
    #             self.filtrngs[f] = list(zip(self.Time[(a & np.roll(~a, 1))],
    #                                         self.Time[(a & np.roll(~a, -1))]))

    #     a = self.filt['combined']
    #     self.filtrngs['combined'] = list(zip(self.Time[(a & np.roll(~a, 1))],
    #                                          self.Time[(a & np.roll(~a, -1))]))

        # print(self.sample, self.filt.keys())

    def filter_distribution(self, analyte, binwidth=0.1, filt=False, transform=None):
        params = locals()
        del(params['self'])

        # generate filter
        if filt:
            ind = self.filt.make_filt(analyte)
        else:
            ind = ~np.isnan(self.focus[analyte])

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

                self.filt.add_filt(name=analyte + '_distribution_{:.0f}'.format(i),
                                   filt=filt,
                                   info=info,
                                   params=params)
                # add to filt_switches
                for a in self.analytes:
                    self.filt_switches[a][analyte + '_distribution_{:.0f}'.format(i)] = True
        else:
            self.filt.add_filt(name=analyte + '_distribution_failed',
                               filt=~np.isnan(self.focus[analyte]),
                               info=analyte + ' is within a single distribution. No data removed.',
                               params=params)
        return

    # def bimodality_fix(self, analytes, mode='lower', report=False, filt=False):
    #     """
    #     Function that checks for bimodality in the data, and excludes either
    #     the higher or lower data.

    #     Inputs:
    #         analytes:                         array-like
    #             list of analytes to check
    #         mode:                         higher|[lower]
    #             higher: keeps the higher distribution
    #             lower: keeps the lower distribution

    #     Returns:
    #         Updates D object with 'filt' dict, containing the exclusions
    #         calculated by the bimodal split cutoff.
    #     """
    #     if not hasattr(self, 'filt'):
    #         self.filt = {}
    #     self.bimodal_limits = {}
    #     if report and ~hasattr(self, 'bimodal_reports'):
    #         self.bimodal_reports = {}
    #     for a in np.array(analytes, ndmin=1):
    #         if type(filt) is bool:
    #             if filt and a in self.filt.keys():
    #                 ind = ~np.isnan(self.focus[a]) & self.filt[a]
    #             else:
    #                 ind = ~np.isnan(self.focus[a])
    #         if type(filt) is str:
    #             ind = ~np.isnan(self.focus[a]) & self.filt[filt]
    #         if sum(ind) <= 1:
    #             ind = ~np.isnan(self.focus[a])  # remove the filter if it takes out all data

    #         kde = gaussian_kde(self.focus[a][ind])
    #         x = np.linspace(np.nanmin(self.focus[a][ind]), np.nanmax(self.focus[a][ind]),
    #                         kde.dataset.size // 3)
    #         yd = kde.pdf(x)
    #         self.bimodal_limits[a] = self.findmins(x, yd)
    #         if self.bimodal_limits[a].size > 0:
    #             self.bimodal_correction = True
    #             if mode is 'lower':
    #                 self.filt[a] = self.focus[a] < self.bimodal_limits[a][0]
    #             if mode is 'upper':
    #                 self.filt[a] = self.focus[a] > self.bimodal_limits[a][-1]
    #         else:
    #             self.filt[a] = np.array([True] * self.focus[a].size)

    #         if report:
    #             self.bimodal_reports[a] = self.bimodality_report(a, mode=mode)

    #     # make 'master' filter
    #     combined = np.array([True] * self.Time.size)
    #     for k, v in self.filt.items():
    #         if k is not 'combined':
    #             combined = combined & v
    #     self.filt['combined'] = combined

    #     # update self.filtrngs
    #     if ~hasattr(self, 'filtrngs'):
    #             self.filtrngs = {}
    #     for f, a in self.filt.items():
    #         if f not in self.filtrngs.keys():
    #             self.filtrngs[f] = list(zip(self.Time[(a & np.roll(~a, 1))],
    #                                         self.Time[(a & np.roll(~a, -1))]))

    #     a = self.filt['combined']
    #     self.filtrngs['combined'] = list(zip(self.Time[(a & np.roll(~a, 1))],
    #                                          self.Time[(a & np.roll(~a, -1))]))
    #     return

    def filter_clustering(self, analytes, mode):
        """
        use clustering algorithms to separate data
        """
        pass

    def filter_correlation(self, x_analyte, y_analyte, r_threshold, p_threshold):
        """
        correlate two analytes, remove regions where they correlate.
        """
        pass


    # Plotting Functions
    def genaxes(self, n, ncol=4, panelsize=[3, 3], tight_layout=True,
                **kwargs):
        """
        Function to generate a grid of subplots for a given set of plots.
        """
        if n % ncol is 0:
            nrow = int(n/ncol)
        else:
            nrow = int(n//ncol + 1)

        fig, axes = plt.subplots(nrow, ncol, figsize=[panelsize[0] * ncol,
                                 panelsize[1] * nrow],
                                 tight_layout=tight_layout,
                                 **kwargs)
        for ax in axes.flat[n:]:
            fig.delaxes(ax)

        return fig, axes

    def pretty_element(self, s):
        """
        Function to format element names nicely.
        """
        g = re.match('([A-Z][a-z]?)([0-9]+)', s).groups()
        return '$^{' + g[1] + '}$' + g[0]

    # def tplot(self, traces=None, figsize=[10, 4], scale=None, filt=False,
    #           ranges=False, plot_filt=None, stats=True, sig='nanmean', err='nanstd', interactive=False):
    #     """
    #     Convenience function for plotting traces.

    #     Parameters:
    #         traces:     list of strings containing names of analytes to plot.
    #                     default = all analytes.
    #         figsize:    tuple-like
    #                     size of final figure.
    #         scale:      str ('log') or blank.
    #                     whether to plot data on a log scale.
    #         filt:       boolean, string or list
    #                     Whether or not to plot the filtered data for all (bool)
    #                     or specific (str, list) analytes.
    #         stats:      boolean
    #                     Whether or not to plot the mean and standard deviation
    #                     for the traces.
    #     """
    #     if interactive:
    #         enable_notebook()  # make the plot interactive
    #     if traces is None:
    #         traces = self.analytes
    #     if type(traces) is str:
    #         traces = [traces]
    #     fig = plt.figure(figsize=figsize)
    #     ax = fig.add_subplot(111)

    #     for t in traces:
    #         x = self.Time
    #         y = self.focus[t]

    #         if type(filt) is bool:
    #             if filt and t in self.filt.keys():
    #                 ind = self.filt[t]
    #             else:
    #                 ind = np.array([True] * x.size)
    #         if type(filt) is str:
    #             ind = self.filt[filt]

    #         if scale is 'log':
    #             ax.set_yscale('log')
    #             y[y == 0] = 1
    #         ax.plot(x, y, color=self.cmap[t], label=t)
    #         if any(~ind):
    #             ax.scatter(x[~ind], y[~ind], s=5, color='k')

    #         # Plot averages and error envelopes
    #         if stats and hasattr(self, 'stats'):
    #             sts = self.stats[sig][0].size
    #             if sts > 1:
    #                 for n in np.arange(self.n):
    #                     x = [self.Time[self.ns==n+1][0], self.Time[self.ns==n+1][-1]]
    #                     y = [self.stats[sig][self.stats['analytes']==t][0][n]] * 2

    #                     yp = [self.stats[sig][self.stats['analytes']==t][0][n] + self.stats[err][self.stats['analytes']==t][0][n]] * 2
    #                     yn = [self.stats[sig][self.stats['analytes']==t][0][n] - self.stats[err][self.stats['analytes']==t][0][n]] * 2

    #                     ax.plot(x, y, color=self.cmap[t], lw=2)
    #                     ax.fill_between(x + x[::-1], yp + yn, color=self.cmap[t], alpha=0.4, linewidth=0)
    #             else:
    #                 x = [self.Time[0], self.Time[-1]]
    #                 y = [self.stats[sig][self.stats['analytes']==t][0]] * 2
    #                 yp = [self.stats[sig][self.stats['analytes']==t][0] + self.stats[err][self.stats['analytes']==t][0]] * 2
    #                 yn = [self.stats[sig][self.stats['analytes']==t][0] - self.stats[err][self.stats['analytes']==t][0]] * 2

    #                 ax.plot(x, y, color=self.cmap[t], lw=2)
    #                 ax.fill_between(x + x[::-1], yp + yn, color=self.cmap[t], alpha=0.4, linewidth=0)

    #     if ranges:
    #         for lims in self.bkgrng:
    #             ax.axvspan(*lims, color='k', alpha=0.1)
    #         for lims in self.sigrng:
    #             ax.axvspan(*lims, color='r', alpha=0.1)

    #         if plot_filt is None:
    #             plot_filt = 'combined'
    #         if hasattr(self, 'filtrngs'):
    #             for lims in self.filtrngs[plot_filt]:
    #                 ax.axvspan(*lims, color='b', alpha=0.1)

    #     ax.text(0.01, 0.99, self.sample, transform=ax.transAxes,
    #             ha='left', va='top')

    #     ax.set_xlabel('Time (s)')

    #     if interactive:
    #         ax.legend()
    #         plugins.connect(fig, plugins.MousePosition(fontsize=14))
    #         display.clear_output(wait=True)
    #         display.display(fig)
    #         input('Press [Return] when finished.')
    #         disable_notebook()  # stop the interactivity
    #     else:
    #         ax.legend(bbox_to_anchor=(1.12, 1))

    #     return fig

    # def statplot(self, analytes=None, figsize=[8, 8], scale=None, vals='nanmean', errs='nanstd'):
    #     """
    #     Plot each trace individually, and the individual mean for comparison.
    #     """

    #     self.unstats = un.uarray(self.stats[vals], self.stats[errs])

    #     means = []
    #     for s in self.unstats:
    #         means.append(s.mean())
    #     means = np.array(means)

    #     fig, ax = plt.subplots(1, 1, figsize=figsize)

    #     x = 0
    #     for a in analytes:
    #         # individual traces
    #         ys = self.unstats[self.analytes == a]
    #         xs = np.linspace(x-0.1, x+0.1, ys.size)
    #         ax.errorbar(xs, un.nominal_values(ys)[0], yerr=un.std_devs(ys)[0],
    #                     color=self.cmap[a], lw=0, elinewidth=1, capsize=0,
    #                     marker='o', markersize=3)

    #         # means of all traces
    #         avx = [x-0.3, x+0.3]
    #         avy = means[self.analytes == a]
    #         ave = [un.nominal_values(avy)[0] + un.std_devs(avy)[0]] * 2 + [un.nominal_values(avy)[0] - un.std_devs(avy)[0]] * 2

    #         ax.plot(avx, [un.nominal_values(avy)] * 2, lw=2, color=self.cmap[a])
    #         ax.fill_between(avx + avx[::-1], ave, lw=0, color=self.cmap[a], alpha=0.4)

    #         x += 1

    #     if scale is not None:
    #         ax.set_yscale(scale)

    #     ax.set_xticklabels([''] + [self.pretty_element(a) for a in analytes])

    #     return fig

    # def crossplot(self, analytes=None, ptype='scatter', bins=25, lognorm=True,
    #               **kwargs):
    #     """
    #     Function for creating scatter crossplots of specified analytes.
    #     Useful for checking for correlations withing the traces, which can
    #     indicate contamination.

    #     Parameters:
    #         analytes:   list of analytes to plot (default = all)
    #         ptype:      'scatter' | 'hist2d'
    #         bins:       (int) Number of bins to use if hist2d
    #         lognorm:    (bool) Log-normalise the data?
    #         **kwargs:   passed to 'scatter' - does nothing for hist2d
    #     """
    #     if analytes is None:
    #         analytes = [a for a in self.analytes if 'Ca' not in a]

    #     numvars = len(analytes)
    #     fig, axes = plt.subplots(nrows=numvars, ncols=numvars,
    #                              figsize=(12, 12))
    #     fig.subplots_adjust(hspace=0.05, wspace=0.05)

    #     for ax in axes.flat:
    #         ax.xaxis.set_visible(False)
    #         ax.yaxis.set_visible(False)

    #         if ax.is_first_col():
    #             ax.yaxis.set_ticks_position('left')
    #         if ax.is_last_col():
    #             ax.yaxis.set_ticks_position('right')
    #         if ax.is_first_row():
    #             ax.xaxis.set_ticks_position('top')
    #         if ax.is_last_row():
    #             ax.xaxis.set_ticks_position('bottom')

    #     if ptype is 'scatter':
    #         # Plot the data.
    #         for i, j in zip(*np.triu_indices_from(axes, k=1)):
    #             for x, y in [(i, j), (j, i)]:
    #                 # set multipliers and units
    #                 mx = my = 1000
    #                 ux = uy = '(mmol/mol)'
    #                 if np.nanmin(self.focus[analytes[x]] * my) < 0.1:
    #                     mx = 1000000
    #                     ux = '($\mu$mol/mol)'
    #                 if np.nanmin(self.focus[analytes[y]] * my) < 0.1:
    #                     my = 1000000
    #                     uy = '($\mu$mol/mol)'
    #                 # make plot
    #                 px = self.focus[analytes[x]] * mx
    #                 py = self.focus[analytes[y]] * my
    #                 axes[x, y].scatter(px, py, color=self.cmap[analytes[x]],
    #                                    **kwargs)
    #                 axes[x, y].set_xlim([np.nanmin(px), np.nanmax(px)])
    #                 axes[x, y].set_ylim([np.nanmin(py), np.nanmax(py)])

    #     if ptype is 'hist2d':
    #         cmlist = ['Blues', 'BuGn', 'BuPu', 'GnBu',
    #                   'Greens', 'Greys', 'Oranges', 'OrRd',
    #                   'PuBu', 'PuBuGn', 'PuRd', 'Purples',
    #                   'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']
    #         for i, j in zip(*np.triu_indices_from(axes, k=1)):
    #             for x, y in [(i, j), (j, i)]:
    #                 # set unit multipliers
    #                 mx = my = 1000
    #                 if np.nanmin(self.focus[analytes[x]] * mx) < 0.1:
    #                     mx = 1000000
    #                 if np.nanmin(self.focus[analytes[y]] * my) < 0.1:
    #                     my = 1000000
    #                 # make plot
    #                 px = self.focus[analytes[x]][~np.isnan(self.focus
    #                                                        [analytes[x]])] * mx
    #                 py = self.focus[analytes[y]][~np.isnan(self.focus
    #                                                        [analytes[y]])] * my
    #                 if lognorm:
    #                     axes[x, y].hist2d(px, py, bins,
    #                                       norm=mpl.colors.LogNorm(),
    #                                       cmap=plt.get_cmap(cmlist[x]))
    #                 else:
    #                     axes[x, y].hist2d(px, py, bins,
    #                                       cmap=plt.get_cmap(cmlist[x]))
    #                 axes[x, y].set_xlim([np.nanmin(px), np.nanmax(px)])
    #                 axes[x, y].set_ylim([np.nanmin(py), np.nanmax(py)])

    #     for i, label in enumerate(analytes):
    #         # assign unit label
    #         unit = '\n(mmol/mol)'
    #         if np.nanmin(self.focus[label] * 1000) < 0.1:
    #             unit = '\n($\mu$mol/mol)'
    #         # plot label
    #         axes[i, i].annotate(label+unit, (0.5, 0.5),
    #                             xycoords='axes fraction',
    #                             ha='center', va='center')

    #     for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
    #         axes[j, i].xaxis.set_visible(True)
    #         for label in axes[j, i].get_xticklabels():
    #             label.set_rotation(90)

    #         axes[i, j].yaxis.set_visible(True)

    #     return fig, axes

    # def bimodality_report(self, mode='lower', filt=False):
    #     """
    #     Function to plot reports for bimodal exclusion checks.
    #     """
    #     fig, axes = self.genaxes(3 * len([i for i in list(self.filt.keys()) if i is not 'combined' and 'thresh' not in i]), 3, [4, 3])

    #     fig.suptitle(self.sample, weight='bold', x=0.1, y=1)

    #     i = 0
    #     for a in sorted([k for k in self.filt.keys() if k in self.analytes]):
    #         if type(filt) is bool:
    #             if filt and a in self.filt.keys():
    #                 ind = ~np.isnan(self.focus[a]) & self.filt[a]
    #             else:
    #                 ind = ~np.isnan(self.focus[a])
    #         if type(filt) is str:
    #             ind = ~np.isnan(self.focus[a]) & self.filt[filt]
    #         if sum(ind) <= 1:
    #             ind = ~np.isnan(self.focus[a])  # remove the filter if it takes out all data

    #         # calculate the multiplier necessary to make units sensible
    #         mean = np.nanmean(self.focus[a][ind])
    #         if mean * 1E3 > 0.1:
    #             m = 1E3
    #             label = self.pretty_element(a) + '/Ca (mmol/mol)'
    #         else:
    #             m = 1E6
    #             label = self.pretty_element(a) + '/Ca ($\mu$mol/mol)'

    #         kde = gaussian_kde(self.focus[a][ind] * m)
    #         bins = x = np.linspace(np.nanmin(self.focus[a][ind]) * m,
    #                                np.nanmax(self.focus[a][ind]) * m,
    #                                kde.dataset.size // 3)
    #         yd = kde.pdf(bins)

    #         bstep = x[1] - x[0]

    #         n, _ = np.histogram(self.focus[a][ind] * m, bins)

    #         if axes.ndim > 1:
    #             ax1, ax2, ax3 = axes[i, 0], axes[i, 1], axes[i, 2]
    #             i += 1
    #         else:
    #             ax1, ax2, ax3 = axes

    #         ax1.bar(bins[:-1], n/(sum(n) * bstep), bstep,
    #                 color=(0, 0, 1, 0.5), lw=0)
    #         ax1.set_ylabel('Density')
    #         ax1.set_xlabel(label)

    #         ax1.plot(x, yd, color='r', lw=2)

    #         ax2.plot(np.sort(self.focus[a][ind] * m),
    #                  color=(0, 0, 1, 0.7))
    #         ax2.set_ylabel(label)
    #         ax2.set_xlabel('Point No')

    #         ax3.plot(self.Time, self.focus[a] * m, color=(0, 0, 0, 0.2))
    #         ax3.set_ylabel(label)
    #         ax3.set_xlabel('Time (s)')
    #         ax3.set_ylim(ax2.get_ylim())

    #         if self.bimodal_limits[a].size > 0:
    #             n = sum(self.filt[a][ind]) - 1

    #             if mode is 'lower':
    #                 ax1.axvline(self.bimodal_limits[a][0] * m, color='k',
    #                             ls='dashed')
    #                 ax1.axvspan(self.bimodal_limits[a][0] * m, ax1.get_xlim()[1],
    #                             color=(1, 0, 0, 0.1))

    #                 ax2.axhline(self.bimodal_limits[a][0] * m, color='k',
    #                             ls='dashed')
    #                 ax2.axvspan(n, ax2.get_xlim()[1], color=(1, 0, 0, 0.1))

    #                 ax3.axhline(self.bimodal_limits[a][0] * m, color='k',
    #                             ls='dashed')
    #                 ax3.axhspan(self.bimodal_limits[a][0] * m, ax1.get_xlim()[1],
    #                             color=(1, 0, 0, 0.1))

    #             if mode is 'upper':
    #                 ax1.axvline(self.bimodal_limits[a][-1] * m, color='k',
    #                             ls='dashed')
    #                 ax1.axvspan(ax1.get_xlim()[0], self.bimodal_limits[a][-1] * m,
    #                             color=(1, 0, 0, 0.1))

    #                 ax2.axhline(self.bimodal_limits[a][-1] * m, color='k',
    #                             ls='dashed')
    #                 ax2.axvspan(ax2.get_xlim()[0], n, color=(1, 0, 0, 0.1))

    #                 ax3.axhline(self.bimodal_limits[a][-1] * m, color='k',
    #                             ls='dashed')
    #                 ax3.axhspan(ax1.get_xlim()[0], self.bimodal_limits[a][-1] * m,
    #                             color=(1, 0, 0, 0.1))

    #             ax2.axvline(n, color='k', ls='dashed')
    #             ax2.text(n+5, ax2.get_ylim()[1]-0.2, 'n = {:.0f}'.format(n),
    #                      va='top', ha='left')

    #             ax3.scatter(self.Time[self.filt[a]],
    #                         self.focus[a][self.filt[a]]*m, s=3, color='k')
    #             ax3.scatter(self.Time[~self.filt[a]],
    #                         self.focus[a][~self.filt[a]]*m, s=3,
    #                         color=(0, 0, 0, 0.2))
    #         else:
    #             ax3.scatter(self.Time, self.focus[a]*m, s=3, color='k')
    #     return fig

class filt(object):
    def __init__(self, size):
        self.size = size
        self.components = {}
        self.info = {}
        self.params = {}

    def get_filtnames(self):
        return dict(zip(self.components.keys(), [True] * len(self.components.keys())))

    def make_filt(self, analyte, switches, mode='and'):
        filt = np.array([True] * self.size)
        for k, v in switches[analyte].items():
            if v:
                if mode == 'and':
                    filt = filt & self.components[k]
                if mode == 'or':
                    filt = filt | self.components[k]
        return filt

    def add_filt(self, name, filt, info='', params=()):
        self.components[name] = filt
        self.info[name] = info
        self.params[name] = params

    def filt_info(self):
        out = ''
        for k in self.components.keys():
            out += '{:s}: {:s}'.format(k, self.info[k]) + '\n'
        return(out)

    def clear_filters(self):
        self.components = {}
        self.info = {}
        self.params = {}
        return


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

