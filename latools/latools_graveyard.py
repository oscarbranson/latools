# Functions removed from main code that MAY still be useful somewhere.


# From Analyse Object
	# def save_ranges(self):
    #     """
    #     Saves signal/background/transition data ranges for each sample.
    #     """
    #     if os.path.isfile(self.param_dir + 'bkg.rng'):
    #         f = input(('Range files already exist. Do you want to overwrite '
    #                    'them (old files will be lost)? [Y/n]: '))
    #         if 'n' in f or 'N' in f:
    #             print('Ranges not saved. Run self.save_ranges() to try again.')
    #             return
    #     bkgrngs = []
    #     sigrngs = []
    #     for d in self.data:
    #         bkgrngs.append(d.sample + ':' + str(d.bkgrng.tolist()))
    #         sigrngs.append(d.sample + ':' + str(d.sigrng.tolist()))
    #     bkgrngs = '\n'.join(bkgrngs)
    #     sigrngs = '\n'.join(sigrngs)

    #     fb = open(self.param_dir + 'bkg.rng', 'w')
    #     fb.write(bkgrngs)
    #     fb.close()
    #     fs = open(self.param_dir + 'sig.rng', 'w')
    #     fs.write(sigrngs)
    #     fs.close()
    #     return

    # def load_ranges(self, bkgrngs=None, sigrngs=None):
    #     """
    #     Loads signal/background/transition data ranges for each sample.

    #     Parameters
    #     ----------
    #     bkgrngs : str or None
    #         A array of size (2, n) specifying time intervals that are
    #         background regions.
    #     sigrngs : str or None
    #         A array of size (2, n) specifying time intervals that are
    #         signal regions.

    #     Returns
    #     -------
    #     None
    #     """
    #     if bkgrngs is None:
    #         bkgrngs = self.param_dir + 'bkg.rng'
    #     bkgs = open(bkgrngs).readlines()
    #     samples = []
    #     bkgrngs = []
    #     for b in bkgs:
    #         samples.append(re.match('(.*):{1}(.*)',
    #                        b.strip()).groups()[0])
    #         bkgrngs.append(eval(re.match('(.*):{1}(.*)',
    #                        b.strip()).groups()[1]))
    #     for s, rngs in zip(samples, bkgrngs):
    #         self.data_dict[s].bkgrng = np.array(rngs)

    #     if sigrngs is None:
    #         sigrngs = self.param_dir + 'sig.rng'
    #     sigs = open(sigrngs).readlines()
    #     samples = []
    #     sigrngs = []
    #     for s in sigs:
    #         samples.append(re.match('(.*):{1}(.*)',
    #                        s.strip()).groups()[0])
    #         sigrngs.append(eval(re.match('(.*):{1}(.*)',
    #                        s.strip()).groups()[1]))
    #     for s, rngs in zip(samples, sigrngs):
    #         self.data_dict[s].sigrng = np.array(rngs)

    #     # number the signal regions (used for statistics and standard matching)
    #     for s in self.data:
    #         # re-create booleans
    #         s.makerangebools()

    #         # make trnrng
    #         s.trn[[0, -1]] = False
    #         s.trnrng = s.Time[s.trn ^ np.roll(s.trn, 1)]

    #         # number traces
    #         n = 1
    #         for i in range(len(s.sig)-1):
    #             if s.sig[i]:
    #                 s.ns[i] = n
    #             if s.sig[i] and ~s.sig[i+1]:
    #                 n += 1
    #         s.n = int(max(s.ns))  # record number of traces

    #     return

# From D Object

	# def bkgrange(self, rng=None):
    #     """
    #     Calculate background boolean array from list of limit pairs.

    #     Generate a background boolean string based on a list of [min,max] value
    #     pairs stored in self.bkgrng.

    #     If `rng` is supplied, these will be added to the bkgrng list before
    #     the boolean arrays are calculated.

    #     Parameters
    #     ----------
    #     rng : array_like
    #         [min,max] pairs defining the upper and lowe limits of background
    #         regions.

    #     Returns
    #     -------
    #     None
    #     """
    #     if rng is not None:
    #         if np.array(rng).ndim is 1:
    #             self.bkgrng = np.append(self.bkgrng, np.array([rng]), 0)
    #         else:
    #             self.bkgrng = np.append(self.bkgrng, np.array(rng), 0)

    #     self.bkg = tuples_2_bool(self.bkgrng, self.Time)
    #     # self.bkg = np.array([False] * self.Time.size)
    #     # for lb, ub in self.bkgrng:
    #     #     self.bkg[(self.Time > lb) & (self.Time < ub)] = True

    #     self.trn = ~self.bkg & ~self.sig  # redefine transition regions
    #     return

    # def sigrange(self, rng=None):
    #     """
    #     Calculate signal boolean array from list of limit pairs.

    #     Generate a background boolean string based on a list of [min,max] value
    #     pairs stored in self.bkgrng.

    #     If `rng` is supplied, these will be added to the sigrng list before
    #     the boolean arrays are calculated.

    #     Parameters
    #     ----------
    #     rng : array_like
    #         [min,max] pairs defining the upper and lowe limits of signal
    #         regions.

    #     Returns
    #     -------
    #     None
    #     """
    #     if rng is not None:
    #         if np.array(rng).ndim is 1:
    #             self.sigrng = np.append(self.sigrng, np.array([rng]), 0)
    #         else:
    #             self.sigrng = np.append(self.sigrng, np.array(rng), 0)

    #     self.sig = tuples_2_bool(self.sigrng, self.Time)
    #     # self.sig = np.array([False] * self.Time.size)
    #     # for ls, us in self.sigrng:
    #     #     self.sig[(self.Time > ls) & (self.Time < us)] = True

    #     self.trn = ~self.bkg & ~self.sig  # redefine transition regions
    #     return

    # def makerangebools(self):
    #     """
    #     Calculate signal and background boolean arrays from lists of limit
    #     pairs.
    #     """
    #     self.sig = tuples_2_bool(self.sigrng, self.Time)
    #     self.bkg = tuples_2_bool(self.bkgrng, self.Time)
    #     self.trn = ~self.bkg & ~self.sig
    #     return

    # def separate(self, analytes=None):
    #     """
    #     Extract signal and backround data into separate arrays.

    #     Isolates signal and background signals from raw data for specified
    #     elements.

    #     Parameters
    #     ----------
    #     analytes : array_like
    #         list of analyte names (default = all analytes)

    #     Returns
    #     -------
    #     None
    #     """
    #     if analytes is None:
    #         analytes = self.analytes
    #     self.data['background'] = {}
    #     self.data['signal'] = {}
    #     for a in analytes:
    #         self.data['background'][a] = self.focus[a].copy()
    #         self.data['background'][a][~self.bkg] = np.nan
    #         self.data['signal'][a] = self.focus[a].copy()
    #         self.data['signal'][a][~self.sig] = np.nan


    # def bkg_subtract(self, bkgs):
    #     """
    #     Subtract provided background from signal (focus stage).

    #     Results is saved in new 'bkgsub' focus stage

    #     Parameters
    #     ----------
    #     bkgs : dict
    #         dict containing background values to subtract from
    #         focus stage of data.


    #     Returns
    #     -------
    #     None
    #     """

    #     if any(a not in bkgs.keys() for a in analytes):
    #         warnings.warn(('Not all analytes have been provided in bkgs.\n' +
    #                        "If you didn't do this on purpose, something is\n" +
    #                        "wrong!"))

    #     self.data['bkgsub'] = {}
    #     for a in self.analytes:
    #         self.data['bkgsub'][a] = self.focus[a] - bkgs[a]
    #     self.setfocus('bkgsub')
    #     return

    # def bkg_correct(self, mode='constant'):
    #     """
    #     Subtract background from signal.

    #     Subtract constant or polynomial background from all analytes.

    #     Parameters
    #     ----------
    #     mode : str or int
    #         'constant' or an int describing the degree of polynomial
    #         background.

    #     Returns
    #     -------
    #     None
    #     """
    #     params = locals()
    #     del(params['self'])
    #     self.bkgcorrect_params = params

    #     self.bkgrange()
    #     self.sigrange()
    #     self.separate()

    #     self.data['bkgsub'] = {}
    #     if mode == 'constant':
    #         for c in self.analytes:
    #             self.data['bkgsub'][c] = \
    #                 (self.data['signal'][c] -
    #                  np.nanmean(self.data['background'][c]))
    #     if (mode != 'constant'):
    #         for c in self.analytes:
    #             p = np.polyfit(self.Time[self.bkg], self.focus[c][self.bkg],
    #                            mode)
    #             self.data['bkgsub'][c] = \
    #                 (self.data['signal'][c] -
    #                  np.polyval(p, self.Time))
    #     self.setfocus('bkgsub')
    #     return


# Helper Functions

# def gauss_inv(y, *p):
#     """
#     Inverse Gaussian function.

#     For determining the x coordinates
#     for a given y intensity (i.e. width at a given height).

#     Parameters
#     ----------
#     y : float
#         The height at which to calculate peak width.
#     *p : parameters unpacked to mu, sigma
#         mu: peak center
#         sigma: peak width

#     Return
#     ------
#     array_like
#         x positions either side of mu where gauss(x) == y.
#     """
#     mu, sigma = p
#     return np.array([mu - 1.4142135623731 * np.sqrt(sigma**2*np.log(1/y)),
#                      mu + 1.4142135623731 * np.sqrt(sigma**2*np.log(1/y))])