import re
import numpy as np
from fuzzywuzzy import fuzz


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

        # self.keys is not added to?
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
            return keys[ratios >= mratio * 0.8]
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
