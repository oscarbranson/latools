"""
An object used for storing, manipulating and modifying data filters.
"""

import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher as seqm
from latools.helpers.helpers import bool_2_indices, Bunch

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
    """
    
    def __init__(self, size, analytes):
        self.size = size
        self.analytes = analytes
        self.maxset = -1
        
        findex = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['N', 'filter'])
        self.fnames = []
        self.filter_table = pd.DataFrame(index=findex, columns=self.analytes)
        self.filter_components = pd.DataFrame(index=np.arange(size), columns=findex)
        
        self.param = Bunch()
        self.info = Bunch()
        self.keydict = Bunch()
        
        self.N = 0

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
        setn : int
            the set number of the filter

        Returns
        -------
        None
        """
        
        if setn is None:
            setn = self.maxset + 1
        self.maxset = setn
        
        # store params and info
        self.param[setn] = params
        self.info[setn] = info
        
        # store switches and filter
        self.filter_table.loc[(setn, name), :] = False
        self.filter_components.loc[:, (setn, name)] = filt
        self.fnames.append(f'{setn}:{name}')
            
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
        raise DeprecationWarning('This no longer works. Use `.filter_clear()` instead, then re-run the filters you want to keep.')

    def clear(self):
        """
        Clear all filters.
        """
        self.__init__(self.size, self.analytes)
    
    def clean(self):
        raise DeprecationWarning('This no longer works.')
    
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
        if analyte is None:
            analyte = self.analytes
        
        if isinstance(filt, str):
            # find filter name
            n, filt = self.fuzzmatch(filt, multi=True)
            
        self.filter_table.loc[(n, filt), analyte] = True
    
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
        if analyte is None:
            analyte = self.analytes
        
        if isinstance(filt, str):
            # find filter name
            n, filt = self.fuzzmatch(filt, multi=True)
            
        self.filter_table.loc[(n, filt), analyte] = False
    
    def fuzzmatch(self, fuzzkey, multi=True):
        """
        Identify a filter by fuzzy string matching.

        Partial ('fuzzy') matching performed by `fuzzywuzzy.fuzzy.ratio`

        Parameters
        ----------
        fuzzkey : str
            A string that partially matches one filter name more than the others.

        Returns
        -------
        The name of the most closely matched filter. : str
        """
        keys, ratios = np.array([(f, seqm(None, fuzzkey, f).ratio()) for f in self.fnames]).T
        mratio = max(ratios)
        
        if multi:
            match = keys[ratios == mratio]
        else:
            if sum(ratios == mratio) == 1:
                match = keys[ratios == mratio][0]
            else:
                raise ValueError("\nThe filter key provided ('{:}') matches two or more filter names equally well:\n".format(fuzzkey) + ', '.join(keys[ratios == mratio]) + "\nBe more specific, or prepend the sequence number?")
        
        n, filt = match[0].split(':')
        return int(n), filt
    
    def make_analyte(self, analyte):
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
        elif analyte is None:
            analyte = self.analytes
            
        key = []
        for n, f in self.filter_table[analyte].index[self.filter_table[analyte].any(1)]:
            key.append(f'{n}:{f}')
        
        return self.make_fromkey('&'.join(key))
    
    def make_fromkey(self, key):
        """
        Make filter from logical expression.

        Takes a logical expression as an input, and returns a filter. Used for advanced
        filtering, where combinations of nested and/or filters are desired. Filter names must
        exactly match the names listed by print(filt).

        Example: ``key = '(Filter_1 | Filter_2) & Filter_3'``
        is equivalent to:
        ``(Filter_1 OR Filter_2) AND Filter_3``
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
                return "self.filter_components.loc[:," + str(tuple(self.fuzzmatch(match.group(0)))) + "]"
            runable = re.sub('[^\(\)|& ]+', make_runable, key)
            return eval(runable).values
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
        if isinstance(analyte, str):
            analyte = [analyte]
        elif analyte is None:
            analyte = self.analytes
        
        for a in analyte:
            key = []
            for n, f in self.filter_table[a].index[self.filter_table[a]]:
                key.append(f'{n}:{f}')
            self.keydict[a] = ' & '.join(key)
    
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
            if filt in self.fnames:
                fkey = self.fuzzmatch(filt)
                if analyte is None:
                    return self.filter_components.loc[fkey].values
                else:
                    if self.filter_table.loc[fkey, analyte]:
                        return self.filter_components.loc[fkey].values
            else:
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
            ind = self.make_analyte(analyte)
        else:
            ind = ~np.zeros(self.size, dtype=bool)
        return ind
    
    def get_components(self, analyte):
        raise DeprecationWarning('This no longer works.')
    
    def get_info(self):
        """
        Get info for all filters.
        """
        out = ''
        for k in sorted(self.info.keys()):
            out += f'{k}: {self.info[k]}\n'
        return(out)
