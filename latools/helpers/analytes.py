import re
import numpy as np
from .stat_fns import nominal_values

def get_analyte_name(s):
    m = re.match('.*?([A-z]{1,3}).*?', s)
    if m:
        return m.groups()[0]
    else:
        return

def get_analyte_mass(s):
    m = re.match('.*?([0-9]{1,3}).*?', s)
    if m:
        return m.groups()[0]
    else:
        return

def analyte_2_namemass(s):
    """
    Converts analytes in format '27Al' to 'Al27'.

    Parameters
    ----------
    s : str
        of format [A-z]{1,3}[0-9]{1,3}

    Returns
    -------
    str
        Name in format [0-9]{1,3}[A-z]{1,3}
    """
    ss = s.split('_')

    out = []

    for si in ss:
        el = re.match('.*?([A-z]{1,3}).*?', si).groups()[0]
        m = re.match('.*?([0-9]{1,3}).*?', si).groups()[0]
        out.append(el + m)

    return '_'.join(out)

def analyte_2_massname(s):
    """
    Converts analytes in format 'Al27' to '27Al'.

    Parameters
    ----------
    s : str
        of format [0-9]{1,3}[A-z]{1,3}

    Returns
    -------
    str
        Name in format [A-z]{1,3}[0-9]{1,3}
    """
    ss = s.split('_')
    out = []

    for si in ss:
        el = re.match('.*?([A-z]{1,3}).*?', si).groups()[0]
        m = re.match('.*?([0-9]{1,3}).*?', si).groups()[0]
        out.append(m + el)

    return '_'.join(out)

def analyte_sort_fn(a):
    m = get_analyte_mass(a)
    if m is not None:
        return int(m)
    
    m = get_analyte_name(a)
    if m is not None:
        return m
    
    return a

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
    els = re.findall('([A-Za-z]{1,3})', s)
    ms = re.findall('([0-9]{1,3})', s)

    pretty = ['^{' + f'{m}' + '}' + f'{el}' for el, m in zip(els, ms)]
    return "$" + "/".join(pretty) + "$"

def unitpicker(a, label=None, focus_stage=None):
    """
    Determines the most appropriate plotting unit for data.

    Parameters
    ----------
    a : float or array-like
        number to optimise. If array like, the 25% quantile is optimised.
    llim : float
        minimum allowable value in scaled data.

    Returns
    -------
    (float, str)
        (multiplier, unit)
    """

    if not isinstance(a, (int, float)):
        a = nominal_values(a)
        a = np.percentile(a[~np.isnan(a)], 25)

    if a == 0:
        raise ValueError("Cannot calculate unit for zero.")

    if label is not None:
        pd = pretty_element(label)
    else:
        pd = ''

    if focus_stage == 'calibrated':
        udict = {0: 'mol/mol ' + pd,
                 3: 'mmol/mol ' + pd,
                 6: '$\mu$mol/mol ' + pd,
                 9: 'nmol/mol ' + pd,
                 12: 'pmol/mol ' + pd,
                 15: 'fmol/mol ' + pd}
    elif focus_stage == 'ratios':
        udict = {0: 'counts/count ' + pd,
                 3: '$10^{-3}$ counts/count ' + pd,
                 6: '$10^{-6}$ counts/count ' + pd,
                 9: '$10^{-9}$ counts/count ' + pd,
                 12: '$10^{-12}$ counts/count ' + pd,
                 15: '$10^{-15}$ counts/count ' + pd}
    elif focus_stage in ('rawdata', 'despiked', 'bkgsub'):
        udict = udict = {0: 'counts',
                         3: '$10^{-3}$ counts',
                         6: '$10^{-6}$ counts',
                         9: '$10^{-9}$ counts',
                         12: '$10^{-12}$ counts',
                         15: '$10^{-15}$ counts'}
    else:
        udict = {0: '', 3: '', 6: '', 9: '', 12: '', 15: ''}

    a = abs(a)
    order = np.log10(a)
    m = np.ceil(-order / 3) * 3
    if np.isnan(m):
        return 1, ''
    else:
        return float(10**m), udict[m]

def analyte_checker(self, analytes=None, check_ratios=True, single=False, focus_stage=None):
    """
    Return valid analytes depending on the analysis stage
    """
    if isinstance(analytes, str):
        analytes = [analytes]

    if focus_stage is None:
        focus_stage = self.focus_stage

    out = set()
    if focus_stage in ['ratios', 'calibrated'] and check_ratios:
        if analytes is None:
            analytes = self.analyte_ratios
        # case 1: provided analytes are an exact match for items in analyte_ratios
        valid1 = self.analyte_ratios.intersection(analytes)
        # case 2: provided analytes are in numerator of ratios
        valid2 = [a for a in self.analyte_ratios if a.split('_')[0] in analytes]
        out = valid1.union(valid2)
    else:
        if analytes is None:
            analytes = self.analytes
        out = self.analytes.intersection(analytes)

    if len(self.uncalibrated) > 0:
        if focus_stage in ['ratios', 'calibrated'] and check_ratios:
            out.difference_update(self.uncalibrated)
        else:
            out.difference_update([u.split('_')[0] for u in self.uncalibrated])
    
    if len(out) == 0:
        raise ValueError(f'{analytes} does not match any valid analyte names.')

    if single:
        if len(out) > 1:
            raise ValueError(f'{analytes} matches more than one valid analyte ({out}). Please be more specific.')
        return out.pop()

    return out

def split_analyte_ratios(ratios):
    out = set()
    if isinstance(ratios, str):
        out.update(ratios.split('_'))
    elif ratios is None:
        return out
    else:
        out.update(*map(split_analyte_ratios, ratios))
    return out