import re
import pkg_resources as pkgrs
import pandas as pd

# masses of all elements.
def elements(all_isotopes=True):
    """
    Loads a DataFrame of all elements and isotopes.

    Scraped from https://www.webelements.com/

    Returns
    -------
    pandas DataFrame with columns (element, atomic_number, isotope, atomic_weight, percent)
    """
    el = pd.read_pickle(pkgrs.resource_filename('latools', 'resources/elements.pkl'))
    if all_isotopes:
        return el.set_index('element')
    else:
        def wmean(g):
            return (g.atomic_weight * g.percent).sum() / 100
        iel = el.groupby('element').apply(wmean)
        iel.name = 'atomic_weight'
        return iel.set_index('element')

def calc_M(molecule):
    """
    Returns molecular weight of molecule.

    Where molecule is in standard chemical notation,
    e.g. 'CO2', 'HCO3' or B(OH)4

    Returns
    -------
    molecular_weight : float
    """

    # load periodic table
    els = elements()

    # define regexs
    parens = re.compile('\(([A-z0-9]+)\)([0-9]+)?')
    stoich = re.compile('([A-Z][a-z]?)([0-9]+)?')

    ps = parens.findall(molecule)  # find subgroups in parentheses
    rem = parens.sub('', molecule)  # get remainder

    m = 0
    # deal with sub-groups
    if len(ps) > 0:
        for sub, ns in ps:
            ms = 0
            for e, n in stoich.findall(sub):
                me = (els.loc[e, 'atomic_weight'] *
                      els.loc[e, 'percent'] / 100).sum()
                if n == '':
                    n = 1
                else:
                    n = int(n)
                ms += me * n
            if ns == '':
                ns = 1
            else:
                ns = int(ns)
            m += ms * ns
    # deal with remainder
    for e, n in stoich.findall(rem):
        me = (els.loc[e, 'atomic_weight'] *
              els.loc[e, 'percent'] / 100).sum()
        if n == '':
            n = 1
        else:
            n = int(n)
        m += me * n
    return m
