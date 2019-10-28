"""
Functions for dealing with chemical formulae, and converting between molar 
ratios and mass fractions.

(c) Oscar Branson : https://github.com/oscarbranson
"""

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
        return iel

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

def decompose_molecule(molecule, n=1):
    """
    Returns the chemical constituents of the molecule, and their number.

    Parameters
    ----------
    molecule : str
        A molecule in standard chemical notation, 
        e.g. 'CO2', 'HCO3' or 'B(OH)4'.
    
    Returns
    -------
    All elements in molecule with their associated counts : dict
    """
    if isinstance(n, str):
        n = int(n)
    
    # define regexs
    parens = re.compile('\(([A-z0-9()]+)\)([0-9]+)?')
    stoich = re.compile('([A-Z][a-z]?)([0-9]+)?')

    ps = parens.findall(molecule)  # find subgroups in parentheses
    rem = parens.sub('', molecule)  # get remainder
    
    if len(ps) > 0:
        for s, ns in ps:
            comp = decompose_molecule(s, ns)
        for k, v in comp.items():
            comp[k] = v * n
    else:
        comp = {}
        
    for e, ns in stoich.findall(rem):
        if e not in comp:
            comp[e] = 0
        if ns == '':
            ns = 1 * n
        else:
            ns = int(ns) * n
        comp[e] += ns

    return comp

def analyte_mass(analyte, in_name=True):
    """
    Returns the mass of a given analyte.

    If the name contains a number (e.g. Ca43), that number is returned. If the name contains
    no number but an element name (e.g. Ca), the average mass of that element is returned. 
    
    Parameters
    ----------
    analyte : str or array-like
        The name or names of the analytes to be considered.
    in_name : bool
        If True, numbers in the analyte name are preferred.
    """
    if isinstance(analyte, str):
        nums = re.findall('[0-9]+', analyte)
        if in_name and nums:
            return float(nums[0])
        else:
            return calc_M(re.findall('[A-z]+', analyte)[0])
    else:
        masses = {}
        for i, a in enumerate(analyte):
            masses[a] = analyte_mass(a, in_name)
        return masses


# functions for converting between mass fraction and molar ratio
def to_molar_ratio(massfrac_numerator, massfrac_denominator, numerator_mass, denominator_mass):
    """
    Converts per-mass concentrations to molar elemental ratios.
    
    Be careful with units.
    
    Parameters
    ----------
    numerator_mass, denominator_mass : float or array-like
        The atomic mass of the numerator and denominator.
    massfrac_numerator, massfrac_denominator : float or array-like
        The per-mass fraction of the numnerator and denominator.
    
    Returns
    -------
    float or array-like : The molar ratio of elements in the material
    """
    return (massfrac_numerator / numerator_mass) / (massfrac_denominator / denominator_mass)

def to_mass_fraction(molar_ratio, massfrac_denominator, numerator_mass, denominator_mass):
    """
    Converts per-mass concentrations to molar elemental ratios.
    
    Be careful with units.
    
    Parameters
    ----------
    molar_ratio : float or array-like
        The molar ratio of elements.
    massfrac_denominator : float or array-like
        The mass fraction of the denominator element
    numerator_mass, denominator_mass : float or array-like
        The atomic mass of the numerator and denominator.
        
    Returns
    -------
    float or array-like : The mass fraction of the numerator element.
    """
    return molar_ratio * massfrac_denominator * numerator_mass / denominator_mass
