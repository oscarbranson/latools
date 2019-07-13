import re

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
    el = re.match('.*?([A-z]{1,3}).*?', s).groups()[0]
    m = re.match('.*?([0-9]{1,3}).*?', s).groups()[0]

    return el + m

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
    el = re.match('.*?([A-z]{1,3}).*?', s).groups()[0]
    m = re.match('.*?([0-9]{1,3}).*?', s).groups()[0]

    return m + el

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
    el = re.match('.*?([A-z]{1,3}).*?', s).groups()[0]
    m = re.match('.*?([0-9]{1,3}).*?', s).groups()[0]

    return '$^{' + m + '}$' + el