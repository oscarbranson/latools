import numpy as np
import pandas as pd

def read_table(srm_file):
    """
    Reads SRM compositional data from file.

    For file format information, see:
    http://latools.readthedocs.io/en/latest/users/configuration/srm-file.html

    Parameters
    ----------
    file : str
        Path to SRM file.

    Returns
    -------
    SRM compositions : pandas.DataFrame
    """
    return pd.read_csv(srm_file).set_index('SRM').dropna(how='all')

def get_defined_srms(srm_file):
    """
    Returns list of SRMS defined in the SRM database
    """
    srms = read_table(srm_file)
    return np.asanyarray(srms.index.unique())
