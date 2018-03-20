import numpy as np
import pandas as pd




def read_latools_data(f):
    md = pd.read_csv(f, index_col=[0,1,2])
    # extract mean only
    md_m = md.loc['mean', :]
    # name index columns
    md_m.index.names = ['sample', 'rep']
    # convert all to mmol/mol
    md_m *= 1e3
    
    return md_m

def load_reference_data(name=None):
    """
    Fetch LAtools reference data from online repository.

    Parameters
    ----------
    name : str<
        Which data to download. Can be one of 'culture_reference',
        'culture_test', 'downcore_reference', 'downcore_test', 'iolite_reference'
        or 'zircon_reference'.
        If None, all are downloaded and returned as a dict.

    Returns
    -------
    pandas.DataFrame or dict.
    """
    base_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQJfCeuqrtFFMAeSpA9rguzLAo9OVuw50AHhAULuqjMJzbd3h46PK1KjF69YiJAeNAAjjMDkJK7wMpG/pub?gid={:}&single=true&output=csv'
    gids = {'culture_reference': '0',
            'culture_test': '1170065442',
            'downcore_reference': '190752797',
            'downcore_test': '721359794',
            'iolite_reference': '483581945',
            'zircon_reference': '1355554964'}

    if name is None:
        out = {}
        for nm, gid in gids.items():
            url = base_url.format(gid)
            tmp = pd.read_csv(url, header=[0], index_col=[0, 1])
            tmp.index.names = ['sample', 'rep']
            tmp.columns.names = ['analyte']
            tmp.sort_index(1, inplace=True)
            out[nm] = tmp
    else:
        gid = gids[name]
        url = base_url.format(gid)
        out = pd.read_csv(url, index_col=[0, 1])
        out.columns.names = ['analyte']
        out.sort_index(1, inplace=True)
    return out
