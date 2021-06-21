"""
Tools for handling file reading/writing.

(c) Oscar Branson : https://github.com/oscarbranson
"""
import os
import json
import datetime as dt
import dateutil as du
import pkg_resources as pkgrs
from .config import read_configuration


def read_dataformat(dataformat, silent=True):
    """
    Takes a dataformat dict, filename or configuration name, and return a valid dataformat.

    Parameters
    ----------
    dataformat : str or dict
        Either a dataformat dict, a dataformat.json file, or the name of an latools configuration.
    silent : bool
        If True, some output is printed about what the function has done.

    Returns
    -------
    dict : dataformat dict
    """
    if isinstance(dataformat, dict):
        if not silent: 
            print('dataformat dict provided - no read necessary.')
        return dataformat
    elif isinstance(dataformat, str):
        if os.path.exists(dataformat):
            if not silent: 
                print('Reading dataformat.json file...')
            dataformat = json.load(open(dataformat))
        else:
            if not silent: 
                print('Getting dataformat from {} configuration...'.format(dataformat))
            config = read_configuration(dataformat)
            df_file = pkgrs.resource_filename('latools', config['dataformat'])
            if not silent: 
                print('Reading dataformat.json file...')
            dataformat = json.load(open(df_file))
        return dataformat
    else:
        raise TypeError("Incorrect 'dataformat' type - must be a dict or a str.")

def get_date(datetime, time_format=None):
    """
    Return a datetime oject from a string, with optional time format.

    Parameters
    ----------
    datetime : str
        Date-time as string in any sensible format.
    time_format : datetime str (optional)
        String describing the datetime format. If missing uses
        dateutil.parser to guess time format.
    """
    if isinstance(datetime, dt.datetime):
        t = datetime
    elif time_format is not None:
        t = dt.datetime.strptime(datetime, time_format)
    else:
        t = du.parser.parse(datetime)        
    return t