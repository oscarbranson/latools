"""
Function for reading LA-ICPMS data files.

(c) Oscar Branson : https://github.com/oscarbranson
"""
import re, os
import numpy as np
from io import BytesIO
import json
from ..helpers.helpers import Bunch

def read_data(data_file, dataformat, name_mode):
    """
    Load data_file described by a dataformat dict.

    Parameters
    ----------
    data_file : str
        Path to data file, including extension.
    dataformat : dict
        A dataformat dict, see example below.
    name_mode : str
        How to identyfy sample names. If 'file_names' uses the
        input name of the file, stripped of the extension. If
        'metadata_names' uses the 'name' attribute of the 'meta'
        sub-dictionary in dataformat. If any other str, uses this
        str as the sample name.

    Example
    -------
    >>>
    {'genfromtext_args': {'delimiter': ',',
                          'skip_header': 4},  # passed directly to np.genfromtxt
     'column_id': {'name_row': 3,  # which row contains the column names
                   'delimiter': ',',  # delimeter between column names
                   'timecolumn': 0,  # which column contains the 'time' variable
                   'pattern': '([A-z]{1,2}[0-9]{1,3})'},  # a regex pattern which captures the column names
     'meta_regex': {  # a dict of (line_no: ([descriptors], [regexs])) pairs
                    "0": (['path'], '(.*)'),
                    "2": (['date', 'method'],  # should include date
                     '([A-Z][a-z]+ [0-9]+ [0-9]{4}[ ]+[0-9:]+ [amp]+).* ([A-z0-9]+\.m)')
                   }
    }

    Returns
    -------
    sample, analytes, data, meta : tuple
    """
    with open(data_file) as f:
        lines = f.readlines()

    meta = Bunch()
    if 'meta_regex' in dataformat.keys():
        for k, v in dataformat['meta_regex'].items():
            if str(k).isdigit():
                # if k is a number, use it to extract that line in the file.
                line = lines[int(k)]
            else:
                # otherwise, find the first line that contains k
                for line in lines:
                    if k in line:
                        break
            
            if re.search(v[-1], line):
                out = re.search(v[-1], line).groups()
                for p, r in zip(v[0], out):
                    meta[p] = r

    # sample name
    if name_mode == 'file_names':
        sample = os.path.basename(os.path.splitext(data_file)[0])
    elif name_mode == 'metadata_names':
        sample = meta['name']
    else:
        sample = name_mode

    # column and analyte names
    if isinstance(dataformat['column_id']['name_row'], int):
        name_row = dataformat['column_id']['name_row']
        column_line = lines[dataformat['column_id']['name_row']].strip()
    elif isinstance(dataformat['column_id']['name_row'], str):
        for name_row, column_line in enumerate(lines):
            if dataformat['column_id']['name_row'] in column_line:
                break
    columns = np.array(column_line.split(dataformat['column_id']['delimiter']))

    if 'pattern' in dataformat['column_id'].keys():
        pr = re.compile(dataformat['column_id']['pattern'])
        analytes = [pr.match(c).groups()[0] for c in columns if pr.match(c)]

    # do any required pre-formatting
    if 'preformat_replace' in dataformat.keys():
        with open(data_file) as f:
            fbuffer = f.read()
        for k, v in dataformat['preformat_replace'].items():
            fbuffer = re.sub(k, v, fbuffer)
        # put modified data into IO object data
        fdata = BytesIO(fbuffer.encode())
    else:
        fdata = data_file
    
    # if skip_header not provided, calculate it.
    if 'skip_header' not in dataformat['genfromtext_args']:
        skip_header = name_row + 1
        while not lines[skip_header].strip():
            skip_header += 1
        dataformat['genfromtext_args']['skip_header'] = skip_header

    # read data
    read_data = np.genfromtxt(fdata, **dataformat['genfromtext_args']).T

    # data dict
    dind = np.zeros(read_data.shape[0], dtype=bool)
    for a in analytes:
        dind[columns == a] = True

    data = Bunch()
    data['Time'] = read_data[dataformat['column_id']['timecolumn']]

    # deal with time units
    if 'time_unit' in dataformat['column_id']:
        if isinstance(dataformat['column_id']['time_unit'], (float, int)):
            time_mult = dataformat['column_id']['time_unit']
        elif isinstance(dataformat['column_id']['time_unit'], str):
            unit_multipliers = {'ms': 1/1000,
                                'min': 60/1,
                                's': 1}
            try:
                time_mult = unit_multipliers[dataformat['column_id']['time_unit']]
            except:
                raise ValueError("In dataformat: time_unit must be a number, 'ms', 'min' or 's'")
        data['Time'] *= time_mult
        
    # convert raw data into counts
    # TODO: Is this correct? Should actually be per-analyte dwell?
    # if 'unit' in dataformat:
    #     if dataformat['unit'] == 'cps':
    #         tstep = data['Time'][1] - data['Time'][0]
    #         read_data[dind] *= tstep
    #     else:
    #         pass
    data['rawdata'] = Bunch(zip(analytes, read_data[dind]))
    data['total_counts'] = np.nansum(read_data[dind], 0)

    return sample, analytes, data, meta

def read_dataformat(file):
    """
    Reads a dataformat .json file and returns it as a dict.

    Parameters
    ----------
    file : str
        Path to dataformat.json file.
    """

    return json.load(open(file))