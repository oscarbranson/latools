"""
Functions to help configure LAtools.

(c) Oscar Branson : https://github.com/oscarbranson
"""

import configparser
import json
import os
import re
import numpy as np
import textwrap as tw
import pkg_resources as pkgrs
from .helpers import Bunch

from io import BytesIO
from shutil import copyfile

line_width = 80

# functions used by latools to read configurations
def read_configuration(config='DEFAULT'):
    """
    Read LAtools configuration file, and return parameters as dict.
    """
    # read configuration file
    _, conf = read_latoolscfg()
    # if 'DEFAULT', check which is the default configuration
    if config == 'DEFAULT':
        config = conf['DEFAULT']['config']

    try:
        # grab the chosen configuration
        conf = dict(conf[config])
        # update config name with chosen
        conf['config'] = config
        return conf
    except KeyError:
        msg = "\nError: '{}' is not a valid configuration.\n".format(config)
        msg += 'Please use one of:\n  DEFAULT\n  ' + '\n  '.join(conf.sections())
        print(msg + '\n')
        raise KeyError('Invalid configuration: {}'.format(config))

def config_locator():
    """
    Returns the path to the file containing your LAtools configurations.
    """
    return pkgrs.resource_filename('latools', 'latools.cfg')

# under-the-hood functions
def read_latoolscfg():
    """
    Reads configuration, returns a ConfigParser object.

    Distinct from read_configuration, which returns a dict.
    """
    config_file = pkgrs.resource_filename('latools', 'latools.cfg')
    cf = configparser.ConfigParser()
    cf.read(config_file)
    return config_file, cf

# convenience functions for configuring LAtools
def locate():
    """
    Prints and returns the location of the latools.cfg file.
    """
    loc = pkgrs.resource_filename('latools', 'latools.cfg')
    print(loc)
    return loc

def print_all():
    """
    Prints all currently defined configurations.
    """
    # read configuration file
    _, conf = read_latoolscfg()

    default = conf['DEFAULT']['config']

    pstr = '\nCurrently defined LAtools configurations:\n\n'
    for s in conf.sections():
        if s == default:
            pstr += s + ' [DEFAULT]\n'
        elif s == 'REPRODUCE':
            pstr += s + ' [DO NOT ALTER]\n'
        else:
            pstr += s + '\n'

        for k, v in conf[s].items():
            if k != 'config':
                if v[:9] == 'resources':
                    v = pkgrs.resource_filename('latools', v)
                pstr += '   ' + k + ': ' + v + '\n'
        pstr += '\n'

    print(pstr)
    return

def copy_SRM_file(destination=None, config='DEFAULT'):
    """
    Creates a copy of the default SRM table at the specified location.

    Parameters
    ----------
    destination : str
        The save location for the SRM file. If no location specified, 
        saves it as 'LAtools_[config]_SRMTable.csv' in the current working 
        directory.
    config : str
        It's possible to set up different configurations with different
        SRM files. This specifies the name of the configuration that you 
        want to copy the SRM file from. If not specified, the 'DEFAULT'
        configuration is used.
    """
    # find SRM file from configuration    
    conf = read_configuration()

    src = pkgrs.resource_filename('latools', conf['srmfile'])

    # work out destination path (if not given)
    if destination is None:
        destination = './LAtools_' + conf['config'] + '_SRMTable.csv'
    
    if os.path.isdir(destination):
        destination += 'LAtools_' + conf['config'] + '_SRMTable.csv'

    copyfile(src, destination)

    print(src + ' \n    copied to:\n      ' + destination)
    return

def create(config_name, srmfile=None, dataformat=None, base_on='DEFAULT', make_default=False):
    """
    Adds a new configuration to latools.cfg.

    Parameters
    ----------
    config_name : str
        The name of the new configuration. This should be descriptive
        (e.g. UC Davis Foram Group)
    srmfile : str (optional)
        The location of the srm file used for calibration.
    dataformat : str (optional)
        The location of the dataformat definition to use.
    base_on : str
        The name of the existing configuration to base the new one on.
        If either srm_file or dataformat are not specified, the new
        config will copy this information from the base_on config.
    make_default : bool
        Whether or not to make the new configuration the default
        for future analyses. Default = False.

    Returns
    -------
    None
    """

    base_config = read_configuration(base_on)

    # read config file
    config_file, cf = read_latoolscfg()
    
    # if config doesn't already exist, create it.
    if config_name not in cf.sections():
        cf.add_section(config_name)
    # set parameter values
    if dataformat is None:
        dataformat = base_config['dataformat']
    cf.set(config_name, 'dataformat', dataformat)

    if srmfile is None:
        srmfile = base_config['srmfile']
    cf.set(config_name, 'srmfile', srmfile)

    # make the parameter set default, if requested
    if make_default:
        cf.set('DEFAULT', 'config', config_name)

    with open(config_file, 'w') as f:
        cf.write(f)

    return

def update(config, parameter, new_value):
    # read config file
    config_file, cf = read_latoolscfg()

    if config == 'REPRODUCE':
        print("Nope. This will break LAtools. Don't do it.")

    pstr = 'Are you sure you want to change the {:s} parameter of the {:s} configuration?\n  It will be changed from:\n    {:s}\n  to:\n    {:s}\n> [N/y]: '

    response = input(pstr.format(parameter, config, cf[config][parameter], new_value))

    if response.lower() == 'y':
        cf.set(config, parameter, new_value)
        with open(config_file, 'w') as f:
            cf.write(f)
        print('  Configuration updated!')
    else:
        print('  Done nothing.')

    return

def delete(config):
    # read config file
    config_file, cf = read_latoolscfg()

    if config == cf['DEFAULT']['config']:
        print("Nope. You're not allowed to delete the default configuration.\n" + 
              "Please change the default configuration, and then try again.")

    if config == 'REPRODUCE':
        print("Nope. This will break LAtools. Don't do it.")

    pstr = 'Are you sure you want to delete the {:s} configuration?\n> [N/y]: '

    response = input(pstr.format(config))

    if response.lower() == 'y':
        cf.remove_section(config)
        with open(config_file, 'w') as f:
            cf.write(f)
        print('  Configuration deleted!')
    else:
        print('  Done nothing.')
    
    return

def change_default(config):
    """
    Change the default configuration.
    """
    config_file, cf = read_latoolscfg()

    if config not in cf.sections():
        raise ValueError("\n'{:s}' is not a defined configuration.".format(config))

    if config == 'REPRODUCE':
        pstr = ('Are you SURE you want to set REPRODUCE as your default configuration?\n' + 
                '     ... this is an odd thing to be doing.')
    else:
        pstr = ('Are you sure you want to change the default configuration from {:s}'.format(cf['DEFAULT']['config']) + 
                'to {:s}?'.format(config))

    response = input(pstr + '\n> [N/y]: ')

    if response.lower() == 'y':
        cf.set('DEFAULT', 'config', config)
        with open(config_file, 'w') as f:
            cf.write(f)
        print('  Default changed!')
    else:
        print('  Done nothing.')

def get_dataformat_template(destination='./LAtools_dataformat_template.json'):
    """
    Copies a data format description JSON template to the specified location.
    """

    template_file = pkgrs.resource_filename('latools', 'resources/data_formats/dataformat_template.json')

    copyfile(template_file, destination)

    return

# tools for developing a valid dataformat description

def test_dataformat(data_file, dataformat_file, name_mode='file_names'):
    """
    Test a data formatfile against a particular data file.

    This goes through all the steps of data import printing out
    the results of each step, so you can see where the import fails.

    Parameters
    ----------
    data_file : str
        Path to data file, including extension.
    dataformat : dict or str
        A dataformat dict, or path to file. See example below.
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
                    0: (['path'], '(.*)'),
                    2: (['date', 'method'],  # MUST include date
                     '([A-Z][a-z]+ [0-9]+ [0-9]{4}[ ]+[0-9:]+ [amp]+).* ([A-z0-9]+\.m)')
                   }
    }

    Returns
    -------
    sample, analytes, data, meta : tuple
    """
    if isinstance(dataformat_file, dict):
        dataformat = dataformat_file
        dataformat_file = '[dict provided]'

    print('*************************************************\n' + 
          'Testing suitability of data format description...\n' + 
          '  Dataformat File: {:s}'.format(dataformat_file) + '\n' +  
          '  Data File: {:s}'.format(data_file) + '\n' +
          '*************************************************')
    
    print('\n  Test: open data file...')
    with open(data_file) as f:
        lines = f.readlines()    
    print('    Success!')

    if dataformat_file != '[dict provided]':
        print('\n  Test: read dataformat file...')
        # if dataformat is not a dict, load the json
        try:
            with open(dataformat_file) as f:
                dataformat = json.load(f)
            print('    Success!')
        except:
            print("        ***PROBLEM: The dataformat file isn't in a valid .json format")
            raise
    
    print("\n  Test: read metadata using 'metadata_regex'...")
    meta = Bunch()
    if 'meta_regex' in dataformat.keys():
        got = []
        for k, v in dataformat['meta_regex'].items():
            rep = '    Line "{:s}":'.format(k)
            try:
                if k.isdigit():
                    line = lines[int(k)]
                else:
                    pattern = k.split('__')[-1]
                    for line in lines:
                        if pattern in line:
                            break
                
                out = re.search(v[-1], line).groups()

                print(rep)
                for i in np.arange(len(v[0])):
                    meta[v[0][i]] = out[i]
                    print('      ' + v[0][i] + ': ' + out[i])
                    got.append(v[0][i])
            except:
                print(rep + '\n    ***PROBLEM in meta_regex:\n' + 
                      '        [' + ', '.join(v[0]) + '] cannot be derived from "' + v[1] + '".\n' + 
                      '        Test regex against: "{:s}"'.format(line.strip()) + '\n' + 
                      '        at https://regex101.com/.')
                raise
        print('    Finished - does the above look correct?')
        if 'date' not in got:
            print('        ***PROBLEM: ' + 
                  'meta_regex must identify data collection start time as "date". LAtools may not behave as expected.')
            # raise ValueError('meta_regex must identify "date" attribute, containing data collection start time')
    else:
        print('        ***PROBLEM: ' + 
              'meta_regex not specified. At minimum, must identify data collection start time as "date". LAtools may not behave as expected.')
        # raise ValueError('meta_regex must identify "date" attribute, containing data collection start time')
    
    # sample name
    print('\n  Test: Sample Name IDs...')
    if name_mode == 'file_names':
        sample = os.path.basename(data_file).split('.')[0]
        print('    Sample name grabbed from file: ' + sample)
    else:
        try:
            sample = meta['name']
            print('   Sample name from metadata_regex: ' + sample)
        except KeyError:
            print('       ***PROBLEM: Sample name not identified by metadata_regex - please include "name".')
            raise
    print('        ***Is the sample name correct?***')
    
    # column and analyte names
    print('\n  Test: Getting Column Names...')
    if isinstance(dataformat['column_id']['name_row'], int):
        print('    Getting names from line {:.0f}: '.format(dataformat['column_id']['name_row']))
        name_row = dataformat['column_id']['name_row']
        column_line = lines[dataformat['column_id']['name_row']]
    elif isinstance(dataformat['column_id']['name_row'], str):
        print('    Column row not provided, using first line containing "{}": '.format(dataformat['column_id']['name_row']))
        for name_row, column_line in enumerate(lines):
            if dataformat['column_id']['name_row'] in column_line:
                break
    columns = np.array(column_line.strip().split(dataformat['column_id']['delimiter']))
    # return tw.wrap(', '.join(columns), line_width)
    print('    Columns found:\n' + '\n'.join(tw.wrap(', '.join(columns), line_width, subsequent_indent='      ', initial_indent='      ')))
    
    if 'pattern' in dataformat['column_id'].keys():
        print('    Cleaning up using column_id/pattern...')
        pr = re.compile(dataformat['column_id']['pattern'])
        analytes = [pr.match(c).groups()[0] for c in columns if pr.match(c)]
        if len(analytes) == 0:
            raise ValueError('no analyte names identified. Check pattern in column_id section.')
        print('    Cleaned Analyte Names: \n' + '\n'.join(tw.wrap(', '.join(analytes), line_width, subsequent_indent='      ', initial_indent='      ')))
        print('    ***This should only contain analyte names... does it?***')
    
    # do any required pre-formatting
    if 'preformat_replace' in dataformat.keys():
        print('\n  Test: preformat_replace...')
        with open(data_file) as f:
            fbuffer = f.read()
        for k, v in dataformat['preformat_replace'].items():
            print('    replacing "' + k + '" with "' + v + '"')
            fbuffer = re.sub(k, v, fbuffer)
            fdata = BytesIO(fbuffer.encode())
        else:
            fdata = data_file
        print('    Done.')

    print('\n  Test: Reading Data...')
    # if skip_header not provided, calculate it.
    if 'skip_header' not in dataformat['genfromtext_args']:
        print('    "skip_header" not specified... finding header size.')
        skip_header = name_row + 1
        while not lines[skip_header].strip():
            skip_header += 1
        dataformat['genfromtext_args']['skip_header'] = skip_header
        print('      -> Header is {} lines long'.format(skip_header))
    
    # read data
    try:
        read_data = np.genfromtxt(data_file,
                                **dataformat['genfromtext_args']).T
        print('      Success!')
    except:
        print('        ***PROBLEM during data read - check genfromtext_args.\n' + 
                '        If they look correct, think about including preformat_replace terms?')
        raise
    
    # print('    checking number of columns...')
    # if read_data.shape[0] != len(analytes) + 1:
    #     print('***\nPROBLEM:\n' + 
    #           'There are {:.0f} data columns, but {:.0f} column names.\n'.format(read_data.shape[0], len(analytes) + 1) +
    #           '   This shouldn't be a problem
    #           '   Check your identification of column names, or your pre-formatting parameters.\n***')
    #     # raise ValueError('Data - Column Name mismatch')
    # else:
    #     print('    Success!')
    
    # data dict
    print('\n  Test: Combine data into dictionary...')    
    dind = np.ones(read_data.shape[0], dtype=bool)
    dind[dataformat['column_id']['timecolumn']] = False

    data = Bunch()
    print('    Time in column {:.0f} ({:s})'.format(dataformat['column_id']['timecolumn'],
                                                  columns[dataformat['column_id']['timecolumn']]))
    data['Time'] = read_data[dataformat['column_id']['timecolumn']]
    
    for a, d in zip(analytes, read_data[dind]):
        data[a] = d

    print('    Calculating total counts...')
    data['total_counts'] = read_data[dind].sum(0)
    print('    Success!')
    
    print('\nTests completed successfully.\n' +
          '  **This does not necessarily mean everything has worked!**\n' +
          '    Look carefully through the output of this function, and\n' +
          '    make sure it looks right.\n\n' +
          'Outputs are: sample_name, analytes, data_dict, metadata_dict')

    return sample, analytes, data, meta