import configparser
import os
import pkg_resources as pkgrs

from shutil import copyfile

# functions used by latools to read configurations
def read_configuration(config='DEFAULT'):
    """
    Read LAtools configuration file, and return parameters as dict.
    """
    # read configuration file
    conf = configparser.ConfigParser()
    conf.read(pkgrs.resource_filename('latools', 'latools.cfg'))
    # if 'DEFAULT', check which is the default configuration
    if config == 'DEFAULT':
        config = conf['DEFAULT']['config']

    # return the chosen configuration    
    return dict(conf[config])

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
    conf = configparser.ConfigParser()
    conf.read(pkgrs.resource_filename('latools', 'latools.cfg'))

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
    conf = read_configuration(config)
    src = pkgrs.resource_filename('latools', conf['srmfile'])

    # work out destination path (if not given)
    if destination is None:
        destination = './LAtools_' + conf['config'] + '_SRMTable.csv'

    copyfile(src, destination)

    print(src + ' copied to ' + destination)
    return

def new(config_name, srmfile=None, dataformat=None, base_on='DEFAULT', make_default=False):
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
    config_file = pkgrs.resource_filename('latools', 'latools.cfg')
    cf = configparser.ConfigParser()
    cf.read(config_file)

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
    config_file = pkgrs.resource_filename('latools', 'latools.cfg')
    cf = configparser.ConfigParser()
    cf.read(config_file)

    pstr = 'Are you sure you want to change the {:s} parameter of the {:s} configuration?\n  It will be changed from:\n    {:s}\n  to:\n    {:s}\n> [N/y]: '

    response = input(pstr.format(parameter, config, cf[config][parameter], new_value))

    if response.lower() == 'y':
        cf.set(config, parameter, new_value)
        with open(config_file, 'w') as f:
            cf.write(f)
        print('Configuration updated!')
    else:
        print('Done nothing.')

    return

def delete(config):
    # read config file
    config_file = pkgrs.resource_filename('latools', 'latools.cfg')
    cf = configparser.ConfigParser()
    cf.read(config_file)

    pstr = 'Are you sure you want to delete the {:s} configuration?\n> [N/y]: '

    response = input(pstr.format(config))

    if response.lower() == 'y':
        cf.remove_section(config)
        with open(config_file, 'w') as f:
            cf.write(f)
        print('Configuration deleted!')
    else:
        print('Done nothing.')
    
    return