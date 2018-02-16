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
def config_locator():
    """
    Prints and returns the location of the latools.cfg file.
    """
    loc = pkgrs.resource_filename('latools', 'latools.cfg')
    print(loc)
    return loc

def print_configs():
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

def new_config(config_name, srmfile=None, dataformat=None, base_on='DEFAULT', make_default=False):
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


def initial_configuration():
    """
    Convenience function for configuring latools.

    Run this function when you first use `latools` to specify the
    location of you SRM data file and your data format file.

    See documentation for full details.
    """
    print(('You will be asked a few questions to configure latools\n'
           'for your specific laboratory needs.'))
    lab_name = input('What would you like to call your configuration? : ')

    params = {}
    OK = False
    while ~OK:
        srmfile = input('Where is your SRM.csv file? [blank = default] : ')
        if srmfile != '':
            if os.path.exists(srmfile):
                params['srmfile'] = srmfile
                OK = True
            else:
                print(("You told us the SRM data file was at: " + srmfile +
                       "\nlatools can't find that file. Please check it "
                       "exists, and \ncheck that the path was correct. "
                       "The file path must be complete, not relative."))
        else:
            print(("No path provided. Using default GeoRem values for "
                   "NIST610, NIST612 and NIST614."))
            OK = True

        OK = False

    while ~OK:
        dataformatfile = input(('Where is your dataformat.dict file? '
                                '[blank = default] : '))
        if dataformatfile != '':
            if os.path.exists(dataformatfile):
                params['srmfile'] = dataformatfile
                OK = True
            else:
                print(("You told us the dataformat file was at: " +
                       dataformatfile + "\nlatools can't find that file. "
                       "Please check it exists, and \ncheck that the path "
                       "was correct. The file path must be complete, not "
                       "relative."))
        else:
            print(("No path provided. Using default dataformat "
                   "for the UC Davis Agilent 7700."))
            OK = True

    make_default = input(('Do you want to use these files as your '
                          'default? [Y/n] : ')).lower() != 'n'

    add_config(lab_name, params, make_default=make_default)

    print("\nConfiguration set. You're good to go!")

    return
