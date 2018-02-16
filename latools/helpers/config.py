import configparser
import os
import pkg_resources as pkgrs

# functions used by latools to read configurations
def read_configuration(config='DEFAULT'):
    # create configparser object
    conf = configparser.ConfigParser()
    conf.read(pkgrs.resource_filename('latools', 'latools.cfg'))

    if config == 'DEFAULT':
        config = conf['DEFAULT']['config']
    
    return dict(conf[config])


# # load configuration parameters
# conf = configparser.ConfigParser()  # read in config file
# conf.read(pkgrs.resource_filename('latools', 'latools.cfg'))
# # load defaults into dict
# pconf = dict(conf.defaults())
# # if no config is given, check to see what the default setting is
# # if (config is None) & (pconf['config'] != 'DEFAULT'):
# #     config = pconf['config']
# # else:
# #     config = 'DEFAULT'

# # if there are any non - default parameters, replace defaults in
# # the pconf dict
# if config != 'DEFAULT':
#     for o in conf.options(config):
#         pconf[o] = conf.get(config, o)
# self.config = config


# convenience functions for configuring LAtools
def config_locator():
    """
    Prints the location of the latools.cfg file.
    """
    print(pkgrs.resource_filename('latools', 'latools.cfg'))
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
    if destination is None:
        destination = './LAtools_' + config + '_SRMTable.csv'



def add_config(config_name, params, config_file=None, make_default=True):
    """
    Adds a new configuration to latools.cfg.

    Parameters
    ----------
    config_name : str
        The name of the new configuration. This should be descriptive
        (e.g. UC Davis Foram Group)
    params : dict
        A (parameter, value) dict defining non - default parameters
        associated with the new configuration.
        Possible parameters include:

        srmfile : str
            Path to srm file used in calibration. Defaults to GeoRem
            values for NIST610, NIST612 and NIST614 provided with latools.
        dataformat : dict (as str)
            See dataformat documentation.
    config_file : str
        Path to the configuration file that will be modified. Defaults to
        latools.cfg in package install location.
    make_default : bool
        Whether or not to make the new configuration the default
        for future analyses. Default = True.

    Returns
    -------
    None
    """

    if config_file is None:
        config_file = pkgrs.resource_filename('latools', 'latools.cfg')
    cf = configparser.ConfigParser()
    cf.read(config_file)

    # if config doesn't already exist, create it.
    if config_name not in cf.sections():
        cf.add_section(config_name)
    # iterate through parameter dict and set values
    for k, v in params.items():
        cf.set(config_name, k, v)
    # make the parameter set default, if requested
    if make_default:
        cf.set('DEFAULT', 'default_config', config_name)

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
    lab_name = input('What is the name of your lab? : ')

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
