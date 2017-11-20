.. _configuration:

##########################
Lab-Specific Configuration
##########################

.. warning:: ``latools`` will not work if incorrectly configured. Follow the instructions below carefully.

Before use, ``latools`` must be configured to work with your particular set up.
When you first install latools, you should run :func:`latools.intial_configuration`

Initial lab-specific configuration can be set up using the :func:`latools.intial_configuration`, which prompts the user for the required parameters.

``latools`` supports multiple simultaneous configurations, which can be specified at the start of data analysis using the ``config`` parameter of :class:`latools.analyse`.
These configurations are stored in the 'latools.cfg' file, which can be found in the latools install location.
The :func:`config_locator` can be used to find this file on your computer.

The configuration file is formatted following the `configparser <https://wiki.python.org/moin/ConfigParserExamples>`_ syntax, with default and customisable parameters.
Further configuration additions and modifications can be made with the :func:`latools.add_config` function, or by manually editing the `latools.cfg` file (not recommended unless you know what you're doing).


Defining SRM Materials
======================

``latools`` comes with `GeoRem <http://georem.mpch-mainz.gwdg.de/>`_ M/Ca ratio values for NIST610, NIST612 and NIST614 glasses.
If you wish to use other SRMs, you must create data tables that ``latools`` can import and use.

The data table must be *.csv* format, in the following format::

    Item, Value, Uncertainty, Uncertainty Type, Unit, Source, SRM
    Ag, 0.0011446252130239381, 2.331206075780284e-05, SE, mol/molCa, GeoReM, NIST610
    ...

Note that all SRM values must be presented in elemental ratios, and that the denominator in these ratios must match the ratio denominator you choose when analysing your data.

The included SRM table can be found within the `resources` subdirectory in the ``latools`` install directory.
:func:`latools.config_locator` can be used to identify the location of this directory.

If you wish to use a different SRM table, the path to the new table must be specified in the configuration file, or on a case-by-case basis when calibrating your data.