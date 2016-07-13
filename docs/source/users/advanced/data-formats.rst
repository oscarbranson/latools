.. _advanced_data_formats:

############
Data Import
############

File Names
==========
``latools`` makes some assumptions about how your data are named and organised, based on the 'best practice' analytical procedure used at UC Davis.
If your data do not conform to these standards, the data may not behave as expected.

Samples
-------
Each data file should contain a measurement of a single sample.
Data files can contain multiple ablations of the same sample.
The file names are used as data labels throughout analysis, so should be unique and meaningful.

.. note:: If you have measured multiple samples in a single data file, all processing steps except data filtering should behave as expected. Some of the more advanced data filtering functions (e.g. clustering algorithms) require that all the data in a single file is from the same sample.

Standards
---------
Standard measurements should be stored in their own data file.
All standards measured during analysis should have a string of characters in their file name, which is used to identify them as non-samples.
The string used to identify standards may be specified using the ``srm_identifier`` parameter of :mod:`latools.analyse` (default is 'STD').
Each standard data file may contain measurements of multiple reference materials.

.. tip:: Before calibration, you will need to identify all the standards you have measured. This is simpler if you always analyse your standards in the same order.

Data Format
===========
``latools`` uses the :func:`numpy.genfromtext` function to read and parse raw data files.
This allows the import of any raw text format, given a few basic parameters that describe the data format.

These parameters must be provided as a dict, which contains three items:

* genfromtext_args: contains a dict of arguments (parameter: value) passed directly to :func:`numpy.genfromtext`. Typical parameters might be ``delimeter``, ``skip_header``, ``skip_footer`` and ``comment``. Refer to :func:`numpy.genfromtext` documentation for a full list of possible parameters.
* column_id: contains information about which line the data column names are on, and which column contains the 'time' variable.
* regex (*optional*): a dict with line numbers as keys, containing sets of parameter names and grouped regular expressions for each line, which allows metadata to be extracted from the header during import. If this is missing from the `dataformat` dict, the header will simply be ignored.

Each time a :mod:`latools.analyse` instance is created, the data format dict is loaded from a text file that contains the dataformat dict.

Metadata and Regex
------------------


Example
-------
Data produced by the UC Davis Agilent 8800 looks like this::

    C:\Path\To\Data.D
    Intensity Vs Time,CPS
    Acquired      : Oct 29 2015  03:11:05 pm using AcqMethod OB102915.m
    Time [Sec],Mg24,Mg25,Al27,Ca43,Ca44,Mn55,Sr88,Ba137,Ba138
    0.367,666.68,25.00,3100.27,300.00,14205.75,7901.80,166.67,37.50,25.00
    [etcetera]

To read this, ``latools`` requires the following parameters::

	{'genfromtext_args': {'delimiter': ',',  # specifies that data are separated by
	                    'skip_header': 4},
	'column_id': {'name_row': 3,
	             'timekey': 'time'},
	'regex': {'0': [('path'), '(.*)'],
	         '1': None,
	         '2': [('date', 'method'),
	                '.*([A-Z][a-z]+ [0-9]+ [0-9]{4}[ ]+[0-9:]+) .*AcqMethod (.*)']}
	}







