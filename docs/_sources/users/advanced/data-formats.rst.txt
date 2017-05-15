.. _advanced_data_formats:

############
Data Import
############

``latools`` can be set up to work with pretty much any conceivable text-based data format.
To get your data into ``latools``, you need to think about two things:
  1. File structure
  2. Data format

File Structure
==========
``latools`` is designed for data that is collected so that each text file contains ablations of a single sample or a (set of) standards, with a name corresponding to the identity of the sample.
The ideal data structure would look something like this:

.. code-block:: bash

	data/
	  STD-1.csv
	  Sample-1.csv
	  Sample-2.csv
	  Sample-3.csv
	  STD-2.csv

Where each of the .csv files within the 'data/' contains one or more ablations of a single sample, or numerous standards (i.e. STD-1 could contain ablations of three different standards).
The names of the .csv files are used to label the data throughout analysis, so should be unique, and meaningful.
Standards are recognised by :mod:`latools.analyse` by the presence of identifying characters that are presend in all standard names, in this case 'STD'.

When importing the data, you give :mod:`latools.analyse` the ``data/`` folder, and some information about the SRM identififier (``srm_identifier='STD'``) and the file extension (``extension='.csv'``), and it imports all data files in the folder.

.. tip:: Some labs save an entire analytical session in a single data file. To use ``latools``, this data will need to be broken up into the format described above. In the near future, there will be a function to do this, where long single files can be broken into individual samples, given the analytical sequence.

Data Format
===========
We tried to make the data import mechanism as simple as possible, but because of the diversity and complexity of formats from different instruments, it can still be a bit tricky to understand. The following will hopefully give you everything you need to write your data format description.

``latools`` uses numpy's incredibly flexible :func:`numpy.genfromtext` function to import data.
This means that it should be able to deal with pretty much any delimited text data format that you can throw at it.

To import your data, you need to give ``latools`` a description of your data format, which enables it to read the metadata and data from the text files.
this takes the form of a python dictionary, with a number of parameters corresponding to different stages of data import.

.. tip:: Data import in ``latools`` makes heavy use of `Regular Expressions <https://en.wikipedia.org/wiki/Regular_expression>`_ to identify different parts of your data. These can be complex, and take a while to get your head around. To help you with this, we suggest using the superb `Regex101 <https://regex101.com/r/HKNavd/1>`_ site to help you design your data format.

Writing a Data Format Description
---------------------------------

Data produced by the UC Davis Agilent 8800 looks like this:

.. code-block:: python
	:linenos:

	C:\Path\To\Data.D
	Intensity Vs Time,CPS
	Acquired      : Oct 29 2015  03:11:05 pm using AcqMethod OB102915.m
	Time [Sec],Mg24,Mg25,Al27,Ca43,Ca44,Mn55,Sr88,Ba137,Ba138
	0.367,666.68,25.00,3100.27,300.00,14205.75,7901.80,166.67,37.50,25.00
	[etcetera]

The minimal `dataformat.dict` required to read this data looks like this:

.. code-block:: python
	:linenos:

	{'genfromtext_args': {'delimiter': ',',
	                      'skip_header': 4},
	 'column_id': {'name_row': 3,
	               'delimeter': ',',
	               'timecolumn': 0,
	               'pattern': '([A-z]{1,2}[0-9]{1,3})'},
	 'meta_regex': {0: (['path'], '(.*)'),
	                2: (['date', 'method'],
	                    '([A-Z][a-z]+ [0-9]+ [0-9]{4}[ ]+[0-9:]+ [amp]+).* ([A-z0-9]+\.m)')}
	}

The dataformat dict has three items:
  - ``genfromtext_args``: A dictionary of parameters passed directly to numpy's ``genfromtext`` function, which does all the work of actually importing your data table. The key parameters here will be ``skip_header``, ``delimeter`` and possibly ``skip_footer`` and ``comments``. These specify how many lines of the file to skip at the start (header) and end (footer) of the data, what the delimeting character is between the data values (``','`` for a csv), and whether there's a special character that denotes a 'comment' in your data, which should be skipped.
  - ``column_id``: A dictionary containing a set of parameters that identify which column of the data is the 'time' variable (``timecolumn``), which row contains the column names (``name_row``), the delimeter between column names (``delimeter``) and a regex pattern that `identifies valid analyte names in a capture group <https://regex101.com/r/gfc09X/2>`_.
  - ``meta_regex``: A dictionary containing information describing aspects of the file metadata that you want to import. The only `essential` item to import here is the ``date`` of the analysis, which is used by ``latools`` for background and drift correction. Everything else is just to preserve information about the data through analysis. The keys of this dictionary are line numbers, with associated ``(labels, regex)`` tuples, where ``labels`` is a list the same length as the number of match groups in the regex. If you're struggling with this, take a look at the Regex101 breakdowns of these two entries `here <https://regex101.com/r/WYcLfZ/1>`_ and `here <https://regex101.com/r/HN1OC9/2>`_. The resulting matches are stored in a dictionary called ``meta`` within the :mod:`latools.analyse` object.

.. warning:: The ``meta_regex`` component of the dataformat description MUST contain an entry that finds the 'date' of the analysis. Background and drift correction depend upon having this information. That is, it must have an entry like ``{N: {['date'], 'regex_string'}}``, where ``N`` is a line number, and ``regex_string`` isolates the analysis date of the file, as demonstrated `here <https://regex101.com/r/jfPV3Z/1>`_.

Additionally, for particularly awkward data formats, you can also include a fourth entry called ``preformat_replace``. This is a dictionary of ``{pattern, replacement}`` regex pairs which are applied to your data before any other import function 'sees' your data. For example, an entry of ``{[\t]{2,}: ','}`` would replace all instances of two tab characters in your data file with a comma.

I've written my dataformat, now what?
-------------------------------------

Once you're happy with your data format description, put it in a text file, and save it as 'my_dataformat.dict' (obviously replace my_dataformat with something meaningful...).
When you want to import data using your newly defined format, you can point ``latools`` towards it by specifying ``dataformat='my_dataformat.dict'`` when starting a data analysis.
Alternatively, you can define a new :ref:`configuration`, to make this the default data format for your setup.

.. note:: If you're stuck on data formats, `submit a question to the mailing list <https://groups.google.com/forum/#!forum/latools>`_ and we'll try to help. If you think you've found a serious problem in the software that will prevent you importing your data, `file an issue on the GitHub project page <https://github.com/oscarbranson/latools/issues/new>`_, and we'll look into updating the software to fix the problem.