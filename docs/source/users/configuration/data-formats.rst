.. _data_formats:

############
Data Formats
############

``latools`` can be set up to work with pretty much any conceivable text-based data format.
To get your data into ``latools``, you need to think about two things:

1. File Structure
=================
At present, ``latools`` is designed for data that is collected so that each text file contains ablations of a single sample or (a set of) standards, with a name corresponding to the identity of the sample.
An ideal data structure would look something like this:

.. code-block :: bash

    data/
      STD-1.csv
      Sample-1.csv
      Sample-2.csv
      Sample-3.csv
      STD-2.csv

Where each of the .csv files within the 'data/' contains one or more ablations of a single sample, or numerous standards (i.e. STD-1 could contain ablations of three different standards).
The names of the .csv files are used to label the data throughout analysis, so should be unique, and meaningful.
Standards are recognised by :meth:`~latools.latools.analyse` by the presence of identifying characters that are present in all standard names, in this case ``'STD'``.

When importing the data, you give :meth:`~latools.latools.analyse` the ``data/`` folder, and some information about the SRM identifier (``srm_identifier='STD'``) and the file extension (``extension='.csv'``), and it imports all data files in the folder.

.. important:: If you data are not in this format (e.g. all your data are stored in one long file), you'll need to convert them into this format to use ``latools``. You can find Information on how to do this in the :ref:`preprocessing` pages.

.. _data_format_description:
2. Data Format
==============
We tried to make the data import mechanism as simple as possible, but because of the diversity and complexity of formats from different instruments, it can still be a bit tricky to understand. The following will hopefully give you everything you need to write your data format description.

Data Format Description : General Principles
--------------------------------------------

The data format description is stored in a plain-text file, in the `JSON <https://en.wikipedia.org/wiki/JSON>`_ format.
In practice, the format description consists of a number of names entries with corresponding values, which are read and interpreted by ``latools``.
A generic JSON file might look something like this:

.. code-block:: JSON

    {
        'entry_1': 'value',
        'entry_2': ['this', 'is', 'a', 'list'],
        'entry_3': (['a', 'set', 'of'], 'three', 'values')
    }

.. tip:: There are a number of characters that are special in the JSON format (e.g. ``/ \ "``). If you want to include these characters in the file, you have to 'escape' them (i.e. mark them as special) by preceding them with a ``\``. If this sounds too confusing, you can just use an `online formatter <https://www.freeformatter.com/json-escape.html>`_ to make sure all your entries are JSON-safe.

Required Sections
^^^^^^^^^^^^^^^^^
  - ``meta_regex`` contains information on how to read the 'metadata' in the file header. Each entry has the form:

    .. code-block:: JSON

        {
            "meta_regex": {
                "line": [["metadata_name"], "Regular Expression with a capture group."],
            }
        }

    Don't worry at this point if 'Regular Expression' and 'capture group' mean nothing to you. :ref:`We'll get to that later <regex>`.

    Replace ``line`` with an identifier that selects the line in the data file that the regex is applied to. There are two ways to do this.

    **What should** ``"line"`` **be?**:
        - A number in quotations to pick out a line in the file, e.g. ``"3"`` to extract the fourth line of the file (remember here that python starts counting at zero). This works well if the file header is *always* the same.
        - A word or string of characters that is *always* in the line (i.e. won't change from file to file). For example you could use ``"Date:"``, and ``latools`` will find the first line in the file that contains ``Date:`` and apply your regular expression to it. This is useful for formats where the header size can vary depending on the analysis.

    .. tip:: The ``meta_regex`` component of the dataformat description should contain an entry that finds the 'date' of the analysis. This is used to define the time scale of the whole session which background and drift correction depend upon. This should be specified as``{"line": {["date"], "regex_string"}}`` where ``regex_string`` isolates the analysis date of the file in a capture group, as demonstrated `here <https://regex101.com/r/jfPV3Z/1>`_. If you don't identify a date in the metadata, ``latools`` will assume all your analyses were done consecutively with no time gaps between them, and in the order of their sample names. This can cause some unexpected behaviour in the analysis...

  - ``column_id`` contains information on where the column names of the data are, and how to interpret them. This requires 4 specific entries, and should look something like:

  .. code-block:: JSON

    {
        "column_id": {
            "delimiter": "Character that separates column headings, e.g. \t (tab) or , (comma)",
            "timecolumn": "Numeric index of time column. Usually zero (the first column). Must be an integer, without quotations.",
            "name_row": "The line number that contains the column headings. Must be an integer, without quotations",
            "pattern": "A Regular Expression that identifies valid analyte names in a capture group."
        }
    }
  - ``genfromtext_args`` contains information on how to read the actual data table. ``latools`` uses Numpy's :func:`~numpy.genfromtxt` function to read the raw data, so this section can contain any valid arguments for the :func:`~numpy.genfromtxt` function. For example, you might include:
  
  .. code-block:: JSON

    {
        "genfromtext_args": {
            "delimiter": "Character that separates data values in rows, e.g. \t (tab) or , (comma)",
            "skip_header": "Integer, without quotations, that specifies the number of lines at the start of the file that *don't* contain data values.",
        }
    }

Optional Sections
^^^^^^^^^^^^^^^^^
  - ``preformat_replace``. Particularly awkward data formats may require some 'cleaning' before they're readable by :func:`~numpy.genfromtxt` (e.g. the removal of non-numeric characters). You can do this by optionally including a ``preformat_replace`` section in your dataformat description. This consists of ``{"regex_expression": "replacement_text"}`` pairs, which are applied to the data before import. For example:
  
  .. code-block:: JSON

    {
        "preformat_replace": {
            "[^0-9, .]+": ""
        }
    }
  will replace all non-numeric characters that are not ``.``, ``,`` or a space with ``""`` (i.e. no text - remove them). The use of ``preformat_replace`` should not be necessary for most dataformats.
  - ``time_format``. ``latools`` attempts to automatically read the ``date`` information identified by ``meta_regex`` (using ``dateutil``'s :func:`~dateutil.parser.parse`), but in rare cases this will fail. If it fails, you'll need to manually specify the date format. Specify the date format using `standard notation for formatting and reading times <https://docs.python.org/3.6/library/datetime.html#strftime-and-strptime-behavior>`_. For example:

  .. code-block:: JSON

    {
        "time_format": "%d-%b-%Y %H:%M:%S"
    }
  will correctly read a time format of "01-Mar-2016 15:23:03".

.. _regex:

Regular Expressions (RegEx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data import in ``latools`` makes use of `Regular Expressions <https://en.wikipedia.org/wiki/Regular_expression>`_ to identify different parts of your data.
Regular expressions are a way of defining *patterns* that allow the computer to extract information from text that isn't exactly the same in every instance.
A very basic example, if you apply the pattern:
::

    "He's not the Mesiah, (.*)"
to ``"He's not the Mesiah, he's a very naughty boy!"``, the expression will *match* the text, and you'll get ``"he's a very naughty boy!"`` in a *capture group*. To break the expression down a bit:

  - ``He's not the Mesiah, `` tells the computer that you're looking for text containing this phrase.
  - ``.`` signifies 'any character'
  - ``*`` signifies 'anywhere between zero and infinity occurrences of ``.``
  - ``()`` identifies the 'capture group'. The expression would still match without this, but you wouldn't be able to isolate the text within the capture group afterwards.
What would the capture group get if you applied the expression to ``He's not the Mesiah, he just thinks he is...``?

Applying this to metadata extraction, imagine you have a line in your file header like:
::

    Acquired      : Oct 29 2015  03:11:05 pm using AcqMethod OB102915.m
And you need to extract the date (``Oct 29 2015  03:11:05 pm``).
You know that the line always starts with ``Acquired [varying number of spaces] :``, and ends with ``using AcqMethod [some text]``.
The expression:
::

    Acquired +: (.*) using.*
will get the date in its capture group! For a full explanation of how this works, have a look at `this breakdown by Regex101 <https://regex101.com/r/C2Qs5z/1>`_ (Note 'Explanation' section in upper right).

Writing your own Regular Expressions can be tricky to get your head around at first.
We suggest using the superb `Regex101 <https://regex101.com/r/HKNavd/1>`_ site to help you design the Regular Expressions in your data format description. Just copy and paste the text you're working with (e.g. line from file header containing the date), play around with the expression until it works as required, and then copy it across to your dataformat file.

.. note:: If you're stuck on data formats, `submit a question to the mailing list <https://groups.google.com/forum/#!forum/latools>`_ and we'll try to help. If you think you've found a serious problem in the software that will prevent you importing your data, `file an issue on the GitHub project page <https://github.com/oscarbranson/latools/issues/new>`_, and we'll look into updating the software to fix the problem.


Writing a new Data Format Description : Step-By-Step
----------------------------------------------------
Data produced by the UC Davis Agilent 8800 looks like this:

.. code-block:: python
    :linenos:

    C:\Path\To\Data.D
    Intensity Vs Time,CPS
    Acquired      : Oct 29 2015  03:11:05 pm using AcqMethod OB102915.m
    Time [Sec],Mg24,Mg25,Al27,Ca43,Ca44,Mn55,Sr88,Ba137,Ba138
    0.367,666.68,25.00,3100.27,300.00,14205.75,7901.80,166.67,37.50,25.00
    ...

This step-by-step guide will go through the process of writing a dataformat description from scratch for the file.

.. tip:: We're working from scratch here for illustrative purposes. When doing this in reality, you might find the :func:`~latools.helpers.config.get_dataformat_template` (accessible via ``latools.config.get_dataformat_template()``), which creates an annotated data format file for you to adapt.

1. Create an empty file, name it, and give it a ``.json`` extension. Open the file in your favourite text editor. Data in ``.json`` files can be stored in lists (comma separated values inside square brackets, e.g. [1,2,3]) or as {'key': 'value'} pairs inside curly brackets.

2. The data format description contains three named sections - ``meta_regex``, ``column_id`` and ``genfromtext_args``, which we'll store as {'key': 'value'} pairs. Create empty entries for these in your new ``.json`` file. Your file should now look like this:

  .. code-block:: JSON

    {
        "meta_regex": {},
        "column_id": {},
        "genfromtext_args": {}
    }

3. Define the start time of the analysis. In this case, it's ``Oct 29 2015  03:11:05 pm``, but it will be different in other files. We therefore use a regular expression' to define a *pattern* that describes the date. To do this, we'll isolate the line containing the date (line 2 - numbers start at ero in Python!), and head on over to `Regex101 to write our expression <https://regex101.com/r/P1chhB/1>`_. Add this expression to the meta_regex ession, with the line number as its key:

.. code-block:: JSON

    {
        "meta_regex": {
            "2": [["date"],
                   "([A-Z][a-z]+ [0-9]+ [0-9]{4}[ ]+[0-9:]+ [amp]+)"]
        },
        "column_id": {},
        "genfromtext_args": {}
    }

.. tip:: Having trouble with Regular Expressions? We really recommend `Regex101 <http://regex101.com>`_!

4. Set some parameters that define where the column names are. ``name_row`` defines which row the column names are in (`3`), ``delimeter`` describes hat character separates the column names (`,`), ``timecolumn`` is the numberical index of the column containing the 'time' data (in this case, `0`). his will grab everything in row 3 that's separated by a comma, and tell ``latools`` that the first column contains the time info. Now we need to tell t which columns contain the analyte names. We'll do this with a regular expression again, copying the entire column over to `Regex101 to help us write he expression <https://regex101.com/r/cOG8dN/1>`_. Put all this information into the "column_id" section:

.. code-block:: JSON

    {
        "meta_regex": {
            "2": [["date"],
                   "([A-Z][a-z]+ [0-9]+ [0-9]{4}[ ]+[0-9:]+ [amp]+)"]
        },
        "column_id": {
            "name_row": 3,
            "delimiter": ",",
            "timecolumn": 0,
            "pattern": "([A-z]{1,2}[0-9]{1,3})"
        },
        "genfromtext_args": {}
    }

5. Finally, we need to add some parameters that tell ``latools`` how to read the actual data table. In this case, we want to skip the first 4 lines, nd then tell it that the values are separated by commas. Add this information to the ``genfromtext_args`` section:

.. code-block:: JSON

    {
        "meta_regex": {
            "2": [["date"],
                   "([A-Z][a-z]+ [0-9]+ [0-9]{4}[ ]+[0-9:]+ [amp]+)"]
        },
        "column_id": {
            "name_row": 3,
            "delimiter": ",",
            "timecolumn": 0,
            "pattern": "([A-z]{1,2}[0-9]{1,3})"
        },
        "genfromtext_args": {
            "delimiter": ",",
            "skip_header": 4
        }
    }

6. Test the format description, using the :func:`~latools.helpers.config.test_dataformat` function. In Python:

.. code-block:: python

    import latools as la

    my_dataformat = 'path/to/my/dataformat.json'
    my_datafile = 'path/to/my/datafile.csv

    la.config.test_dataformat(my_datafile, my_dataformat)

This will go through the data import process for you file, printing out the results of each stage, so if it fails you can see *where* if failed, and ix the problem.

7. Fix any errors, and you're done! You have a working data description.


I've written my dataformat, now what?
-------------------------------------

Once you're happy with your data format description, put it in a text file, and save it as 'my_dataformat.json' (obviously replace my_dataformat with something meaningful...).
When you want to import data using your newly defined format, you can point ``latools`` towards it by specifying ``dataformat='my_dataformat.dict'`` when starting a data analysis.
Alternatively, you can define a new :ref:`manage-configurations`, to make this the default data format for your setup.