.. _preprocessing:

##################
Preprocessing Data
##################

``latoools`` expects data to be organised in a :ref:`particular way\ <data_formats>`. If your data do not meet these expectations, you'll have to do some pre-processing to get your data into a format that latools can deal with. Read on to learn about the 'preprocessing' tools included in ``latools``.

If you've got data that can't be processed with the functions below, please `let us know <https://groups.google.com/forum/#!forum/latools>`_ and we'll work out how to accommodate your data.

=============================
1. Data in a single long file
=============================
To work with this data, you have to split it up into numerous shorter files, each containing ablations of a single sample. This can be done using :meth:`latools.preprocessing.split.long_file`.

What this function does:
------------------------
 1. Import your data, and provide a list of sample names.
 2. Apply :meth:`~latools.processes.signal_id.autorange` to identify ablations.
 3. Match up the sample names and ablations.
 4. Save a single file for each sample in an output folder, which can be imported by :meth:`~latools.latools.analyse`

You'll end up with a single directory containing one file for each sample in the sample list, named with the sample names that you provide. The number of samples in the list and the number of ablations in the long file obviously have to match.

Example usage:

.. code-block :: python

    import latools as la

    la.preprocessing.long_file('path/to/long_data_file.csv', dataformat='DEFAULT', 
                                sample_list=sample_list)

`dataformat` can be a dataformat dictionary, or the name of a file or ``latools`` configuration.


2. Data in multiple files without sample names
==============================================
Some instruments store file names as sequential numbered files, with sample id stored either externally (e.g. in a laser log file), or in the file headers.

External log file
-----------------
.. TODO:: Not implemented yet. This will be implemented when there is demand for it.

File names in headers
---------------------
Currently supported.