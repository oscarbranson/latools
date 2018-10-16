.. _preprocessing:

##################
Preprocessing Data
##################

``latoools`` expects data to be organised in a :ref:`particular way\ <data_formats>`. If your data do not meet these expectations, you'll have to do some pre-processing to get your data into a format that latools can deal with.

We've come across a couple of situations where data aren't organised in a way that latools can make sense of.
1. Data in a single file either stored as multiple individual collections separated by an identifying line, or as one long collection with no separation.
2. Data in multiple files, with unmeaningful names (e.g. numbers), sometimes with the sample ID stored in the file header.

If your data fits into one of these categories, read on to learn about the 'preprocessing' tools included in ``latools``. If not, please `let us know <https://groups.google.com/forum/#!forum/latools>`_ and we'll work out how to accommodate your data.

1. Data in a single file
========================
This can occur if all the data are collected in a single, long analysis file on the mass spec, or if the mass spec software exports all analyses into a single output file.

Single, long analysis
---------------------
.. TODO:: Not implemented yet. This will be implemented when there is demand for it.

Multiple export in single file
------------------------------
In this case, the most straightforward solution is to split your long files into its component parts.
To help with this, we've got a function that helps with splitting files.

2. Data in multiple files
=========================
Some instruments store file names as sequential numbered files, with sample id stored either externally (e.g. in a laser log file), or in the file headers.

External log file
-----------------
.. TODO:: Not implemented yet. This will be implemented when there is demand for it.

File names in headers
---------------------
In this case, you'll need 