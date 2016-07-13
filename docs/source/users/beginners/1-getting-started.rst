.. _getting_started:

###############
Getting Started
###############

This guide will take you through the analysis of some example data included with `latools`.
We recommend working through these examples to understand the mechanics of the software before setting up your :ref:`configuration`, and working on your own data.

Starting `latools`
==================

First, create an empty folder that will be used for processing the sample data.

Next, you need to start a python interpreter in this directory.
You can run ''latools'' from any python interpreter.
However, we highly recommend you try out `Jupyter Notebook <http://jupyter.org/>`_, a browser-based front end for python, which provides a nice clean interface for writing python code and taking notes.
This comes pre-installed with packaged installations of python, like `Anaconda <https://www.continuum.io/downloads>`_ or `Canopy <https://www.enthought.com/products/canopy/>`_.
To start a jupyter notebook run :code:`jupyter notebook` in the terminal window you opened above.

Once python is running, import latools into your environment::

	import latools as la

All the functions of latools are now accessible from within the `la` prefix.

.. tip:: To evaluate code in a jupyter notebook, you must 'run' the cell containing the code. to do this, type:

	* [ctrl] + [return] run the selected cell.
	* [shift] + [return] run the selected cell, and moves the focus to the next cell
	* [alt] + [return] run the selected cell, and creates a new empty cell underneath.

.. _example_data:
Example Data
============

Once you've imported `latools`, extract the example dataset to an empty directory in a convenient location::

	# copy the example data to your current directory.
	la.get_example_data('./')

Take a look at the contents of the directory.
You should see four .csv files, which are raw data files from an Agilent 7700 Quadrupole mass spectrometer, outputting the counts per second of each analyte as a function of time.
Notice that each .csv file either has a sample name, or is called 'STD'.
This is how you should prepare your data for analysis with `latools`.

.. note:: Each data file should contain data from a single sample, and data files containing measurements of standards should contain 'STD' in the name. For more information, see :ref:`advanced_data_formats`.

