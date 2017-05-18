.. _getting_started:

###############
Getting Started
###############

This guide will take you through the analysis of some example data included with ``latools``, with explanatory notes telling you what the software is doing at each step.
We recommend working through these examples to understand the mechanics of the software before setting up your :ref:`configuration`, and working on your own data.

The Fundamentals: Python
========================
`Python <https://www.python.org/>`_ is an open-source (free) general purpose programming language, with growing application in science.
``latools`` is a python `module` - a package of code containing a number of Python `objects` and functions, which run within Python.
That means that you need to have a working copy of Python to use ``latools``.

If you don't already have this (or are unsure if you do), we recommend that you install one of the pre-packaged science-oriented Python distributions, like Continuum's `Anaconda <https://www.continuum.io/downloads>`_ (recommended).
Both of these provide a complete working installation of Python, and all the pre-requisites you need to run latools.

``latools`` has been developed and tested in Python 3.5. 
It *should* also run on 2.7, but we can't guarantee that it will behave.

''latools'' should work in any python interpreter, but we recommend either `Jupyter Notebook <http://jupyter.org/>`_ or `iPython <https://ipython.org/>`_.
Jupyter is a browser-based interface for ipython, which provides a nice clean interactive front-end for writing code, taking notes and viewing plots.

For simplicity, the rest of this guide will assume you're using Jupyter notebook, although it should translate directly to other Python interpreters.

For a full walk through of getting ``latools`` set up on your system, head on over to the :ref:`install` guide.

Preparation
===========
Before we start ``latools``, you should create a folder to contain everything we're going to do in this guide.
For example, you might create a folder called ``latools_demo/`` on your Desktop - we'll refer to this folder as ``latools_demo/`` from now on, but you can call it whatever you want.
Remember where this folder is - we'll come back to it a lot later.

.. tip:: As you process data with ``latools``, new folders will be created in this directory containing plots and exported data. This works best (i.e. is least cluttered) if you put your data in a single directory inside a parent directory (in this case ``latools_demo``), so all the directories created during analysis will also be in the same place, without lots of other files.

Starting `latools`
==================
Next, launch a Jupyter notebook in this folder. To do this, open a terminal window, and run:

.. code-block:: bash

    cd ~/path/to/latools_demo/
    jupyter notebook

This should open a browser window, showing the Jupyter main screen. 
From here, start a new Python notebook by clicking 'New' in the top right, and selecting your Python version (preferably 3.5+).
This will open a new browser tab containing your Python notebook.

Once python is running, import latools into your environment::

	import latools as la

All the functions of latools are now accessible from within the ``la`` prefix.

.. tip:: To run code in a Jupyter notebook, you must 'evaluate' the cell containing the code. to do this, type:

	* [ctrl] + [return] evaluate the selected cell.
	* [shift] + [return] evaluate the selected cell, and moves the focus to the next cell
	* [alt] + [return] evaluate the selected cell, and creates a new empty cell underneath.

.. _example_data:
Example Data
============

Once you've imported ``latools``, extract the example dataset to a ``data/`` folder within ``latools_demo/``::

	# copy the example data to your current directory.
	la.get_example_data('./data')

Take a look at the contents of the directory.
You should see four .csv files, which are raw data files from an Agilent 7700 Quadrupole mass spectrometer, outputting the counts per second of each analyte as a function of time.
Notice that each .csv file either has a sample name, or is called 'STD'.

.. note:: Each data file should contain data from a single sample, and data files containing measurements of standards should all contain an identifying set of characters (in this case 'STD') in the name. For more information, see :ref:`advanced_data_formats`.