.. _import:

##############
Importing Data
##############

Once you have unpacked the :ref:`example_data`, you must set up an ''latools'' analysis session.
To do this, create an instance of the :class:`latools.analyse` class, and assign it to the variable :code:`eg`::

	eg = la.analyse('./')

This finds all the data files in the specified folder and loads them into a data `object`, along with various default parameters.
If it has worked correctly, you should see the output::

	latools analysis using "DEFAULT" configuration:
	  4 Data Files Loaded: 1 standards, 3 samples
	  Analytes: Mg24 Mg25 Al27 Ca43 Ca44 Mn55 Sr88 Ba137 Ba138

The three main actions on data import are:

1. Read the latools.cfg file. This tells ''latools'' the format of the data files, and location of the SRM data file.
2. Find and import each data file in the directroy as a :class:`latools.D` data object.
3. Create 'params' and 'reports' directories for storing analysis outputs later on.

Check inside your data directory, there should now be new new folders called 'params' and 'reports' alongside the .csv files.

The :code:`eg` object created here now contains all the data, and all the processing functions you will apply to the data.