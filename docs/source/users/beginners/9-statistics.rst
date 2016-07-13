.. _statistics:

#################
Sample Statistics
#################

Now we have determined which regions of the data are 'good' and 'bad', we can calculate the mean and standard deviation for each analysis spot in each sample::

	# calculate sample statistics.
	eg.stat_samples(stat_fns=[np.nanmean, np.nanstd], eachtrace=True, filt=True)

This calculates the mean and standard deviation (the numpy functions for nan-ignoring mean and standard deviations are specified in the ``stat_fns`` variable) for each analysis spot (``eachtrace=True``) independently, after it has applied the filters we just created and turned on (``filt=True``).

We've now calculated the statistics, but they are still trapped inside the 'analyse' data object (``eg``).
To get them out into a more useful form::

	stats =	eg.getstats()

This returns a :class:`pandas.DataFrame` containing all the statistics we just calculated.
You can either keep this data in python and continue your analysis (reccommended), or export it to a .csv file for analysis in *your_favourite_stats_program*.

To export the data, you can either specifing the ``path`` variable in :meth:`~latools.analyse.getstats`.
Or you can use the pandas built in export methods like :meth:`~pandas.DataFrame.to_csv` or :meth:`~pandas.DataFrame.to_excel` to take your data straight to a number of formats, for example::

	stats.to_csv('reports/stats.csv') # .csv format
	stats.to_excel('reports/stats.csv') # excel format

.. hint:: To use pandas :meth:`~pandas.DataFrame.to_excel` method, you must have `xlsxwriter <http://xlsxwriter.readthedocs.io/>`_ installed in python.