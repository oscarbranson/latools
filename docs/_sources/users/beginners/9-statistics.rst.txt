.. _statistics:

#################
Sample Statistics
#################

After filtering, you can calculated and export integrated compositional values for your analyses::

	eg.sample_stats(stat_fns=[np.nanmean, np.nanstd], filt=True)

Where ``stat_fns`` specifies which functions you would like to use to calculate the statistics (in this case the ``nanmean`` and ``nanstd`` of the ``numpy`` package). 
You can specify any function that accepts an array and returns a single value here.
``filt`` can either be True (applies all active filters), or a specific filter number or partially matching name to apply a specific filter.
In combination with data subsets, and the ability to specify different combinations of filters for different subsets, this provides a flexible way to explore the impact of different filters on your integrated values.

We've now calculated the statistics, but they are still trapped inside the 'analyse' data object (``eg``).
To get them out into a more useful form::

	stats =	eg.getstats()

This returns a :class:`pandas.DataFrame` containing all the statistics we just calculated.
You can either keep this data in python and continue your analysis, or export the integrated values to an external file for analysis and plotting in *your_favourite_program*.

To export the data, you can either specifying the ``filename`` variable in :meth:`~latools.analyse.getstats`, which will be saved in the 'export_data' directory, or you can use the pandas built in export methods like :meth:`~pandas.DataFrame.to_csv` or :meth:`~pandas.DataFrame.to_excel` to take your data straight to a variety of formats, for example::

	stats.to_csv('reports/stats.csv') # .csv format
	stats.to_excel('reports/stats.csv') # excel format

.. hint:: To use pandas :meth:`~pandas.DataFrame.to_excel` method, you must have `xlsxwriter <http://xlsxwriter.readthedocs.io/>`_ installed in python.