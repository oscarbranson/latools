.. _filters-trim:

##################
Trimming/Expansion
##################

This either expands or contracts the currently active filters by a specified number of points.

Trimming a filter can be a useful tool to make selections more conservative, and more effectively remove contaminants.

Related Functions
-----------------

* :meth:`~latools.latools.analyse.trace_plots` with option ``filt=True`` creates plots of all data, showing which regions are selected/rejected by the active filters.
* :meth:`~latools.latools.analyse.filter_on` and :meth:`~latools.latools.analyse.filter_off` turn filters on or off.
* :meth:`~latools.latools.analyse.filter_clear` deletes all filters.