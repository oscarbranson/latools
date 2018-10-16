.. _filters-defrag:

###############
Defragmentation
###############

Occasionally, filters can become 'fragmented' and erroneously omit or include lots of small data fragments. For example, if a signal oscillates either side of a threshold value. The defragmentation filter provides a way to either include incorrectly removed missing data regions, or exclude data in fragmented regions.

.. code :: python

    eg.filter_threshold('Al27', 0.65e-4)
    eg.filter_on('Al27_below')

.. image :: figs/5-fragmented-filter.png

Notice how this filter has removed lots of small data regions, where Al27 oscillates around the threshold value.

If you think these regions should be included in the selection, the defragmentation filter can be used in 'include' mode to create a contiguous data selection:

.. code :: python

    eg.filter_defragment(10, mode='include')

    eg.filter_off('Al27')  # deactivate the original Al filter
    eg.filter_on('defrag')  # activate the new defragmented filter

.. image :: figs/5-fragmented-include.png

This identifies all regions removed by the currently active filters that are 10 points or less in length, and includes them in the data selection.

.. tip :: The defragmentation filter acts on all currently active filters, so pay attention to which filters are turned 'on' or 'off' when you use it. You'll also need to de-activate the filters used to create the defragmentation filter to see its effects.

If, on the other hand, the proximity of Al27 in this sample to the threshold value might suggest contamination, you can use 'exclude' mode to remove small regions of selected data.

.. code :: python

    eg.filter_threshold('Al27', 0.65e-4)
    eg.filter_on('Al27_below')


    eg.filter_defragment(10, mode='exclude')

    eg.filter_off()  # deactivate the original Al filter
    eg.filter_on('defrag')  # activate the new defragmented filter

.. image :: figs/5-fragmented-exclude.png

This removes all fragments fragments of selected data that are 10-points or less, and removes them. 

Related Functions
-----------------

* :meth:`~latools.latools.analyse.trace_plots` with option ``filt=True`` creates plots of all data, showing which regions are selected/rejected by the active filters.
* :meth:`~latools.latools.analyse.filter_on` and :meth:`~latools.latools.analyse.filter_off` turn filters on or off.
* :meth:`~latools.latools.analyse.filter_clear` deletes all filters.