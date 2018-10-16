.. _filters-optimise:

###################
Signal Optimisation
###################

This is the most complex filter available within ``latools``, but can produce some of the best results.

The filter aims to identify the longest contiguous region within each ablation where the concentration of target analyte(s) is either maximised or minimised, and standard deviation is minimised.

First, we calculate the mean and standard deviation for the target analyte over all sub-regions of the data.

.. figure :: ./figs/4-optimise-statcalc.gif
    :align: center

    The mean and standard devation of Al27 is calculated using an N-point rolling window, then N+1, N+2 and etc., until N equals the number of data points. In the 'Mean' plot, darker regions contain the lowest values, and in the 'Standard Deviation' plot red regions contain a higher standard deviation.

Next, we use the distributions of the calculated means and standard deviations to define some threshold values to identify the optimal region to select. For example, if the goal is to minimise the concentration of an analyte, the threshold concentration value will be the lowest disinct peak in the histogram of region means. The location of this peak defines the 'mean' threshold. Similarly, as the target is always to minimise the standard deviation, the standard deviation threshold will also be the lowest distinct peak in the histogram of standard devaiations of all calculated regions. Once identified, these thresholds are used to 'filter' the calculated means and standard deviations. The 'optimal' selection has a mean and standard deviation below the calculated threshold values, and contains the maximum possible number of data points.

.. figure :: ./figs/4-optimise-statcalc.png
    :align: center

    After calculating the mean and standard deviation for all regions, the optimal region is identified using threshold values derived from the distributions of sub-region means and standard deviations. These thresholds are used to 'filter' the calculated means and standard deviations - regions where they are above the threshold values are greyed out in the top row of plots. The optimal selection is the largest region where both the standard deviation and mean are below the threshold values.

Related Functions
-----------------

* :meth:`~latools.latools.analyse.optimisation_plots` creates plots similar to the one above, showing the action of the optimisation algorithm.
* :meth:`~latools.latools.analyse.trace_plots` with option ``filt=True`` creates plots of all data, showing which regions are selected/rejected by the active filters.
* :meth:`~latools.latools.analyse.filter_on` and :meth:`~latools.latools.analyse.filter_off` turn filters on or off.
* :meth:`~latools.latools.analyse.filter_clear` deletes all filters.