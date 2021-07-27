.. _filters-correlation:

###########
Correlation
###########

Correlation filters identify regions in the signal where two analytes increase or decrease in tandem. This can be useful for removing ablation regions contaminated by a phase with similar composition to the host material, which influences more than one element. 

For example, the tests of foraminifera (biomineral calcium carbonate) are known to be relatively homogeneous in Mn/Ca and Al/Ca. When preserved in marine sediments, the tests can become contaminated with clay minerals that are enriched in Mn and Al, and unknown concentrations of other elements. Thus, regions where Al/Ca and Mn/Ca co-vary are likely contaminated by clay materials. A Al vs. Mn correlation filter can be used to exclude these regions.

For example:

.. code-block :: python

    eg.filter_correlation(x_analyte='Al27', y_analyte='Mn55', window=51, r_threshold=0.5, p_threshold=0.05)
    eg.filter_on('AlMn')

    eg.data['Sample-1'].trace_plot(filt=True)

.. image :: ./figs/2-correlation.png

The top panel shows the two target analytes. The Pearson correlation coefficient (R) between these elements, along with the significance level of the correlation (p) is calculated for 51-point rolling window across the data. Data are excluded in regions R is greater than ``r_threshold`` and p is less than ``p_threshold``.

The second panel shows the Pearson R value for the correlation between these elements. Regions where R is above the ``r_threshold`` value are excluded.

The third panel shows the significance level of the correlation (p). Regions where p is less than ``p_threshold`` are excluded.

The bottom panel shows data regions excluded by the combined R and p filters.

Choosing R and p thresholds
---------------------------

The pearson R value ranges between -1 and 1, where 0 is no correlation, -1 is a perfect negative, and 1 is a perfect positive correlation. 
The R values of the data will be effected by both the degree of correlation between the analytes, and the noise in the data.
Choosing an absolute R threshold is therefore not straightforward.

.. tip :: The filter does not discriminate between positive and negative correlations, but considers the absolute R value - i.e. an ``r_threshold`` of 0.9 will remove regions where R is greater than 0.9, and less than -0.9.

Similarly, the p value of the correlation will depend on the strength of the correlation, the window size used, and the noise in the data.

The best way to choose thresholds is by looking at the correlation values, using :meth:`~latools.latools.analyse.correlation_plots` to inspect inter-analyte correlations before creating the filter.

For example:

.. code-block :: python

    eg.correlation_plots(x_analyte='Al27', y_analyte='Mn55', window=51)

Will produce pdf plots like the following for all samples. 

.. image :: ./figs/2-correlation_plot.png

Related Functions
-----------------

* :meth:`~latools.latools.analyse.correlation_plots` creates plots of the local correlation between two analytes.
* :meth:`~latools.latools.analyse.crossplot` creates a cross-plot of all analytes, showing relationships within the data at the population-level (all samples). This can be useful when choosing a threshold value.
* :meth:`~latools.latools.analyse.trace_plots` with option ``filt=True`` creates plots of all data, showing which regions are selected/rejected by the active filters.
* :meth:`~latools.latools.analyse.filter_on` and :meth:`~latools.latools.analyse.filter_off` turn filters on or off.
* :meth:`~latools.latools.analyse.filter_clear` deletes all filters.