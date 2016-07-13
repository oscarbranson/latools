.. _filtering:

############################
Data Selection and Filtering
############################

Your data is now identified, background corrected, normalised to a particular element, and calibrated.
Now begins the most important aspect of your data processing: **data selection**.

Not all data in a laser ablation analysis will be 'good' data.
Contaminants are often present in the sample, and can vary significantly
through a laser ablation pit.
Thus, some parts of your data will be 'good', and others will be 'bad'.
The manual discrimination between 'good' and 'bad' data introduces significant subjectivity into laser ablation analysis.
``latools`` offers a way to remove the subjectivity from data selection, and quantify which data you have selected in a transparrent way so that your analysis can be examined and reproduced by others.

Data selection is done using **filters**.
Filters apply a function to your data which splits your data into several categories, each of which is identified by a boolean key (array of True/False values the same length as your data).
Once created these filters can be turned on or off to select specific parts of the data.

Several filtering functions are available (in order of increasing complexity):

* :meth:`~latools.analyse.filter_threshold`: Creates two filter keys identifying where a specific analyte is above or below a given threshold.
* :meth:`~latools.analyse.filter_distribution`: Finds separate `populations` within the measured concentration of a single analyte within by creating a Probability Distribution Function (PDF) of the analyte within each sample. Local minima in the PDF identify the boundaries between distinct concentrations of that analyte within your sample.
* :meth:`~latools.analyse.filter_clustering`: A more sophisticated version of :meth:`~latools.analyse.filter_distribution`, which uses data clustering algorithms from the `sklearn <http://scikit-learn.org/>`_ module to find statistically distinguishable populations in your data. This can consider multiple analytes at once, allowing for the robust detection of distinct compositional zones in your data using robust, n-dimensional clustering algorithms.
* :meth:`~latools.analyse.filter_correlation`: Finds regions in your data where two analytes correlate locally. For example, if your analyte of interest strongly covaries with an analyte that is a known contaminant indicator, the signal is likely contaminated, and shoule be discarded.

For a full account of these filters and how they should be used, see :ref:`advanced_filtering`.

For the purposes of this demonstration, we will apply a simple theshold filter to exclude regions where the Al/Ca ratios > 0.5 mmol/mol::

	# create a threshold filter at 0.5 mmol/mol Al/Ca
	eg.filter_threshold('Al27', 0.5E-3)
	# turn on the 'below' filter, so only data below the threshold are reported.
	eg.filter_on('Al27_thresh_below')

.. note:: Filters do not delete any data. They simply create a *mask* which tells the statistic funcions which data to use, and which to ignore.
