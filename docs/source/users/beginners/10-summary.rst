.. _beginner-summary:

#######
Summary
#######

If we put all the preceding steps together::

	eg = la.analyse('./')
	eg.despike()
	eg.autorange()
	eg.bkg_correct()
	eg.ratio()
	eg.calibrate()

	# create a threshold filter at 0.5 mmol/mol Al/Ca
	eg.filter_threshold('Al27', 0.5E-3)
	# turn off the 'above' filter, so only data below the threshold is kept.
	eg.filter_off('Al27_thresh_above')

	# calculate sample statistics.
	eg.stat_samples()
	stats =	eg.getstats()
	stats.to_csv('reports/stats.csv')

Here we processed just 4 files, but the same procedure can be applied to an entire day of analyses, and takes just a few seconds longer.

The processing stage most likely to modify your results is filtering.
There are a number of filters available, ranging from simple concentration thresholds (:meth:`~latools.analyse.filter_threshold`, as above) to advanced multi-dimensional clustering algorithms (:meth:`~latools.analyse.filter_clustering`).
We reccommend you read and understand the section on :ref:`advanced_filtering` before applying filters to your data.

Before You Go
=============

Before you try to analyse your own data, you must configure latools to work with your particular instrument/standards.
To do this, follow the :ref:`configuration` guide.

We also highly reccommend that you read through the :ref:`advanced_topics`, so you understand how ``latools`` works before you start using it.