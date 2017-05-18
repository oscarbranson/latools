.. _beginner-summary:

#######
Summary
#######

If we put all the preceding steps together::

	eg = la.analyse(data_folder='./data/', 
                    config='DEFAULT', 
                    internal_standard='Ca43', 
                    srm_identifier='STD')
	eg.despike(expdecay_despiker=True, 
               noise_despiker=True)
	eg.autorange(on_mult=[1.5, 0.8], 
                 off_mult=[0.8, 1.5])
	eg.bkg_calc_weightedmean(weight_fwhm=300, 
                             n_min=10)
	eg.bkg_subtract()
	eg.ratio()
	eg.calibrate(drift_correct=False, 
                 poly_n=0,
                 srms_used=['NIST610', 'NIST612', 'NIST614'])
	
	# create a threshold filter at 0.1 mmol/mol Al/Ca
	eg.filter_threshold(analyte='Al27', threshold=0.1e-3)
	# turn off the 'above' filter, so only data below the threshold is kept.
	eg.filter_on(filt=0)

	# calculate sample statistics.
	eg.sample_stats()
	# get statistics into a dataframe
	stats =	eg.getstats()
	# save statistics to a csv file
	stats.to_csv('data_export/stats.csv')

Here we processed just 3 files, but the same procedure can be applied to an entire day of analyses, and takes just a little longer.

The processing stage most likely to modify your results is filtering.
There are a number of filters available, ranging from simple concentration thresholds (:meth:`~latools.analyse.filter_threshold`, as above) to advanced multi-dimensional clustering algorithms (:meth:`~latools.analyse.filter_clustering`).
We recommend you read and understand the section on :ref:`advanced_filtering` before applying filters to your data.

Before You Go
=============

Before you try to analyse your own data, you must configure latools to work with your particular instrument/standards.
To do this, follow the :ref:`configuration` guide.

We also highly recommend that you read through the :ref:`advanced_topics`, so you understand how ``latools`` works before you start using it.