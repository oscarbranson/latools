.. _plotting:

########
Plotting
########

At this stage, you might like to plot you data and check what it looks like. This is achieved using :meth:`~latools.analyse.trace_plots`::

	eg.trace_plots()

This will generate 4 plots of your raw data in a newly generated folder called reports/rawdata.
This function can be customisedto produce plots of any stage of your data analysis.
If you run it without any parameters (as above) it plots all the analytes measures on each of your samples on a log scale, and saves the plots in a subfolder of the 'reports' directory named for whichever stage of dataprocessing has just been performed (in this case, no processing has happened, so it plots 'rawdata'.