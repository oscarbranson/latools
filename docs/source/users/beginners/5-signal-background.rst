.. _bkgcorrect:

#####################
Background Correction
#####################

Next, the data must be background corrected. Before this can be done, ``latools`` must determine which parts of your data are `background` vs `signal`.
This is achived using :meth:`~latools.analyse.autorange`, which uses a combination of data thresholding to identify approximate signal and background regions, and a complex curve-fitting routine to exclude the 'transition' between background and singal.
The end result is a series of boolean 'keys' (arrays of True/False values the same length of your data), which identify sections of your data as 'signal', 'background' or 'transition'.
For a full account of how this function works, see :ref:`autorange`.

The function is applied to your data by running::

	eg.autorange()

Once you have identified the background and signal regions of your data, you may perform a background correction using :meth:`~latools.analyse.bkg_correct`::

	eg.bkg_correct()

Background correction can either subtract a constant background, or an n\ :sup:`th` order polynomial background.
This is specified in the ``mode`` argument of :meth:`~latools.analyse.bkg_correct`, which may be either:

* ``'constant'``: subtract the mean background counts for each analyte in each sample from the signal.
* ``n`` (where n is an integer): fit an n\ :sup:`th` order polynomial to the background counts of each analyte for each sample, and subtract this from the signal.

.. warning:: Use extreme caution with polynomial backgrounds of n>1. You should only use this if you know you have significant non-linear drift in your background, which you understand but cannot be dealt with by changing you analytical procedure.

.. tip:: Remember that you can plot the data and examine it at any stage of your processing. running ``eg.trace_plot()`` now would create a new subdirectory called 'bkgcorrect' in your 'reports' directory, and plot all the background corrected data.