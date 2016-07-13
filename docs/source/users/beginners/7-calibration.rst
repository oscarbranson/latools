.. _calibration:

###########
Calibration
###########

Once the ratios are calculated and data are in the same units as the SRM file, you may calibrate your data to give mol/mol values. This is done using the :meth:`~latools.analyse.calibrate` method::

	eg.calibrate()

.. warn:: At present, this only works in the Jupyter Notebook.

First, ``latools`` must determine the names of each SRM you have measured, so it can match it against SRMs in the databse.
To do this, it will create a plot of your first standard file, and ask you to name each SRM you have measured in it.
After you have labelled the SRMs in this standard file, it will ask you if the SRMs in all the standard files were measured in the same order. If they were, answer yes, and it will assign SRM labels to all the other standards.

.. note:: The SRM names you give must *exactly* match the SRM names in the SRM table. Names are case sensitive.

.. tip:: Measure all the SRMs in the same order in each of your standard files, to make labelling them easier.

Once you have identified the SRMs in your standards, ``latools`` will import your SRM data table (defined in the :ref:`.cfg file <cfg_file>`), calculate a calibration curve for each analyte based on your measured and known SRM vaues, and apply the calibration to all samples.
You can specify the form of the calibration curve used with the the ``n_poly`` paramter.
If ``n_poly=0``, a zero-intercept line function is used, and if ``n_poly>=1`` an n\ :sup:`th` order polynomial is used.

The calibration lines for each analyte can be plotted using::

	eg.calibration_plot()

.. todo:: At present, a single calibration is applied to all your data. In future it will be possible to assign a time-sensitive calibration factor, to accommodate sensitivity drift throughout you analysis period.