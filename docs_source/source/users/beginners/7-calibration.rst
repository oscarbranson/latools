.. _calibration:

###########
Calibration
###########

Once all your data are normalised to an internal standard, you're ready to calibrate the data.
This is done by creating a calibration curve for each element based on SRMs measured throughout your analysis session, and a table of known SRM values.
You can either calculate a single calibration from a combination of all your measured standards, or generate a time-sensitive calibration to account for sensitivity drift through an analytical session.
The latter is achieved by creating a separate calibration curve for each element in each SRM measurement, and linearly extrapolating these calibrations between neighbouring standards.

Calibration is performed using the :meth:`~latools.analyse.calibrate` method:

.. literalinclude:: ../../../../tests/test_beginnersGuide.py
   :language: python
   :dedent: 4
   :lines: 36-37

In this simple example case, our analytical session is very short, so we are not worried about sensitivity drift (``drift_correct=False``). ``poly_n=0`` is fitting a polynomial calibration line to the data that is forced through zero. Changing this number alters the order of polynomial used during calibration. Because of the wide-scale linearity of ICPM-MS detectors, ``poly_n=0`` should normally provide an adequate calibration line. If it does not, it suggests that either one of your 'known' SRM values may be incorrect, or there is some analytical problem that needs to be investigated (e.g. interferences from other elements). Finally, ``srms_used`` contains the names of the SRMs measured throughout analysis. The SRM names you give must *exactly* (case sensitive) match the SRM names in the SRM table.

.. note:: For calibration to work, you must have an SRM table containing the element/internal_standard ratios of the standards you've measured, whose location is specified in the latools configuration. You should only need to do this once for your lab, but it's important to ensure that this is done correctly. For more information, see the :ref:`configuration` section.

First, ``latools`` will automatically determine the identity of measured SRMs throughout your analysis session using a relative concentration matrix (see :ref:`SRM Identification <srm_id>` section for details).
Once you have identified the SRMs in your standards, ``latools`` will import your SRM data table (defined in the :ref:`configuration file <cfg_file>`), calculate a calibration curve for each analyte based on your measured and known SRM values, and apply the calibration to all samples.

The calibration lines for each analyte can be plotted using:

.. literalinclude:: ../../../../tests/test_beginnersGuide.py
   :language: python
   :dedent: 4
   :lines: 39

Which should look something like this:

.. image:: ./figs/calibration.png

Where each panel shows the measured counts/count (x axis) vs. known mol/mol (y axis) for each analyte with associated errors, with the fitted calibration line, equation and R2 of the fit. The axis on the right of each panel contains a histogram of the raw data from each sample, showing where your sample measurements lie compared to the range of the standards.
