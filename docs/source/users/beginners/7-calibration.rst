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


In this simple example case, our analytical session is very short, so we are not worried about sensitivity drift (``drift_correct=False``). 

There is also a default parameter ``poly_n=0``, which specifies that the polynomial calibration line fitted to the data that is forced through zero. Changing this number alters the order of polynomial used during calibration. Because of the wide-scale linearity of ICPM-MS detectors, ``poly_n=0`` should normally provide an adequate calibration line. If it does not, it suggests that either one of your 'known' SRM values may be incorrect, or there is some analytical problem that needs to be investigated (e.g. interferences from other elements). Finally, ``srms_used`` contains the names of the SRMs measured throughout analysis. The SRM names you give must *exactly* (case sensitive) match the SRM names in the SRM table.

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

###############################
Mass Fraction (ppm) Calculation
###############################

After calibration, all data are in units of mol/mol.
For many use cases (e.g. carbonate trace elements) this will be sufficient, and you can continue on to :ref:`filtering`.
In other cases, you might prefer to work mass fractions (e.g. ppm).
If so, the next step is to convert your mol/mol ratios to mass fractions.

This requires knowledge of the concentration of the internal standard in all your samples, which we must provide to latools.
First, generate a list of samples in a spreadsheet:

.. code-block :: python

    eg.get_sample_list()

This will create a file containing a list of all samples in your analysis, with an empty column to provide the mass fraction (or % or ppm) of the internal standard for each individual sample. Enter this information for each sample, and save the file without changing its format (.csv) - remember where you saved it, you'll use it in the next step! Pay attention to units here - the calculated mass fraction values for your samples will have the same units as you provide here.

.. tip :: If all your samples have the same concentration of internal standard, you can skip this step and just enter a single mass fraction value at the calculation stage.

Next, import this information and use it to calculate the mass fraction of each element in each sample:

.. code-block :: python

    eg.calculate_mass_fraction('/path/to/internal_standard_massfrac.csv')

Replace `path/to/interninternal_standard_massfrac.csv` with the location of the file you edited in the previous step). This will calculate the mass fractions of all analytes in all samples in the same units as the provided internal standard concentrations. If you know that all your samples have the same internal standard concentration, you could just provide a number instead of a file path here.