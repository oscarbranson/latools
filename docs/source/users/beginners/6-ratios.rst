.. _ratios:

#################
Ratio Calculation
#################

Next, you must standardise your data against a particular analyte.
Do this using :meth:`~latools.analyse.ratio`::

	eg.ratio(denominator='Ca43')

This divides all analytes in all samples by the Ca43 counts.
This converts all data to counts/Ca43 counts format, the same units as used in the SRM table.

.. note:: At present, ``latools`` is designed to work with ratio data, and it has not been tested with absolute data. However, with minor adjustments (replacing the ratio SRM file with an absolute SRM file, and tweaking the calibration function) it could be made to work with absolute data. If there is a need for this, it can be implemented.