.. _ratios:

#################
Ratio Calculation
#################

Next, you must normalise your data to an internal standard, using :meth:`~latools.analyse.ratio`:

.. literalinclude:: ../../../../tests/test_beginnersGuide.py
   :language: python
   :dedent: 4
   :lines: 34

The internal standard is specified during data import, but can also be changed here by specifying ``internal_standard`` in :meth:`~latools.analyse.ratio`.
In this case, the internal standard is Ca43, so all analytes are divided by Ca43.

.. note:: ``latools`` works entirely in ratios from here on. This avoids cumbersome assumptions regarding bulk sample composition required to attain absolute analyte concentrations, and makes processing and error propagation numerically simpler. If you require absolute concentrations, these may be calculated from the ratio values at the end of data processing, *as long as you know the concentration of the internal standard in your samples*.