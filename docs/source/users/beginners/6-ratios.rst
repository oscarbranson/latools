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
