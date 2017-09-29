.. _beginner-summary:

#######
Summary
#######

If we put all the preceding steps together:

.. literalinclude:: ../../../../tests/test_beginnersGuide.py
   :language: python
   :dedent: 4
   :lines: 15-60

Here we processed just 3 files, but the same procedure can be applied to an entire day of analyses, and takes just a little longer.

The processing stage most likely to modify your results is filtering.
There are a number of filters available, ranging from simple concentration thresholds (:meth:`~latools.analyse.filter_threshold`, as above) to advanced multi-dimensional clustering algorithms (:meth:`~latools.analyse.filter_clustering`).
We recommend you read and understand the section on :ref:`advanced_filtering` before applying filters to your data.

Before You Go
=============

Before you try to analyse your own data, you must configure latools to work with your particular instrument/standards.
To do this, follow the :ref:`configuration` guide.

We also highly recommend that you read through the :ref:`advanced_topics`, so you understand how ``latools`` works before you start using it.