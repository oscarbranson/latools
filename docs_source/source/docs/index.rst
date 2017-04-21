.. _latools_docs:

#####################
Latools Documentation
#####################

Classes
=======

* :class:`latools.D` contains the data from a single analysis file, and all the processing fucntions that can be applied to it.
* :class:`latools.analyse` combines numerous 'D' data objects into a single analysis session.
* :class:`latools.filt` is for storing, choosing and applying data selection filters.

:class:`latools.D`
==================

.. autoclass:: latools.D
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: plotting, colormaps

:class:`latools.analyse`
========================

.. autoclass:: latools.analyse
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: plotting, colormaps

:class:`latools.filt`
=====================

.. autoclass:: latools.filt
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: plotting, colormaps

Other Functions
===============

.. autofunction:: latools.collate_data
.. autofunction:: latools.unitpicker
.. autofunction:: latools.pretty_element
.. autofunction:: latools.bool_2_indices
.. autofunction:: latools.tuples_2_bool
.. autofunction:: latools.config_locator
.. autofunction:: latools.add_config
.. autofunction:: latools.intial_configuration
.. autofunction:: latools.get_example_data
