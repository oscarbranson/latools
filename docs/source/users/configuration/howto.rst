.. _configuration:

##################
Setting up LAtools
##################

.. warning:: ``latools`` will not work if incorrectly configured. Follow the instructions below carefully.

Like all software, ``latools`` is stupid.
It won't be able to read your data unless you tell it how to, and it won't know the composition of your reference materials unless you tell it. Before you can use it, ``latools`` must be configured so that it can read your data files, and understand your reference materials.

This involves three key steps:


1. Data Format Description
==========================
Mass specs can produce a baffling array of different data formats, which can also be customised by the user.
Because of this, out built-in data format descriptions are unlikely to work with your data, and you'll need to write a data format description.

The complexity of this data format description will depend on the format of your data, and can vary from a simple 2-3 line snippet, to a baffling array of heiroglyphics. We appreciate that this may be something of a barrier to the beginner. To make this process as painless as possible, a step-by-step guide on how to approach this is in the :ref:`advanced_data_formats` section. 

.. tip:: If you're having difficulties after going through the :ref:`advanced_data_formats` guide, send an example file to the mailing list, and we'll do our best to help out.

2. SRM database File
====================
This contains raw compositional values for the SRMs you use in analysis, and is essential for calibrating your data.

``latools`` comes with `GeoRem <http://georem.mpch-mainz.gwdg.de/>`_ 'preferred' compositions for NIST610, NIST612 and NIST614 glasses.
If you use any other standards, or are unhappy with the GeoRem 'preferred' values, you'll have to add them.

Instructions on how to do this are in :ref:`srm_file` guide.

3. Make an LAtools Configuration
================================
Once you've got a data description and SRM database that you're happy with, you can create a configuration in `latools`, so it always knows where to find them.
Whenever you go to import data, you can tell it which configuration to use by setting the ``config`` parameters of :class:`latools.analyse`.

Instructions on how to do this are in the :ref:`manage-configurations` section.

I've configured latools for one instrument, what about my other one?
====================================================================

``latools`` supports multiple simultaneous configurations, which can be specified at the start of data analysis using the ``config`` parameter of :class:`latools.analyse`.
This makes it easy to switch between processing data from different instruments, or using different sets of SRMs.

There are number of functions to make creating, modifying and deleting configurations as simple as possible in :module:`latools.helpers.config`.
As a starting point, you can view all defined configurations using :func:`latools.config.print_all`.