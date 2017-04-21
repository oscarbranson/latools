.. _structure:

###############################################
Package Structure
###############################################

``latools`` consists of three interlinked 'objects' (classes), which contain all your data, the functions necessary to process it and a record of all processing done to the data:

* :mod:`latools.analyse`
	This is a high-level container for processing entire datasets.
	It is designed to facilitate the simultaneous importing and processing of numerous samples with corresponding standards.
	Each data file loaded into an :mod:`latools.analyse` instance is contained in a :mod:`latools.D` class.
	Calls to data processing `methods` are passed directly to corresponding methods of the data-containing :mod:`latools.D` class, where the work actually happens.
	Only two `methods` in the :mod:`latools.analyse` class are not contained within the :mod:`latools.D` class: :meth:`latools.analyse.calibrate`, which pulls together multiple standards and applies it to the data, and :meth:`latools.analyse.sample_stats`, which collates statistics for all samples.
* :mod:`latools.D`
	This contains an individual data file, and all the required `methods` to process that data.
* :mod:`latools.filt`
	This class contains all the data selection filters created for a particular sample, information about the filter, and a record of whether it is 'on' or 'off'. These individual components can then be combined to generate a data mask from the selected filters. This class comes into play then creating data filters (any ``filter_`` command), and is used when any function with a ``filt`` parameter is called.

.. note:: All of the above are python 'classes'. Each has `attributes` (variables stored inside it) and `methods` (functions that take inputs, and act on the attributes). When you create a variable from one of these classes, you are making a `class instance` - a version of that class with a particular set of inputs.

When an an instance of :mod:`latools.analyse` is created, you give it a path to a ``data_folder``.
The class searches this folder for data files, and creates a :mod:`latools.D` instance for each data file that it finds.
These :mod:`latools.D` instances are stored within the in ``data_dict`` within :mod:`latools.analyse`.
When creating a :mod:`latools.analyse` instance it also reads the :ref:`configuration file <cfg_file>`, which contains the paths to the :ref:`SRM data table <srm_file>` and :ref:`data format description <advanced_data_formats>` that are essential for analysis.

The :mod:`latools.analyse` class is essentially a 'wrapper' for dealing with numerous data files at once.
The ability to process mutliple data files is essential if you wish to calibrate your data (i.e. you must have a number of samples, and at least one standard in the .

When each instance of :mod:`latools.D` is created, it is given a single ``data_file``, and a dict containing
