############
Introduction
############

.. only:: html

    :Release: |version|
    :Date: |today|

Laser Ablation Tools (``latools``) is a Python toolbox for processing Laser Ablations Mass Spectrometry (LA-MS) data.

Why Use latools?
================
At present, most LA-MS data requires a degree of manual processing.
This introduces subjectivity in data analysis, and independent expert analysts can obtain significantly different results from the same raw data.
At present, there is no standard way of reporting LA-MS data analysis, which would allow an independent user to obtain the same results from the same raw data.
``latools`` is designed to tackle this problem.

``latools`` automatically handles all the routine aspects of LA-MS data reduction:

1. Signal De-spiking
2. Signal / Background Identification
3. Background Subtraction
4. Normalisation to internal standard
5. Calibration to SRMs

These processing steps perform the same basic functions as :ref:`other LA-MS processing software <latools_alternatives>`.
If your end goal is calibrated ablation profiles, these can be exported at this stage for external plotting an analysis.
The real strength of ``latools`` comes in the systematic identification and removal of contaminant signals, and calculation of integrated values for ablation spots.
This is accomplished with two significant new features.

6. Systematic data selection using quantitative data selection `filters`.
7. Analyses can be fully reproduced by independent users through the export and import of analytical sessions.

These features provide the user with systematic tools to reduce laser ablation profiles to per-ablation integrated averages. At the end of processing, ``latools`` can export a set of parameters describing your analysis, along with a minimal dataset containing the SRM table and all raw data required to reproduce your analysis (i.e. only analytes explicitly used during processing).


Very Important Warning
======================
If used correctly, ``latools`` will allow the high-throughput, semi-automated processing of LA-MS data in a systematic, reproducible manner.
Because it is semi-automated, it is very easy to treat it as a 'black box'.
**You must not do this.**
The data you get at the end will only be valid if processed *appropriately*.
Because ``latools`` brings reproducibility to LA-MS processing, it will be very easy for peers to examine your data processing methods, and identify any shortfalls.
In essence: to appropriately use ``latools``, you must understand how it works!

The best way to understand how it works will be to play around with data processing, but before you do that there are a few things you can do to start you off in the right direction:

1. Read and understand the following 'Overview' section. This will give you a basic understanding of the architecture of ``latools``, and how its various components relate to each other.
2. Work through the 'Getting Started' guide. This takes you step-by-step through the analysis of an example dataset.
3. Be aware of the extensive :ref:`documentation <latools_docs>` that describes the action of each function within ``latools``, and tells you what each of the input parameters does.


Overview: Understand ``latools``
================================
``latools`` is a Python 'module'. 
You do not need to be fluent in Python to understand ``latools``, as understanding *what* each processing step does to your data is more important than *how* it is done.
That said, an understanding of Python won't hurt!

Architecture
------------
The ``latools`` module contains two core 'objects' that interact to process LA-MS data:

* :class:`latools.D` is the most 'basic' object, and is a 'container' for the data imported from a single LA-MS data file.
* :class:`latools.analyse` is a higher-level object, containing numerous :class:`latools.D` objects. This is the object you will interact with most when processing data, and it contains all the functions you need to perform your analysis.

This structure reflects the hierarchical nature of LA-MS analysis. 
Each ablation contains an measurements of a single sample (i.e. the 'D' object), but data reduction requires consideration of multiple ablations of samples and standards collected over an analytical session (i.e. the 'analyse' object).
In line with this, some data processing steps (de-spiking, signal/background identification, normalisation to internal standard) can happen at the individual analysis level (i.e. within the :class:`latools.D` object), while others (background subtraction, calibration, filtering) require a more holistic approach that considers the entire analytical session (i.e. at the :class:`latools.analyse` level).

How it works
------------
In practice, you will do all data processing using the :class:`latools.analyse` object, which contains all the data processing functionality you'll need.
To start processing data, you create an :class:`latools.analyse` object and tell it which folder your data are stored in.
:class:`latools.analyse` then imports all the files in the data folder as :class:`latools.D` objects, and labels them by their file names.
The :class:`latools.analyse` object contains all of the :class:`latools.D` objects withing a 'dictionary' called ``latools.analyse.data_dict``, where the each individual :class:`latools.D` object can be accessed via its name.
Data processing therefore works best when ablations of each individual sample or standard are stored in a single data folder, named according to what was measured.

.. todo:: In the near future, ``latools`` will also be able to cope with multiple ablations stored in a single, long data file, as long as a list of sample names is provided to identify each ablation.

When you're performing a processing step that can happen at an individual-sample level (e.g. de-spiking), the :class:`latools.analyse` object passes the task directly on to the :class:`latools.D` objects,
whereas when you're performing a step that requires consideration of the *entire* analytical session (e.g. calibration), the :class:`latools.analyse` object will coordinate the interaction of the different :class:`latools.D` objects (i.e. calculate calibration curves from SRM measurements, and apply them to quantify the compositions of your unknown samples).

Filtering
---------
Finally, there is an additional 'object' attached to each :class:`latools.D` object, specifically for handling data filtering.
This :class:`latools.filt` object contains all the information about filters that have been calculated for the data, and allows you to switch filters on or off for individual samples, or subsets of samples.
This is best demonstrated by example, so we'll return to this in more detail in the :ref:`filtering` section of the :ref:`beginners_guide`

Where next?
===========

Hopefully, you now have a rudimentary understanding of how ``latools`` works, and how it's put together. To start using ``latools``, :ref:`install <install>` it on your system, then work through the step-by-step example in the :ref:`beginners_guide` guide to begin getting to grips with how ``latools`` works. If you already know what you're doing and are looking for more in-depth information, head to :ref:`advanced_topics`, or use the search bar in the top left to find specific information.
