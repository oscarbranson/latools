.. _filtering:

############################
Data Selection and Filtering
############################

The data are now background corrected, normalised to an internal standard, and calibrated.
Now we can get into some of the new features of ``latools``, and start thinking about **data filtering**.

What is Data Filtering?
=======================
Laser ablation data are spatially resolved.
In heterogeneous samples, this means that the concentrations of different analytes will change within a single analysis.
This compositional heterogeneity can either be natural and expected (e.g. Mg/Ca variability in foraminifera), or caused by compositionally distinct contaminant phases indluded in the sample structure.
If the end goal of your analysis is to get integrated compositional estimates for each ablation analysis, how you deal with sample heterogeneity becomes central to data processing, and can have a profound effect on the resulting integrated values.
So far, heterogeneous samples tend to be processed manually, by choosing regions to integrate by eye, based on a set of criteria and knowledge of the sample material.
While this is a valid approach to data reduction, it is *highly* irreproducible: if two 'expert analysts' were to process the data, the resulting values would not be quantitatively identical.
Reproducibility is fundamental to sound science, and the inability to reproduce integrated values from identical raw data is a fundamental flaw in Laser Ablation studies.
In short, this is a serious problem.

To get round this, we have developed 'Data Filters'.
Data Filters are systematic selection criteria, which can be applied to all samples to select specific regions of ablation data for integration.
For example, the analyst might apply a filter that removes all regions where a particular analyte exceeds a threshold concentration, or exlcude regions where two contaminant elements co-vary through the ablation.
Ultimately, the choice of selection criteria remains entirely subjective, but because these criteria are quantitative they can be uniformly applied to all specimens, and most importantly, reported and reproduced by an independent researcher.
This removes significant possibilities for 'human error' from data analysis, and solves the long-standing problem of reproducibility in LA-MS data processing.

Data Filters
============
``latools`` includes several filtering functions, which can be applied in any order, repetitively and in any sequence.
By their combined application, it should be possible to islate any specific region within the data that is systematically identified by patterns in the ablation profile.
These filter are (in order of increasing complexity):

* :meth:`~latools.analyse.filter_threshold`: Creates two filter keys identifying where a specific analyte is above or below a given threshold.
* :meth:`~latools.analyse.filter_distribution`: Finds separate `populations` within the measured concentration of a single analyte within by creating a Probability Distribution Function (PDF) of the analyte within each sample. Local minima in the PDF identify the boundaries between distinct concentrations of that analyte within your sample.
* :meth:`~latools.analyse.filter_clustering`: A more sophisticated version of :meth:`~latools.analyse.filter_distribution`, which uses data clustering algorithms from the `sklearn <http://scikit-learn.org/>`_ module to identify compositionally distinct 'populations' in your data. This can consider multiple analytes at once, allowing for the robust detection of distinct compositional zones in your data using n-dimensional clustering algorithms.
* :meth:`~latools.analyse.filter_correlation`: Finds regions in your data where two analytes correlate locally. For example, if your analyte of interest strongly covaries with an analyte that is a known contaminant indicator, the signal is likely contaminated, and shoule be discarded.

For a full account of these filters, how they work and how they can be used, see :ref:`advanced_filtering`.

Simple Demonstration
====================

Choosing a filter
-----------------
The foraminifera analysed in this example dataset are from culture experiments and have been thoroughly cleaned.
There should not be any contaminants in these samples, and filtering is relatively straightforward.
The first step in choosing a filter is to *look* at the data.
You can look at the calibrated profiles manually to get a sense of the patterns in the data (using ``trace_plots()``):

.. image:: ./figs/calibrated_Sample-3.png

Or alternatively, you can make a 'crossplot' (using ``eg.crossplot()``) of your data, to examine how all the trace elements in your samples relate to each other:

.. image:: ./figs/crossplot.png

This plots every analyte in your ablation profiles, plotted against every other analyte. The axes in each panel are described by the diagonal analyte names. The colour intensity in each panel corresponds to the data density (i.e. it's a 2D histogram!).

Within these plots, you should focus on the behaviour of 'conaminant indicator' elements, i.e. elements that are normally within a known concentration range, or are known to be associated with a possible contaminant phase.
As these are foraminifera, we will pay particularly close attention to the concentrations of Al, Mn and Ba in the ablations, which are all normally low and homogeneous in foraminifera samples, but are prone to contamination by clay particles.
In these samples, the Ba and Mn are relatively uniform, but the Al increases towards the end of each ablation.
This is because the tape that the specimens were mounted on contains a significant amount of Al, which is picked up by the laser as it ablates through the shell.
We know from experience that the tape tends to have very low concentration of other elements, but to be safe we should exclude regions with hi Al/Ca from our analysis.

Creating a Filter
-----------------
We wouldn't expect cultured foraminifera to have a Al/Ca of ~100 µmol/mol, so we therefore want to remove all data from regions with an Al/Ca above this.
We'll do this with a threshold filter::

    eg.filter_threshold(analyte='Al27', threshold=100e-6)  # remember that all units are in mol/mol!

This goes through *all* the samples in our analysis, and works out which analyses have an Al/Ca both greater than and less than 200 µmol/mol.
This function calculates the filters, but does not apply them - that happens later.
You can check which filters have been calculated, and which are active for individual analytes by typing::

    eg.filter_status()

Which will return::

    Subset All_Samples:
    Samples: Sample-1, Sample-2, Sample-3

    n  Filter Name          Mg24   Mg25   Al27   Ca43   Ca44   Mn55   Sr88   Ba137  Ba138  
    0  Al27_thresh_below    False  False  False  False  False  False  False  False  False  
    1  Al27_thresh_above    False  False  False  False  False  False  False  False  False

This produces a grid showing the filter numbers, names, and which analytes they are active for (for each analyte False = inactive, True = active). 
The ``filter_threshold`` function has generated two filters: one identifying data above the threshold, and the other below it.
Finally, notice also that it says 'Subset: All_Samples' at the top, and lists which samples they are. 
You can apply different filters to different subsets of samples... We'll come back to this later.
This display shows all the filters you've calculated, and which analytes they are applied to. 

Before we think about applying the filter, we should check what it has acutally done to the data.

.. note:: Filters do not delete any data. They simply create a *mask* which tells latools funcions which data to use, and which to ignore.

Checking a Filter
-----------------
You can do this in three ways:

1. Plot the traces, with ``filt=True``. This plots the calibrated traces, with areas excluded by the filter shaded out in grey. Specifying ``filt=True`` shows the net effect of all active filters. By setting ``filt`` as a number or filter name, the effect of one individual filter will be shown.
2. Crossplot with ``filt=True`` will generate a new crossplot containing only data that remains after filtering. This can be useful for refining filter choices during multiple rounds of filtering. You can also set ``filt`` to be a filter name or a number, as with trace plotting.
3. The most sophisticated way of looking at a filter is by creating a 'filter_report'. This generates a plot of each analysis, showing which regions are selected by particular filters::

    eg.filter_reports(analytes='Al27', filt_str='thresh')

Where ``analytes`` specifies which analytes you want to see the influence of the filters on, and ``filt_str`` identifies which filters you want to see.
``filt_str`` supports partial filter name matching, so 'thresh' will pick up any filter with 'thresh' in the name - i.e. if you'd calculated multiple thresholds, it would plot each on a different plot.
If all has gone to plan, it will look something like this:

.. image:: ./figs/thresh_Sample-3_Al27.png

In the case of a threshold filter report, the dashed line shows the threshold, and the legend identifies which data regions are selected by the different filters (in this case '0_below' or '1_above').
The reports for different types of filter are slightly different, and often include numerous groups of data.
In this case, the 100 µmol/mol threshold seems to do a good job of excluding extraneously high Al/Ca values, so we'll use the '0_Al27_thresh_below' filter to select these data.

Applying a Filter
-----------------
Once you've identified which filter you want ot apply, you must turn that filter 'on' using::

    eg.filter_on(filt=0)

Where ``filt`` can either be the filter number (as here), or a partially matching string (e.g. you could use ``filt='below'`` to turn on all filters with 'below' in the name).
There is also a counterpart ``eg.filter_off()`` function, which works in the inverse.
These functions will turn the threshold filter on for all analytes measured in all samples.
The status of filtering can be checked with ``eg.filter_status()`` (as above), which should now return::

    Subset All_Samples:
    Samples: Sample-1, Sample-2, Sample-3

    n  Filter Name          Mg24   Mg25   Al27   Ca43   Ca44   Mn55   Sr88   Ba137  Ba138  
    0  Al27_thresh_below    True   True   True   True   True   True   True   True   True   
    1  Al27_thresh_above    False  False  False  False  False  False  False  False  False  

In some cases, you might have a sample where one analyte is effected by a contaminant that does not alter other analytes.
If this is the case, you can switch a filter on or off for a specific analyte::

    eg.filter_off(filt=0, analyte='Mg25')

    eg.filter_status()


    Subset All_Samples:
    Samples: Sample-1, Sample-2, Sample-3

    n  Filter Name          Mg24   Mg25   Al27   Ca43   Ca44   Mn55   Sr88   Ba137  Ba138  
    0  Al27_thresh_below    True   False  True   True   True   True   True   True   True   
    1  Al27_thresh_above    False  False  False  False  False  False  False  False  False  

Notice how filter '0' is now deactivated for Mg25.

Finally, let's return to the 'Subsets', which we skpped over earlier.

Sample Subsets
--------------
It is quite common to analyse distinct sets of samples in the same analytical session.
To accommodate this, you can create data 'subsets' during analysis, and treat them in different ways.
For example, imagine that 'Sample-1' in our test dataset was a different type of sample, that needs to be filtered in a different way.
We can identify this as a subset by::

    eg.make_subset(samples='Sample-1', name='set1')
    eg.make_subset(samples=['Sample-2', 'Sample-3'], name='set2')

And filters can be turned on and off independently for each subset::

    eg.filter_on(filt=0, subset='set1')
    eg.filter_off(filt=0, subset='set2')

    eg.filter_status(subset=['set1', 'set2'])

    Subset set1:
    Samples: Sample-1

    n  Filter Name          Mg24   Mg25   Al27   Ca43   Ca44   Mn55   Sr88   Ba137  Ba138  
    0  Al27_thresh_below    True   True   True   True   True   True   True   True   True   
    1  Al27_thresh_above    False  False  False  False  False  False  False  False  False  

    Subset set2:
    Samples: Sample-2, Sample-3

    n  Filter Name          Mg24   Mg25   Al27   Ca43   Ca44   Mn55   Sr88   Ba137  Ba138  
    0  Al27_thresh_below    False  False  False  False  False  False  False  False  False  
    1  Al27_thresh_above    False  False  False  False  False  False  False  False  False

To see which subsets have been defined::

    eg.subsets

    {'All_Analyses': ['Sample-1', 'Sample-2', 'Sample-3', 'STD-1', 'STD-2'],
     'All_Samples': ['Sample-1', 'Sample-2', 'Sample-3'],
     'STD': ['STD-1', 'STD-2'],
     'set1': ['Sample-1'],
     'set2': ['Sample-2', 'Sample-3']}

.. note:: The filtering above is relatively simplistic. More complex filters require quite a lot more thought and care in their application. For examples of how to use clustering, distribution and correlation filters, see the :ref:`Advanced Filtering <advanced_filtering>` section.


