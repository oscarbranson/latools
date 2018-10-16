.. _filters-clustering:

##########
Clustering
##########

The clustering filter provides a convenient way to separate compositionally distinct materials within your ablations, using multi-dimensional clustering algorithms.

Two algorithms are currently available in ``latools``:
* `K-Means <http://scikit-learn.org/stable/modules/clustering.html#k-means>`_ will divide the data up into N groups of equal variance, where N is a known number of groups.
* `Mean Shift <http://scikit-learn.org/stable/modules/clustering.html#mean-shift>`_ will divide the data up into an arbitrary number of clusters, based on the characteristics of the data.

For an in-depth explanation of these algorithms and how they work, take a look at the `Scikit-Learn clustering pages <http://scikit-learn.org/stable/modules/clustering.html>`_.

For most cases, we recommend the K-Means algorithm, as it is relatively intuitive and produces more predictable results.

2D Clustering Example
=====================

For illustrative purposes, consider some 2D synthetic data:

.. figure :: ./figs/3-clustering_synthetic.png
    :align: center

    The left panel shows two signals (A and B) which transition from an initial state where B > A (<40 s) to a second state where A > B (>60 s).
    In laser ablation terms, this might represent a change in concentration of two analytes at a material boundary.
    The right panel shows the relationship between the A and B signals, ignoring the time axis.

Two 'clusters' in composition are evident in the data, which can be separated by clustering algorithms.

.. figure :: ./figs/3-clustering-example.png
    :scale: 50%
    :align: center
    
    In the left panel, the K-Means algorithm has been used to find the boundary between two distinct materials.
    In the right panel, the Mean Shift algorithm has automatically detected three materials.


The main difference here is that the MeanShift algorithm has identified the transition points (orange) as a separate cluster.

Once the clusters are identified, they can be translated back into the time-domain to separate the signals in the original data:

.. figure :: ./figs/3-clustering_solution.png
    :scale: 50%
    :align: center

    Horizontal bars denote the regions identified by the K-Means and MeanShift clustering algorithms.

For simplicity, the example above considers the relationship between two signals (i.e. 2-D).
When creating a clustering filter on real data, multiple analytes may be included (i.e. N-D).
The only limits on the number of analytes you can include is the number of analytes you've measured, and how much RAM your computer has.

If, for example, your ablation contains three distinct materials with variations in five analytes, you might create a K-Means clustering filter that takes all five analytes, and separates them into three clusters.

When to use a Clustering Filter
===============================

Clustering filters should be used to discriminate between clearly different materials in an analysis.
Results will be best when they are based on signals with clear sharp changes, and high signal/noise (as in the above example).

Results will be poor when data are noisy, or when the transition between materials is very gradual.
In these cases, clustering filters may still be useful after you have used other filters to remove the transition regions - for example gradient-threshold or correlation filters.

Clustering Filter Design
========================

A good place to start when creating a clustering filter is by looking at a cross-plot of your analytes:

.. code-block :: python
    
    eg.crossplot()

.. figure :: ./figs/3-crossplot.png
    :align: center

    A crossplot showing relationships between all measured analytes in all samples.
    Data are presented as 2D histograms, where the intensity of colour relates to the number of data points in that pixel.
    In this example, a number of clusters are evident in both Sr88 and Mn55, which are candidates for clustering filters.

A crossplot provides an overview of your data, and allows you to easily identify relationships between analytes.
In this example, multiple levels of Sr88 concentration are evident, which we might want to separate.
Three Sr88 groups are evident, so we will create a K-Means filter with three clusters:

.. code-block :: python

    eg.filter_clustering(analyte='Sr88', level='population', method='kmeans', n_clusters=3)

    eg.filter_status()

    > Subset: 0
    > Samples: Sample-1, Sample-2, Sample-3
    > 
    > n  Filter Name      Mg24   Mg25   Al27   Ca43   Ca44   Mn55   Sr88   Ba137  Ba138  
    > 0  Sr88_kmeans_0    False  False  False  False  False  False  False  False  False  
    > 1  Sr88_kmeans_1    False  False  False  False  False  False  False  False  False  
    > 2  Sr88_kmeans_2    False  False  False  False  False  False  False  False  False  

The clustering filter has used the population-level data to identify three clusters in Sr88 concentration, and created a filter based on these concentration levels.

We can directly see the influence of this filter:

.. code-block :: python

    eg.crossplot_filters('Sr88_kmeans')

.. figure :: ./figs/3-crossplot-filters.png
    :align: center

    A crossplot of all the data, highlighting the clusters identified by the filter.

.. tip :: You can use ``crossplot_filter`` to see the effect of any created filters - not just clustering filters!

Here, we can see that the filter has picked out three Sr concentrations well, but that these clusters don't seem to have any systematic relationship with other analytes.
This suggests that Sr might not be that useful in separating different materials in these data.
(In reality, the Sr variance in these data comes from an incorrectly-tuned mass spec, and tells us nothing about the sample!)