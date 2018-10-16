.. _filters-downhole:

###################
Down-Hole Exclusion
###################

This filter is specifically designed for spot analyses were, because of side-wall ablation effects, data collected towards the end of an ablation will be influenced by data collected at the start of an ablation.

This filter provides a means to exclude all material 'down-hole' of the first excluded contaminant.

For example, to continue the example from the :ref:`filters-defrag` Filter, you may end up with a selection that looks like this:

.. image :: figs/5-fragmented-pre.png

In the first ablation of this example, the defragmentation filter has left four data regions selected.
Because of down-hole effects, data in the second, third and fourth regions will be influenced by the material ablated at the start of the sample.
If there is a contaminant at the start of the sample, this contaminant will also have a minor influence on these regions, and they should be excluded.
This can be done using the Down-Hole Exclusion filter:

.. code :: python

    eg.filter_exclude_downhole(threshold=5)
    # threshold sets the number of consecutive excluded points after which
    # all data should be excluded.

    eg.filter_off()
    eg.filter_on('downhole')

.. image :: figs/5-fragmented-post.png

This filter is particularly useful if, for example, there is a significant contaminated region in the middle of an ablation, but threshold filters do not effectively exclude the post-contaminant region.