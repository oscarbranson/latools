.. _reproducibility:

###############
Reproducibility
###############

A key new feature of ``latools`` is making your analysis quantitatively reproducible.
As you go through your analysis, ``latools`` keeps track of everything you're doing in a command log, which stores the sequence and parameters of every step in your data analysis.
These can be exported, alongside an SRM table and your raw data, and be imported and reproduced by an independent user.

If you are unwilling to make your entire raw dataset available, it is also possible to export a 'minimal' dataset, which only includes the elements required for your analyses (i.e. any analyte used during filtering or processing, combined with the analytes of interest that are the focus of the reduction).

Minimal Export
==============
The minimum parameters and data to reproduce you're analysis can be exported by::

    eg.minimal_export(ks)

This will create a new folder inside the ``data_export`` folder, called ``minimal export``. This will contain your complete dataset, or a subset of your dataset containing only the analytes you specify, the SRM values used to calibrate your data, and a ``.log`` file that contains a record of everything you've done to your data.

This entire folder should be compressed (e.g. .zip), and included alongside your publication.

.. tip:: When someone else goes to reproduce your analysis, `everything` you've done to your data will be re-calculated. However, analysis is often an iterative process, and an external user does not need to experience `all` these iterations. We therefore recommend that after you've identified all the processing and filtering steps you want to apply to the data, you reprocess your entire dataset using `only` these steps, before performing a minimal export.

Import and Reproduction
=======================
To reproduce someone else's analysis, download a compressed minimal_export folder, and unzip it.
Next, in a new python window, run::
    
    import latools as la

    rep = la.reproduce('path/to/analysis.log')

This will reproduce the entire analysis, and call it 'rep'. You can then experiment with different data filters and processing techniques to see how it modifies their results.