.. _reproducibility:

###############
Reproducibility
###############

A key new feature of ``latools`` is making your analysis quantitatively reproducible.
As you go through your analysis, ``latools`` keeps track of everything you're doing in a command log, which stores the sequence and parameters of every step in your data analysis.
These can be exported, alongside an SRM table and your raw data, and be imported and reproduced by an indepedent user.

If you are unwilling to make your entire raw dataset available, it is also possible to export a 'minimal' dataset, which only includes the elements required for your analyses (i.e. any analyte used during filtering or processing, combined with the analytes of interest that are the focus of the reduction).

Minimal Export
==============
The minimum parameters and data to reproduce youre analysis can be exported by::

    eg.minimal_export(ks)


Import and Reproduction
=======================