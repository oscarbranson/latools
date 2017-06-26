.. _import:

##############
Importing Data
##############

Once you have Python running in your ``latools_demo/`` directory and have unpacked the :ref:`example_data`, you're ready to start an ``latools`` analysis session.
To do this, run:

.. literalinclude:: ../../../../tests/test_beginnersGuide.py
   :language: python
   :dedent: 4
   :lines: 15-18

This imports all the data files within the ``data/`` folder into an :class:`latools.analyse` object called ``eg``, along with several parameters describing the dataset and how it should be imported:

* ``config='DEFAULT'``: The configuration contains information about the data file format and the location of the SRM table. Multiple configurations can be set up and chosen during data import, allowing ``latools`` to flexibly work with data from different instruments.
* ``internal_standard='Ca43'``: This specifies the internal standard element within your samples. The internal standard is used at several key stages in analysis (signal/background identification, normalisation), and should be relatively abundant and homogeneous in your samples.
* ``srm_identifier='STD'``: This identifies which of your analyses contain standard reference materials (SRMs). Any data file with 'STD' in its name will be flagged as an SRM measurement.

If it has worked correctly, you should see the output::

    latools analysis using "DEFAULT" configuration:
      5 Data Files Loaded: 2 standards, 3 samples
      Analytes: Mg24 Mg25 Al27 Ca43 Ca44 Mn55 Sr88 Ba137 Ba138
      Internal Standard: Ca43

In this output, ``latools`` reports that 5 data files were imported from the ``data/`` directory, two of which were standards (names contained 'STD'), and tells you which analytes are present in these data.
Each of the imported data files is stored in a :class:`latools.D` object, which are 'managed' by the :class:`latools.analyse` object that contains them.

Check inside the ``latools_demo`` directory. 
There should now be two new folders called ``reports_data/`` and ``export_data/`` alongside the ``data/`` folder.
Note that the '_data' suffix will be the same as the name of the folder that contains your data - i.e. the names of these folders will change, depending on the name of your data folder.
``latools`` saves data and plots to these folders throughout analysis:

* ``data_export`` will contain exported data: traces, per-ablation averages and minimal analysis exports.
* ``data_reports`` will contain all plots generated during analysis.

