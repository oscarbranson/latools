.. _srm_file:

#############
The SRM File
#############

The SRM file contains compositional data for standards. To calibrate raw data standards measured during analysis must be in this database.

File Location
=============

The default SRM table is stored in the `resources` directory within the ``latools`` install location.

If you wish to use a different SRM table, the path to the new table must be specified in the configuration file or on a case-by-case basis when calibrating your data.

File Format
===========

The SRM file must be stores as a ``.csv`` file (comma separated values). The full default table has the following columns:

+-------+-------+-------+-------------+-----------------+------+----------------+-------------------+-----+--------+---------+----------+-----------+
|Item   | SRM   | Value | Uncertainty | Uncertainty_Type| Unit | GeoReM_bibcode | Reference         | M   | g/g    | g/g_err | mol/g    | mol/g_err |
+=======+=======+=======+=============+=================+======+================+===================+=====+========+=========+==========+===========+
|Se     |NIST610|138.0  | 42.0        |95%CL            |ug/g  |GeoReM 5211     |Jochum et al 2011  |78.96|0.000138| 4.2e-05 | 1.747e-06|5.319e-07  |
+-------+-------+-------+-------------+-----------------+------+----------------+-------------------+-----+--------+---------+----------+-----------+

For completeness, the full SRM file contains a lot of info. You don't need to complete *all* the columns for a new SRM.

Essential Data
--------------

The *essential* columns that must be included for ``latools`` to use a new SRM are:

+-------+-------+----------+-----------+
|Item   | SRM   | mol/g    | mol/g_err |
+=======+=======+==========+===========+
|Se     |NIST610| 1.747e-06|5.319e-07  |
+-------+-------+----------+-----------+

Other columns may be left blank, although we recommend at least adding a note as to where the values come from in the ``Reference`` column.

Creating/Modifying an SRM File
==============================

To create a new table you can either start from scratch (not recommended), or modify a copy of the existing SRM table (recommended).

To get a copy of the existing SRM table, in Python:

.. code-block:: python

    import latools as la

    la.config.copy_SRM_file('path/to/save/location', config='DEFAULT')

This will create a copy of the default SRM table, and save it to the specified location. You can then modify the copy as necessary.

To use your new SRM database, you can either specify it manually at the start of a new analysis:

.. code-block:: python

    import latools as la

    eg = la.analyse('data/', srm_file='path/to/srmfile.csv')

Or :ref:`specify it as part of a configuration <manage-configurations>`, so that ``latools`` knows where it is automatically.
