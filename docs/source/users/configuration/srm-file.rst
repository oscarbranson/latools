.. _srm_file:

#############
The SRM file
#############

The SRM file contains compositional data for standards. To calibrate raw data standards measured during analysis must be in this database.

File Location
=============

The default SRM table is stored in the `resources` directory within the ``latools`` install location.

If you wish to use a different SRM table, the path to the new table must be specified in the configuration file or on a case-by-case basis when calibrating your data.

To access the SRM table for a particular configuration, you can :func:`~latools.helpers.config.copy_SRM_file` to a new location, and modify the copied file as desired. The new file can then be used during a single analysis (be specifying it in the ``srm_file`` argument of :class:`~latools.latools.analyse`), or incorporated as part of ``latools`` system-wide :ref:`configurations`.

File Format
===========

The SRM file must be stores as a ``.csv`` file (comma separated values). The full default table has the following columns:

+-------+-------+-------+-------------+-----------------+------+----------------+-------------------+-----+--------+---------+----------+-----------+
|Item   | SRM   | Value | Uncertainty | Uncertainty_Type| Unit | GeoReM_bibcode | Reference         | M   | g/g    | g/g_err | mol/g    | mol/g_err |
+=======+=======+=======+=============+=================+======+================+===================+=====+========+=========+==========+===========+
|Se     |NIST610|138.0  | 42.0        |95%CL            |ug/g  |GeoReM 5211     |Jochum et al 2011  |78.96|0.000138| 4.2e-05 | 1.747e-06|5.319e-07  |
+-------+-------+-------+-------------+-----------------+------+----------------+-------------------+-----+--------+---------+----------+-----------+

Essential Data
==============

For completeness, the full SRM file contains a lot of info. The essential information that must be included for ``latools`` to use the SRM is:

+-------+-------+----------+-----------+
|Item   | SRM   | mol/g    | mol/g_err |
+=======+=======+==========+===========+
|Se     |NIST610| 1.747e-06|5.319e-07  |
+-------+-------+----------+-----------+

Other columns may be left blank when adding new SRMS, although we recommend at least adding a note as to where the values come from in the ``Reference`` column.