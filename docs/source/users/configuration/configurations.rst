.. _manage-configurations:

#######################
Managing Configurations
#######################

A 'configuration' is how ``latools`` stores the location of a data format description and SRM file to be used during data import and analysis. In labs working with a single LA-ICPMS system, you can set a default configuration, and then leave this alone. If you're running multiple LA-ICPMS systems, or work with different data formats, you can specify multiple configurations, and specify which one you want to use at the start of analysis, like this:

.. code-block:: python

    import latools as la

    eg = la.analyse('data', config='MY-CONFIG-NAME')

Viewing Existing Configurations
-------------------------------

You can see a list of currently defined configurations at any time:

.. code-block:: python

    import latools as la

    la.config.print_all()

        Currently defined LAtools configurations:
        
        REPRODUCE [DO NOT ALTER]
        dataformat: /latools/install/location/resources/data_formats/repro_dataformat.json
        srmfile: /latools/install/location/resources/SRM_GeoRem_Preferred_170622.csv
        
        UCD-AGILENT [DEFAULT]
        dataformat: /latools/install/location/resources/data_formats/UCD_dataformat.json
        srmfile: /latools/install/location/resources/SRM_GeoRem_Preferred_170622.csv

Note how each configuration has a ``dataformat`` and ``srmfile`` specified.
The ``REPRODUCE`` configuration is a special case, and should not be modified.
All other configurations are listed by name, and the default configuration is marked (in this case there's only one, and it's the default).
If you *don't* specify a configuration when you start an analysis, it will use the default one.

Creating a Configuration
------------------------

Once you've created your own :ref:`dataformat description <data_formats>` and/or :ref:`SRM File <srm_file>`, you can set up a configuration to use them:

.. code-block:: python

    import latools as la

    # create new config
    la.config.create('MY-FANCY-CONFIGURATION',
                     srmfile='path/to/srmfile.csv',
                     dataformat='path/to/dataformat.json',
                     base_on='DEFAULT', make_default=False)

    # check it's there
    la.config.print_all()

        Currently defined LAtools configurations:
        
        REPRODUCE [DO NOT ALTER]
        dataformat: /latools/install/location/resources/data_formats/repro_dataformat.json
        srmfile: /latools/install/location/resources/SRM_GeoRem_Preferred_170622.csv
        
        UCD-AGILENT [DEFAULT]
        dataformat: /latools/install/location/resources/data_formats/UCD_dataformat.json
        srmfile: /latools/install/location/resources/SRM_GeoRem_Preferred_170622.csv
    
        MY-FANCY-CONFIGURATION
        dataformat: path/to/dataformat.json
        srmfile: path/to/srmfile.csv

You should see the new configuration in the list, and unless you specified ``make_default=True``, the default should not have changed.
The ``base_on`` argument tells ``latools`` which existing configuration the new one is based on.
This only matters if you're only specifying one of ``srmfile`` or ``dataformat`` - whichever you *don't* specify is copied from the ``base_on`` configuration.

.. important:: When making a configuration, make sure you store the dataformat and srm files somewhere permanent - if you move or rename these files, the configuration will stop working.


Modifying a Configuration
-------------------------
Once created, configurations can be modified...

.. code-block:: python

    import latools as la

    # modify configuration
    la.config.update('MY-FANCY-CONFIGURATION', 'srmfile', 'correct/path/to/srmfile.csv')

        Are you sure you want to change the srmfile parameter of the MY-FANCY-CONFIGURATION configuration?
        It will be changed from:
            path/to/srmfile.csv
        to:
            correct/path/to/srmfile.csv
        > [N/y]: y
        Configuration updated!

    # check it's updated
    la.config.print_all()

        Currently defined LAtools configurations:
        
        REPRODUCE [DO NOT ALTER]
        dataformat: /latools/install/location/resources/data_formats/repro_dataformat.json
        srmfile: /latools/install/location/resources/SRM_GeoRem_Preferred_170622.csv
        
        UCD-AGILENT [DEFAULT]
        dataformat: /latools/install/location/resources/data_formats/UCD_dataformat.json
        srmfile: /latools/install/location/resources/SRM_GeoRem_Preferred_170622.csv
    
        MY-FANCY-CONFIGURATION
        dataformat: path/to/dataformat.json
        srmfile: correct/path/to/srmfile.csv


Deleting a Configuration
------------------------
Or deleted...

.. code-block:: python
    :linenos:

    import latools as la

    # delete configuration
    la.config.delete('MY-FANCY-CONFIGURATION')
    
        Are you sure you want to delete the MY-FANCY-CONFIGURATION configuration?
        > [N/y]: y
        Configuration deleted!

    # check it's gone
    la.config.print_all()

        Currently defined LAtools configurations:
        
        REPRODUCE [DO NOT ALTER]
        dataformat: /latools/install/location/resources/data_formats/repro_dataformat.json
        srmfile: /latools/install/location/resources/SRM_GeoRem_Preferred_170622.csv
        
        UCD-AGILENT [DEFAULT]
        dataformat: /latools/install/location/resources/data_formats/UCD_dataformat.json
        srmfile: /latools/install/location/resources/SRM_GeoRem_Preferred_170622.csv