.. _install:

************
Installation
************

.. only:: html

    :Release: |version|
    :Date: |today|

====================
Prerequisite: Python
====================

Before you install ``latools``, you'll need to make sure you have a working installation of Python, preferably version 3.5+. 
If you don't already have this (or are unsure if you do), we reccommend that you install one of the pre-packaged science-oriented Python distributions, like Continuum's `Anaconda <https://www.continuum.io/downloads>`_ or Enthought's `Canopy <https://www.enthought.com/products/canopy/>`_.
These provide a working copy of Python, and most of the modules that ``latools`` relies on.

If you already have a working Python installation or don't want to install one of the pre-packaged Python distributions, everything below `should` work.

======================
Installing ``latools``
======================

There are two ways to install ``latools``. We recommend the first method, which will allow you to easily keep your installation of ``latools`` up to date with new developments.

----------------------------------
1. Online Repository (Recommended)
----------------------------------

All the code for ``latools`` is open source, and hosted on `Github <https://github.com/>`_. You can look at, and contribute to the code by visiting the `latools Github page <https://github.com/oscarbranson/latools>`_.

The simplest way to install the latest version of ``latools`` and keep it up to date with new releases, is by grabbing it directly from the the github repository. To do this, you need to `install git <https://git-scm.com/downloads>`_. Make sure you've done this before continuing to the next step.

Once git is installed, you can install ``latools`` directly from the online project repository using the ``pip`` package manager:

1. Open a terminal window.
2. Type (or copy & paste):

.. code-block:: bash

    pip install git+https://github.com/oscarbranson/latools.git@master

3. Press [Return]

This will download and install the latest version of ``latools`` from Github. In the future if you'd like to update ``latools`` to the latest version it's as adding ``--upgrade`` to the end of the code above.

.. Tip:: The ``@master`` at the end of the command installs the most up-to-date version of the software. If you want to install a specific version of ``latools``, replace ``@master`` with the version number (e.g. ``@v0.2.2a`` will get you the very first release of ``latools``).

-------------------
2. Pre-built Binary
-------------------

This method is more 'manual', and is really only for people who want to install a specific, past version of ``latools``, or don't want to install git.

1. Go to the `Distributions <https://github.com/oscarbranson/latools/tree/master/dist>`_ page of the Github project, and download a pre-built version of ``latools`` to a ``convenient_folder/``. Download ``latools-latest.tar.gz`` for the most up-to-date version, or ``latools-[version.number].tar.gz`` for a specific, past version. 
2. Open a terminal window, and navigate to ``convenient_folder/``.
3. Type (or copy & paste):

.. code-block:: bash

    pip install latools-latest.tar.gz

or

.. code-block:: bash

    pip install latools-[version.number].tar.gz

4. Press [Return]

This will install the downloaded version of ``latools`` on your system, and you can now delete the .tar.gz file you saved to ``conventient_location/``.

If you want to update ``latools`` in future, or install a different version, you'll have to repeat these steps, and re-download a new .tar.gz file.

==========
Next Steps
==========

If this is your first time, read through the :ref:`getting_started` guide. Otherwise, get analysing!