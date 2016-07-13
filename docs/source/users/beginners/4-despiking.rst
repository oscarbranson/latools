.. _despiking:

##############
Data Despiking
##############

After your data is imported, you may despike your data, to remove physically unrealistic outliers from your data (i.e. higher or lower than is physically possible based on your system setup).

Two despiking methods are available:

* :meth:`~latools.D.expdecay_despiker` removes low outliers, based on the signal washout time of your laser cell. The signal washout is described using an exponential decay function. If the measured signal decreases faster than physically possible based on your laser setup, these points are removed, and replaced with the average of the adjacent values.
* :meth:`~latools.D.noise_despiker` removes high outliers by calculating a rolling mean and standard deviation, and replacing points that are greater than `n` standard deviations above the mean with the mean of the adjacent data points.

These functions can both be applied at once, using :meth:`~latools.analyse.despike`::

	eg.despike()

By default, this applies :meth:`~latools.D.expdecay_despiker` followed by :meth:`~latools.D.noise_despiker` to all samples. You can specify several parameters that change the behaviour of these despiking routines.

.. tip:: The exponential decay constant used by :meth:`~latools.D.expdecay_despiker` will be specific to your laser setup. If you don't know what this is, :meth:`~latools.analyse.despike` determines it automatically by fitting an exponential decay function to the washout phase of measured SRMs in your data. You can look at this fit by passing ``exponent_plot=True`` to the function.