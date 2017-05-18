.. _despiking:

##############
Data De-spiking
##############

The first step in data reduction is the 'de-spike' the raw data to remove physically unrealistic outliers from the data (i.e. higher than is physically possible based on your system setup).

Two de-spiking methods are available:

* :meth:`~latools.D.expdecay_despiker` removes low outliers, based on the signal washout time of your laser cell. The signal washout is described using an exponential decay function. If the measured signal decreases faster than physically possible based on your laser setup, these points are removed, and replaced with the average of the adjacent values.
* :meth:`~latools.D.noise_despiker` removes high outliers by calculating a rolling mean and standard deviation, and replacing points that are greater than `n` standard deviations above the mean with the mean of the adjacent data points.

These functions can both be applied at once, using :meth:`~latools.analyse.despike`::

    eg.despike(expdecay_despiker=True, 
               noise_despiker=True)

By default, this applies :meth:`~latools.D.expdecay_despiker` followed by :meth:`~latools.D.noise_despiker` to all samples. 
You can specify several parameters that change the behaviour of these de-spiking routines.

The :meth:`~latools.D.expdecay_despiker` relies on knowing the exponential decay constant that describes the washout characteristics of your laser ablation cell.
If this values is missing (as here), ``latools`` calculates it by fitting an exponential decay function to the internal standard at the on-off laser transition at the end of ablations of standards. 
If this has been done, you will be informed. In this case, it should look like::

    Calculating exponential decay coefficient
    from SRM Ca43 washouts...
      -2.28

.. tip:: The exponential decay constant used by :meth:`~latools.D.expdecay_despiker` will be specific to your laser setup. If you don't know what this is, :meth:`~latools.analyse.despike` determines it automatically by fitting an exponential decay function to the washout phase of measured SRMs in your data. You can look at this fit by passing ``exponent_plot=True`` to the function.