import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp

def mass_bias(self, analyte_ratios):
    
    if isinstance(analyte_ratios, str):
        analyte_ratios = [analyte_ratios]

    if analyte_ratios is None:
        # analytes = self._calib_analytes
        analyte_ratios = self.analytes_sorted(self._srm_id_analyte_ratios)
        # analytes = self.analytes_sorted(self.analytes.difference([self.internal_standard]))

    n = len(analyte_ratios)
    
    fig, axs = plt.subplots(n, 1, figsize=(10, 2*n), sharex=True)
    if n == 1:
        axs = [axs]

    for a, ax in zip(analyte_ratios, axs):
        # ax.scatter(self.calib_params.index, unp.nominal_values(self.calib_params[a]), color=self.cmaps[a])
        ax.errorbar(self.calib_params.index, unp.nominal_values(self.calib_params[a]).flat, unp.std_devs(self.calib_params[a]).flat, fmt='o', color=self.cmaps[a])

        ax.set_xlim(*ax.get_xlim())
        xn = np.linspace(*ax.get_xlim(), 200)
        
        pred = self.calib_ps[a]['m'].new(xn)
        
        ax.plot(xn, unp.nominal_values(pred), color=self.cmaps[a])
        ax.fill_between(xn, unp.nominal_values(pred) - unp.std_devs(pred), unp.nominal_values(pred) + unp.std_devs(pred), color=self.cmaps[a], alpha=0.2)

        ax.set_ylabel(a)
    
    axs[-1].set_xlabel('Time (s)')