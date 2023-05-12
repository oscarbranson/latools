import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score

import uncertainties as un
import uncertainties.unumpy as unp

nom = unp.nominal_values
err = unp.std_devs

from ..helpers import Bunch
from ..helpers.stat_fns import uncertainty_to_std

def make_d11b_offset_table(self, xvar, b_correction_srms=['MACS', 'JCT', 'JCP'], yvar='11B_10B', uncertainty_metric='SE'):
    
    self.d11b_xvar = xvar
    self.d11b_yvar = yvar
    self.d11b_correction_srms = b_correction_srms
    
    self.make_subset(match=b_correction_srms, name='b_correction_srms')

    b_srm_data = self._srmdat_raw.loc[b_correction_srms]
    b_srm_data = b_srm_data.loc[b_srm_data.Item == yvar]
    self.d11b_srm_data = b_srm_data

    b_offset_table = pd.DataFrame(index=self.subsets['b_correction_srms'], columns=['uTime', 'x', 'measured', 'srm_id', 'srm_val'])

    for s in b_offset_table.index:
        sam = self.data[s]
        
        mdat = sam.data['calibrated'][yvar]
        measured = np.nanmean(mdat)
        if uncertainty_metric.lower() == 'se':
            n_measured = np.sum(~unp.isnan(mdat))
            measured = un.ufloat(measured.nominal_value, measured.std_dev() / np.sqrt(n_measured))
        
        b_offset_table.loc[s, 'measured'] = measured
        
        xdat = sam.data['ratios'][xvar]
        x = np.nanmean(xdat)
        if uncertainty_metric.lower() == 'se':
            n_measured = np.sum(~unp.isnan(xdat))
            x = un.ufloat(x.nominal_value, x.std_dev() / np.sqrt(n_measured))
        b_offset_table.loc[s, 'x'] = x
            
        srm_id = [i for i in b_correction_srms if i in s]
        if len(srm_id) == 0:
            raise ValueError('The sample name does not contain any of the SRM names in b_correction_srms')
        elif len(srm_id) > 1:
            raise ValueError('The sample name is ambiguous, and contains more than one SRM listed in b_correction_srms')
        else:
            srm_id = srm_id[0]
        
        srm_val = un.ufloat(b_srm_data.loc[srm_id, 'Value'], uncertainty_to_std(b_srm_data.loc[srm_id, 'Uncertainty'], b_srm_data.loc[srm_id, 'Uncertainty_Type']))
        
        b_offset_table.loc[s, 'srm_id'] = srm_id
        b_offset_table.loc[s, 'srm_val'] = srm_val
        
        # get measurment time
        utime = sam.uTime.mean()    
        b_offset_table.loc[s, 'uTime'] = utime
        
    b_offset_table['alpha'] = b_offset_table.measured / b_offset_table.srm_val
    b_offset_table['delta'] = (b_offset_table.alpha - 1) * 1e3
    
    self.d11b_offset_table = b_offset_table
    
    return b_offset_table
    
def d11B_Ca_offset(x, a, b):
    return x * a / (1 + x * a) + b

def calc_d11b_offset_parameters(b_offset_table, fn=None):
    up = un.correlated_values(
        *curve_fit(
            fn,
            xdata=nom(b_offset_table.x),
            ydata=nom(b_offset_table.alpha),
            sigma=err(b_offset_table.alpha),
            p0=(1, 0))
        )
    
    return up

def plot_d11b_offset_calibration(b_offset_table, up=None, fn=None, xlabel=None, confidence_interval=0.95):
    if fn is None:
        fn = d11B_Ca_offset
    if up is None:
        up = calc_d11b_offset_parameters(b_offset_table, fn=fn)
    if xlabel is None:
        xlabel = 'x'
    
    fig, (ax, rax) = plt.subplots(2, 1, figsize=[6, 5], constrained_layout=True, sharex=True)
        
    for srm, g in b_offset_table.groupby('srm_id'):
        ax.errorbar(nom(g.x), nom(g.alpha), yerr=err(g.alpha), xerr=err(g.x), fmt='o', label=srm, markersize=2)
        
    xlim = ax.get_xlim()
    ax.set_xlim(xlim)
    
    nx = np.linspace(*xlim, 200)
    pred = fn(nx, *up)
    
    r2 = r2_score(nom(b_offset_table.alpha), fn(nom(b_offset_table.x), *nom(up)), sample_weight=1/err(b_offset_table.alpha)**2)
    ax.text(0.02, 0.95, f'$R^2$ {r2:.3f}', transform=ax.transAxes, ha='left', va='top')
    
    ax.plot(nx, nom(pred), 'k-', lw=1)
    
    ci_m = stats.t.interval(confidence_interval, b_offset_table.shape[0] - 2)[1]
    ax.fill_between(nx, nom(pred) - err(pred) * ci_m, nom(pred) + err(pred) * ci_m, color=(.8,.8,.8), lw=0)
    
    ax.fill_between(nx, nom(pred) - err(pred), nom(pred) + err(pred), color=(.6,.6,.6), lw=0)
    
    ax.set_ylabel(r'$\alpha^{11}B$')

    ax.axhline(1, color='k', lw=1, ls='--')

    tax = ax.twinx()
    ylim = (np.array(ax.get_ylim()) - 1) * 1000
    yticks = (ax.get_yticks() - 1) * 1000 

    tax.set_yticks(yticks)
    tax.set_ylim(ylim)

    tax.set_ylabel(r'$\Delta \delta^{11}B$')
    
    ax.legend(fontsize=8)
    
    resid = b_offset_table.alpha - fn(b_offset_table.x, *up)
    for srm, g in b_offset_table.groupby('srm_id'):
        ind = b_offset_table.srm_id == srm
        
        rax.errorbar(nom(g.x), nom(resid[ind]), yerr=err(resid[ind]), xerr=err(g.x), fmt='o', markersize=2)
        
    rax.axhline(0, color='k', lw=1, label='Fit', zorder=-1)
    rax.fill_between(nx, -err(pred), err(pred), color=(.6,.6,.6), lw=0, label='$1 \sigma$', zorder=-2)
    rax.fill_between(nx, -err(pred) * ci_m, err(pred) * ci_m, color=(.8,.8,.8), lw=0, label=f'{confidence_interval*1e2:.0f}% CI', zorder=-3)
    
    rax.legend(fontsize=8)

    rax.set_xlabel(xlabel)
    rax.set_ylabel(r'Residual $\alpha^{11}B$')

    tax = rax.twinx()
    ylim = (np.array(rax.get_ylim())) * 1000
    yticks = (rax.get_yticks()) * 1000 

    tax.set_yticks(yticks)
    tax.set_ylim(ylim)

    tax.set_ylabel(r'Residual $\Delta \delta^{11}B$')

    return fig, (ax, rax)

def correct_d11b_Ca_offset(self, xvar, b_correction_srms=['MACS', 'JCT', 'JCP'], yvar='11B_10B', uncertainty_metric='SE', fn=None, plot=True):
    
    _ = make_d11b_offset_table(self, xvar=xvar, b_correction_srms=b_correction_srms, yvar=yvar, uncertainty_metric=uncertainty_metric)
    
    if fn is None:
        fn = d11B_Ca_offset
    self.d11b_offset_fn = fn
    
    self.d11b_offset_parameters = calc_d11b_offset_parameters(self.d11b_offset_table, fn=fn)
    
    if plot:
        plot_d11b_offset_calibration(b_offset_table=self.d11b_offset_table, up=self.d11b_offset_parameters, fn=fn, xlabel=self.d11b_xvar)
    
    for s in self.samples:
        sam = self.data[s]
        
        if 'd11b_corrected' not in sam.data:
            sam.data['d11b_corrected'] = Bunch()
        
        correction_alpha = self.d11b_offset_fn(sam.data['ratios'][self.d11b_xvar], *self.d11b_offset_parameters)
        
        sam.data['d11b_corrected']['11B_10B'] = sam.data['calibrated']['11B_10B'] / correction_alpha
        
        sam.setfocus('d11b_corrected')

    self.stages_complete.update(['d11b_correction'])
    self.focus_stage = 'd11b_correction'
    
def delta_to_R(delta, SRM_ratio=4.04367):
    """
    Convert delta notation (d) to an isotope ratio (R).

    Parameters
    ----------
    delta : array-like
        The isotope ratio expressed in delta notation.
    SRM_ratio : float, optional
        The isotope ratio (R) of the SRM, by default NIST951 which is 4.04367

    Returns
    -------
    array-like
       Delta notation expressed as isotope ratio (R).
    """
    return (delta / 1000 + 1) * SRM_ratio

def R_to_delta(R, SRM_ratio=4.04367):
    """
    Convert an isotope ratio (R) to delta notation (d).

    Parameters
    ----------
    R : array-like
        The isotope ratio.
    SRM_ratio : float, optional
        The isotope ratio (R) of the SRM, by default NIST951 which is 4.04367

    Returns
    -------
    array-like
        R11 expressed in delta notation (d11).
    """
    return (R / SRM_ratio - 1) * 1000