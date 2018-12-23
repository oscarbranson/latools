import unittest
import os, shutil, sys
import latools as la
# IF YOU CHANGE THIS, MUST UPDATE LINE NO CALLS IN DOCS_SOURCE BEGINNERS GUIDE!!!
class test_docscode(unittest.TestCase):
    """test examples in documentation - WARNING don't change line numbers. Copied directly to docs.
    """
    print('\n\nTesting Beginners Guide code examples.')
    if not os.path.exists('./latools_demo_tmp'):
        os.mkdir('./latools_demo_tmp')
    os.chdir('latools_demo_tmp')

    la.get_example_data('./latools_demo_tmp')

    eg = la.analyse(data_folder='./latools_demo_tmp',
                    config='DEFAULT',
                    internal_standard='Ca43',
                    srm_identifier='STD')
    eg.trace_plots()

    eg.despike(expdecay_despiker=True,
               noise_despiker=True)

    eg.autorange(on_mult=[1.5, 0.8],
                 off_mult=[0.8, 1.5])

    eg.bkg_calc_weightedmean(weight_fwhm=300,
                             n_min=10)

    eg.bkg_plot()

    eg.bkg_subtract()

    eg.ratio()

    eg.calibrate(drift_correct=False,
                 srms_used=['NIST610', 'NIST612', 'NIST614'])

    eg.calibration_plot()

    eg.filter_threshold(analyte='Al27', threshold=100e-6)  # remember that all units are in mol/mol!

    eg.filter_reports(analytes='Al27', filt_str='thresh')

    eg.filter_on(filt='Albelow')

    eg.filter_off(filt='Albelow', analyte='Mg25')

    eg.make_subset(samples='Sample-1', name='set1')
    eg.make_subset(samples=['Sample-2', 'Sample-3'], name='set2')

    eg.filter_on(filt=0, subset='set1')

    eg.filter_off(filt=0, subset='set2')

    eg.sample_stats(stats=['mean', 'std'], filt=True)

    stats = eg.getstats()

    eg.minimal_export()

    # OK To change line numbers after here.

    # clean up
    os.chdir('..')
    shutil.rmtree('latools_demo_tmp')

    print('\nDone.\n\n')


if __name__ == '__main__':
    unittest.main()
