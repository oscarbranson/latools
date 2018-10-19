import shutil
import unittest
import latools as la


class test_latools(unittest.TestCase):
    """
    Test whole of latools
    """
    print('\n\nTest LATOOLS on static data.')

    # load test data
    d = la.analyse('./tests/test_dir/test_data', internal_standard="Ca43")

    # despike
    d.despike(expdecay_despiker=True, noise_despiker=True)

    # autorange
    d.autorange(on_mult=[1.5, 0.8],
                off_mult=[0.8, 1.5])

    # trace plotting
    d.trace_plots(ranges=True)

    # calc background
    d.bkg_calc_weightedmean(weight_fwhm=300, n_min=10)

    # background plot
    fig, ax = d.bkg_plot()

    # subtract background
    d.bkg_subtract()

    # ratio
    d.ratio()

    # calibrate
    d.calibrate(drift_correct=False, n_min=10,
                srms_used=['NIST610', 'NIST612', 'NIST614'])
    # calibration plot
    fig, axs = d.calibration_plot()

    # crossplot
    fig, axs = d.crossplot(save=True)

    # filtering
    # analysis-level classifier
    d.fit_classifier('test', ['Al27', 'Mn55'], 'kmeans', n_clusters=2)
    d.apply_classifier('test')
    d.filter_clear()

    # threshold filter
    d.filter_threshold('Al27', 100e-6)

    d.filter_on('Albelow')

    # calculate stats
    d.sample_stats(stats=['mean', 'std', 'se', 'H15_mean', 'H15_std', 'H15_se'], filt=True)
    s = d.getstats()

    # minimal export
    d.minimal_export()

    # clean up
    shutil.rmtree('./tests/test_dir/test_data_reports')

    print('\nDone.\n\n')


class test_reproduce(unittest.TestCase):
    """
    Test data reproduction.
    """
    print('\n\nTesting latools.reproduce')
    d = la.reproduce('./tests/test_dir/test_data_export/minimal_export.zip')

    # clean up
    shutil.rmtree('./tests/test_dir/test_data_export')

    print('\nDone.\n\n')


if __name__ == '__main__':
    unittest.main()
