# Changelog
All significant changes to the software will be documented here.

## [0.3.20]

### New
- Integrated long file splitting into analysis function.
- Stackhist plot

### Changed
- Improved handling of analytes missing from SRM database (analyte is removed from further calculations, rather than throwing an error)
- Improved gradient plot.

## [0.3.19]

### Changed
- Filters now use pandas to keep track of filter components and switches.
- Quality of life improvements using pandas DataFrames to display filter status
- Improvements to the calculation of rolling statistics. Can choose justification window relative to data position, and rolling gradient calculation now handles nans better (excludes them).
- Improvements to `gradient_threshold` filters. Can now set `absolute_gradient=False` to select 'up' or 'down' components separately. `gplot` now shows the effect of filters directly, similar to `tplot`.

## [0.3.18]

### New
 - Change ratio calibration framework to support multiple internal standards. Internal standard is now stored alongside analyte name after ratio calculation as 'XX_YY'. This involved a lot of tweaks to discrimnate between pre- and post- ratio treatment of data. All tests passing & manuscript supplement notebooks run fine, **but some more advanced features may be broken**. Please log an issue if you find anything.

### Changed
 - `latools.preprocessing.split.long_file` can now accept wildcards in sample names to allow for repeat ablations of single samples.

## [0.3.17]

### Changed
 - Removed extraneous in with Manifest.in

## [0.3.16] - 18/01/2021

### New
 - `clear_subsets` function.

## [0.3.12] - 28/10/2019

### Changed
 - Fixed analyte ordering in stat export.

## [0.3.11] - 04/07/2019
Fixing minor bugs introduced in 0.3.10.

### Changed
 - Fixed broken meta_regex parsing when key was numeric instead of str.
 - Fixed default behaviour of mass fraction calculation
 - Removed extrapolation from interp1d background calculation to avoid negative backgrounds/stderrs.

## [0.3.10] - 14/06/2019

### New
- Fully functional long-file splitting, with documentation
- Ability to calculate mass fraction (i.e. ppm)
- Configurations for GEOMAR Atom and FIN2.

### Changed
- Improved dataformat description flexibility
- Improved handling of missing SRMs
- clear_calibration now works as intended
- Minor bug fixes throughout
- Parsed out io functions

## [0.3.9] - 13/02/2019

### Changed
- Modified read_data function to properly identify and remove non-analyte columns.

## [0.3.8] - 23/12/2018

### Changed
- Calibration now treats blocks of sequentially measured SRMs as single calibration points in drift correction.
- Re-wrote calibration function to allow flexible inclusion of multiple SRMS, which need not contain data for all measured analytes.
- Improved scaling in calibration_plot to better highlight spurious values (e.g. counts/count less than 0)
- Calibration plot shows individual SRMs in different symbols.
- Calibration plot can show individual calibration groups.

## [0.3.7] - 19/12/2018

### Changed
- Modified calibration process to cope with mismatches between measured analytes and those in the SRM database. 

## [0.3.6] - 11/12/2018

### New
- Changelog!
- Beta file-splitting capabilities in `latools.preprocessing`.

### Changed
- Minor improvements and fixes.

## [0.3.5] - 16/10/2018 - Manuscript Revision
The complete working version associated with the published manuscript:

>[LAtools: a data analysis package for the reproducible reduction of LA-ICPMS data. 2018. Branson, O., Fehrenbacher, J., Vetter, L., Sadekov, A.Y., Eggins, S.M., Spero, H.J. *Chemical Geology, Accepted Manuscript.* doi:10.1016/j.chemgeo.2018.10.029](https://doi.org/10.1016/j.chemgeo.2018.10.029)

### New
- Ability to correct for spectral interferences.
- Extensive filtering documentation.

### Changed
- Improved saving/loading .zip files.

## [0.3.4] - 20/03/2018 - Manuscript Submission
A complete working version of LATools.