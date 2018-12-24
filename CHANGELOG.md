# Changelog
All significant changes to the software will be documented here.

## [0.3.8] - 23/12/2018

## Changed
- Calibration now treats blocks of sequentially measured SRMs as single calibration points in drift correction.
- Re-wrote calibration function to allow flexible inclusion of multiple SRMS, which need not contain data for all measured analytes.
- Improved scaling in calibration_plot to better highlight spurious values (e.g. counts/count less than 0)
- Calibration plot shows individual SRMs in different symbols.
- Calibration plot can show individual calibration groups.

## [0.3.7] - 19/12/2018

## Changed
- Modified calibration process to cope with mismatches between measured analytes and those in the SRM database. 

## [0.3.6] - 11/12/2018

### Added
- Changelog!
- Beta file-splitting capabilities in `latools.preprocessing`.

### Changed
- Minor improvements and fixes.

## [0.3.5] - 16/10/2018 - Manuscript Revision
The complete working version associated with the published manuscript:

>[LAtools: a data analysis package for the reproducible reduction of LA-ICPMS data. 2018. Branson, O., Fehrenbacher, J., Vetter, L., Sadekov, A.Y., Eggins, S.M., Spero, H.J. *Chemical Geology, Accepted Manuscript.* doi:10.1016/j.chemgeo.2018.10.029](https://doi.org/10.1016/j.chemgeo.2018.10.029)

### Added
- Ability to correct for spectral interferences.
- Extensive filtering documentation.

### Changed
- Improved saving/loading .zip files.

## [0.3.4] - 20/03/2018 - Manuscript Submission
A complete working version of LATools.