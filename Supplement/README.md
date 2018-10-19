# Supplementary Data and Analysis

This folder contains supplementary data and data analysis information for the examples presented in Branson et al (sub.).

Data processing with `latools`, and subsequent and comparison to other processing methods, are described in four [Jupyter](http://jupyter.org/) notebooks:
- [**Cultured Foraminifera**](http://nbviewer.jupyter.org/github/oscarbranson/latools/blob/master/Supplement/cultured_foram_manual.ipynb) (Manual)
- [**Fossil Foraminfifera**](http://nbviewer.jupyter.org/github/oscarbranson/latools/blob/master/Supplement/fossil_foram_manual.ipynb) (Manual)
- [**Fossil Foraminifera**](http://nbviewer.jupyter.org/github/oscarbranson/latools/blob/master/Supplement/fossil_foram_iolite.ipynb) [(Iolite<sup>TM</sup>)](https://iolite-software.com/)
- [**Zircons**](http://nbviewer.jupyter.org/github/oscarbranson/latools/blob/master/Supplement/zircon_manual.ipynb) (Manual)

## Raw Data

All raw LA-ICPMS data are contained in the [raw_data](raw_data/) folder.

## Summary Statistics

Within the [raw_data](raw_data/) folder, there are subfolders with `_export` appended onto the name. 
These folders contain `stat_export.csv` files, which contain summary statistics for each sample produced by `latools` at the end of processing.

Also within these folders, you will find `minimal_export.zip` files, which contain everything needed to reproduce the analysis workflow. To use them, download the `.zip` files to your computer, start latools, and run `latools.reproduce('path/to/minimal_export.zip')`.

## Sample information

Data collection paramter tables are stored in HTML format in the [Parameter_Tables](Parameter_Tables/) folder, and are also displayed in the analysis notebooks.

## Figures

Plots of the data comparison results (as shown in the manuscript) are saved in [Figures](Figures/)

## Technical Details

Data comparisons in the notebooks rely on functions within the [comparison_tools](comparison_tools) folder. If you are trying to run these examples, download the entire 'Supplement' folder (including the [comparison_tools](comparison_tools) folder), and run them within this folder on your computer.


# Try it yourself!

If you'd like to compare the results of `Your Favourite Data Processing Software` with the output of `latools`, please do! Here are a few easy steps to do this:

1. Download one of ablation datasets from the [raw_data](raw_data/) folder.
2. Process the datset with `Your Favourite Data Processing Software`
3. Download the `latools` summary statistics file corresponding to the dataset you downloaded (also within the [raw_data](raw_data/) folder, with the same name as the data folder, appended by `_export`).
4. Compare your results to the `latools` values. 