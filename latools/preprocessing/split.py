"""
Functions for splitting long files into multiple short ones.

(c) Oscar Branson : https://github.com/oscarbranson
"""
import re
import os
import json
import datetime
import dateutil
import numpy as np
import pandas as pd
import pkg_resources as pkgrs
from warnings import warn
from ..processes import read_data, autorange
from ..helpers.helpers import bool_2_indices
from ..helpers.analyte_names import analyte_2_namemass
from ..helpers.io import read_dataformat

import matplotlib.pyplot as plt

def by_regex(file, outdir=None, split_pattern=None, global_header_rows=0, fname_pattern=None, trim_tail_lines=0, trim_head_lines=0):
    """
    Split one long analysis file into multiple smaller ones.

    Parameters
    ----------
    file : str
        The path to the file you want to split.
    outdir : str
        The directory to save the split files to.
        If None, files are saved to a new directory
        called 'split', which is created inside the
        data directory.
    split_pattern : regex string
        A regular expression that will match lines in the
        file that mark the start of a new section. Does
        not have to match the whole line, but must provide
        a positive match to the lines containing the pattern.
    global_header_rows : int
        How many rows at the start of the file to include
        in each new sub-file.
    fname_pattern : regex string
        A regular expression that identifies a new file name
        in the lines identified by split_pattern. If none,
        files will be called 'noname_N'. The extension of the
        main file will be used for all sub-files.
    trim_head_lines : int
        If greater than zero, this many lines are removed from the start of each segment
    trim_tail_lines : int
        If greater than zero, this many lines are removed from the end of each segment

    Returns
    -------
    Path to new directory : str
    """
    # create output sirectory
    if outdir is None:
        outdir = os.path.join(os.path.dirname(file), 'split')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # read input file
    with open(file, 'r') as f:
        lines = f.readlines()
    
    # get file extension
    extension = os.path.splitext(file)[-1]
    
    # grab global header rows
    global_header = lines[:global_header_rows]

    # find indices of lines containing split_pattern
    starts = []
    for i, line in enumerate(lines):
        if re.search(split_pattern, line):
            starts.append(i)    
    starts.append(len(lines))  # get length of lines

    # split lines into segments based on positions of regex
    splits = {}
    for i in range(len(starts) - 1):
        m = re.search(fname_pattern, lines[starts[i]])
        if m:
            fname = m.groups()[0].strip()
        else:
            fname = 'no_name_{:}'.format(i)

        splits[fname] = global_header + lines[starts[i]:starts[i+1]][trim_head_lines:trim_tail_lines]
    
    # write files
    print('Writing files to: {:}'.format(outdir))
    for k, v in splits.items():
        fname = (k + extension).replace(' ', '_')
        with open(os.path.join(outdir, fname), 'w') as f:
            f.writelines(v)
        print('  {:}'.format(fname))
    
    print('Done.')

    return outdir

def long_file(data_file, dataformat, sample_list, analyte='total_counts', savedir=None, srm_id=None, combine_same_name=True, defrag_to_match_sample_list=True, min_points=0, plot=True, **autorange_args):
    """
    Split single long files containing multiple analyses into multiple files containing single analyses.

    Imports a long datafile and uses `latools.processes.autorange` to
    identify ablations in the long file based on your chosen analyte.
    The data are then saved as multiple files each containing a single
    ablation, named using the list of names you provide.

    Data will be saved in latools' 'REPRODUCE' format.

    WARNING: This functionality is currently *very beta*. Use carefully.

    TODO: Check for existing files in savedir, don't overwrite?

    Parameters
    ----------
    data_file : str
        The path to the data file you want to read.
    dataformat : dataformat dict
        A valid dataformat dict. See online documentation for more details.
    sample_list : array-like
        A list of strings that will be used to name the individual files.
    analyte : str
        The analyte that autorange uses to identify ablations. Can be any valid
        analyte in the data. Defaults to 'total_counts'.
    savedir : str
        The directory to save the data in. Defaults to the name of the data_file,
        appended with '_split'.
    srm_id : str
        If given, all file names containing srm_id will be replaced with srm_id.
    **autorange_args
        Additional arguments passed to la.processes.autorange used for identifying ablations.
    Returns
    -------
    None
    """
    if isinstance(sample_list, str):
        if os.path.exists(sample_list):
            sample_list = np.genfromtxt(sample_list, dtype=str)
        else:
            raise ValueError('File {} not found.')
    else:
        sample_list = np.asanyarray(sample_list)
        
    if srm_id is not None:
        srm_replace = []
        for s in sample_list:
            if srm_id in s:
                s = srm_id
            srm_replace.append(s)
        sample_list = srm_replace
    
    dataformat = read_dataformat(dataformat, silent=False)
                    
    _, _, dat, meta = read_data(data_file, dataformat=dataformat, name_mode='file')
    
    if 'date' in meta:
        d = dateutil.parser.parse(meta['date'])
    else:
        d = datetime.datetime.now()

    # analyte handling
    if analyte == 'total_counts':
        y_data = dat['total_counts']
    elif analyte in dat['rawdata'].keys():
        y_data = dat['rawdata'][analyte]
    else:
        valid = list(dat['rawdata'].keys()) + ['total_counts']
        raise ValueError("'{}' is not a valid analyte. Please use one of:\n  {}".format(analyte, valid))
    
    # autorange
    bkg, sig, _, _ = autorange(dat['Time'], y_data, **autorange_args)
    
    ns = np.zeros(sig.size)
    ns[sig] = np.cumsum((sig ^ np.roll(sig, 1)) & sig)[sig]
    n = int(max(ns))

    nsamples = len(sample_list)

    if nsamples != n:
        print('Number of samples in list ({}) does not match number of ablations ({}).'.format(nsamples, n))
        if nsamples < n:
            print('  -> There are more ablations than samples...')
            if defrag_to_match_sample_list:
                print('     Removing data fragments to match sample list length.')
                while nsamples < n:
                    min_points += 1
                    sig = sig & np.roll(sig, min_points)
                    ns = np.zeros(sig.size)
                    ns[sig] = np.cumsum((sig ^ np.roll(sig, 1)) & sig)[sig]
                    n = int(max(ns))
                print('       (Removed data fragments < {} points long)'.format(min_points))
        elif isinstance(min_points, (int, float)):
            # minimum point filter
            sig = sig & np.roll(sig, min_points)
            ns = np.zeros(sig.size)
            ns[sig] = np.cumsum((sig ^ np.roll(sig, 1)) & sig)[sig]
            n = int(max(ns))
        else:
            print('  -> There are more samples than ablations...')
            print('     Check your sample list is correct. If so, consider')
            print('     adding autorange_params to change the signal detection.')
            return

    minn = min([len(sample_list), n])

    # calculate split boundaries
    bounds = []
    lower = 0
    sn = 0
    next_sample = ''
    for ni in range(minn-1):
        sample = sample_list[sn]
        next_sample = sample_list[sn + 1]
        
        if not combine_same_name or sample != next_sample:
            current_end = np.argwhere(dat['Time'] == dat['Time'][ns == ni + 1].max())[0]
            next_start = np.argwhere(dat['Time'] == dat['Time'][ns == ni + 2].min())[0]
            upper = (current_end + next_start) // 2

            bounds.append((sample, (int(lower), int(upper))))

            lower = upper + 1
                
        sn += 1

    bounds.append((sample_list[-1], (int(upper) + 1, len(ns))))

    # split up data
    sections = {}
    seen = {}
    for s, (lo, hi) in bounds:
        if s not in seen:
            seen[s] = 0
        else:
            seen[s] += 1
            s += '_{}'.format(seen[s])
        sections[s] = {'oTime': dat['Time'][lo:hi]}
        sections[s]['Time'] = sections[s]['oTime'] - np.nanmin(sections[s]['oTime'])
        sections[s]['rawdata'] = {}
        for k, v in dat['rawdata'].items():
            sections[s]['rawdata'][k] = v[lo:hi]
        sections[s]['starttime'] = d + datetime.timedelta(seconds=np.nanmin(sections[s]['oTime']))
        sections[s]['total_counts'] = dat['total_counts'][lo:hi]
    
    # save output
    if savedir is None:
        savedir = os.path.join(os.path.dirname(os.path.abspath(data_file)), os.path.splitext(os.path.basename(data_file))[0] + '_split')
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    
    header = ['# Long data file split by latools on {}'.format(datetime.datetime.now().strftime('%Y:%m:%d %H:%M:%S'))]
    if 'date' not in meta:
        header.append('# Warning: No date specified in file - Analysis Times are date file was split. ')
    else:
        header.append('# ')

    header.append('# ')
    header.append('# ')
    
    flist = []
    for s, sdat in sections.items():
        iheader = header.copy()
        iheader.append('# Sample: {}'.format(s))
        iheader.append('# Analysis Time: {}'.format(sdat['starttime'].strftime('%Y-%m-%d %H:%M:%S')))
    
        iheader = '\n'.join(iheader) + '\n'
        
        out = pd.DataFrame({analyte_2_namemass(k): v for k, v in sdat['rawdata'].items()}, index=sdat['Time'])
        out.index.name = 'Time'
        csv = out.to_csv()
        
        with open('{}/{}.csv'.format(savedir, s), 'w') as f:
            f.write(iheader)
            f.write(csv)
        flist.append('   {}.csv'.format(s))
    
    print("Success! File split into {} sections.".format(n))
    print("New files saved to:\n{}/\n{}\n\nImport the split files using the 'REPRODUCE' configuration.".format(os.path.relpath(savedir), '\n'.join(flist)))
    
    if plot:
        return plot_long_file_split(dat, sig, bkg, sections)
    else:
        return None
    # return dat, sig, sections

def plot_long_file_split(dat, sig, bkg, sections):
    n = len(sections)

    fig, ax = plt.subplots(1, 1, figsize=(n * 1.5, 2.5))

    ax.plot(dat['Time'], dat['total_counts'], c=(0,0,0,0))
    ax.set_yscale('log')
    ax.set_xlim(dat['Time'].min(), dat['Time'].max())

    ylim = ax.get_ylim()
    yrng = np.ptp(ylim)
    xlim = ax.get_xlim()
    xrng = np.ptp(xlim)

    for s, d in sections.items():
        line = ax.plot(d['oTime'], d['total_counts'])
        ax.axvline(d['oTime'][0], color=line[0].get_color())
        ax.text(d['oTime'][0] + 0.02 * xrng / n, ylim[0] + 0.95 * yrng, s, rotation=90, ha='left', va='top', color=line[0].get_color())

    sigs = bool_2_indices(sig)
    for slo, shi in sigs:
        ax.axvspan(dat['Time'][slo], dat['Time'][shi], zorder=-2, color=(.8,.7,0,0.15), lw=0)

    bkgs = bool_2_indices(bkg)
    for blo, bhi in bkgs:
        ax.axvspan(dat['Time'][blo], dat['Time'][bhi], zorder=-2, color=(.2,.2,0,0.1), lw=0)
    
    fig.tight_layout()
    return fig, ax