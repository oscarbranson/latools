import re
import os
import datetime
import dateutil
import numpy as np
import pandas as pd
from warnings import warn
from ..processes import read_data, autorange
from ..helpers.helpers import analyte_2_namemass

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

def long_file(data_file, dataformat, sample_list, savedir=None, srm_id=None, **autorange_args):
    """
    TODO: Check for existing files in savedir, don't overwrite?
    """
    if isinstance(sample_list, str):
        if os.path.exists(sample_list):
            sample_list = np.genfromtxt(sample_list, dtype=str)
        else:
            raise ValueError('File {} not found.')
    elif not isinstance(sample_list, (list, np.ndarray)):
        raise ValueError('sample_list should be an array_like or a file.')
        
    if srm_id is not None:
        srm_replace = []
        for s in sample_list:
            if srm_id in s:
                s = srm_id
            srm_replace.append(s)
        sample_list = srm_replace
                
    _, _, dat, meta = read_data(data_file, dataformat=dataformat, name_mode='file')
    
    if 'date' in meta:
        d = dateutil.parser.parse(meta['date'])
    else:
        d = datetime.datetime.now()
    # autorange
    bkg, sig, trn, _ = autorange(dat['Time'], dat['total_counts'], **autorange_args)
    
    ns = np.zeros(sig.size)
    ns[sig] = np.cumsum((sig ^ np.roll(sig, 1)) & sig)[sig]
    
    n = int(max(ns))
    
    if len(sample_list) != n:
        warn('Length of sample list does not match number of ablations in file.\n' + 
             'We will continue, but please make sure the assignments are correct.')
    
    # calculate split boundaries
    bounds = []
    lower = 0
    sn = 0
    next_sample = ''
    for ni in range(n-1):
        sample = sample_list[sn]
        next_sample = sample_list[sn + 1]
                
        if sample != next_sample:
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
    
    flist = [savedir]
    for s, dat in sections.items():
        iheader = header.copy()
        iheader.append('# Sample: {}'.format(s))
        iheader.append('# Analysis Time: {}'.format(dat['starttime'].strftime('%Y-%m-%d %H:%M:%S')))
    
        iheader = '\n'.join(iheader) + '\n'
        
        out = pd.DataFrame({analyte_2_namemass(k): v for k, v in dat['rawdata'].items()}, index=dat['Time'])
        out.index.name = 'Time'
        csv = out.to_csv()
        
        with open('{}/{}.csv'.format(savedir, s), 'w') as f:
            f.write(iheader)
            f.write(csv)
        flist.append('   {}.csv'.format(s))
    
    print("File split into {} sections.\n Saved to: {}\n\n Import using the 'REPRODUCE' configuration.".format(n, '\n'.join(flist)))
    return None