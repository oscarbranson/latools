import re
import os

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