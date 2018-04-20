import os
import re
import shutil
import zipfile

def zipdir(directory, name=None, delete=False):
    """
    Compresses the target directory, and saves it to ../name.zip

    Parameters
    ----------
    directory : str
        Path to the directory you want to compress.
        Compressed file will be saved at directory/../name.zip
    name : str (default=None)
        The name of the resulting zip file. If not specified, the
        name of the directory to be compressed is used.
    delete : bool
        If True, the uncompressed directory is deleted after the zip file
        has been created. Defaults to False.

    Returns
    -------
    None
    """
    if not os.path.isdir(directory) or not os.path.exists(directory):
        raise ValueError('Please provide a valid directory.')
    if name is None:
        name = os.path.dirname(directory).split('/')[-1]
    
    savepath = os.path.join(directory, os.path.pardir)
    
    # create zipfile
    with zipfile.ZipFile(os.path.join(savepath, name + '.zip'), 'w', zipfile.ZIP_DEFLATED) as zipf:
        print(os.path.join(savepath, name + '.zip'))
        for root, dirs, files in os.walk(directory):
            for f in files:
                zipf.write(os.path.join(root, f), os.path.join(root.replace(directory, ''), f))
    if delete:
        shutil.rmtree(directory)

    return None

def extract_zipdir(zip_file):
    """
    Extract contents of zip file into subfolder in parent directory.
    
    Parameters
    ----------
    zip_file : str
        Path to zip file
    
    Returns
    -------
        None
    """
    if not os.path.exists(zip_file):
        raise ValueError('{} does not exist'.format(zip_file))
    directory = os.path.dirname(zip_file)
    file = os.path.basename(zip_file)

    with zipfile.ZipFile(zip_file, 'r', zipfile.ZIP_DEFLATED) as zipf:
        zipf.extractall(os.path.join(directory, file.replace('.zip', '')))

    return None

def read_logfile(log_file):
    dirname = os.path.dirname(log_file) + '/'
    
    with open(log_file, 'r') as f:
        rlog = f.readlines()

    hashind = [i for i, n in enumerate(rlog) if '#' in n]

    pathread = re.compile('(.*) :: (.*)\n')
    paths = (pathread.match(l).groups() for l in rlog[hashind[0] + 1:hashind[-1]] if pathread.match(l))
    paths = {k: os.path.join(dirname, v) for k, v in paths}
    # paths = {k: os.path.abspath(v) for k, v in paths}

    logread = re.compile('([a-z_]+) :: args=(\(.*\)) kwargs=(\{.*\})')
    runargs = {}
    for line in rlog[hashind[1] + 1:]:
        fname, args, kwargs = (logread.match(line).groups())
        runargs[fname] = {'args': eval(args), 'kwargs': eval(kwargs)}
        
        if fname == '__init__':
            runargs[fname]['kwargs']['config'] = 'REPRODUCE'
            runargs[fname]['kwargs']['dataformat'] = None
            runargs[fname]['kwargs']['data_folder'] = paths['data_folder']

    return runargs, paths