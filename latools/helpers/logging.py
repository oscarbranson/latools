import os
import re
from functools import wraps

# Logging Function
def _log(func):
    """
    Function for logging method calls and parameters
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        a = func(self, *args, **kwargs)
        self.log.append(func.__name__ + ' :: args={} kwargs={}'.format(args, kwargs))
        return a
    return wrapper

def write_logfile(log, header, file_name):
    """
    Write and analysis log to a file.

    Parameters
    ----------
    log : list
        latools.analyse analysis log
    header : list
        File header lines.
    file_name : str
        Destination file. If no file extension
        specified, uses '.lalog'

    Returns
    -------
    None
    """
    path, ext = os.path.splitext(file_name)
    if ext == '':
        ext = '.lalog'
    
    with open(path + ext, 'w') as f:
        f.write('\n'.join(header))
        f.write('\n'.join(log))
    
    return path + ext

def read_logfile(log_file):
    """
    Reads an latools analysis.log file, and returns dicts of arguments.

    Parameters
    ----------
    log_file : str
        Path to an analysis.log file produced by latools.
    
    Returns
    -------
    runargs, paths : tuple
        Two dictionaries. runargs contains all the arguments required to run each step
        of analysis in the form (function_name, {'args': (), 'kwargs': {}}). paths contains
        the locations of the data directory and the SRM database used for analysis.
    """
    dirname = os.path.dirname(log_file) + '/'
    
    with open(log_file, 'r') as f:
        rlog = f.readlines()

    hashind = [i for i, n in enumerate(rlog) if '#' in n]

    pathread = re.compile('(.*) :: (.*)\n')
    paths = (pathread.match(l).groups() for l in rlog[hashind[0] + 1:hashind[-1]] if pathread.match(l))
    paths = {k: os.path.join(dirname, v) for k, v in paths}
    # paths = {k: os.path.abspath(v) for k, v in paths}

    logread = re.compile('([a-z_]+) :: args=(\(.*\)) kwargs=(\{.*\})')
    runargs = []
    for line in rlog[hashind[1] + 1:]:
        fname, args, kwargs = (logread.match(line).groups())
        runargs.append((fname ,{'args': eval(args), 'kwargs': eval(kwargs)}))
        
        if fname == '__init__':
            runargs[-1][-1]['kwargs']['config'] = 'REPRODUCE'
            runargs[-1][-1]['kwargs']['dataformat'] = None
            runargs[-1][-1]['kwargs']['data_folder'] = paths['data_folder']
            if 'srm_table' in paths:
                runargs[-1][-1]['kwargs']['srm_file'] = paths['srm_table']

    return runargs, paths