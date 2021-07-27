"""
File handling and os-interface functions

(c) Oscar Branson : https://github.com/oscarbranson
"""
import os
import re
import shutil
import zipfile
import pkg_resources as pkgrs

# Bunch modifies dict to allow item access using dot (.) operator
class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch, self).__init__(*args, **kwds)
        self.__dict__ = self

# warnings monkeypatch
# https://stackoverflow.com/questions/2187269/python-print-only-the-message-on-warnings
def _warning(message, category=UserWarning,
             filename='', lineno=-1,
             file=None, line=None):
    print(message)


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
        name = directory.split('/')[-1]
    
    savepath = os.path.join(directory, os.path.pardir)
    
    # create zipfile
    with zipfile.ZipFile(os.path.join(savepath, name + '.zip'), 'w', zipfile.ZIP_DEFLATED) as zipf:
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
        str : folder where the zip was extracted
    """
    if not os.path.exists(zip_file):
        raise ValueError('{} does not exist'.format(zip_file))
    directory = os.path.dirname(zip_file)
    filename = os.path.basename(zip_file)
    dirpath = os.path.join(directory, filename.replace('.zip', ''))

    with zipfile.ZipFile(zip_file, 'r', zipfile.ZIP_DEFLATED) as zipf:
        zipf.extractall(dirpath)

    return dirpath

def collate_data(in_dir, extension='.csv', out_dir=None):
    """
    Copy all csvs in nested directroy to single directory.

    Function to copy all csvs from a directory, and place
    them in a new directory.

    Parameters
    ----------
    in_dir : str
        Input directory containing csv files in subfolders
    extension : str
        The extension that identifies your data files.
        Defaults to '.csv'.
    out_dir : str
        Destination directory

    Returns
    -------
    None
    """
    if out_dir is None:
        out_dir = './' + re.search('^\.(.*)', extension).groups(0)[0]

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for p, d, fs in os.walk(in_dir):
        for f in fs:
            if extension in f:
                shutil.copy(p + '/' + f, out_dir + '/' + f)
    return

def get_example_data(destination_dir):
    if os.path.isdir(destination_dir):
        overwrite = input(destination_dir +
                          ' already exists. Overwrite? [N/y]: ').lower() == 'y'
        if overwrite:
            shutil.rmtree(destination_dir)
        else:
            print(destination_dir + ' was not overwritten.')

    shutil.copytree(pkgrs.resource_filename('latools', 'resources/test_data'),
                    destination_dir)

    return