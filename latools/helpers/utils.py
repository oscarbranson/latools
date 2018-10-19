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
