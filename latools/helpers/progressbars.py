"""
Convenience wrappers for progress bars to allow interface with pyQT.

(c) Oscar Branson : https://github.com/oscarbranson
"""

from tqdm.autonotebook import tqdm

class progressbar(tqdm):
    def __init__(self):
        super().__init__()
    
    def set(self, iterable=None, total=None, desc=None):
        return tqdm(iterable=iterable,
                    total=total,
                    desc=desc)
    