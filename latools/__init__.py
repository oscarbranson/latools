"""
LAtools: Python tools for processing Laser Ablation mass spectrometry data

User Guide: https://latools.readthedocs.io/en/latest/

Citation:
LAtools: a data analysis package for the reproducible reduction of LA-ICPMS data. 2018. Branson, O., Fehrenbacher, J., Vetter, L., Sadekov, A.Y., Eggins, S.M., Spero, H.J. Chemical Geology 504: 83-95. doi:10.1016/j.chemgeo.2018.10.029

(c) Oscar Branson : https://github.com/oscarbranson
"""

from .latools import analyse, reproduce
from .latools import analyse as analyze  # for the yanks
from .helpers.helpers import get_example_data
from .helpers.stat_fns import nominal_values, std_devs
from .helpers import config
from .helpers import chemistry
from . import preprocessing

__version__ = '0.3.12'

def cite(output='text'):
    """
    Citation for LAtools.
    """
    if output == 'bibtex':
        print(
"""@article{Branson_2019_LAtools,
title = "LAtools: A data analysis package for the reproducible reduction of LA-ICPMS data",
journal = "Chemical Geology",
volume = "504",
pages = "83 - 95",
year = "2019",
issn = "0009-2541",
doi = "https://doi.org/10.1016/j.chemgeo.2018.10.029",
url = "http://www.sciencedirect.com/science/article/pii/S0009254118305461",
author = "Oscar Branson and Jennifer S. Fehrenbacher and Lael Vetter and Aleksey Y. Sadekov and Stephen M. Eggins and Howard J. Spero",
keywords = "Geochemistry, Laser ablation, Data processing"}"""
)
    else:
        print("LAtools: a data analysis package for the reproducible reduction of LA-ICPMS data. (2018)\n" +
              "Branson, O., Fehrenbacher, J., Vetter, L., Sadekov, A.Y., Eggins, S.M., Spero, H.J.\n" + 
              "Chemical Geology 504: 83-95. doi:10.1016/j.chemgeo.2018.10.029")

# from . import pca
# from . import helpers
