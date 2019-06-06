"""
Core LAtools functions that perform key data processing steps.

(c) Oscar Branson : https://github.com/oscarbranson
"""
from .signal_id import autorange
from .data_read import read_data
from .despiking import noise_despike, expdecay_despike