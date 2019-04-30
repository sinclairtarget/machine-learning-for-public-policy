"""
This module contains functions that explore distributions of values and
relationships between columns.
"""
import numpy as np

def extrema(narray):
    """
    Returns a (min, max) tuple for the narry.
    """
    return (np.min(narray), np.max(narray))
