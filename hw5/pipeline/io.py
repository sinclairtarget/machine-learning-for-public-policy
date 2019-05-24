"""
This module handles the reading and writing of data in various formats.

Methods in this module default to assuming that all data files are stored under
a /data directory in the project root.
"""
import pandas as pd

def read_csv(filename, prefix='data/'):
    return pd.read_csv(prefix + filename)

def write_csv(df, filename, prefix='data/'):
    df.to_csv(prefix + filename, index=False)
