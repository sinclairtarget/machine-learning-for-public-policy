"""
This module handles the reading and writing of data in various formats.

Methods in this module default to assuming that all data files are stored under
a /data directory in the project root.
"""
import pandas as pd
from datetime import datetime

def datetime_converter(date_format):
    return lambda d: datetime.strptime(d, date_format)

def read_csv(filename, prefix='data/', **kwargs):
    return pd.read_csv(prefix + filename, **kwargs)

def write_csv(df, filename, prefix='data/'):
    df.to_csv(prefix + filename, index=False)
