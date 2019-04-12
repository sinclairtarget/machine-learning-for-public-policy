import pandas as pd

def read_csv(filename, prefix='data/'):
    return pd.read_csv(prefix + filename)
