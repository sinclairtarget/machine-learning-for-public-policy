import pandas as pd

def read_csv(filename, prefix='data/'):
    return pd.read_csv(prefix + filename)

def write_csv(df, filename, prefix='data/'):
    df.to_csv(prefix + filename, index=False)
