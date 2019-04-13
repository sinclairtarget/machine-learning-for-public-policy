import numpy as np
import pandas as pd

def impute(df, col, how='avg'):
    subst = np.nan

    if how == 'avg':
        subst = np.average(df.loc[df[col].notnull(), col])
    else:
        raise Exception(f"\"how\" argument of \"{how}\" is not supported.")

    df.loc[df[col].isnull(), col] = subst

def bin(df, col, bins, labels=None):
    new_col = col + '_binned'
    max_val = np.max(df[col])
    df[new_col] = pd.cut(df[col],
                         bins + [max_val],
                         include_lowest=True,
                         labels=labels)

def dummify(df, col):
    uniqs = pd.unique(df[col])
    for uniq in uniqs:
        new_col = col + '_is_' + str(uniq)
        df[new_col] = df[col] == uniq
