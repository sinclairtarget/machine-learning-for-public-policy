import numpy as np

def impute(df, col, how='avg'):
    subst = np.nan

    if how == 'avg':
        subst = np.average(df.loc[df[col].notnull(), col])
    else:
        raise Exception(f"\"how\" argument of \"{how}\" is not supported.")

    df.loc[df[col].isnull(), col] = subst
