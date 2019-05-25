import pandas as pd
from sklearn import preprocessing

class Binner:
    """
    Bins continuous columns into bins with the same number of members.
    """
    def __init__(self, n_bins, colnames):
        self.n_bins = n_bins
        self.colnames = colnames
        self.new_colnames = self._new_colnames()
        self.est = preprocessing.KBinsDiscretizer(n_bins=n_bins)


    def fit(self, df):
        df_sub = df[self.colnames]
        self.est = self.est.fit(df_sub.values)


    def transform(self, df):
        df_sub = df[self.colnames]
        m = self.est.transform(df_sub.values).todense()
        df_discretized = pd.DataFrame(m, columns=self.new_colnames)
        return pd.merge(df, df_discretized, left_index=True, right_index=True)


    def _new_colnames(self):
        new_colnames = []
        for colname in self.colnames:
            for i in range(self.n_bins):
                new_colnames.append(f"{colname}_bin_{i + 1}")

        return new_colnames
