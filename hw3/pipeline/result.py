from pandas import DataFrame, Series
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

class PredictionResult:
    """
    Wraps a dataframe that contains prediction results.

    The dataframe is supposed to have at least the following columns:
    - actual
    - predict
    """
    def __init__(self, df):
        self.df = df


    def baseline(self):
        """Returns the ratio of actually true outcomes to total outcomes."""
        count_true = len(self.df[self.df.actual == 1])
        return count_true / len(self.df)


    def accuracy(self):
        return metrics.accuracy_score(self.df.actual.values,
                                      self.df.predict.values)


    def precision(self):
        return metrics.precision_score(self.df.actual.values,
                                       self.df.predict.values)


    def recall(self):
        return metrics.recall_score(self.df.actual.values,
                                    self.df.predict.values)


    def f1(self):
        return metrics.f1_score(self.df.actual.values,
                                self.df.predict.values)


    def matrix(self):
        m = metrics.confusion_matrix(self.df.actual.values,
                                     self.df.predict.values)
        return DataFrame(m,
                         index=['false', 'true'],
                         columns=['negative', 'positive'])


    def as_series(self):
        return Series({
            'accuracy': self.accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'f1': self.f1()
        })


    def stack(results, index=None):
        index = index or list(range(1, len(results) + 1))
        return DataFrame([r.as_series() for r in results],
                         index=index)


class ResultCollection:
    """
    Wrapper around a collection of PredictionResults that can draw graphs.
    """
    def __init__(self):
        self.df = DataFrame()
        self.suffixes = []


    def add(self, suffix, result):
        local_result = result.copy(deep=False)
        local_result.columns = \
            [colname + '_' + suffix for colname in local_result.columns]

        self.df = self.df.join(local_result, how='right')
        self.suffixes.append(suffix)


    def plot_statistic(self, stat_name):
        stat_df = self.df.filter(regex=stat_name)
        stat_df.columns = self.suffixes

        for suffix in self.suffixes:
            plt.plot([str(x) for x in stat_df.index.values],
                     stat_df[suffix].values,
                     label=suffix)

        plt.xlabel('split')
        plt.ylabel(stat_name)
        plt.legend()
        plt.show()
