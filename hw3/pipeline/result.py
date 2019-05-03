from pandas import DataFrame, Series
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

class PredictionResult:
    """
    Wraps a dataframe that contains prediction results.

    The dataframe is supposed to have at least the following columns:
    - actual
    - score
    - predict

    Any threshold passed to functions should be a percentage threshold.
    """
    def __init__(self, df):
        self.df = df.sort_values('score', ascending=False)


    def baseline(self):
        """Returns the ratio of actually true outcomes to total outcomes."""
        count_true = len(self.df[self.df.actual == 1])
        return count_true / len(self.df)


    def accuracy(self, threshold=None):
        if threshold:
            return metrics.accuracy_score(self.df.actual.values,
                                          self._threshold_predict(threshold))
        else:
            return metrics.accuracy_score(self.df.actual.values,
                                          self.df.predict.values)


    def precision(self, threshold=None):
        if threshold:
            return metrics.precision_score(self.df.actual.values,
                                           self._threshold_predict(threshold))
        else:
            return metrics.precision_score(self.df.actual.values,
                                           self.df.predict.values)


    def recall(self, threshold=None):
        if threshold:
            return metrics.recall_score(self.df.actual.values,
                                        self._threshold_predict(threshold))
        else:
            return metrics.recall_score(self.df.actual.values,
                                        self.df.predict.values)


    def f1(self, threshold=None):
        if threshold:
            return metrics.f1_score(self.df.actual.values,
                                    self._threshold_predict(threshold))
        else:
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


    def _threshold_predict(self, threshold):
        # Stolen from Rayid Ghani
        cutoff_index = int(len(self.df) * (threshold / 100.0))
        return [1 if i < cutoff_index else 0 for i in range(len(self.df))]


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

        if len(stat_df.index) > 1:
            for suffix in self.suffixes:
                plt.plot([str(x) for x in stat_df.index.values],
                         stat_df[suffix].values,
                         label=suffix)

            plt.xlabel('split')
            plt.legend()
        else:
            plt.bar(stat_df.columns, stat_df.iloc[0].values)
            plt.xlabel('models')

        plt.ylabel(stat_name)
        plt.show()
