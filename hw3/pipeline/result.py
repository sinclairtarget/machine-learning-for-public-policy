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

    Threshold should be a percentage.
    """
    def __init__(self, df, threshold=None):
        self.df = df.sort_values('score', ascending=False)
        self.threshold = threshold


    def with_threshold(self, threshold):
        return PredictionResult(self.df, threshold)


    def with_thresholds(self, thresholds):
        return [self.with_threshold(t) for t in thresholds]


    def baseline(self):
        """Returns the ratio of actually true outcomes to total outcomes."""
        count_true = len(self.df[self.df.actual == 1])
        return count_true / len(self.df)


    def accuracy(self):
        if self.threshold:
            return metrics.accuracy_score(self.df.actual.values,
                                          self._threshold_predict())
        else:
            return metrics.accuracy_score(self.df.actual.values,
                                          self.df.predict.values)


    def precision(self):
        if self.threshold:
            return metrics.precision_score(self.df.actual.values,
                                           self._threshold_predict())
        else:
            return metrics.precision_score(self.df.actual.values,
                                           self.df.predict.values)


    def recall(self):
        if self.threshold:
            return metrics.recall_score(self.df.actual.values,
                                        self._threshold_predict())
        else:
            return metrics.recall_score(self.df.actual.values,
                                        self.df.predict.values)


    def f1(self):
        if self.threshold:
            return metrics.f1_score(self.df.actual.values,
                                    self._threshold_predict())
        else:
            return metrics.f1_score(self.df.actual.values,
                                    self.df.predict.values)


    def auc(self):
        if self.threshold:
            return metrics.roc_auc_score(self.df.actual.values,
                                         self._threshold_predict())
        else:
            return metrics.roc_auc_score(self.df.actual.values,
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
            'f1': self.f1(),
            'auc': self.auc()
        })


    def _threshold_predict(self):
        # Stolen from Rayid Ghani
        cutoff_index = int(len(self.df) * (self.threshold / 100.0))
        return [1 if i < cutoff_index else 0 for i in range(len(self.df))]


class ResultCollection:
    """
    Wrapper around a collection of PredictionResult data that can draw graphs.
    """
    def __init__(self, df=DataFrame()):
        self.df = df
        self.suffixes = []


    def join(self, suffix, result_or_collection):
        if isinstance(result_or_collection, PredictionResult):
            collection = ResultCollection.from_stack([result_or_collection])
        else:
            collection = result_or_collection

        incoming_df = collection.df.copy(deep=False)

        incoming_df.columns = \
            [colname + '_' + suffix for colname in incoming_df.columns]

        self.df = self.df.join(incoming_df, how='right')
        self.suffixes.append(suffix)


    def plot_statistic(self, stat_name, xlabel='split'):
        stat_df = self.df.filter(regex=stat_name)
        stat_df.columns = self.suffixes

        if len(stat_df.index) > 1:
            for suffix in self.suffixes:
                plt.plot([str(x) for x in stat_df.index.values],
                         stat_df[suffix].values,
                         label=suffix)
            plt.legend()
        else:
            plt.figure(figsize=(9, 3))
            plt.bar(stat_df.columns, stat_df.iloc[0].values)

        plt.xlabel(xlabel)
        plt.ylabel(stat_name)
        plt.show()


    def from_stack(results, index=None):
        index = index or list(range(1, len(results) + 1))
        return ResultCollection(DataFrame([r.as_series() for r in results],
                                          index=index))
