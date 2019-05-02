from pandas import DataFrame
import sklearn.metrics as metrics

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


    def matrix(self):
        m = metrics.confusion_matrix(self.df.actual.values,
                                     self.df.predict.values)
        return DataFrame(m,
                         index=['false', 'true'],
                         columns=['negative', 'positive'])
