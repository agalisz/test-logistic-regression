import numpy as np
import pandas as pd
import datetime
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from bisect import bisect_right

class OneHotDummyTransformer(TransformerMixin, BaseEstimator):
    '''
    A transformer for one hot encoding.
    Supports strings, handles unknown values (keeps the fit columns).
    '''

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        X_oneHot = pd.get_dummies(X, columns=self.columns)
        self.fit_columns = X_oneHot.columns
        return self

    def transform(self, X):
        X_oneHot = pd.get_dummies(X, columns=self.columns)
        X_oneHot_columns = set(X_oneHot.columns)
        for c in self.fit_columns:
            if c not in X_oneHot_columns:
                X_oneHot[c] = 0
        return X_oneHot[self.fit_columns]
