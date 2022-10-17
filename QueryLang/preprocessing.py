from collections import defaultdict
from numbers import Integral
import numpy as np
import pandas as pd
from scipy import stats


class MultiColumnEncoder:
    def __init__(self, target=True, m=100):
        self.is_te = target
        self.m = m
        self.dtypes = None
        self.features = None
        self.target = None
        self.columns = None
        self.mads = {}
        self.cat_vars_ord = defaultdict()
        self.encoder = defaultdict()
        self.decoder = defaultdict()
        self.label_encoder = defaultdict()
        self.label_decoder = defaultdict()

    def is_category(self, feature):
        return feature in self.cat_vars_ord

    def fit(self, X, y):
        self.columns = [col for col in X.columns if X[col].dtype == object]
        self.target = y.name
        self.features = X.columns
        self.dtypes = X.dtypes
        mean = y.mean()
        df = X.join(y)
        for i, col in enumerate(self.columns):
            # Compute the number of values and the mean of each group
            agg = df.groupby(col)[self.target].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']
            # Compute the "smoothed" means
            self.encoder[col] = (counts * means + self.m * mean) / (counts + self.m)
            self.label_encoder[col] = {k: i for i, (k, v) in enumerate(self.encoder[col].items())}
            self.cat_vars_ord[i] = len(self.encoder[col])
            self.decoder[col] = pd.Series(self.encoder[col].index.values, index=self.encoder[col])
            self.label_decoder[col] = {i: k for k, i in self.label_encoder[col].items()}
        for f in self.features:
            if f in self.columns:
                continue
            mad = stats.median_absolute_deviation(df[f])
            self.mads[f] = 1 if mad == 0 else mad
        if not self.is_te:
            self.encoder, self.decoder = self.label_encoder, self.label_decoder
        return self

    def is_category(self, i):
        return i < len(self.columns)

    def is_integer(self, feature):
        if self.dtypes[feature] == np.int64 or self.dtypes[feature] == np.int32:
            return True
        return False

    def _transform(self, X, encoder):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = output[col].map(encoder[col])
        return output

    def transform(self, X):
        return self._transform(X, self.encoder)

    def transform_to_project(self, X):
        return self._transform(X, self.label_encoder)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def _inverse_transform(self, X, decoder):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = output[col].map(decoder[col])
        return output

    def inverse_transform(self, X):
        return self._inverse_transform(X, self.decoder)

    def inverse_transform_to_project(self, X):
        return self._inverse_transform(X, self.label_decoder)

    def closest(self, col, threshold: float, bigger=True):
        if isinstance(col, Integral):
            col = self.features[col]
        arr = list(self.decoder[col].keys())
        if bigger:
            if max(arr) <= threshold:
                return max(arr)  # can't find bigger value
            return min(filter(lambda x: x > threshold, arr))
        else:
            if min(arr) > threshold:
                return min(arr)  # can't find smaller value
            return max(filter(lambda x: x <= threshold, arr))
