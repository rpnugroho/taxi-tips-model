import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one


class DataFrameUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans, X=X, y=y, weight=weight, **fit_params
            )
            for name, trans, weight in self._iter()
        )
        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        Xs = self.merge_dataframes_by_column(Xs)
        # get all fitted columns
        self.fitted_columns = Xs.columns

        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        X = X.copy()
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(transformer=trans, X=X, y=None, weight=weight)
            for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        X_ = self.merge_dataframes_by_column(Xs)
        # Create data for missing columns
        missing_vars = [var for var in self.fitted_columns if var not in X_.columns]
        if len(missing_vars) != 0:
            for var in missing_vars:
                X_[var] = 0
        # Matching transformed column and fitted columns
        X_ = X_[self.fitted_columns].copy()
        return X_


class ColumnsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variables, dtype=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.dtype:
            na_mask = X.isna()
            _X = X[self.variables].astype(self.dtype)
            return _X.mask(na_mask, np.nan)
        else:
            return X[self.variables]


class ColumnsDroper(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.variables)
