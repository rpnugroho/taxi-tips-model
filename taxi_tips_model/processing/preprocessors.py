from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer as SkSimpleImputer
from sklearn.preprocessing import KBinsDiscretizer


class ExtractHourDay(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str], drop: bool = True):
        """Extract hour and day of week from timestamp and create new column.

        Args:
            variables (List[str]): Timestamp column name to extract.
            suffix (bool, optional): If True, use timestamp column name as suffix.
                Defaults to False.
            drop (bool, optional): If True, drop timestamp column. Defaults to True.
        """
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.drop = drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].apply(pd.to_datetime)
        for feature in self.variables:
            X[f"{feature}_day"] = X[feature].dt.dayofweek.astype(float)
            X[f"{feature}_hour"] = X[feature].dt.hour.astype(float)
        if self.drop:
            return X.drop(columns=self.variables)
        return X


class ExtractLastWord(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str], suffix: str, drop: bool = True):
        """Extract last word from a column

        Args:
            variables (List): Columns name to extract.
            suffix (str): Suffix of created columns
            drop (bool, optional): If True, drop timestamp column. Defaults to True.
        """
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.suffix = suffix
        self.drop = drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for feature in self.variables:
            X[f"{feature}_{self.suffix}"] = X[feature].squeeze().str.split(" ").str[-1]
        if self.drop:
            return X.drop(columns=self.variables)
        return X


class ExtractLatLngBucket(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        variables: List[str],
        lat_bins: int = 5,
        lng_bins: int = 5,
        name: str = "latlng_bucket",
        drop: bool = True,
    ):
        """Extract/Bucketize Latitude and Longitude into one columns.

        Args:
            variables (List): Latitude and Longitude columns name.
            lat_bins (int, optional): Latitude bins. Defaults to 5.
            lng_bins (int, optional): Longitude bins. Defaults to 5.
            name (str, optional): Name of bucketize feature. Defaults to "latlng_bucket".
            drop (bool, optional): If True, drop Latitude and Longitude columns.
                Defaults to True.

        """
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.lat_bins = lat_bins
        self.lng_bins = lng_bins
        self.name = name
        self.drop = drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        X = X.copy()
        # Change dtypes to float
        X[self.variables] = X[self.variables].astype(float)
        # Deal with missing values
        self.imputer = SkSimpleImputer(strategy="constant", fill_value=0)
        X[self.variables] = self.imputer.fit_transform(X[self.variables])
        # Create Discretizer
        self.lat_encoder = KBinsDiscretizer(n_bins=self.lat_bins, encode="ordinal")
        self.lng_encoder = KBinsDiscretizer(n_bins=self.lng_bins, encode="ordinal")
        self.lat_encoder.fit(X[[self.variables[0]]])
        self.lng_encoder.fit(X[[self.variables[1]]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        na_mask = X[self.variables[0]].isna()
        # Change dtypes
        X[self.variables] = X[self.variables].astype(float)
        # Deal with missing values
        X[self.variables] = self.imputer.transform(X[self.variables])
        # Discretizer
        X[[self.variables[0]]] = self.lat_encoder.transform(X[[self.variables[0]]])
        X[[self.variables[1]]] = self.lng_encoder.transform(X[[self.variables[1]]])
        # Category Crossings
        X[self.name] = X[self.variables].astype(int).astype(str).agg("X".join, axis=1)
        X[self.name] = X[self.name].mask(na_mask, np.nan)
        if self.drop:
            return X.drop(columns=self.variables)
        return X


class ExtractWithFn(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str], fn, name: str, drop: bool = True):
        """Extract a new feature by applying a function.

        Args:
            variables (List): Columns name to extract.
            name (str): Name of created columns
            fn : Function to Extract new feature.
            drop (bool, optional): If True, drop timestamp column. Defaults to True.
        """
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.name = name
        self.fn = fn
        self.drop = drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.name] = self.fn(X[self.variables])
        if self.drop:
            return X.drop(columns=self.variables)
        return X


class ExtractWithApply(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str], fn, suffix: str, drop: bool = True):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.suffix = suffix
        self.fn = fn
        self.drop = drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[f"{feature}_{self.suffix}"] = X[feature].apply(self.fn)
        if self.drop:
            return X.drop(columns=self.variables)
        return X


class DtypeSetter(TransformerMixin, BaseEstimator):
    def __init__(self, variables: List[str] = [], dtype=int):
        """Set column into desire dtypes.

        Args:
            variables (List): Columns name to transform.
            dtype ([type], optional): [description]. Defaults to int.
        """
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.dtype = dtype

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = X.columns
        na_mask = X.isna()
        if self.variables == []:
            X[cols] = X.astype(self.dtype)
        else:
            X[self.variables] = X[self.variables].astype(self.dtype)
        if self.dtype == int:
            return X
        return X.mask(na_mask, np.nan)


class StringRemover(BaseEstimator, TransformerMixin):
    def __init__(self, variables: List[str], regex: str):
        """Remove string in a columns.

        Args:
            variables (List): Columns name to transform.
            regex (str): Regex
        """
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.regex = regex

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].squeeze().str.replace(self.regex, "", regex=True)
        return X


class SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        variables: List[str] = [],
        missing_values=np.nan,
        strategy: str = "constant",
        fill_value=None,
    ):
        """Impute missing value with SimpleImputer.

        Args:
            variables (List): Columns name to impute.
            missing_values ([type], optional): Value to input. Defaults to np.nan.
            strategy (str, optional): Impute strategy. Defaults to "constant".
            fill_value ([type], optional): Fill value. Defaults to None.
        """
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.missing_values = missing_values
        self.strategy = strategy
        if fill_value is not None:
            self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.imputer = SkSimpleImputer(
            missing_values=self.missing_values,
            strategy=self.strategy,
            fill_value=self.fill_value,
        )
        if self.variables == []:
            self.imputer.fit(X)
        else:
            self.variable_ = np.intersect1d(self.variables, X.columns).tolist()
            self.imputer.fit(X[self.variable_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = X.columns
        if self.variables == []:
            X[cols] = self.imputer.transform(X)
        else:
            X[self.variable_] = self.imputer.transform(X[self.variable_])
        return X


class RareLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self, variables: List[str], th_rare: float = 0.05, fill_value: str = "Rare"
    ):
        """Encode poorly represented label into new category.

        Args:
            variables (List): Columns name to encode.
            th_rare (float, optional): Rare label threshold. Defaults to 0.05.
            fill_value (str, optional): Fill value. Defaults to "Rare".
        """
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.th_rare = th_rare
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # persist frequent labels in dictionary
        self.encoder_dict_ = {}
        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float_(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.th_rare].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]),
                X[feature],
                self.fill_value,
            )
        return X


class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        variables: List[str],
        tail: str = "right",
        fold: float = 1.5,
        outlier_mark: bool = False,
        mark_suffix: str = "outlier",
        drop: bool = False,
    ):
        """Capper outlier into 1.5 IQR or 0.5 IQR.

        Args:
            variables (List): Column to cap.
            tail (str, optional): Whether to cap outliers on the "right",
                "left" or "both". Defaults to "right".
            outlier_mark (bool, optional): If True, reate marker columns.
                Defaults to False.
            mark_suffix (str, optional): Suffix marker column. Defaults to "mark".
            drop (bool, optional): If True, drop timestamp column. Defaults to False.
        """
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.tail = tail
        self.outlier_mark = outlier_mark
        self.mark_suffix = mark_suffix
        self.fold = fold
        self.drop = drop

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.right_tail_caps_ = {}
        self.left_tail_caps_ = {}

        if self.tail in ["right", "both"]:
            IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
            self.right_tail_caps_ = (
                X[self.variables].quantile(0.75) + (IQR * self.fold)
            ).to_dict()

        if self.tail in ["left", "both"]:
            IQR = X[self.variables].quantile(0.75) - X[self.variables].quantile(0.25)
            self.left_tail_caps_ = (
                X[self.variables].quantile(0.25) - (IQR * self.fold)
            ).to_dict()

        self.input_shape_ = X.shape
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()  # replace outliers
        for feature in self.right_tail_caps_.keys():
            if self.outlier_mark:
                X[f"{feature}_{self.mark_suffix}"] = np.where(
                    X[feature] > self.right_tail_caps_[feature], 1, 0
                )
            X[feature] = np.where(
                X[feature] > self.right_tail_caps_[feature],
                self.right_tail_caps_[feature],
                X[feature],
            )

        for feature in self.left_tail_caps_.keys():
            if self.outlier_mark:
                X[f"{feature}_{self.mark_suffix}"] = np.where(
                    X[feature] < self.left_tail_caps_[feature], 1, 0
                )
            X[feature] = np.where(
                X[feature] < self.left_tail_caps_[feature],
                self.left_tail_caps_[feature],
                X[feature],
            )

        if self.drop:
            return X.drop(columns=self.variables)
        return X
