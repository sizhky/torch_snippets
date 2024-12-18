# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/sklegos.ipynb.

# %% auto 0
__all__ = ['ColumnSelector', 'GroupedPredictor', 'EstimatorTransformer', 'train_test_split', 'MakeFrame', 'ImputeMissingValues',
           'LambdaTransformer', 'Cat2Num', 'SplitDateColumn']

# %% ../nbs/sklegos.ipynb 3
from . import *
from collections import defaultdict

try:
    from sklearn.model_selection import train_test_split as tts
except ModuleNotFoundError:
    logger.Exception("Please `pip install scikit-learn sklego` to use this submodule")


def train_test_split(*args, **kwargs):
    outputs = tts(*args, **kwargs)
    outputs = [i.reset_index(drop=True) for i in outputs]
    return outputs

# %% ../nbs/sklegos.ipynb 4
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin

from sklego.preprocessing import ColumnSelector

ColumnSelector = ColumnSelector

# %% ../nbs/sklegos.ipynb 5
from fastcore.basics import patch_to
from sklego.meta import (
    # GroupedEstimator,
    GroupedPredictor,
    GroupedTransformer,
    EstimatorTransformer,
)

# GroupedEstimator = GroupedEstimator


# @patch_to(GroupedEstimator)
# def transform(self, X, y=None):
#     return self.predict(X)


GroupedPredictor = GroupedPredictor


@patch_to(GroupedPredictor)
def transform(self, X, y=None):
    return self.predict(X)


EstimatorTransformer = EstimatorTransformer

# %% ../nbs/sklegos.ipynb 6
class MakeFrame(BaseEstimator, TransformerMixin):
    """Convert sklearn's output to a pandas dataframe
    Especially useful when working with an ensemble of models
    """

    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return X

    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.column_names)

# %% ../nbs/sklegos.ipynb 8
class ImputeMissingValues(BaseEstimator, TransformerMixin):
    """DataFrame input - DataFrame output
    During fit -
    1. Store imputable value for each column
    During transform -
    2. Impute missing values with imputable value
    3. Create a '{col}_na' boolean column to tell if cells contained missing value
    """

    def __init__(self, num_mode=np.mean, cat_mode="MISSING"):
        self.num_mode = num_mode
        self.cat_mode = lambda x: cat_mode if isinstance(cat_mode, str) else cat_mode

    def fit(self, trn_df, y=None):
        assert isinstance(
            trn_df, pd.DataFrame
        ), """
        Transform is a df-input df-output transform
        """.strip()
        self.columns = trn_df.columns
        self.imputable_values = {}
        for col in self.columns:
            _col = trn_df[col]
            a = ~_col.isna()
            ixs = a[a].index
            _col = _col[ixs]
            if _col.dtype != "object":
                self.imputable_values[col] = self.num_mode(_col.values)
            else:
                self.imputable_values[col] = self.cat_mode(_col.values)
        return trn_df

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col not in X.columns:
                continue
            ixs = X[col].isna()
            jxs = ixs[ixs].index
            X.loc[jxs, col] = [self.imputable_values[col]] * len(jxs)
            X[f"{col}_na"] = ixs
        return X

    def fit_transform(self, trn_df, y=None):
        return self.transform(self.fit(trn_df, y))

# %% ../nbs/sklegos.ipynb 9
class LambdaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y=None):
        return X

    def predict(self, X, y=None):
        return self.fn(X)

    def predict_proba(self, X, y=None):
        return self.fn(X)

    def transform(self, X, y=None):
        return self.fn(X)

    def fit_transform(self, X, y=None):
        return self.fit(self.transform(X))

# %% ../nbs/sklegos.ipynb 10
class MakeFrame(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return X

    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.column_names)

    def predict_proba(self, X, y=None):
        return self.transform(X)

    def predict(self, X, y=None):
        return self.transform(X)

# %% ../nbs/sklegos.ipynb 11
class Cat2Num(BaseEstimator, TransformerMixin):
    def __init__(self): ...

    def fit(self, df, y=None):
        self.cat_cols = df.select_dtypes("object").columns
        self.ids = {}
        for col in self.cat_cols:
            _d = defaultdict(lambda: 0)  # 0 is reserved for the unknown
            _d.update({id: ix + 1 for ix, id in enumerate(df[col].unique())})
            self.ids[col] = _d
        return df

    def transform(self, df, y=None):
        for col in self.cat_cols:
            df.loc[:, col] = df[col].map(lambda x: self.ids[col][x])
        return df

    def fit_transform(self, trn_df, y=None):
        return self.transform(self.fit(trn_df, y))

# %% ../nbs/sklegos.ipynb 12
class SplitDateColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column_names, has_date, has_time, date_format=None):
        self.column_names = (
            column_names if isinstance(column_names, list) else [column_names]
        )
        self.has_date = has_date
        self.has_time = has_time

    def fit(self, X, y=None):
        return X

    def transform(self, X, y=None):
        dfs = {}
        for col in self.column_names:
            _col = pd.DatetimeIndex(X[col])
            attrs = []
            if self.has_date:
                attrs = attrs + ["day", "month", "year", "weekday", "weekofyear"]
            if self.has_time:
                attrs = attrs + ["hour", "minute", "second"]
            dfs.update({f"{col}_{attr}": getattr(_col, attr) for attr in attrs})
        _df = pd.DataFrame(dfs)
        _df.index = X.index
        X = pd.concat([X, _df], axis=1)
        return pd.DataFrame(X)

    def predict_proba(self, X, y=None):
        return self.transform(X)

    def predict(self, X, y=None):
        return self.transform(X)
