from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from taxi_tips_model.config.core import config
from taxi_tips_model.processing import pipeline_helper as ph
from taxi_tips_model.processing import preprocessors as pp

FEATURES = config.model_config.features
TIME_VARS = config.model_config.time_vars
CATEGORICAL_VARS = config.model_config.categorical_vars
CATEGORICAL_VARS_WITH_NA = config.model_config.categorical_vars_with_na
NUMERICAL_VARS = config.model_config.numerical_vars
LATLNG_VARS = config.model_config.latlng_vars
LATLNG_PICKUP_VARS = config.model_config.latlng_pickup_vars
LATLNG_DROPOFF_VARS = config.model_config.latlng_dropoff_vars
DROPED_VARS = config.model_config.drop_vars

time_pipe = Pipeline(
    [
        ("selector", ph.ColumnsSelector(variables=TIME_VARS)),
        ("extract", pp.ExtractHourDay(variables=TIME_VARS, drop=True)),
        # ("imputer", pp.SimpleImputer(fill_value=0)), # NA not allowed
        ("dtype", pp.DtypeSetter(dtype=float)),
    ]
)


cat_pipe = Pipeline(
    [
        ("selector", ph.ColumnsSelector(variables=CATEGORICAL_VARS)),
        (
            "dtype",
            pp.DtypeSetter(variables=CATEGORICAL_VARS_WITH_NA, dtype=str),
        ),
        (
            "imputer_community_area",
            pp.SimpleImputer(
                variables=CATEGORICAL_VARS_WITH_NA,
                fill_value="Outside",
            ),
        ),
        ("dtype_", pp.DtypeSetter(dtype="category")),
    ]
)


extras = lambda x: 0 if x > 0 else 1  # noqa: E731

num_pipe = Pipeline(
    [
        ("selector", ph.ColumnsSelector(variables=NUMERICAL_VARS)),
        (
            "outlier",
            pp.OutlierCapper(
                variables=["trip_seconds", "trip_miles", "fare"],
                outlier_mark=True,
            ),
        ),
        (
            "extras",
            pp.ExtractWithApply(variables=["extras"], fn=extras, suffix="bin", drop=True),
        ),
        # ("imputer", pp.SimpleImputer(fill_value=-9)), # NA not allowed
        (
            "dtype",
            pp.DtypeSetter(
                variables=[
                    "trip_seconds_outlier",
                    "trip_miles_outlier",
                    "fare_outlier",
                    "extras_bin",
                ],
                dtype=int,
            ),
        ),
    ]
)


latlng_pipe = Pipeline(
    [
        ("selector", ph.ColumnsSelector(variables=LATLNG_VARS)),
        (
            "pickup",
            pp.ExtractLatLngBucket(
                variables=LATLNG_PICKUP_VARS,
                name="pickup_latlng",
                lat_bins=8,
                lng_bins=8,
            ),
        ),
        (
            "dropoff",
            pp.ExtractLatLngBucket(
                variables=LATLNG_DROPOFF_VARS,
                name="dropoff_latlng",
                lat_bins=8,
                lng_bins=8,
            ),
        ),
        ("imputer", pp.SimpleImputer(fill_value="Outside")),
        ("dtype", pp.DtypeSetter(dtype="category")),
    ]
)


def fn_fare_per_mile(X):
    return X.fare / (X.trip_miles + 1)


def fn_fare_per_sec(X):
    return X.fare * 100 / (X.trip_seconds + 1)


def fn_mile_per_sec(X):
    return X.trip_miles * 1000 / (X.trip_seconds + 1)


craft_pipe = Pipeline(
    [
        ("selector", ph.ColumnsSelector(variables=NUMERICAL_VARS)),
        (
            "fare_per_mile",
            pp.ExtractWithFn(
                variables=["fare", "trip_miles"],
                fn=fn_fare_per_mile,
                name="fare_per_mile",
                drop=False,
            ),
        ),
        (
            "fare_per_sec",
            pp.ExtractWithFn(
                variables=["fare", "trip_seconds"],
                fn=fn_fare_per_sec,
                name="fare_per_sec",
                drop=False,
            ),
        ),
        (
            "mile_per_sec",
            pp.ExtractWithFn(
                variables=["trip_miles", "trip_seconds"],
                fn=fn_mile_per_sec,
                name="mile_per_sec",
                drop=False,
            ),
        ),
        ("droper", ph.ColumnsDroper(variables=NUMERICAL_VARS)),
        # ("imputer", pp.SimpleImputer(fill_value=-9)),
        (
            "outlier",
            pp.OutlierCapper(variables=["fare_per_mile", "fare_per_sec", "mile_per_sec"]),
        ),
        (
            "dtype",
            pp.DtypeSetter(
                variables=["fare_per_mile", "fare_per_sec", "mile_per_sec"],
                dtype=float,
            ),
        ),
    ]
)


data_union = ph.DataFrameUnion(
    [
        ("time", time_pipe),
        ("cat", cat_pipe),
        ("num", num_pipe),
        ("latlng", latlng_pipe),
        ("craft", craft_pipe),
    ],
    verbose=True,
)

clf = LGBMClassifier(
    objective=config.model_config.objective,
    is_unbalance=config.model_config.is_unbalance,
    random_state=config.model_config.random_state,
    # Improve
    num_leaves=config.model_config.num_leaves,
    max_depth=config.model_config.max_depth,
    learning_rate=config.model_config.learning_rate,
    n_estimators=config.model_config.n_estimators,
    colsample_bytree=config.model_config.colsample_bytree,
    force_row_wise=True,
    verbose=2,
)


model_pipeline = Pipeline(
    [
        ("selector", ph.ColumnsSelector(variables=FEATURES)),
        # ("dropper", ph.ColumnsDroper(variables=DROPED_VARS)),
        ("data", data_union),
        ("clf", clf),
    ],
    verbose=True,
)
