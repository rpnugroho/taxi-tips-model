from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from taxi_tips_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    vars_na_not_allowed = config.model_config.features_na_not_allowed

    validated_data.dropna(subset=vars_na_not_allowed, inplace=True)

    return validated_data


def drop_unused_columns(*, input_data: pd.DataFrame) -> pd.DataFrame:
    validated_data = input_data.copy()
    validated_data = validated_data.drop(columns=config.model_config.drop_vars)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()

    errors = None

    try:
        validated_data = MultipleTaxiDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        ).dict()

    except ValidationError as error:
        errors = error.json()

    if errors is None:
        validated_data = pd.DataFrame(
            [dict(s) for s in validated_data.get("inputs")]
        ).replace({None: np.nan})

    return validated_data, errors


class TaxiDataInputSchema(BaseModel):
    trip_id: str  # NaN not allowed
    taxi_id: Optional[str]
    trip_start_timestamp: Optional[str]
    trip_end_timestamp: Optional[str]
    trip_seconds: Optional[float]
    trip_miles: Optional[float]
    fare: Optional[float]
    tips: Optional[float]
    tolls: Optional[float]
    extras: Optional[float]
    trip_total: Optional[float]
    company: Optional[str]
    pickup_community_area: Optional[float]
    dropoff_community_area: Optional[float]
    pickup_centroid_latitude: Optional[float]
    pickup_centroid_longitude: Optional[float]
    pickup_centroid_location: Optional[str]
    dropoff_centroid_latitude: Optional[float]
    dropoff_centroid_longitude: Optional[float]
    dropoff_centroid_location: Optional[str]
    pickup_census_tract: Optional[str]
    dropoff_census_tract: Optional[float]


class MultipleTaxiDataInputs(BaseModel):
    inputs: List[TaxiDataInputSchema]
