from typing import Union

import pandas as pd

from taxi_tips_model import __version__ as _version
from taxi_tips_model.config.core import config
from taxi_tips_model.processing.data_manager import load_pipeline
from taxi_tips_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_model_pipeline = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)

    predictions = None

    if errors:
        results = {
            "trip_id": validated_data["trip_id"].values.tolist(),
            "predictions": predictions,
            "version": _version,
            "errors": errors,
        }
    else:

        predictions = _model_pipeline.predict(X=validated_data)
        results = {
            "trip_id": validated_data["trip_id"].values.tolist(),
            "predictions": predictions.tolist(),
            "version": _version,
            "errors": errors,
        }

    return results
