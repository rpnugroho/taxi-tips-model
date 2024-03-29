import logging
import typing as t

import dill
import pandas as pd
from sklearn.pipeline import Pipeline

from taxi_tips_model import __version__ as _version
from taxi_tips_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{DATASET_DIR}/{file_name}")
    return _data


def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    # Remove old pipeline
    remove_old_pipelines(files_to_keep=[save_file_name])
    # Save pipeline using dill (joblib didnt work with lambda function)
    with open(save_path, "wb") as f:
        dill.dump(pipeline_to_persist, f)
    logging.info(f"saved pipeline: {save_file_name}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    # Load pipeline
    with open(file_path, "rb") as f:
        trained_model = dill.load(f)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.

    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    However, we do also include the immediate previous
    pipeline version for differential testing purposes.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
