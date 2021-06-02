import logging
import os
import pathlib

import numpy as np
import pandas as pd
from sodapy import Socrata

import taxi_tips_model

_logger = logging.getLogger(__name__)


PACKAGE_ROOT = pathlib.Path(taxi_tips_model.__file__).resolve().parent

DATE_START = "2021-04-05"
DATE_END = "2021-04-11"
QUERY = (
    f"trip_start_timestamp between '{DATE_START}T00:00:00' and '{DATE_END}T23:59:59'"
)

RAW_DATASET = PACKAGE_ROOT / f"datasets/raw/{DATE_START}_to_{DATE_END}.csv"
INTERIM_DATASET = PACKAGE_ROOT / f"datasets/interim/{DATE_START}_to_{DATE_END}.csv"

CASH_DATASET = PACKAGE_ROOT / "datasets/processed/test.csv"
NON_CASH_DATASET = PACKAGE_ROOT / "datasets/processed/train.csv"


def download_dataset(query, raw_dataset):
    client = Socrata("data.cityofchicago.org", None, timeout=60)
    results = client.get("wrvz-psew", limit=99999, where=query)
    df = pd.DataFrame.from_records(results)
    df.to_csv(raw_dataset, index=False)
    _logger.info("Download completed")


def modify_dataset(raw_dataset, interim_dataset, tips_th=0, drop_columns=None):

    df = pd.read_csv(raw_dataset)
    # Create label
    logging.info("Create label")
    df["target"] = np.where(df["tips"].astype(float) > tips_th, 1, 0)
    # Drop unused columns
    if drop_columns is not None:
        df = df.drop(columns=drop_columns, errors="ignore")
    # Save modified dataset
    df.to_csv(interim_dataset, index=False)
    _logger.info("Interim dataset saved")


def separate_dataset(interim_dataset, cash_dataset, non_cash_dataset):
    df = pd.read_csv(interim_dataset)
    # Separate data
    _logger.info("Separate dataset")
    non_cash = df[df["payment_type"] != "Cash"]
    cash = df[df["payment_type"] == "Cash"]
    non_cash = non_cash.drop(columns=["payment_type"])
    cash = cash.drop(columns=["payment_type"])
    # Save modified dataset
    non_cash.to_csv(non_cash_dataset, index=False)
    cash.to_csv(cash_dataset, index=False)
    _logger.info("Separated dataset saved")


def check_execution_path():
    directory = "taxi_tips_model/"
    if not os.path.exists(directory):
        _logger.error(
            "Don't execute the script from a sub-directory."
            "Switch to the root of the project folder"
        )
        return False
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _logger.info("Started download script")

    if check_execution_path():
        download_dataset(query=QUERY, raw_dataset=RAW_DATASET)
        modify_dataset(
            raw_dataset=RAW_DATASET,
            interim_dataset=INTERIM_DATASET,
            tips_th=0,
            drop_columns=None,
        )
        separate_dataset(
            interim_dataset=INTERIM_DATASET,
            cash_dataset=CASH_DATASET,
            non_cash_dataset=NON_CASH_DATASET,
        )
    logging.info("Finished")
