from taxi_tips_model.config.core import config
from taxi_tips_model.processing.pipeline_helper import ColumnsDroper, ColumnsSelector


def test_columns_selector(input_data):
    data = input_data.head()
    ph = ColumnsSelector(config.model_config.numerical_vars)
    transformed_data = ph.fit_transform(data)
    assert all(
        elem in config.model_config.numerical_vars
        for elem in transformed_data.columns.tolist()
    )


def test_columns_dropper(input_data):
    data = input_data.head()
    ph = ColumnsDroper(config.model_config.numerical_vars)
    transformed_data = ph.fit_transform(data)
    assert config.model_config.numerical_vars not in transformed_data.columns.tolist()
