from taxi_tips_model.config.core import config
from taxi_tips_model.pipeline import model_pipeline, time_pipe


def test_time_pipeline(sample_input_data):
    data = time_pipe.fit_transform(sample_input_data)

    assert len(data) == len(sample_input_data)
    assert len(data.columns) != len(sample_input_data.columns)


def test_pipeline_drops_unnecessary_features(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs

    # When
    X_transformed = model_pipeline[:-1].fit_transform(X_train, y_train)

    # Then
    assert config.model_config.drop_vars not in X_transformed.columns.tolist()
