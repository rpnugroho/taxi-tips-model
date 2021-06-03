import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from taxi_tips_model import pipeline
from taxi_tips_model.config.core import config
from taxi_tips_model.processing.data_manager import load_dataset, save_pipeline
from taxi_tips_model.processing.validation import drop_na_inputs


def run_training() -> None:
    """Train the model."""
    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    # filter missing data in some columns
    data = drop_na_inputs(input_data=data)
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
    pipeline.model_pipeline.fit(X_train, y_train)

    y_pred = pipeline.model_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    save_pipeline(pipeline_to_persist=pipeline.model_pipeline)


if __name__ == "__main__":
    run_training()
