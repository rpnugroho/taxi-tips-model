from taxi_tips_model.predict import make_prediction


def test_make_single_prediction(input_data):
    # Given
    single_test_input = input_data[-1:]

    # When
    result = make_prediction(input_data=single_test_input)

    # Then
    assert result is not None
    assert isinstance(result.get("trip_id"), list)
    assert isinstance(result.get("trip_id")[0], str)
    assert isinstance(result.get("predictions"), list)
    assert isinstance(result.get("predictions")[0], int)
    assert result.get("predictions")[0] == 1
    assert result.get("errors") is None


def test_make_multiple_predictions(input_data):
    # Given
    multiple_test_input = input_data.head(100)

    expected_no_predictions = 100

    # When
    result = make_prediction(input_data=multiple_test_input)

    # Then
    assert result is not None
    assert isinstance(result.get("trip_id"), list)
    assert isinstance(result.get("trip_id")[0], str)
    assert isinstance(result.get("predictions"), list)
    assert isinstance(result.get("predictions")[0], int)
    assert len(result.get("predictions")) == expected_no_predictions
