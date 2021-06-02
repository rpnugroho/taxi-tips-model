from taxi_tips_model.processing.validation import validate_inputs


def test_validate_inputs(input_data):
    # When
    validated_inputs, errors = validate_inputs(input_data=input_data)

    # Then
    assert not errors
    # we expect no row removed
    assert len(input_data) == len(validated_inputs)


def test_validate_inputs_identifies_errors(input_data):
    # Given
    test_inputs = input_data[0:2].copy()

    # introduce errors
    test_inputs["fare"] = test_inputs["fare"].astype(str)
    test_inputs.at[0, "fare"] = "outside"
    # When
    validated_inputs, errors = validate_inputs(input_data=test_inputs)

    # Then
    assert errors
    # assert errors[1] == {"BldgType": ["Not a valid string."]}
