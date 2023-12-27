import pandas as pd
import pytest
from unittest.mock import Mock, patch
try:
    from premier_league import preprocessing
except ModuleNotFoundError:
    import preprocessing

def test_custom_transformer_init():
    transformer = preprocessing.CustomTransformer(
        dtypes=['int', 'float'])
    assert transformer.dtypes == ['int', 'float']

def test_custom_transformer_fit(mocker):
    # Mock preprocessing_helpers.run_pipeline
    mock_pipeline = mocker.patch(
        'premier_league.preprocessing_helpers.run_pipeline')
    mock_pipeline.return_value = pd.DataFrame(columns=['col1', 'col2'])

    # Create a sample DataFrame
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': ['a', 'b', 'c']
    })

    transformer = preprocessing.CustomTransformer()
    transformer.fit(df)

    assert transformer.input_columns == ['col3']
    assert list(transformer.final_columns) == ['col1', 'col2']

def test_custom_transformer_transform(mocker):
    # Mock preprocessing_helpers.run_pipeline
    mock_pipeline = mocker.patch(
        'premier_league.preprocessing_helpers.run_pipeline')
    mock_pipeline.return_value = pd.DataFrame(columns=['col1', 'col2'])

    # Create sample dataframes for fit and transform
    df_fit = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3': ['a', 'b']})
    df_transform = pd.DataFrame({'col1': [5, 6], 'col2': [7, 8]})

    transformer = preprocessing.CustomTransformer()
    transformer.fit(df_fit)
    transformed = transformer.transform(df_transform)

    # Check if the transformed dataframe has the correct columns
    assert list(transformed.columns) == ['col1', 'col2']
    assert 'col3' not in transformed.columns

def test_custom_transformer_get_feature_names():
    transformer = preprocessing.CustomTransformer()
    transformer.final_columns = ['col1', 'col2']
    feature_names = transformer.get_feature_names()
    assert feature_names == ('col1', 'col2')

def test_split_data(mocker):
    # Mocking logger
    mocker.patch('premier_league.logger_config.logger')

    # Create a sample DataFrame
    df = pd.DataFrame({
        'team': ['Arsenal', 'Liverpool', 'Chelsea', 'Man City', 'Fulham'],
        'goals': [1,4,2,5,4]
    })

    # Call the split_data function
    train_ratio = 0.8
    train_data, test_data = preprocessing.split_data(df, train_ratio)

    # Test total length
    assert len(train_data) + len(test_data) == len(df)

    # Test train ratio
    assert len(train_data) == 4

    # Test mutual exclusivity
    assert len(test_data) == 1

