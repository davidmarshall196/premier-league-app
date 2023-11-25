
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from typing import Tuple
try:
    from premier_league import (
        preprocessing_helpers,
        logger_config
    )
except ImportError:
    import preprocessing_helpers
    import logger_config

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dtypes=None):
        self.input_columns = None
        self.final_columns = None
        if dtypes is None:
            dtypes = [object, 'category']
        self.dtypes = dtypes

    def fit(self, X, y=None, **kwargs):
        logger_config.logger.info("Fitting Transformer")
        self.input_columns = list(X.select_dtypes(self.dtypes).columns)
        X = preprocessing_helpers.run_pipeline(X)
        self.final_columns = X.columns
        return self
        
    def transform(self, X, y=None, **kwargs):
        logger_config.logger.info("Transforming Data")
        X = preprocessing_helpers.run_pipeline(X)
        X_columns = X.columns
        # if columns in X had values not in the data set used during
        # fit add them and set to 0
        missing = set(self.final_columns) - set(X_columns)
        for c in missing:
            X[c] = 0
        # remove any new columns that may have resulted from values in
        # X that were not in the data set when fit
        return X[self.final_columns]
    
    def get_feature_names(self):
        return tuple(self.final_columns)    
    
def fit_transformers(x_train: pd.DataFrame):
    """Fit an sklearn ColumnTransformer on the data and return it with the tranformed data.

    Args:
        x_train (pandas.DataFrame): Data upon which to train.

    Returns:
        (sklearn.compose.ColumnTransformer): Transformed data and fitted transformer.
    """
    logger_config.logger.info("Fitting Transformer")
    transformers = CustomTransformer() 
    x_train = transformers.fit(x_train)
    return transformers

def transform_data(
    data: pd.DataFrame, 
    transformer,
) -> pd.DataFrame:
    """Transform data using provided transformer.

    Args:
        data (pandas.DataFrame): Data to transform.
        transformer: scikit-learn style transformer.
        target_column (str): Target column of model. If specified, column is
            set as column 0 in the returned data.

    Returns:
        transformed_data (pandas.DataFrame): Transformed data.
    """
    logger_config.logger.info("Transforming Data")
    transformed_data = transformer.transform(
        data
    )
    
    return transformed_data


def split_data(
    data: pd.DataFrame, 
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split passed data into training and testing datasets.

    Args:
        data (pandas.DataFrame): Data to split.
        train_ratio (float): Ratio of data to assign to training data. Min 0, max 1.

    Returns:
        (pandas.DataFrame): Training dataset.
        (pandas.DataFrame): Testing dataset.
    """
    logger_config.logger.info("Splitting Data")
    train_data = data.sample(frac=train_ratio, axis=0)
    test_data = data[~data.isin(train_data)].dropna().reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)
    return train_data, test_data


def validate_data(
        data: pd.DataFrame, 
        data_expectations: dict
    ) -> dict:
    """Validate data using predifined Great Expectations suite
    
    Args:
        data (pandas.DataFrame): Data to validate.
        data_expectations (dict): Expectations used for the validation.

    Returns:
        (dict): Validation results.
    """
    logger_config.logger.info("Validating Data")
    data = ge.from_pandas(data, expectation_suite=data_expectations)
    validation_results = data.validate()
    return validation_results.to_json_dict()



