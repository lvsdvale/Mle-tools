"""this file contains the fixtures for evaluation"""
import pytest

import os
import sys
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(PARENT_DIR)
sys.path.append(PROJECT_DIR)

from mock_model import MockModel


@pytest.fixture
def mock_empty_dataframe():
    """
    Returns an empty DataFrame.
    """
    return pd.DataFrame()


@pytest.fixture
def mock_dataframe_with_categorical():
    """
    Returns a DataFrame with one categorical column.
    """
    data = {"category": pd.Categorical(["A", "B", "C", "A", "B"])}
    return pd.DataFrame(data)


@pytest.fixture
def mock_dataframe():
    """
    Returns a DataFrame with at least 4 columns:
    - 3 columns of type float
    - 1 column of type int
    All column names are generic.
    """
    data = {
        "float_column_1": np.random.rand(10),
        "float_column_2": np.random.rand(10),
        "target_column": np.random.rand(10),
        "int_column": np.random.randint(0, 1, size=10),
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_model():
    return MockModel()
