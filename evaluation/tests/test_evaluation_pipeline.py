"""File for evaluation pipeline test"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

from mock_evaluation_pipeline import MockEvaluationPipeline
from fixtures import *  # noqa: F403


def test_class_attributes(mock_dataframe, mock_model):
    """Test if the attributes were create correctly."""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    required_attributes = [
        "data",
        "target_column",
        "models",
        "kfold",
        "n_splits",
        "trained",
        "report",
    ]
    print(evaluation_pipeline_instance.__dict__.keys())

    for attr in required_attributes:
        assert hasattr(
            evaluation_pipeline_instance, attr
        ), f"attribute {attr} not found in Evaluation Class."


def test_data_init_with_categorical(mock_dataframe_with_categorical, mock_model):
    """Test data setter with categorical"""
    with pytest.raises(
        ValueError, match="data can´t have any categorical column for model training"
    ):
        evaluation_pipeline_instance = MockEvaluationPipeline(
            mock_dataframe_with_categorical, "target_column", [mock_model, mock_model]
        )
        del evaluation_pipeline_instance


def test_data_setter_with_categorical(
    mock_dataframe_with_categorical, mock_model, mock_dataframe
):
    """Test data setter with categorical"""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    with pytest.raises(
        ValueError, match="data can´t have any categorical column for model training"
    ):
        evaluation_pipeline_instance.data = mock_dataframe_with_categorical


def test_data_setter_with_empty_data(mock_empty_dataframe, mock_model):
    """Test data setter with empty data"""
    with pytest.raises(
        ValueError, match="data can´t be empty, please insert a valid dataframe"
    ):
        evaluation_pipeline_instance = MockEvaluationPipeline(
            mock_empty_dataframe, "target_column", [mock_model, mock_model]
        )
        del evaluation_pipeline_instance


def test_target_column_setter_column_not_in_data_columns(mock_model, mock_dataframe):
    """Test if target column setter is working when target column is not in data columns when we change the attribute value"""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    with pytest.raises(
        KeyError,
        match="target column is not in data columns, please insert a valid target column",
    ):
        evaluation_pipeline_instance.target_column = "anything"


def test_target_column_init_column_not_in_data_columns(mock_dataframe, mock_model):
    """Test if target column setter is working when target column is not in data columns when initiate the instance"""
    with pytest.raises(
        KeyError,
        match="target column is not in data columns, please insert a valid target column",
    ):
        evaluation_pipeline_instance = MockEvaluationPipeline(
            mock_dataframe, "anything", [mock_model, mock_model]
        )
        del evaluation_pipeline_instance


def test_target_column_init_with_wrong_type(mock_dataframe, mock_model):
    """Test if target column setter is working when target column is with a wrong type when initiate the instance"""
    with pytest.raises(ValueError, match="target_column must be a string"):
        evaluation_pipeline_instance = MockEvaluationPipeline(
            mock_dataframe, 1, [mock_model, mock_model]
        )
        del evaluation_pipeline_instance


def test_target_column_setter_with_wrong_type(mock_model, mock_dataframe):
    """Test if target column setter is working when target column is with a wrong type when we change the attribute value"""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    with pytest.raises(ValueError, match="target_column must be a string"):
        evaluation_pipeline_instance.target_column = 1
