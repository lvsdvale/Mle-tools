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
    with pytest.raises(TypeError, match="target_column must be a string"):
        evaluation_pipeline_instance = MockEvaluationPipeline(
            mock_dataframe, 1, [mock_model, mock_model]
        )
        del evaluation_pipeline_instance


def test_target_column_setter_with_wrong_type(mock_model, mock_dataframe):
    """Test if target column setter is working when target column is with a wrong type when we change the attribute value"""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    with pytest.raises(TypeError, match="target_column must be a string"):
        evaluation_pipeline_instance.target_column = 1


def test_models_init_with_empty_list(mock_dataframe, mock_model):
    """Test if models setter is working when models list is empty when initiate the instance"""
    with pytest.raises(
        ValueError,
        match="models can´t be a empty list",
    ):
        evaluation_pipeline_instance = MockEvaluationPipeline(
            mock_dataframe, "target_column", []
        )
        del evaluation_pipeline_instance


def test_models_init_with_wrong_type(mock_dataframe, mock_model):
    """Test if models setter is working when models is with a wrong type when initiate the instance"""
    with pytest.raises(TypeError, match="models must be a list with models to train"):
        evaluation_pipeline_instance = MockEvaluationPipeline(
            mock_dataframe, "target_column", dict()
        )
        del evaluation_pipeline_instance


def test_models_setter_with_empty_list(mock_dataframe, mock_model):
    """Test if models setter is working when models list is empty when we change the attribute"""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    with pytest.raises(
        ValueError,
        match="models can´t be a empty list",
    ):
        evaluation_pipeline_instance.models = []


def test_models_setter_with_wrong_type(mock_dataframe, mock_model):
    """Test if models setter is working when models is with a wrong type when we change the attribute"""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    with pytest.raises(TypeError, match="models must be a list with models to train"):
        evaluation_pipeline_instance.models = dict()


def test_models_init_a_model_with_no_fit_method(mock_dataframe, mock_model):
    """Test if models setter is working when models is with a model that has not fit when we initiate the class"""
    with pytest.raises(AttributeError, match="the model has not fit method"):
        evaluation_pipeline_instance = MockEvaluationPipeline(
            mock_dataframe, "target_column", [mock_model, "anything"]
        )
        del evaluation_pipeline_instance


def test_models_setter_with_a_model_with_no_fit_method(mock_dataframe, mock_model):
    """Test if models setter is working when models is with a model that has not fit when we change the attribute"""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    with pytest.raises(AttributeError, match="the model has not fit method"):
        evaluation_pipeline_instance.models = [mock_model, "anything"]


def test_kfold_init_with_wrong_type(mock_dataframe, mock_model):
    """Test if kfold setter is working when models is with a wrong type when we change the attribute"""
    with pytest.raises(TypeError, match="kfold must be a boolean"):
        evaluation_pipeline_instance = MockEvaluationPipeline(
            mock_dataframe, "target_column", [mock_model, mock_model], None
        )
        del evaluation_pipeline_instance


def test_kfold_setter_with_wrong_type(mock_dataframe, mock_model):
    """Test if kfold setter is working when models is with a wrong type when we change the attribute"""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    with pytest.raises(TypeError, match="kfold must be a boolean"):
        evaluation_pipeline_instance.kfold = None


def test_trained_setter_with_wrong_type(mock_dataframe, mock_model):
    """Test if trained setter is working when models is with a wrong type when we change the attribute"""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    with pytest.raises(TypeError, match="trained must be a boolean"):
        evaluation_pipeline_instance.trained = None


def test_report_setter_with_wrong_type(mock_dataframe, mock_model):
    """Test if report setter is working when models is with a wrong type when we change the attribute"""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    with pytest.raises(TypeError, match="report must be a pandas dataframe"):
        evaluation_pipeline_instance.report = "anything"


def test_n_splits_init_with_wrong_type(mock_dataframe, mock_model):
    """Test if n_splits setter is working when models is with a wrong type when we initiate the class"""
    with pytest.raises(TypeError, match="n_splits must be an integer"):
        evaluation_pipeline_instance = MockEvaluationPipeline(
            mock_dataframe,
            "target_column",
            [mock_model, mock_model],
            n_splits="anything",
        )
        del evaluation_pipeline_instance


def test_n_splits_setter_with_wrong_type(mock_dataframe, mock_model):
    """Test if n_splits setter is working when models is with a wrong type when we change the attribute"""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    with pytest.raises(TypeError, match="n_splits must be an integer"):
        evaluation_pipeline_instance.n_splits = "anything"


def test_run_training(mock_dataframe, mock_model):
    """Test if n_splits setter is working when models is with a wrong type when we change the attribute"""
    evaluation_pipeline_instance = MockEvaluationPipeline(
        mock_dataframe, "target_column", [mock_model, mock_model]
    )
    evaluation_pipeline_instance.run_training()
    assert evaluation_pipeline_instance.trained
