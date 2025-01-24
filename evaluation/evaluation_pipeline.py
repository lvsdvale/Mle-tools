"""This file contains evaluation pipeline abstract class"""

from abc import ABC, abstractmethod
from typing import Optional
from sklearn.model_selection import train_test_split

import pandas as pd


class EvaluationPipeline(ABC):
    """
    Abstract class for Evaluation pipeline.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        models: list,
        kfold=False,
        n_splits: Optional[int] = 4,
    ) -> None:
        self.data_setter(data)
        self.target_column_setter(target_column)
        self.models_setter(models)
        self.kfold_setter(kfold)
        self.n_splits_setter(n_splits)
        self.trained_setter(False)
        self.report_setter(None)

    @property
    def data(self) -> pd.DataFrame:
        """return data attribute"""
        return self.__data

    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        """verify if data attribute is correct

        Parameters:
            data (pd.DataFrame): The DataFrame to analyze.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas dataframe")

        if data.empty:
            raise ValueError("data can´t be empty, please insert a valid dataframe")

        if self.check_if_has_categorical_column(data):
            raise ValueError(
                "data can´t have any categorical column for model training"
            )

        self.__data = data

    def data_setter(self, data: pd.DataFrame) -> None:
        """verify if data attribute is correct in init

        Parameters:
            data (pd.DataFrame): The DataFrame to analyze.
        """
        self.data = data

    @property
    def target_column(self) -> str:
        """return target column attribute"""
        return self.__target_column

    @target_column.setter
    def target_column(self, target_column: str):
        """target column setter
        Parameters:
            target_column (str): The target column name
        """
        if not isinstance(target_column, str):
            raise ValueError("target_column must be a string")

        if not self.check_if_target_column_is_in_data_columns(target_column):
            raise KeyError(
                "target column is not in data columns, please insert a valid target column"
            )
        self.__target_column = target_column

    def target_column_setter(self, target_column: str):
        """target column setter
        Parameters:
            target_column (str): The target column name
        """
        self.target_column = target_column

    @property
    def kfold(self):
        """return kfold attribute"""
        return self.__kfold

    @kfold.setter
    def kfold(self, kfold):
        """kfold setter
        Parameters:
            kfold (bool): if the evaluater is using kfold or not.
        """
        if not isinstance(kfold, bool):
            raise ValueError("kfold must be a boolean")
        self.__kfold = kfold

    def kfold_setter(self, kfold):
        """kfold setter
        Parameters:
            kfold (bool): if the evaluater is using kfold or not.
        """
        self.kfold = kfold

    @property
    def n_splits(self) -> int:
        """return n_splits attribute"""
        return self.__n_splits

    @n_splits.setter
    def n_splits(self, n_splits: int) -> None:
        """n_splits setter
        Parameters:
            n_splits (Optional[int]): numbers of splits for kfold
        """
        if not isinstance(n_splits, int) and not None:
            raise ValueError("n_splits must be an integer")
        if self.__kfold:
            self.__n_splits = n_splits
        else:
            self.__n_splits = None

    def n_splits_setter(self, n_splits: int) -> None:
        """n_splits setter
        Parameters:
            n_splits (Optional[int]): numbers of splits for kfold.
        """
        self.n_splits = n_splits

    @property
    def models(self) -> list:
        """return models attribute"""
        return self.__models

    @models.setter
    def models(self, models: list) -> None:
        """models setter
        Parameters:
            models (list): list of models to train.
        """
        if not isinstance(models, list) and not None:
            raise ValueError("models must be a list with models to train")

        if not models:
            raise ValueError("models can´t be a empty list")
        self.__models = models

    def models_setter(self, models: list) -> None:
        """models setter
        Parameters:
            models (list): list of models to train.
        """
        self.models = models

    @property
    def trained(self) -> bool:
        """return trained attribute"""
        return self.__trained

    @trained.setter
    def trained(self, trained):
        """trained setter
        Parameters:
            trained (bool): if the models are trained
        """
        if not isinstance(trained, bool) and not None:
            raise ValueError("trained must be a boolean")
        self.__trained = trained

    def trained_setter(self, trained):
        """trained setter
        Parameters:
            trained (bool): if the models are trained
        """
        self.trained = trained

    @property
    def report(self) -> pd.DataFrame:
        """return report attribute"""
        return self.__report

    @report.setter
    def report(self, report: pd.DataFrame) -> None:
        """verify if report attribute is correct

        Parameters:
            report (pd.DataFrame): The DataFrame o metrics.
        """
        if not isinstance(report, pd.DataFrame) and report is not None:
            raise ValueError("report must be a pandas dataframe")
        self.__report = report

    def report_setter(self, report: pd.DataFrame) -> None:
        """verify if report attribute is correct

        Parameters:
            report (pd.DataFrame): The DataFrame o metrics.
        """
        self.report = report

    def run_training(self, test_size: float = 0.3) -> None:
        """training pipeline
        Parameters:
            test_size (float): the percentage of data that will be used for test
        """
        X = self.__data.drop(self.__target_column, axis=1)
        y = self.__data[self.__target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        for model_index in range(len(self.__models)):
            self.__models[model_index].fit(X_train, y_train)
        self.__trained = True

    @abstractmethod
    def create_report(self):
        """abstract method for report"""
        pass

    def check_if_target_column_is_in_data_columns(self, target_column: str) -> bool:
        """verify if target column is in data columns
        Parameters:
            target_column (str): The target column name
        Returns:
            bool: True if target column is in data, False otherwise.
        """
        return target_column in self.data.columns

    @staticmethod
    def check_if_has_categorical_column(data: pd.DataFrame) -> bool:
        """verify if any of data columns in categoric
        Parameters:
            data (pd.DataFrame): The DataFrame to analyze
        Returns:
            bool: True if there is at least one categorical column, False otherwise.
        """
        return any(
            pd.api.types.is_categorical_dtype(data[column]) for column in data.columns
        ) or any(data[column].dtype == "object" for column in data.columns)
