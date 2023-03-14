from imports import *
from utils import *

class ClassifierOptimizer:
    def __init__(self, data:pd.DataFrame, target_column:str, model, params:dict):
        """
        Classifier Optimizer class init 
        param data: pd.Dataframe
            'data param is the dataframe that will be used for our fitting
        :param target: str 
            the target param is used to find the target column of our prediction
        :param model: machine learning model
            the model param is used as machine learning algorithm model
        :param **params: dict 
            the params dict param is the dict with the hyperparameters of our model
        """
        self.__data = data
        self.__target_column = target_column
        self.__model = model
        self.__params = params
    

    def recall_objective(self, trial) -> float:
        """
        An Optimization function using recall as metric value
        :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
        return
            a float of recall value
        """
        train_data = self.__data.drop(target, axis = 1)
        target = self.__data[self.__target_column]
        train_x, valid_x, train_y, valid_y = train_test_split(train_data, target, test_size=0.33, random_state=42)
        params = params
        classifier = self.__model(**self.__params)
        classifier.fit(train_x, train_y)
        y_pred = classifier.predict(valid_x)
        recall = recall_score(valid_y, y_pred)
        return recall
    
    def f1_score_objective(self, trial) -> float:
        """
        An Optimization function using f1 as metric value
        
        return:
            a float of f1 value
        """
        train_data = self.__data.drop(target, axis = 1)
        target = self.__data[self.__target_column]
        train_x, valid_x, train_y, valid_y = train_test_split(train_data, target, test_size=0.33, random_state=42)
        params = params
        classifier = self.__model(**self.__params)
        classifier.fit(train_x, train_y)
        y_pred = classifier.predict(valid_x)
        f1 = f1_score(valid_y, y_pred)
        return f1
    
    def precision_score_objective(self, trial) -> float:
        """
        An Optimization function using precision as metric value
        return:
            a float of precision value
        """
        train_data = self.__data.drop(target, axis = 1)
        target = self.__data[self.__target_column]
        train_x, valid_x, train_y, valid_y = train_test_split(train_data, target, test_size=0.33, random_state=42)
        params = params
        classifier = self.__model(**self.__params)
        classifier.fit(train_x, train_y)
        y_pred = classifier.predict(valid_x)
        precision = precision_score(valid_y, y_pred)
        return precision
    
    def accuracy_score_objective(self, trial) -> float:
        """
        An Optimization function using precision as metric value
        return:    
            a float of Accuracy value
        """
        train_data = self.__data.drop(target, axis = 1)
        target = self.__data[self.__target_column]
        train_x, valid_x, train_y, valid_y = train_test_split(train_data, target, test_size=0.33, random_state=42)
        params = params
        classifier = self.__model(**self.__params)
        classifier.fit(train_x, train_y)
        y_pred = classifier.predict(valid_x)
        accuracy = accuracy_score(valid_y, y_pred)
        return accuracy
    
    
    def optmize(self, metric:Literal['f1', 'recall', 'precision', 'accuracy'], direction:Literal['minimize', 'maximize'], n_trials:int, timeout:int) -> dict:
        if direction not in ['minimize', 'maximize']:
            raise ValueError("invalid direction, must be 'minimize' or 'maximize'")

        if metric not in ['f1', 'recall', 'precision', 'accuracy']:
            raise ValueError("invalid metric, must be 'f1', 'recall', 'precision' or 'accuracy'")
        
        study = optuna.create_study(
        direction=direction,
        )
        if metric == "f1":
            study.optimize(lambda trial:self.f1_score_objective(trial), n_trials=n_trials, timeout=timeout)
        elif metric == 'recall':
            study.optimize(lambda trial:self.recall_objective(trial), n_trials=n_trials, timeout=timeout)
        elif metric == 'precision':
            study.optimize(lambda trial:self.precision_score_objective(trial), n_trials=n_trials, timeout=timeout)
        elif metric == 'accuracy':
            study.optimize(lambda trial:self.accuracy_score_objective(trial), n_trials=n_trials, timeout=timeout)
        return study.best_params
    
    


class RegressorOptimizer:
    def __init__(self, data:pd.DataFrame, target_column:str, model, params:dict):
        """
        Regressor Optimizer class init 
        param data: pd.Dataframe
            'data param is the dataframe that will be used for our fitting
        :param target: str 
            the target param is used to find the target column of our prediction
        :param model: machine learning model
            the model param is used as machine learning algorithm model
        :param **params: dict 
            the params dict param is the dict with the hyperparameters of our model
        """
        self.__data = data
        self.__target_column = target_column
        self.__model = model
        self.__params = params
    

    def r2_score_objective(self, trial) -> float:
        """
        An Optimization function using r2 score as metric value
        :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
        return
            a float of r2 score value
        """
        train_data = self.__data.drop(target, axis = 1)
        target = self.__data[self.__target]
        train_x, valid_x, train_y, valid_y = train_test_split(train_data, target, test_size=0.33, random_state=42)
        params = params
        regressor = self.__model(**self.__params)
        regressor.fit(train_x, train_y)
        y_pred = regressor.predict(valid_x)
        r2 = r2_score(valid_y, y_pred)
        return r2
    
    def adjusted_r2_score_objective(self, trial) -> float:
        """
        An Optimization function using adjusted_r2 score as metric value
        :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
        return
            a float of adjusted_r2_score value
        """
        train_data = self.__data.drop(target, axis = 1)
        target = self.__data[self.__target]
        train_x, valid_x, train_y, valid_y = train_test_split(train_data, target, test_size=0.33, random_state=42)
        params = params
        regressor = self.__model(**self.__params)
        regressor.fit(train_x, train_y)
        y_pred = regressor.predict(valid_x)
        r2 = adjusted_r2_score(valid_y, y_pred)
        return r2
    
    def mean_absolute_percentage_error_objective(self, trial) -> float:
        """
        An Optimization function using mean absolute percentage error as metric value
        return:
            a float of mean absolute percentage error value
        """
        train_data = self.__data.drop(target, axis = 1)
        target = self.__data[self.__target]
        train_x, valid_x, train_y, valid_y = train_test_split(train_data, target, test_size=0.33, random_state=42)
        params = params
        regressor = self.__model(**self.__params)
        regressor.fit(train_x, train_y)
        y_pred = regressor.predict(valid_x)
        mape = mean_absolute_percentage_error(valid_y, y_pred)
        return mape
    
    def mean_absolute_error_objective(self, trial) -> float:
        """
        An Optimization function using mean absolute error as metric value
        return:
            a float of mean absolute error value
        """
        train_data = self.__data.drop(target, axis = 1)
        target = self.__data[self.__target]
        train_x, valid_x, train_y, valid_y = train_test_split(train_data, target, test_size=0.33, random_state=42)
        params = params
        regressor = self.__model(**self.__params)
        regressor.fit(train_x, train_y)
        y_pred = regressor.predict(valid_x)
        mae = mean_absolute_error(valid_y, y_pred)
        return mae
    
    def mean_squared_error_objective(self, trial) -> float:
        """
        An Optimization function using mean squared error as metric value
        return:
            a float of mean squared error value
        """
        train_data = self.__data.drop(target, axis = 1)
        target = self.__data[self.__target]
        train_x, valid_x, train_y, valid_y = train_test_split(train_data, target, test_size=0.33, random_state=42)
        params = params
        regressor = self.__model(**self.__params)
        regressor.fit(train_x, train_y)
        y_pred = regressor.predict(valid_x)
        mse = mean_squared_error(valid_y, y_pred)
        return mse
    

    
    
    def optmize(self, metric:Literal['adjusted_r2', 'r2', 'MAE','MAPE', 'MSE'], direction:Literal['minimize', 'maximize'], n_trials:int, timeout:int) -> dict:
        if direction not in ['minimize', 'maximize']:
            raise ValueError("invalid direction, must be 'minimize' or 'maximize'")

        if metric not in ['adjusted_r2', 'r2', 'MAE','MAPE', 'MSE']:
            raise ValueError("invalid metric, must be 'adjusted_r2', 'r2', 'MAE','MAPE' or'MSE' ")
        
        study = optuna.create_study(
        direction=direction,
        )
        if metric == "r2":
            study.optimize(lambda trial:self.r2_score_objective(trial), n_trials=n_trials, timeout=timeout)
        elif metric == 'adjusted_r2':
            study.optimize(lambda trial:self.adjusted_r2_score_objective(trial), n_trials=n_trials, timeout=timeout)
        elif metric == 'MAPE':
            study.optimize(lambda trial:self.mean_absolute_percentage_error_objective(trial), n_trials=n_trials, timeout=timeout)
        elif metric == 'MAE':
            study.optimize(lambda trial:self.mean_absolute_error_objective(trial), n_trials=n_trials, timeout=timeout)
        elif metric == 'MSE':
            study.optimize(lambda trial:self.mean_squared_error_objective(trial), n_trials=n_trials, timeout=timeout)
        return study.best_params