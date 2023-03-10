from imports import *
from utils import adjusted_r2

def r2_objective(trial, data:pd.DataFrame, target:str, params:dict, model) -> float:
    """
    An Optimization function using r2 as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    return
        a float of r2 value
    """
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    regressor = model(**params)
    regressor.fit(train_x, train_y)
    y_pred = regressor.predict(valid_x)
    r2 = r2_score(valid_y, y_pred)
    return r2

def adjusted_r2_objective(trial, data:pd.DataFrame, target:str, params:dict, model)-> float:
    """
    An Optimization function using adjusted r2 as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    return
        a float of adjusted r2 value
    """
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    regressor = model(**params)
    regressor.fit(train_x, train_y)
    y_pred = regressor.predict(valid_x)
    r2 = adjusted_r2(valid_y, y_pred, train_x)
    return r2

def mape_objective(trial, data:pd.DataFrame, target:str, params:dict, model)-> float:
    """
    An Optimization function using mean absolute percentage error as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    return
        a float of mean absolute percentage error value
    """
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    regressor = model(**params)
    regressor.fit(train_x, train_y)
    y_pred = regressor.predict(valid_x)
    mape = mean_absolute_percentage_error(valid_y, y_pred)
    return mape

def mae_objective(trial, data:pd.DataFrame, target:str, params:dict, model)-> float:
    """
    An Optimization function using mean absolute error as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    return
        a float of mean absolute error value
    """
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    regressor = model(**params)
    regressor.fit(train_x, train_y)
    y_pred = regressor.predict(valid_x)
    mape = mean_absolute_error(valid_y, y_pred)
    return mape

def mse_objective(trial, data:pd.DataFrame, target:str, params:dict, model)-> float:
    """
    An Optimization function using mean squared error as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    return
        a float of mean squared error value
    """
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    regressor = model(**params)
    regressor.fit(train_x, train_y)
    y_pred = regressor.predict(valid_x)
    mape = mean_squared_error(valid_y, y_pred)
    return mape