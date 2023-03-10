from imports import *

def recall_objective(trial, data:pd.DataFrame, target:str, params:dict, model) -> float:

    """
    An Optimization function using recall as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    return
        a float of recall value
    """
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    classifier = model(**params)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(valid_x)
    recall = recall_score(valid_y, y_pred)
    return recall

def f1_objective(trial, data:pd.DataFrame, target:str, params:dict, model) -> float:
    """
    An Optimization function using f1 as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    return
        a float of recall value
    """

    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    classifier = model(**params)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(valid_x)
    f1 = f1_score(valid_y, y_pred)
    return f1

def precision_objective(trial, data:pd.DataFrame, target:str, params:dict, model) -> float:

    """
    An Optimization function using precision as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    return
        a float of precision value
    """
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    classifier = model(**params)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(valid_x)
    precision = precision_score(valid_y, y_pred)
    return precision


def accuracy_objective(trial, data:pd.DataFrame, target:str, params:dict, model) -> float:
    """
    An Optimization function using accuracy as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    return
        a float of accuracy value
    """
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    classifier = model(**params)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(valid_x)
    accuracy = accuracy_score(valid_y, y_pred)
    return accuracy