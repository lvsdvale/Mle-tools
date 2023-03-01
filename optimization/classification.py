from imports import *

def recall_objective(trial, data:pd.DataFrame, target_columns:str, params:dict, model):
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    classifier = model(**params)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(valid_x)
    recall = recall_score(valid_y, y_pred)
    return recall

def f1_objective(trial, data:pd.DataFrame, target_columns:str, params:dict, model):
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    classifier = model(**params)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(valid_x)
    f1 = f1_score(valid_y, y_pred)
    return f1

def precision_objective(trial, data:pd.DataFrame, target_columns:str, params:dict, model):
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    classifier = model(**params)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(valid_x)
    precision = precision_score(valid_y, y_pred)
    return precision


def accuracy_objective(trial, data:pd.DataFrame, target_columns:str, params:dict, model):
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    classifier = model(**params)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(valid_x)
    accuracy = accuracy_score(valid_y, y_pred)
    return accuracy