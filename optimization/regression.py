from imports import *

def r2_objective(trial, data:pd.DataFrame, target_columns:str, params:dict, model):
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    regressor = model(**params)
    regressor.fit(train_x, train_y)
    y_pred = regressor.predict(valid_x)
    r2 = r2_score(valid_y, y_pred)
    return r2

def mape_objective(trial, data:pd.DataFrame, target_columns:str, params:dict, model):
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    regressor = model(**params)
    regressor.fit(train_x, train_y)
    y_pred = regressor.predict(valid_x)
    mape = mean_absolute_percentage_error(valid_y, y_pred)
    return mape

def mae_objective(trial, data:pd.DataFrame, target_columns:str, params:dict, model):
    data = data.drop(target, axis = 1)
    target = data[target]
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.33, random_state=42)
    params = params
    regressor = model(**params)
    regressor.fit(train_x, train_y)
    y_pred = regressor.predict(valid_x)
    mape = mean_absolute_error(valid_y, y_pred)
    return mape