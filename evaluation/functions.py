from imports import *

def r2_score_evaluation_function(**kwargs) -> float:
    """
    An evaluation function using r2 as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    :param is_validation: boolean  
        the is valitadion param is a boolean to use validation data for evaluation
    :param validation_data: pd.Dataframe
        the is valitadion param is the data for validation
    :param n_splits: pd.Dataframe
        the is n_split param is the number of splits for kfold
    
    return
        a float of r2 value
    """
    data: pd.DataFrame = kwargs.get('data')
    target: str = kwargs.get('target')
    params = kwargs.get('params', None)
    model = kwargs.get('model')
    is_validation = kwargs.get('is_validation', False)
    x = data.drop(target, axis=1)
    y = data[target]
    if is_validation is False:
        n_splits = kwargs.get('n_splits')
        kfold = KFold(n_splits=n_splits)
        if params is not None:
            model.set_params(**params)
        kfold.get_n_splits(x)
        scores = list()
        for train_index, test_index in kfold.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            scores.append(r2_score(y_test, y_pred))
        return np.mean(scores)
    validation_data: pd.DataFrame = kwargs.get('validation_data')
    model.fit(x, y)
    x_validation = validation_data.drop(target, axis=1)
    y_validation = validation_data[target]
    y_pred = model.predict(x_validation)
    return r2_score(y_validation, y_pred)


def mean_absolute_percente_error_evaluation_function(**kwargs) -> float:
    """
    An evaluation function using mean absolute percente error as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    :param is_validation: boolean  
        the is valitadion param is a boolean to use validation data for evaluation
    :param validation_data: pd.Dataframe
        the is valitadion param is the data for validation
    :param n_splits: pd.Dataframe
        the is n_split param is the number of splits for kfold
    
    return
        a float of mean absolute percente error value
    """
    data: pd.DataFrame = kwargs.get('data')
    target: str = kwargs.get('target')
    params = kwargs.get('params', None)
    model = kwargs.get('model')
    is_validation = kwargs.get('is_validation', False)
    x = data.drop(target, axis=1)
    y = data[target]
    if is_validation is False:
        n_splits = kwargs.get('n_splits')
        kfold = KFold(n_splits=n_splits)
        if params is not None:
            model.set_params(**params)
        kfold.get_n_splits(x)
        scores = list()
        for train_index, test_index in kfold.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            scores.append(mean_absolute_percentage_error(y_test, y_pred))
        return np.mean(scores)
    validation_data: pd.DataFrame = kwargs.get('validation_data')
    model.fit(x, y)
    x_validation = validation_data.drop(target, axis=1)
    y_validation = validation_data[target]
    y_pred = model.predict(x_validation)
    return mean_absolute_percentage_error(y_validation, y_pred)

def mean_absolute_error_evaluation_function(**kwargs) -> float:
    """
    An evaluation function using mean absolute error as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    :param is_validation: boolean  
        the is valitadion param is a boolean to use validation data for evaluation
    :param validation_data: pd.Dataframe
        the is valitadion param is the data for validation
    :param n_splits: pd.Dataframe
        the is n_split param is the number of splits for kfold
    
    return
        a float of mean absolute error value
    """
    data: pd.DataFrame = kwargs.get('data')
    target: str = kwargs.get('target')
    params = kwargs.get('params', None)
    model = kwargs.get('model')
    is_validation = kwargs.get('is_validation', False)
    x = data.drop(target, axis=1)
    y = data[target]
    if is_validation is False:
        n_splits = kwargs.get('n_splits')
        kfold = KFold(n_splits=n_splits)
        if params is not None:
            model.set_params(**params)
        kfold.get_n_splits(x)
        scores = list()
        for train_index, test_index in kfold.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            scores.append(mean_absolute_error(y_test, y_pred))
        return np.mean(scores)
    validation_data: pd.DataFrame = kwargs.get('validation_data')
    model.fit(x, y)
    x_validation = validation_data.drop(target, axis=1)
    y_validation = validation_data[target]
    y_pred = model.predict(x_validation)
    return mean_absolute_error(y_validation, y_pred)

def recall_evaluation_function(**kwargs) -> float:
    """
    An evaluation function using recall as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    :param is_validation: boolean  
        the is valitadion param is a boolean to use validation data for evaluation
    :param validation_data: pd.Dataframe
        the is valitadion param is the data for validation
    :param n_splits: pd.Dataframe
        the is n_split param is the number of splits for kfold
    
    return
        a float of recall value
    """
    data: pd.DataFrame = kwargs.get('data')
    target: str = kwargs.get('target')
    params = kwargs.get('params', None)
    model = kwargs.get('model')
    is_validation = kwargs.get('is_validation', False)
    x = data.drop(target, axis=1)
    y = data[target]
    if is_validation is False:
        n_splits = kwargs.get('n_splits')
        kfold = KFold(n_splits=n_splits)
        if params is not None:
            model.set_params(**params)
        kfold.get_n_splits(x)
        scores = list()
        for train_index, test_index in kfold.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            scores.append(recall_score(y_test, y_pred))
        return np.mean(scores)
    validation_data: pd.DataFrame = kwargs.get('validation_data')
    model.fit(x, y)
    x_validation = validation_data.drop(target, axis=1)
    y_validation = validation_data[target]
    y_pred = model.predict(x_validation)
    return recall_score(y_validation, y_pred)


def precision_evaluation_function(**kwargs) -> float:
    """
    An evaluation function using precision as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    :param is_validation: boolean  
        the is valitadion param is a boolean to use validation data for evaluation
    :param validation_data: pd.Dataframe
        the is valitadion param is the data for validation
    :param n_splits: pd.Dataframe
        the is n_split param is the number of splits for kfold
    
    return
        a float of precision value
    """
    data: pd.DataFrame = kwargs.get('data')
    target: str = kwargs.get('target')
    params = kwargs.get('params', None)
    model = kwargs.get('model')
    is_validation = kwargs.get('is_validation', False)
    x = data.drop(target, axis=1)
    y = data[target]
    if is_validation is False:
        n_splits = kwargs.get('n_splits')
        kfold = KFold(n_splits=n_splits)
        if params is not None:
            model.set_params(**params)
        kfold.get_n_splits(x)
        scores = list()
        for train_index, test_index in kfold.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            scores.append(precision_score(y_test, y_pred))
        return np.mean(scores)
    validation_data: pd.DataFrame = kwargs.get('validation_data')
    model.fit(x, y)
    x_validation = validation_data.drop(target, axis=1)
    y_validation = validation_data[target]
    y_pred = model.predict(x_validation)
    return precision_score(y_validation, y_pred)


def f1_score_evaluation_function(**kwargs) -> float:
    """
    An evaluation function using F1 score as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    :param is_validation: boolean  
        the is valitadion param is a boolean to use validation data for evaluation
    :param validation_data: pd.Dataframe
        the is valitadion param is the data for validation
    :param n_splits: pd.Dataframe
        the is n_split param is the number of splits for kfold
    
    return
        a float of F1 score value
    """
    data: pd.DataFrame = kwargs.get('data')
    target: str = kwargs.get('target')
    params = kwargs.get('params', None)
    model = kwargs.get('model')
    is_validation = kwargs.get('is_validation', False)
    x = data.drop(target, axis=1)
    y = data[target]
    if is_validation is False:
        n_splits = kwargs.get('n_splits')
        kfold = KFold(n_splits=n_splits)
        if params is not None:
            model.set_params(**params)
        kfold.get_n_splits(x)
        scores = list()
        for train_index, test_index in kfold.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            scores.append(f1_score(y_test, y_pred))
        return np.mean(scores)
    validation_data: pd.DataFrame = kwargs.get('validation_data')
    model.fit(x, y)
    x_validation = validation_data.drop(target, axis=1)
    y_validation = validation_data[target]
    y_pred = model.predict(x_validation)
    return f1_score(y_validation, y_pred)

def accuracy_evaluation_function(**kwargs) -> float:
    """
    An evaluation function using accuracy as metric value

    param data: pd.Dataframe
        'data param is the dataframe that will be used for our fitting
    :param target: str 
        the target param is used to find the target column of our prediction
    :param **params: dict 
        the params dict param is the dict with the hyperparameters of our model
    :param is_validation: boolean  
        the is valitadion param is a boolean to use validation data for evaluation
    :param validation_data: pd.Dataframe
        the is valitadion param is the data for validation
    :param n_splits: pd.Dataframe
        the is n_split param is the number of splits for kfold
    
    return
        a float of accuracy value
    """
    data: pd.DataFrame = kwargs.get('data')
    target: str = kwargs.get('target')
    params = kwargs.get('params', None)
    model = kwargs.get('model')
    is_validation = kwargs.get('is_validation', False)
    x = data.drop(target, axis=1)
    y = data[target]
    if is_validation is False:
        n_splits = kwargs.get('n_splits')
        kfold = KFold(n_splits=n_splits)
        if params is not None:
            model.set_params(**params)
        kfold.get_n_splits(x)
        scores = list()
        for train_index, test_index in kfold.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            scores.append(accuracy_score(y_test, y_pred))
        return np.mean(scores)
    validation_data: pd.DataFrame = kwargs.get('validation_data')
    model.fit(x, y)
    x_validation = validation_data.drop(target, axis=1)
    y_validation = validation_data[target]
    y_pred = model.predict(x_validation)
    return accuracy_score(y_validation, y_pred)


