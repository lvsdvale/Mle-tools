from imports import *

def r2_score_evaluation_function(**kwargs):
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


def mean_absolute_percente_error_evaluation_function(**kwargs):
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

def mean_absolute_error_evaluation_function(**kwargs):
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

def recall_evaluation_function(**kwargs):
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


def precision_evaluation_function(**kwargs):
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


def f1_score_evaluation_function(**kwargs):
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

def accuracy_evaluation_function(**kwargs):
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


