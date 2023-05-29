from imports import *
from utils import *

class ClassifierEvaluator:
    def __init__(self, data:pd.DataFrame, target_column:str, models:list):
        self.__data = data
        self.__target_columns = target_column
        self.__models = models
        
    def f1_score_evaluation(self, n_splits:int=1) -> pd.DataFrame:
        """
        An evaluation function using f1 as metric value

        :param n_splits: pd.Dataframe
            the is n_split param is the number of splits for kfold
    
        return
            a pandas dataframe with all scores
        """
        data = self.__data
        target_column = self.__target_columns 
        models = self.__models
        x = data.drop(target_column, axis=1)
        y = data[target_column]
        list_of_series = list()
        if n_splits !=0:
            kfold = KFold(n_splits=n_splits)
            for model in models:
                scores = list()
                for train_index, test_index in kfold.split(x):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    scores.append(f1_score(y_test, y_pred))
                score = np.mean(scores)
                list_of_series.append(pd.Series([model ,score], index=['Model','f1 Score']))
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
            for model in models:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = f1_score(y_test, y_pred)
                list_of_series.append(pd.Series([model ,score], index=['Model','f1 Score']))
        df_models = pd.DataFrame(list_of_series, columns = ['Model','f1 Score'])
        return df_models
    
    def recall_score_evaluation(self, n_splits:int=1) -> pd.DataFrame:
        """
        An evaluation function using recall as metric value

        :param n_splits: pd.Dataframe
            the is n_split param is the number of splits for kfold
    
        return
            a pandas dataframe with all scores
        """
        data = self.__data
        target_column = self.__target_columns 
        models = self.__models
        x = data.drop(target_column, axis=1)
        y = data[target_column]
        list_of_series = list()
        if n_splits !=0:
            kfold = KFold(n_splits=n_splits)
            for model in models:
                scores = list()
                for train_index, test_index in kfold.split(x):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    scores.append(recall_score(y_test, y_pred))
                score = np.mean(scores)
                list_of_series.append(pd.Series([model ,score], index=['Model','recall Score']))
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
            for model in models:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = recall_score(y_test, y_pred)
                list_of_series.append(pd.Series([model ,score], index=['Model','recall Score']))
        df_models = pd.DataFrame(list_of_series, columns = ['Model','recall Score'])
        return df_models
    
    def precision_score_evaluation(self, n_splits:int=1) -> pd.DataFrame:
        """
        An evaluation function using precision as metric value

        :param n_splits: pd.Dataframe
            the is n_split param is the number of splits for kfold
    
        return
            a pandas dataframe with all scores
        """
        data = self.__data
        target_column = self.__target_columns 
        models = self.__models
        x = data.drop(target_column, axis=1)
        y = data[target_column]
        list_of_series = list()
        if n_splits !=0:
            kfold = KFold(n_splits=n_splits)
            for model in models:
                scores = list()
                for train_index, test_index in kfold.split(x):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    scores.append(precision_score(y_test, y_pred))
                score = np.mean(scores)
                list_of_series.append(pd.Series([model ,score], index=['Model','precision Score']))
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
            for model in models:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = precision_score(y_test, y_pred)
                list_of_series.append(pd.Series([model ,score], index=['Model','precision Score']))
        df_models = pd.DataFrame(list_of_series, columns = ['Model','precision Score'])
        return df_models
    
    def accuracy_score_evaluation(self, n_splits:int=1) -> pd.DataFrame:
        """
        An evaluation function using accuracy as metric value

        :param n_splits: pd.Dataframe
            the is n_split param is the number of splits for kfold
    
        return
            a pandas dataframe with all scores
        """
        data = self.__data
        target_column = self.__target_columns 
        models = self.__models
        x = data.drop(target_column, axis=1)
        y = data[target_column]
        list_of_series = list()
        if n_splits !=0:
            kfold = KFold(n_splits=n_splits)
            for model in models:
                scores = list()
                for train_index, test_index in kfold.split(x):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    scores.append(accuracy_score(y_test, y_pred))
                score = np.mean(scores)
                list_of_series.append(pd.Series([model ,score], index=['Model','accuracy Score']))
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
            for model in models:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = accuracy_score(y_test, y_pred)
                list_of_series.append(pd.Series([model ,score], index=['Model','accuracy Score']))
        df_models = pd.DataFrame(list_of_series, columns = ['Model','accuracy Score'])
        return df_models


class RegressorEvaluator:
    def __init__(self, data:pd.DataFrame, target_column:str, models:list):
        self.___data = data
        self.__target_columns = target_column
        self__models = models

    def r2_score_evaluation(self, n_splits:int=1) -> pd.DataFrame:
        """
        An evaluation function using r2 as metric value

        :param n_splits: pd.Dataframe
            the is n_split param is the number of splits for kfold
    
        return
            a pandas dataframe with all scores
        """
        data = self.__data
        target_column = self.__target_columns 
        models = self.__models
        x = data.drop(target_column, axis=1)
        y = data[target_column]
        list_of_series = list()
        if n_splits !=0:
            kfold = KFold(n_splits=n_splits)
            for model in models:
                scores = list()
                for train_index, test_index in kfold.split(x):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    scores.append(r2_score(y_test, y_pred))
                score = np.mean(scores)
                list_of_series.append(pd.Series([model ,score], index=['Model','R2 Score']))
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
            for model in models:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = r2_score(y_test, y_pred)
                list_of_series.append(pd.Series([model ,score], index=['Model','R2 Score']))
        df_models = pd.DataFrame(list_of_series, columns = ['Model','R2 Score'])
        return df_models
    
    def adjusted_r2_score_evaluation(self, n_splits:int=1) -> pd.DataFrame:
        """
        An evaluation function using adjusted r2 as metric value

        :param n_splits: pd.Dataframe
            the is n_split param is the number of splits for kfold
    
        return
            a pandas dataframe with all scores
        """
        data = self.__data
        target_column = self.__target_columns 
        models = self.__models
        x = data.drop(target_column, axis=1)
        y = data[target_column]
        list_of_series = list()
        if n_splits !=0:
            kfold = KFold(n_splits=n_splits)
            for model in models:
                scores = list()
                for train_index, test_index in kfold.split(x):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    scores.append(adjusted_r2_score(y_test, y_pred))
                score = np.mean(scores)
                list_of_series.append(pd.Series([model ,score], index=['Model','Adjusted R2 Score']))
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
            for model in models:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = adjusted_r2_score(y_test, y_pred)
                list_of_series.append(pd.Series([model ,score], index=['Model','Adjusted R2 Score']))
        df_models = pd.DataFrame(list_of_series, columns = ['Model','Adjusted R2 Score'])
        return df_models
    
    
    def mean_absolute_percentage_error_evaluation(self, n_splits):
        """
        An evaluation function using mean absolute percentage error as metric value

        :param n_splits: pd.Dataframe
            the is n_split param is the number of splits for kfold
    
        return
            a pandas dataframe with all scores
        """
        data = self.__data
        target_column = self.__target_columns 
        models = self.__models
        x = data.drop(target_column, axis=1)
        y = data[target_column]
        list_of_series = list()
        if n_splits !=0:
            kfold = KFold(n_splits=n_splits)
            for model in models:
                scores = list()
                for train_index, test_index in kfold.split(x):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    scores.append(mean_absolute_percentage_error(y_test, y_pred))
                score = np.mean(scores)
                list_of_series.append(pd.Series([model ,score], index=['Model','MAPE']))
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
            for model in models:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = mean_absolute_percentage_error(y_test, y_pred)
                list_of_series.append(pd.Series([model ,score], index=['Model','MAPE']))
        df_models = pd.DataFrame(list_of_series, columns = ['Model','MAPE'])
        return df_models
    
    def mean_absolute_error_evaluation(self, n_splits):
        """
        An evaluation function using mean absolute error as metric value

        :param n_splits: pd.Dataframe
            the is n_split param is the number of splits for kfold
    
        return
            a pandas dataframe with all scores
        """
        data = self.__data
        target_column = self.__target_columns 
        models = self.__models
        x = data.drop(target_column, axis=1)
        y = data[target_column]
        list_of_series = list()
        if n_splits !=0:
            kfold = KFold(n_splits=n_splits)
            for model in models:
                scores = list()
                for train_index, test_index in kfold.split(x):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    scores.append(mean_absolute_error(y_test, y_pred))
                score = np.mean(scores)
                list_of_series.append(pd.Series([model ,score], index=['Model','MAE']))
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
            for model in models:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = mean_absolute_error(y_test, y_pred)
                list_of_series.append(pd.Series([model ,score], index=['Model','MAE']))
        df_models = pd.DataFrame(list_of_series, columns = ['Model','MAE'])
        return df_models
    
    def mean_squared_error_evaluation(self, n_splits):
        """
        An evaluation function using mean squared error as metric value

        :param n_splits: pd.Dataframe
            the is n_split param is the number of splits for kfold
    
        return
            a pandas dataframe with all scores
        """
        data = self.__data
        target_column = self.__target_columns 
        models = self.__models
        x = data.drop(target_column, axis=1)
        y = data[target_column]
        list_of_series = list()
        if n_splits !=0:
            kfold = KFold(n_splits=n_splits)
            for model in models:
                scores = list()
                for train_index, test_index in kfold.split(x):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    scores.append(mean_squared_error(y_test, y_pred))
                score = np.mean(scores)
                list_of_series.append(pd.Series([model ,score], index=['Model','MSE']))
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
            for model in models:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = mean_squared_error(y_test, y_pred)
                list_of_series.append(pd.Series([model ,score], index=['Model','MSE']))
        df_models = pd.DataFrame(list_of_series, columns = ['Model','MSE'])
        return df_models