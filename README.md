# MLE tools

<h1>What is it?</h1>

<p>MLE tools is a Python package that helps to train and validate models faster, It aims to be the fundamental high-level building block for doing practical, real world model training and validation in Python.</p>

<h1>Dependencies</h1>
<li>Numpy</li>
<li>Pandas</li>
<li>Matplotlib</li>
<li>Seaborn</li>
<li>Scikit-Learn</li>
<li>Optuna</li>


<h1>Documentation</h1>

<h1>Evaluation</h1>

<h2>Evaluation.ClassifierEvaluator</h2>
<p>class ClassifierEvaluator (data:pd.DataFrame, target_column:str, models:list)

this class is used to evaluate different classification models, has many methods that uses kfold and all the models that you want to compare and give you a dataframe with the mean of the metric tha you use. 

Parameters:

    data:DataFrame
        data that will be used to train and validate our models

    Target:str
        the target column in data
    
    models:list
        a list of sklearn models to train and compare.
</p>

<h3>Methods</h3>
<p>

    f1_score_evaluation(n_splits:int)

    recall_score_evaluation(n_splits:int)

    precision_score_evaluation(n_splits:int)

    accuracy_score_evaluation(n_split:int)
</p>

<h2>Evaluation.ClassifierEvaluator.f1_score_evaluation</h2>
<p>
Evaluation.ClassifierEvaluator.f1_score_evaluation(n_splits)

this function uses the class data, kfold and f1 score to return a dataframe with f1 score mean of all the models given in class init. 
Parameters:

    n_splits:int
        kfold n_splits

return:
    Pandas Dataframe.
</p>

<h2>Evaluation.ClassifierEvaluator.recall_score_evaluation</h2>
<p>
Evaluation.ClassifierEvaluator.recall_score_evaluation(n_splits)

this function uses the class data, kfold and recall score to return a dataframe with recall score mean of all the models given in class init.

Parameters:

    n_splits:int
        kfold n_splits

return:
    Pandas Dataframe.
</p>

<h2>Evaluation.ClassifierEvaluator.precision_score_evaluation</h2>
<p>
Evaluation.ClassifierEvaluator.precision_score_evaluation(n_splits)

this function uses the class data, kfold and precision score to return a dataframe with precion score mean of all the models given in class init.

Parameters:

    n_splits:int
        kfold n_splits

return:
    Pandas Dataframe.
</p>

<h2>Evaluation.ClassifierEvaluator.accuracy_score_evaluation</h2>
<p>
Evaluation.ClassifierEvaluator.accuracy_score_evaluation(n_splits)

this function uses the class data, kfold and Accuracy score to return a dataframe with Accuracy score mean of all the models given in class init.

Parameters:

    n_splits:int
        kfold n_splits

return:
    Pandas Dataframe.
</p>

<h2>Evaluation.RegressorEvaluator</h2>
<p>class RegressorEvaluator (data:pd.DataFrame, target_column:str, models:list)

this class is used to evaluate different regression models, has many methods that uses kfold and all the models that you want to compare and give you a dataframe with the mean of the metric you use. 

Parameters:

    data:DataFrame
        data that will be used to train and validate our models

    Target:str
        the target column in data
    
    models:list
        a list of sklearn models to train and compare.
</p>

<h3>Methods</h3>
<p>

    r2_score_evaluation(n_splits:int)

    adjusted_r2_score_evaluation(n_splits:int)

    mean_absolute_error_evaluation(n_splits:int)

    mean_absolute_percentage_error_evaluation(n_split:int)

    mean_squared_error_evaluation(n_split:int)

</p>

<h2>Evaluation.RegressorEvaluator.r2_score_evaluation</h2>
<p>
Evaluation.RegressorEvaluator.r2_score_evaluation(n_splits)

this function uses the class data, kfold and r2 score to return a dataframe with r2 score mean of all the models given in class init. 
Parameters:

    n_splits:int
        kfold n_splits

return:
    Pandas Dataframe.
</p>

<h2>Evaluation.RegressorEvaluator.adjusted_r2_score_evaluation</h2>
<p>
Evaluation.RegressorEvaluator.adjusted_r2_score_evaluation(n_splits)

this function uses the class data, kfold and adjusted r2 score to return a dataframe with adjusted r2 score mean of all the models given in class init.

Parameters:

    n_splits:int
        kfold n_splits

return:
    Pandas Dataframe.
</p>

<h2>Evaluation.RegressorEvaluator.mean_absolute_error_evaluation</h2>
<p>
Evaluation.RegressorEvaluator.mean_absolute_error_evaluation(n_splits)

this function uses the class data, kfold and precision score to return a dataframe with mean absolute error mean of all the models given in class init.

Parameters:

    n_splits:int
        kfold n_splits

return:
    Pandas Dataframe.
</p>

<h2>Evaluation.RegressorEvaluator.mean_absolute_percentage_error_evaluation</h2>
<p>
Evaluation.RegressorEvaluator.mean_absolute_percentage_error_evaluation(n_splits)

this function uses the class data, kfold and precision score to return a dataframe with mean absolute percentage error mean of all the models given in class init.

Parameters:

    n_splits:int
        kfold n_splits

return:
    Pandas Dataframe.
</p>

<h2>Evaluation.RegressorEvaluator.mean_squared_error_evaluation</h2>
<p>
Evaluation.RegressorEvaluator.mean_absolute_squared_evaluation(n_splits)

this function uses the class data, kfold and precision score to return a dataframe with mean squared error mean of all the models given in class init.

Parameters:

    n_splits:int
        kfold n_splits

return:
    Pandas Dataframe.
</p>

<h1>Optimization</h1>

<h2>optimization.ClassifierOptimizer</h2>
<p>class ClassifierOptimizer (data:pd.DataFrame, target_column:str, models:list)

this class is used to optimize a classification model, using optuna, the methods are made as objectives of optuna, to optimize the model.

Parameters:

    data:DataFrame
        data that will be used to train and validate our models.

    Target:str
        the target column in data.
    
    model: scikit learning model
        Machine learning model will be optimize.

    params:dict
        the params dict param is the dict with the hyperparameters of our model, this params <b>must</b> be an optuna dict with the trials to make. 
</p>

<h3>Methods</h3>
<p>

    f1_score_objective(trial:optuna.trial)

    recall_score_objective(trial:optuna.trial)

    precision_score_objective(trial:optuna.trial)

    accuracy_score_objective(trial:optuna.trial)

    optimize(metric:Literal['f1', 'recall', 'precision', 'accuracy'], direction:Literal['minimize', 'maximize'], n_trials:int, timeout:int)
</p>

<h2>optimization.ClassifierOptimizer.f1_score_objective</h2>
<p>
optimization.ClassifierOptimizer.f1_score_objective(trial)

this function uses the class data, target and optuna suggested params to train the model and than return  f1 score. 
Parameters:

    trial: optuna.trial
        this is an optuna necessary param for optimization

return:
    f1 score float value.
</p>

<h2>optimization.ClassifierOptimizer.recall_score_objective</h2>
<p>
optimization.ClassifierOptimizer.recall_score_objective(trial)

this function uses the class data, target and optuna suggested params to train the model and than return recall score. 
Parameters:

    trial: optuna.trial
        this is an optuna necessary param for optimization

return:
    recall score float value.
</p>

<h2>optimization.ClassifierOptimizer.precision_score_objective</h2>
<p>
optimization.ClassifierOptimizer.precision_score_objective(trial)

this function uses the class data, target and optuna suggested params to train the model and than return recall score. 
Parameters:

    trial: optuna.trial
        this is an optuna necessary param for optimization

return:
    precision score float value.
</p>

<h2>optimization.ClassifierOptimizer.accuracy_score_objective</h2>
<p>
optimization.ClassifierOptimizer.accuracy_score_objective(trial)

this function uses the class data, target and optuna suggested params to train the model and than return an accuracy score. 
Parameters:

    trial: optuna.trial
        this is an optuna necessary param for optimization

return:
    accuracy score float value.
</p>

<h2>optimization.ClassifierOptimizer.optimize</h2>
<p>
optimization.ClassifierOptimizer.optimize(metric:Literal['f1', 'recall', 'precision', 'accuracy'], direction:Literal['minimize', 'maximize'], n_trials:int, timeout:int)

this function uses class functions as metric and direction, maximize to maximize your metric and minimize to minimize your metric.  
Parameters:

    metric:Literal['f1', 'recall', 'precision', 'accuracy']
        metric param is used to choose the metric.
    
    direction:Literal['minimize', 'maximize']
        param used to maximize or minimize to your metric during the optimization.

    n_trials:int
        number of trials that optuna will try to optimize the model.
        
    timeout:int
        max time in seconds that optuna will have to optimize the model

return:
    None    
</p>

<h2>optimization.RegressorOptimizer</h2>
<p>class RegressorOptimizer (data:pd.DataFrame, target_column:str, models:list)

this class is used to optimize a classification model, using optuna, the methods are made as objectives of optuna, to optimize the model.

Parameters:

    data:DataFrame
        data that will be used to train and validate our models.

    Target:str
        the target column in data.
    
    model: scikit learning model
        Machine learning model will be optimize.

    params:dict
        the params dict param is the dict with the hyperparameters of our model, this params <b>must</b> be an optuna dict with the trials to make. 
</p>

<h3>Methods</h3>
<p>

    r2_score_objective(trial:optuna.trial)

    adjusted_r2_score_objective(trial:optuna.trial)

    mean_absolute_error_objective(trial:optuna.trial)

    mean_absolute_percentage_error_objective(trial:optuna.trial)

    mean_squared_error_objective(trial:optuna.trial)
</p>

<h2>optimization.RegressorOptimizer.r2_score_objective</h2>
<p>
optimization.RegressorOptimizer.r2_score_objective(trial)

this function uses the class data, target and optuna suggested params to train the model and than return r2 score. 
Parameters:

    trial: optuna.trial
        this is an optuna necessary param for optimization

return:
    r2 score float value.
</p>

<h2>optimization.RegressorOptimizer.adjusted_r2_score_objective</h2>
<p>
optimization.RegressorOptimizer.adjusted_r2_score_objective(trial)

this function uses the class data, target and optuna suggested params to train the model and than return r2 score. 
Parameters:

    trial: optuna.trial
        this is an optuna necessary param for optimization

return:
    adjusted r2 score float value.
</p>

<h2>optimization.RegressorOptimizer.mean_absolute_error_objective</h2>
<p>
optimization.RegressorOptimizer.mean_absolute_error_objective(trial)

this function uses the class data, target and optuna suggested params to train the model and than return mean_absolute_error. 
Parameters:

    trial: optuna.trial
        this is an optuna necessary param for optimization

return:
    mean_absolute_error float value.
</p>

<h2>optimization.RegressorOptimizer.mean_percentage_absolute_error_objective</h2>
<p>
optimization.RegressorOptimizer.mean_percentage_absolute_error_objective(trial)

this function uses the class data, target and optuna suggested params to train the model and than return mean absolute percentage error. 
Parameters:

    trial: optuna.trial
        this is an optuna necessary param for optimization

return:
    mean absolute percentage error float value.
</p>

<h2>optimization.RegressorOptimizer.mean_squared_error_objective</h2>
<p>
optimization.RegressorOptimizer.mean_squared_error_objective(trial)

this function uses the class data, target and optuna suggested params to train the model and than return mean squared error. 
Parameters:

    trial: optuna.trial
        this is an optuna necessary param for optimization

return:
    mean squared error float value.
</p>

<h2>optimization.RegressorOptimizer.optimize</h2>
<p>
optimization.RegressorOptimizer.optimize(metric:Literal['adjusted_r2', 'r2', 'MAE','MAPE', 'MSE'], direction:Literal['minimize', 'maximize'], n_trials:int, timeout:int)

this function uses class functions as metric and direction, maximize to maximize your metric and minimize to minimize your metric.  
Parameters:

    metric:Literal['adjusted_r2', 'r2', 'MAE','MAPE', 'MSE']
        metric param is used to choose the metric.
    
    direction:Literal['minimize', 'maximize']
        param used to maximize or minimize to your metric during the optimization.

    n_trials:int
        number of trials that optuna will try to optimize the model.
        
    timeout:int
        max time in seconds that optuna will have to optimize the model

return:
    None    
</p>


