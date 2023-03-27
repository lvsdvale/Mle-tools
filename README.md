# UbiMle

<h1>What is it?</h1>

<p>UbiMle is a Python package from ubivis that helps to train and validate models faster, It aims to be the fundamental high-level building block for doing practical, real world model training and validation in Python.</p>

<h1>Dependencies</h1>
<li>Numpy</li>
<li>Pandas</li>
<li>Matplotlib</li>
<li>Seaborn</li>
<li>Scikit-Learn</li>
<li>Optuna</li>


<h1>Documentation</h1>
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