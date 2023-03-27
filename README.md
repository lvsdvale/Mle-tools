# UbiMle

<h1>What is it?</h1>

<p>UbiMle is a Python package from ubivis that helps to train and validate models faster, It aims to be the fundamental high-level building block for doing practical, real world model training and validation in Python.</p>

<h1>Dependencies</h1>
<li>Numpy</li>
<li>Pandas</li>
<li>Matplotlib</li>
<li>Seaborn</li>
<li>Scikit-Learn</li>


<h1>Documentation</h1>
<h2>Evaluation.ClassifierEvaluator</h2>
<p>class ClassifierEvaluator (data:pd.DataFrame, target_column:str, models:list)

this class has many methods that uses kfold and all the models that you want to compare and give you a dataframe with the mean of the metric tha you use 

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
this function uses the class data, kfold and f1 score to return a dataframe with f1 score mean of all the models trained. 
Parameters:
    n_splits:int
        kfold n_splits
return:
    Pandas Dataframe.
</p>

<h2>Evaluation.ClassifierEvaluator.recall_score_evaluation</h2>
<p>
this function uses the class data, kfold and recall score to return a dataframe with recall score mean of all the models trained. 
Parameters:
    n_splits:int
        kfold n_splits
return:
    Pandas Dataframe.
</p>

<h2>Evaluation.ClassifierEvaluator.precision_score_evaluation</h2>
<p>
this function uses the class data, kfold and precision score to return a dataframe with precion score mean of all the models trained. 
Parameters:
    n_splits:int
        kfold n_splits
return:
    Pandas Dataframe.
</p>

<h2>Evaluation.ClassifierEvaluator.Acurracy_score_evaluation</h2>
<p>
this function uses the class data, kfold and Acurracy score to return a dataframe with Acurracy score mean of all the models trained. 
Parameters:
    n_splits:int
        kfold n_splits
return:
    Pandas Dataframe.
</p>