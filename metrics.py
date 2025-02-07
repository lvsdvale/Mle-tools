from imports import *

def classification_metrics(model, real_target, predicted_target) -> pd.DataFrame:
    """
    An metric function using f1, recall, accuracy, precision, true negatives, true positive, false negative e false positive values.

    param model: sklean model
        'model param is the machine learning algorithm that was used for our fitting
    :param real_target: np.array
        the real_target param is our target column of real data
    param predicted_target: np.array
        the predicted_target param is our prediction  of target column 
    return
        a dataframe with all metrics values above
    """
    metrics_dict = dict()
    metrics_dict['Model'] = [model,]
    metrics_dict['F1'] = [f1_score(real_target, predicted_target),]
    metrics_dict['Accuracy'] = [accuracy_score(real_target, predicted_target),]
    metrics_dict['Precision'] = [precision_score(real_target, predicted_target),]
    metrics_dict['Recall'] = [recall_score(real_target, predicted_target),]
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(real_target, predicted_target).ravel()
    metrics_dict['True Positive'] = true_positive
    metrics_dict['False Positive'] = false_positive
    metrics_dict['True Negative'] = true_negative
    metrics_dict['False Negative'] =false_negative
    metrics_dataframe = pd.DataFrame.from_dict(metrics_dict)
    return metrics_dataframe


from imports import *

def regression_metrics(model, real_target, predicted_target) -> pd.DataFrame:
    """
    An metric function using r2, mean absolute error, mean absolute percentage error, % Of Predictions With Error < 5, 10, 15 and 30 values.
    param model: sklean model
        'model param is the machine learning algorithm that was used for our fitting
    :param real_target: np.array
        the real_target param is our target column of real data
    param predicted_target: np.array
        the predicted_target param is our prediction  of target column 
    return
        a dataframe with all metrics values above
    """
    metrics_dict = dict()
    metrics_dict['Model'] = [model,]
    metrics_dict['R2'] = [r2_score(real_target, predicted_target),]
    metrics_dict['MAE'] = [mean_absolute_error(real_target, predicted_target),]
    metrics_dict['MAPE'] = [mean_absolute_percentage_error(real_target, predicted_target),]
    results = pd.DataFrame(columns=['y_real', 'y_pred'])
    results['y_real'] = real_target
    results['y_pred'] = predicted_target
    results['relative error (%)'] = np.abs(((results['y_real']-results['y_pred'])/results['y_real'])*100)
    less_than_5 = results.loc[results['relative error (%)']<5]['y_real'].count()
    less_than_10 = results.loc[results['relative error (%)']<10]['y_real'].count()
    less_than_15 = results.loc[results['relative error (%)']<15]['y_real'].count()
    less_than_20 = results.loc[results['relative error (%)']<20]['y_real'].count()  
    total_samples = results.shape[0]
    metrics_dict['% Of Predictions With Error < 5%'] = [(less_than_5/total_samples)*100,]
    metrics_dict['% Of Predictions With Error < 10%'] = [(less_than_10/total_samples)*100,]
    metrics_dict['% Of Predictions With Error < 15%'] = [(less_than_15/total_samples)*100,]
    metrics_dict['% Of Predictions With Error < 20%'] = [(less_than_20/total_samples)*100,]
    metrics_dataframe = pd.DataFrame.from_dict(metrics_dict)
    return metrics_dataframe