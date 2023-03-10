from imports import *

def metrics(model, real_target, predicted_target) -> pd.DataFrame:
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
