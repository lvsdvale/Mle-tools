from imports import * 

def probability_mass_function(data:pd.DataFrame, column:str)->None:
    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    if column not in data.columns:
        raise ValueError('given column not in data columns')
    sns.histplot(data=data, x=column, stat='probability')
    plt.show()

def scatterplot_prediction_compare(real_target, predicted_target)->None:
    fg, axs = plt.subplots(figsize=(12,6))
    plt.title("IA Prediction Scatterplot")
    plt.ylabel(ylabel='Values')
    sns.scatterplot(x=list(len(real_target)), y= real_target, ax=axs, label='real target')
    sns.scatterplot(x=list(len(predicted_target)), y= predicted_target, ax=axs, label='predicted target', color='red')
    axs.legend(loc='best')
    plt.show()

def lineplot_prediction_compare(real_target, predicted_target)->None:
    fg, axs = plt.subplots(figsize=(12,6))
    plt.title("IA Prediction lineplot")
    plt.ylabel(ylabel='Values')
    sns.lineplot(x=list(len(real_target)), y= real_target, ax=axs, label='real target')
    sns.lineplot(x=list(len(predicted_target)), y= predicted_target, ax=axs, label='predicted target', color='red')
    axs.legend(loc='best')
    plt.show()