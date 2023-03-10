from imports import *

def adjusted_r2(real_target, predicted_target, train_data):
    adj_r2 = (1 - ((1 - r2_score(real_target, predicted_target)) * (len(real_target) - 1)) / (len(real_target) - train_data.shape[1] - 1))
    return adj_r2