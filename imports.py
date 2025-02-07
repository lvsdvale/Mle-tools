import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Literal
import seaborn as sns
import warnings
from matplotlib.ticker import PercentFormatter
import optuna
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, plot_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score , KFold 
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, LinearRegressor, TheilSenRegressor
from sklearn.svm import SVR