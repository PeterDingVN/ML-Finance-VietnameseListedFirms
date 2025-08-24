from sklearn.metrics import \
    roc_auc_score as roc_auc, precision_score as precision, recall_score as recall, \
    mean_absolute_percentage_error as mape, r2_score as r2, root_mean_squared_error as rmse, balanced_accuracy_score, \
    make_scorer
from sklearn.ensemble import RandomForestRegressor as rf
from xgboost import XGBRegressor as xgb_reg


'''
    This file is reserved only for Machine Learning algorithm
    Deep Learning params will be in another file
'''

# Criteria
score_class = {
    'roc_auc': make_scorer(roc_auc),
    'precision': make_scorer(precision),
    'recall': make_scorer(recall),
    'accuracy': make_scorer(balanced_accuracy_score)
}

score_reg = {
    'r2': make_scorer(r2),
    'rmse': make_scorer(rmse),
    'mape': make_scorer(mape),
}

# Hyper params list
    # Random forest
rf_param = {
    'n_estimators': [100], # 300, 150, 200],
    'max_depth': [None], # 5, 10],
    'min_samples_split': [2], # 5],
    'min_samples_leaf': [1] #, 5]
}

    # XGBoost
xgb_param = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth' : [2, 5, 8, 10],
    'subsample' : [0.5]
}

# Regressor
algorithm = {
    'XG_reg' : (xgb_reg(), xgb_param)
    # 'RF' : (rf(), rf_param)
}