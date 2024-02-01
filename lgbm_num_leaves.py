import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np

# get important features from lightGBM model
def get_model(X_train, y_train):
    # Specify your configurations as a dict
    models = []
    lgb_params = {
    'objective': 'binary', # fixed
    'metric': 'binary_logloss', # 'auc' 'binary_error' 
    'boosting_type': 'gbdt', # 'dart' 'goss'
    'learning_rate': 0.01, # 0.01 ~ 0.3
    'num_leaves': 511, # 64 128 256 512 1024 2048
    'max_depth': -1, # +1 -1~8
    'min_data_in_leaf': 20, # 20 ~ 900
    'is_unbalance': True, # 'scale_pos_weight'
    'max_bin': 511, # +100 or + 200 255 ~ 1024 
    'verbose': -1, 
    'random_state': 1 # 0 or 1
    }
    num_boost_round=1000
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, val_index in tqdm(kf.split(X_train), desc=f'PHQ-9 - {len(X_train)} samples: '):
        X_train_inner, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_inner, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        lgb_train = lgb.Dataset(X_train_inner, y_train_inner)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        gbm = lgb.train(lgb_params, lgb_train, num_boost_round=num_boost_round, valid_sets=lgb_eval,)

        models.append(gbm)
    
    return models