import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, log_loss
import random
import math
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

def objective_function(X_train, y_train, X_test, y_test, feature_subset):
    # Convert feature_subset to a list if X is a DataFrame
    if isinstance(X_train, pd.DataFrame):
        feature_subset = list(feature_subset)

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, feature_subset], y, test_size=0.2, random_state=42)
    X_train = X_train.iloc[:, feature_subset]
    X_test = X_test.iloc[:, feature_subset]
    
    # Train LightGBM
    # model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)
    models = []
    lgb_params = {
    'objective': 'binary', # fixed
    'metric': 'binary_logloss', # 'auc' 'binary_error' 
    'boosting_type': 'gbdt', # 'dart' 'goss'
    'learning_rate': 0.01, # 0.01 ~ 0.3
    'num_leaves': 31, # 64 128 256 512 1024 2048
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

    y_true_total = y_test
    y_pred_proba_total = np.zeros((len(y_test), 5))
    y_pred_total = np.zeros((len(y_test), 5))
    y_final_pred = np.zeros(len(y_test))
    for i, model in enumerate(models):
        y_pred_proba = model.predict(X_test)

        y_pred_proba_total[:, i] = y_pred_proba
        
    inner_threshold = 0.5
                                            
    y_pred_total = np.where(y_pred_proba_total > inner_threshold, 1, 0)
    y_final_proba = y_pred_proba_total.mean(axis=1)
    y_final_pred = y_pred_total.sum(axis=1)
    threshold = 4
    y_group_pred = np.where(y_final_pred > threshold, 1, 0)
    acc = accuracy_score(y_true_total, y_group_pred)
    pre = precision_score(y_true_total, y_group_pred)
    rec = recall_score(y_true_total, y_group_pred)
    loss = log_loss(y_true_total, y_final_proba)

    # Predict and evaluate
    # return acc, 1/(math.sqrt(acc) + math.sqrt(pre) + math.sqrt(rec))
    return acc, acc
