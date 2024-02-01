import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np

# get important features from lightGBM model
def get_model(X_train, y_train):
    # Specify your configurations as a dict
    models_list = []
    models = []
    
    # num_boost_round=1000
    for lr, num_boost_round in zip([0.001, 0.01], [500, 100]):
        for num_leaves in [63, 127]:
            for min_data_in_leaf in [5, 10]:
                for max_bin in [255, 1024]:
                    lgb_params = {
                    'objective': 'binary', # fixed
                    'metric': 'binary_logloss', # 'auc' 'binary_error' 
                    'boosting_type': 'gbdt', # 'dart' 'goss'
                    'learning_rate': lr, # 0.01 ~ 0.3
                    'num_leaves': num_leaves, # 63 127 255 511 1023 2047
                    'max_depth': -1, # +1 -1~8
                    'min_data_in_leaf': min_data_in_leaf, # 20 ~ 900
                    'is_unbalance': True, # 'scale_pos_weight'
                    'max_bin': max_bin, # +100 or + 200 255 ~ 1024 
                    'verbose': -1, 
                    'random_state': 1 # 0 or 1
                    }
    
                    kf = KFold(n_splits=5, shuffle=True, random_state=0)
                    for train_index, val_index in tqdm(kf.split(X_train), desc=f'PHQ-9 - {len(X_train)} samples: '):
                        X_train_inner, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
                        y_train_inner, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

                        lgb_train = lgb.Dataset(X_train_inner, y_train_inner)
                        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

                        gbm = lgb.train(lgb_params, lgb_train, num_boost_round=num_boost_round, valid_sets=lgb_eval,)

                        models.append(gbm)
                    models_list.append(models)
    
    return models_list, lr, num_boost_round,num_leaves,min_data_in_leaf,max_bin