import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, log_loss


def get_acc(X_test, y_test, X_val, y_val, feature_subset, lgbm_models, feature_importance_df, column_index_mapping):
    if isinstance(X_test, pd.DataFrame):
        feature_subset = list(feature_subset)
    
    inner_threshold = 0.5 # for prob
    threshold = 1         # for sum
    
    ############################################################
    
    y_test_true_total = y_test
    y_test_pred_proba_total = np.zeros((len(y_test), 5))
    
    for i, model in enumerate(lgbm_models):
        y_test_pred_proba = model.predict(X_test)

        y_test_pred_proba_total[:, i] = y_test_pred_proba
        # print(model.feature_importance())
        feature_importance_df[f'fold_{i}'] = model.feature_importance()
        # feature_importance_df[f'fold_{i}'] = pd.Series(model.feature_importance(), index=feature_importance_df.index)
        
    feature_importance_df.index = [column_index_mapping.get(index) for index in feature_importance_df.index]
    feature_importance_df['mean'] = feature_importance_df.mean(axis=1)
    # feature_importance_df['index'] = feature_subset
    feature_importance_df['normalised'] = feature_importance_df['mean'] / feature_importance_df['mean'].sum()
    # df_reset = feature_importance_df.reset_index()
    feature_importance_dict = pd.Series(feature_importance_df['normalised'].values, index=feature_importance_df.index).to_dict()
                                            
    y_test_pred_total = np.where(y_test_pred_proba_total > inner_threshold, 1, 0)
    y_test_final_proba = y_test_pred_proba_total.mean(axis=1)
    y_test_final_pred = y_test_pred_total.sum(axis=1)
    y_test_group_pred = np.where(y_test_final_pred > threshold, 1, 0)
    
    acc_test = accuracy_score(y_test_true_total, y_test_group_pred)
    # pre_test = precision_score(y_test_true_total, y_test_group_pred)
    # rec_test = recall_score(y_test_true_total, y_test_group_pred)
    # loss_test = log_loss(y_test_true_total, y_test_final_proba)
    # f1_test = f1_score(y_test_true_total, y_test_group_pred)
    # auc_test = roc_auc_score(y_test_true_total, y_test_final_proba)
    
    ############################################################

    y_val_true_total = y_val
    y_val_pred_proba_total = np.zeros((len(y_val), 5))
    
    for i, model in enumerate(lgbm_models):
        y_val_pred_proba = model.predict(X_val)

        y_val_pred_proba_total[:, i] = y_val_pred_proba
                                            
    y_val_pred_total = np.where(y_val_pred_proba_total > inner_threshold, 1, 0)
    y_val_final_proba = y_val_pred_proba_total.mean(axis=1)
    y_val_final_pred = y_val_pred_total.sum(axis=1)
    y_val_group_pred = np.where(y_val_final_pred > threshold, 1, 0)
    
    acc_val = accuracy_score(y_val_true_total, y_val_group_pred)
    pre_val = precision_score(y_val_true_total, y_val_group_pred)
    rec_val = recall_score(y_val_true_total, y_val_group_pred)
    # loss_val = log_loss(y_val_true_total, y_val_final_proba)
    f1_val = f1_score(y_val_true_total, y_val_group_pred)
    auc_val = roc_auc_score(y_val_true_total, y_val_group_pred)
    
    ############################################################
    
    return acc_test,acc_val,pre_val,rec_val,f1_val,auc_val,feature_importance_dict

def get_pre(X_test, y_test, X_val, y_val, feature_subset, lgbm_models_list, feature_importance_df, column_index_mapping):
    if isinstance(X_test, pd.DataFrame):
        feature_subset = list(feature_subset)
    
    inner_threshold = 0.5 # for prob
    threshold = 1         # for sum
    
    ############################################################
    
    y_test_true_total = y_test
    y_test_pred_proba_total = np.zeros((len(y_test), 5))
    
    precision_metric_list = []
    total_acc_list,total_pre_list,total_rec_list,total_f1_list,total_auc_list = [],[],[],[],[]
    
    for lgbm_models in lgbm_models_list:
        for i, model in enumerate(lgbm_models):
            y_test_pred_proba = model.predict(X_test)

            y_test_pred_proba_total[:, i] = y_test_pred_proba
            # print(model.feature_importance())
            feature_importance_df[f'fold_{i}'] = model.feature_importance()
            # feature_importance_df[f'fold_{i}'] = pd.Series(model.feature_importance(), index=feature_importance_df.index)
            
        feature_importance_df.index = [column_index_mapping.get(index) for index in feature_importance_df.index]
        feature_importance_df['mean'] = feature_importance_df.mean(axis=1)
        # feature_importance_df['index'] = feature_subset
        feature_importance_df['normalised'] = feature_importance_df['mean'] / feature_importance_df['mean'].sum()
        # df_reset = feature_importance_df.reset_index()
        feature_importance_dict = pd.Series(feature_importance_df['normalised'].values, index=feature_importance_df.index).to_dict()
                                                
        y_test_pred_total = np.where(y_test_pred_proba_total > inner_threshold, 1, 0)
        y_test_final_proba = y_test_pred_proba_total.mean(axis=1)
        y_test_final_pred = y_test_pred_total.sum(axis=1)
        y_test_group_pred = np.where(y_test_final_pred > threshold, 1, 0)
        
        # acc_test = accuracy_score(y_test_true_total, y_test_group_pred)
        pre_test = precision_score(y_test_true_total, y_test_group_pred)
        # rec_test = recall_score(y_test_true_total, y_test_group_pred)
        # loss_test = log_loss(y_test_true_total, y_test_final_proba)
        # f1_test = f1_score(y_test_true_total, y_test_group_pred)
        # auc_test = roc_auc_score(y_test_true_total, y_test_final_proba)
        precision_metric_list.append(pre_test)
        ############################################################

        y_val_true_total = y_val
        y_val_pred_proba_total = np.zeros((len(y_val), 5))
        
        for i, model in enumerate(lgbm_models):
            y_val_pred_proba = model.predict(X_val)

            y_val_pred_proba_total[:, i] = y_val_pred_proba
                                                
        y_val_pred_total = np.where(y_val_pred_proba_total > inner_threshold, 1, 0)
        y_val_final_proba = y_val_pred_proba_total.mean(axis=1)
        y_val_final_pred = y_val_pred_total.sum(axis=1)
        y_val_group_pred = np.where(y_val_final_pred > threshold, 1, 0)
        
        acc_val = accuracy_score(y_val_true_total, y_val_group_pred)
        pre_val = precision_score(y_val_true_total, y_val_group_pred)
        rec_val = recall_score(y_val_true_total, y_val_group_pred)
        # loss_val = log_loss(y_val_true_total, y_val_final_proba)
        f1_val = f1_score(y_val_true_total, y_val_group_pred)
        auc_val = roc_auc_score(y_val_true_total, y_val_group_pred)
        
        total_acc_list.append(acc_val)
        total_pre_list.append(pre_val)
        total_rec_list.append(rec_val)
        total_f1_list.append(f1_val)
        total_auc_list.append(auc_val)
        
        ############################################################
        
    pre_test_max = max(precision_metric_list)
    index = precision_metric_list.index(pre_test_max)
    acc = total_acc_list[index]
    pre = total_pre_list[index]
    rec = total_rec_list[index]
    f1 = total_f1_list[index]
    auc = total_auc_list[index]
    
    return pre_test_max,acc,pre,rec,f1,auc,feature_importance_dict

def get_rec(X_test, y_test, X_val, y_val, feature_subset, lgbm_models, feature_importance_df, column_index_mapping):
    if isinstance(X_test, pd.DataFrame):
        feature_subset = list(feature_subset)
    
    inner_threshold = 0.5 # for prob
    threshold = 1         # for sum
    
    ############################################################
    
    y_test_true_total = y_test
    y_test_pred_proba_total = np.zeros((len(y_test), 5))
    
    for i, model in enumerate(lgbm_models):
        y_test_pred_proba = model.predict(X_test)

        y_test_pred_proba_total[:, i] = y_test_pred_proba
        # print(model.feature_importance())
        feature_importance_df[f'fold_{i}'] = model.feature_importance()
        # feature_importance_df[f'fold_{i}'] = pd.Series(model.feature_importance(), index=feature_importance_df.index)
        
    feature_importance_df.index = [column_index_mapping.get(index) for index in feature_importance_df.index]
    feature_importance_df['mean'] = feature_importance_df.mean(axis=1)
    # feature_importance_df['index'] = feature_subset
    feature_importance_df['normalised'] = feature_importance_df['mean'] / feature_importance_df['mean'].sum()
    # df_reset = feature_importance_df.reset_index()
    feature_importance_dict = pd.Series(feature_importance_df['normalised'].values, index=feature_importance_df.index).to_dict()
                                            
    y_test_pred_total = np.where(y_test_pred_proba_total > inner_threshold, 1, 0)
    y_test_final_proba = y_test_pred_proba_total.mean(axis=1)
    y_test_final_pred = y_test_pred_total.sum(axis=1)
    y_test_group_pred = np.where(y_test_final_pred > threshold, 1, 0)
    
    # acc_test = accuracy_score(y_test_true_total, y_test_group_pred)
    # pre_test = precision_score(y_test_true_total, y_test_group_pred)
    rec_test = recall_score(y_test_true_total, y_test_group_pred)
    # loss_test = log_loss(y_test_true_total, y_test_final_proba)
    # f1_test = f1_score(y_test_true_total, y_test_group_pred)
    # auc_test = roc_auc_score(y_test_true_total, y_test_final_proba)
    
    ############################################################

    y_val_true_total = y_val
    y_val_pred_proba_total = np.zeros((len(y_val), 5))
    
    for i, model in enumerate(lgbm_models):
        y_val_pred_proba = model.predict(X_val)

        y_val_pred_proba_total[:, i] = y_val_pred_proba
                                            
    y_val_pred_total = np.where(y_val_pred_proba_total > inner_threshold, 1, 0)
    y_val_final_proba = y_val_pred_proba_total.mean(axis=1)
    y_val_final_pred = y_val_pred_total.sum(axis=1)
    y_val_group_pred = np.where(y_val_final_pred > threshold, 1, 0)
    
    acc_val = accuracy_score(y_val_true_total, y_val_group_pred)
    pre_val = precision_score(y_val_true_total, y_val_group_pred)
    rec_val = recall_score(y_val_true_total, y_val_group_pred)
    # loss_val = log_loss(y_val_true_total, y_val_final_proba)
    f1_val = f1_score(y_val_true_total, y_val_group_pred)
    auc_val = roc_auc_score(y_val_true_total, y_val_group_pred)
    
    ############################################################
    
    return rec_test,acc_val,pre_val,rec_val,f1_val,auc_val,feature_importance_dict

def get_f1(X_test, y_test, X_val, y_val, feature_subset, lgbm_models, feature_importance_df, column_index_mapping):
    if isinstance(X_test, pd.DataFrame):
        feature_subset = list(feature_subset)
    
    inner_threshold = 0.5 # for prob
    threshold = 1         # for sum
    
    ############################################################
    
    y_test_true_total = y_test
    y_test_pred_proba_total = np.zeros((len(y_test), 5))
    
    for i, model in enumerate(lgbm_models):
        y_test_pred_proba = model.predict(X_test)

        y_test_pred_proba_total[:, i] = y_test_pred_proba
        # print(model.feature_importance())
        feature_importance_df[f'fold_{i}'] = model.feature_importance()
        # feature_importance_df[f'fold_{i}'] = pd.Series(model.feature_importance(), index=feature_importance_df.index)
        
    feature_importance_df.index = [column_index_mapping.get(index) for index in feature_importance_df.index]
    feature_importance_df['mean'] = feature_importance_df.mean(axis=1)
    # feature_importance_df['index'] = feature_subset
    feature_importance_df['normalised'] = feature_importance_df['mean'] / feature_importance_df['mean'].sum()
    # df_reset = feature_importance_df.reset_index()
    feature_importance_dict = pd.Series(feature_importance_df['normalised'].values, index=feature_importance_df.index).to_dict()
                                            
    y_test_pred_total = np.where(y_test_pred_proba_total > inner_threshold, 1, 0)
    y_test_final_proba = y_test_pred_proba_total.mean(axis=1)
    y_test_final_pred = y_test_pred_total.sum(axis=1)
    y_test_group_pred = np.where(y_test_final_pred > threshold, 1, 0)
    
    # acc_test = accuracy_score(y_test_true_total, y_test_group_pred)
    # pre_test = precision_score(y_test_true_total, y_test_group_pred)
    # rec_test = recall_score(y_test_true_total, y_test_group_pred)
    # loss_test = log_loss(y_test_true_total, y_test_final_proba)
    f1_test = f1_score(y_test_true_total, y_test_group_pred)
    # auc_test = roc_auc_score(y_test_true_total, y_test_final_proba)
    
    ############################################################

    y_val_true_total = y_val
    y_val_pred_proba_total = np.zeros((len(y_val), 5))
    
    for i, model in enumerate(lgbm_models):
        y_val_pred_proba = model.predict(X_val)

        y_val_pred_proba_total[:, i] = y_val_pred_proba
                                            
    y_val_pred_total = np.where(y_val_pred_proba_total > inner_threshold, 1, 0)
    y_val_final_proba = y_val_pred_proba_total.mean(axis=1)
    y_val_final_pred = y_val_pred_total.sum(axis=1)
    y_val_group_pred = np.where(y_val_final_pred > threshold, 1, 0)
    
    acc_val = accuracy_score(y_val_true_total, y_val_group_pred)
    pre_val = precision_score(y_val_true_total, y_val_group_pred)
    rec_val = recall_score(y_val_true_total, y_val_group_pred)
    # loss_val = log_loss(y_val_true_total, y_val_final_proba)
    f1_val = f1_score(y_val_true_total, y_val_group_pred)
    auc_val = roc_auc_score(y_val_true_total, y_val_group_pred)
    
    ############################################################
    
    return f1_test,acc_val,pre_val,rec_val,f1_val,auc_val,feature_importance_dict

def get_auc(X_test, y_test, X_val, y_val, feature_subset, lgbm_models, feature_importance_df, column_index_mapping):
    if isinstance(X_test, pd.DataFrame):
        feature_subset = list(feature_subset)
    
    inner_threshold = 0.5 # for prob
    threshold = 1         # for sum
    
    ############################################################
    
    y_test_true_total = y_test
    y_test_pred_proba_total = np.zeros((len(y_test), 5))
    
    for i, model in enumerate(lgbm_models):
        y_test_pred_proba = model.predict(X_test)

        y_test_pred_proba_total[:, i] = y_test_pred_proba
        # print(model.feature_importance())
        feature_importance_df[f'fold_{i}'] = model.feature_importance()
        # feature_importance_df[f'fold_{i}'] = pd.Series(model.feature_importance(), index=feature_importance_df.index)
        
    feature_importance_df.index = [column_index_mapping.get(index) for index in feature_importance_df.index]
    feature_importance_df['mean'] = feature_importance_df.mean(axis=1)
    # feature_importance_df['index'] = feature_subset
    feature_importance_df['normalised'] = feature_importance_df['mean'] / feature_importance_df['mean'].sum()
    # df_reset = feature_importance_df.reset_index()
    feature_importance_dict = pd.Series(feature_importance_df['normalised'].values, index=feature_importance_df.index).to_dict()
                                            
    y_test_pred_total = np.where(y_test_pred_proba_total > inner_threshold, 1, 0)
    y_test_final_proba = y_test_pred_proba_total.mean(axis=1)
    y_test_final_pred = y_test_pred_total.sum(axis=1)
    y_test_group_pred = np.where(y_test_final_pred > threshold, 1, 0)
    
    # acc_test = accuracy_score(y_test_true_total, y_test_group_pred)
    # pre_test = precision_score(y_test_true_total, y_test_group_pred)
    # rec_test = recall_score(y_test_true_total, y_test_group_pred)
    # loss_test = log_loss(y_test_true_total, y_test_final_proba)
    # f1_test = f1_score(y_test_true_total, y_test_group_pred)
    auc_test = roc_auc_score(y_test_true_total, y_test_final_proba)
    
    ############################################################

    y_val_true_total = y_val
    y_val_pred_proba_total = np.zeros((len(y_val), 5))
    
    for i, model in enumerate(lgbm_models):
        y_val_pred_proba = model.predict(X_val)

        y_val_pred_proba_total[:, i] = y_val_pred_proba
                                            
    y_val_pred_total = np.where(y_val_pred_proba_total > inner_threshold, 1, 0)
    y_val_final_proba = y_val_pred_proba_total.mean(axis=1)
    y_val_final_pred = y_val_pred_total.sum(axis=1)
    y_val_group_pred = np.where(y_val_final_pred > threshold, 1, 0)
    
    acc_val = accuracy_score(y_val_true_total, y_val_group_pred)
    pre_val = precision_score(y_val_true_total, y_val_group_pred)
    rec_val = recall_score(y_val_true_total, y_val_group_pred)
    # loss_val = log_loss(y_val_true_total, y_val_final_proba)
    f1_val = f1_score(y_val_true_total, y_val_group_pred)
    auc_val = roc_auc_score(y_val_true_total, y_val_group_pred)
    
    ############################################################
    
    return auc_test,acc_val,pre_val,rec_val,f1_val,auc_val,feature_importance_dict