import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, log_loss, confusion_matrix

def get_acc(X_test, y_test, X_val, y_val, feature_subset, lgbm_models, feature_importance_df, column_index_mapping):
    if isinstance(X_test, pd.DataFrame):
        feature_subset = list(feature_subset)
    
    inner_threshold = 0.5 # for prob
    ############################################################
    
    y_test_true_total = y_test
    
    acc_test_list, pre_test_list, rec_test_list, f1_test_list, auc_test_list = [],[],[],[],[]

    for i, model in enumerate(lgbm_models):
        y_test_pred = model.predict(X_test)
        y_test_pred_total = np.where(y_test_pred > inner_threshold, 1, 0)
        test_acc = accuracy_score(y_test_true_total, y_test_pred_total)
        # test_pre = precision_score(y_test_true_total, y_test_pred)
        # test_rec = recall_score(y_test_true_total, y_test_pred)
        # test_f1 = f1_score(y_test_true_total, y_test_pred)
        # test_auc = roc_auc_score(y_test_true_total, y_test_pred)
        
        acc_test_list.append(test_acc)
        # pre_test_list.append(test_pre)
        # rec_test_list.append(test_rec)
        # f1_test_list.append(test_f1)
        # auc_test_list.append(test_auc)
        
        feature_importance_df[f'fold_{i}'] = model.feature_importance()
        # feature_importance_df[f'fold_{i}'] = pd.Series(model.feature_importance(), index=feature_importance_df.index)
        
        
    feature_importance_df.index = [column_index_mapping.get(index) for index in feature_importance_df.index]
    feature_importance_df['mean'] = feature_importance_df.mean(axis=1)
    # feature_importance_df['index'] = feature_subset
    feature_importance_df['normalised'] = feature_importance_df['mean'] / feature_importance_df['mean'].sum()
    # df_reset = feature_importance_df.reset_index()
    feature_importance_dict = pd.Series(feature_importance_df['normalised'].values, index=feature_importance_df.index).to_dict()

    ############################################################

    y_val_true_total = y_val
    
    acc_val_list, pre_val_list, rec_val_list, f1_val_list, auc_val_list = [],[],[],[],[]
    
    for i, model in enumerate(lgbm_models):
        y_val_pred = model.predict(X_val)
        y_val_pred_total = np.where(y_val_pred > inner_threshold, 1, 0)
        val_acc = accuracy_score(y_val_true_total, y_val_pred_total)
        val_pre = precision_score(y_val_true_total, y_val_pred_total)
        val_rec = recall_score(y_val_true_total, y_val_pred_total)
        val_f1 = f1_score(y_val_true_total, y_val_pred_total)
        val_auc = roc_auc_score(y_val_true_total, y_val_pred_total)
        
        acc_val_list.append(val_acc)
        pre_val_list.append(val_pre)
        rec_val_list.append(val_rec)
        f1_val_list.append(val_f1)
        auc_val_list.append(val_auc)
        
        cm = confusion_matrix(y_val_true_total, y_val_pred_total)
        print(cm)
                                            
    ############################################################
    
    return sum(acc_test_list)/5,sum(acc_val_list)/5,sum(pre_val_list)/5,sum(rec_val_list)/5,sum(f1_val_list)/5,sum(auc_val_list)/5,feature_importance_dict

def get_pre(X_test, y_test, X_val, y_val, feature_subset, lgbm_models, feature_importance_df, column_index_mapping):
    if isinstance(X_test, pd.DataFrame):
        feature_subset = list(feature_subset)

    ############################################################
    
    y_test_true_total = y_test
    
    acc_test_list, pre_test_list, rec_test_list, f1_test_list, auc_test_list = [],[],[],[],[]

    for i, model in enumerate(lgbm_models):
        y_test_pred = model.predict(X_test)
        # test_acc = accuracy_score(y_test_true_total, y_test_pred)
        test_pre = precision_score(y_test_true_total, y_test_pred)
        # test_rec = recall_score(y_test_true_total, y_test_pred)
        # test_f1 = f1_score(y_test_true_total, y_test_pred)
        # test_auc = roc_auc_score(y_test_true_total, y_test_pred)
        
        # acc_test_list.append(test_acc)
        pre_test_list.append(test_pre)
        # rec_test_list.append(test_rec)
        # f1_test_list.append(test_f1)
        # auc_test_list.append(test_auc)
        
        feature_importance_df[f'fold_{i}'] = model.feature_importance()
        # feature_importance_df[f'fold_{i}'] = pd.Series(model.feature_importance(), index=feature_importance_df.index)
        
    feature_importance_df.index = [column_index_mapping.get(index) for index in feature_importance_df.index]
    feature_importance_df['mean'] = feature_importance_df.mean(axis=1)
    # feature_importance_df['index'] = feature_subset
    feature_importance_df['normalised'] = feature_importance_df['mean'] / feature_importance_df['mean'].sum()
    # df_reset = feature_importance_df.reset_index()
    feature_importance_dict = pd.Series(feature_importance_df['normalised'].values, index=feature_importance_df.index).to_dict()

    ############################################################

    y_val_true_total = y_val
    
    acc_val_list, pre_val_list, rec_val_list, f1_val_list, auc_val_list = [],[],[],[],[]
    
    for i, model in enumerate(lgbm_models):
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val_true_total, y_val_pred)
        val_pre = precision_score(y_val_true_total, y_val_pred)
        val_rec = recall_score(y_val_true_total, y_val_pred)
        val_f1 = f1_score(y_val_true_total, y_val_pred)
        val_auc = roc_auc_score(y_val_true_total, y_val_pred)
        
        acc_val_list.append(val_acc)
        pre_val_list.append(val_pre)
        rec_val_list.append(val_rec)
        f1_val_list.append(val_f1)
        auc_val_list.append(val_auc)
        
        cm = confusion_matrix(y_test_true_total, y_test_pred)
        print(cm)
                                            
    ############################################################
    
    return sum(pre_test_list)/5,sum(acc_val_list)/5,sum(pre_val_list)/5,sum(rec_val_list)/5,sum(f1_val_list)/5,sum(auc_val_list)/5,feature_importance_dict

def get_rec(X_test, y_test, X_val, y_val, feature_subset, lgbm_models, feature_importance_df, column_index_mapping):
    if isinstance(X_test, pd.DataFrame):
        feature_subset = list(feature_subset)
    
    ############################################################
    
    y_test_true_total = y_test
    
    acc_test_list, pre_test_list, rec_test_list, f1_test_list, auc_test_list = [],[],[],[],[]

    for i, model in enumerate(lgbm_models):
        y_test_pred = model.predict(X_test)
        # test_acc = accuracy_score(y_test_true_total, y_test_pred)
        # test_pre = precision_score(y_test_true_total, y_test_pred)
        test_rec = recall_score(y_test_true_total, y_test_pred)
        # test_f1 = f1_score(y_test_true_total, y_test_pred)
        # test_auc = roc_auc_score(y_test_true_total, y_test_pred)
        
        # acc_test_list.append(test_acc)
        # pre_test_list.append(test_pre)
        rec_test_list.append(test_rec)
        # f1_test_list.append(test_f1)
        # auc_test_list.append(test_auc)
        
        feature_importance_df[f'fold_{i}'] = model.feature_importance()
        # feature_importance_df[f'fold_{i}'] = pd.Series(model.feature_importance(), index=feature_importance_df.index)
        
    feature_importance_df.index = [column_index_mapping.get(index) for index in feature_importance_df.index]
    feature_importance_df['mean'] = feature_importance_df.mean(axis=1)
    # feature_importance_df['index'] = feature_subset
    feature_importance_df['normalised'] = feature_importance_df['mean'] / feature_importance_df['mean'].sum()
    # df_reset = feature_importance_df.reset_index()
    feature_importance_dict = pd.Series(feature_importance_df['normalised'].values, index=feature_importance_df.index).to_dict()

    ############################################################

    y_val_true_total = y_val
    
    acc_val_list, pre_val_list, rec_val_list, f1_val_list, auc_val_list = [],[],[],[],[]
    
    for i, model in enumerate(lgbm_models):
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val_true_total, y_val_pred)
        val_pre = precision_score(y_val_true_total, y_val_pred)
        val_rec = recall_score(y_val_true_total, y_val_pred)
        val_f1 = f1_score(y_val_true_total, y_val_pred)
        val_auc = roc_auc_score(y_val_true_total, y_val_pred)
        
        acc_val_list.append(val_acc)
        pre_val_list.append(val_pre)
        rec_val_list.append(val_rec)
        f1_val_list.append(val_f1)
        auc_val_list.append(val_auc)
        
        cm = confusion_matrix(y_test_true_total, y_test_pred)
        print(cm)
                                            
    ############################################################
    
    return sum(rec_test_list)/5,sum(acc_val_list)/5,sum(pre_val_list)/5,sum(rec_val_list)/5,sum(f1_val_list)/5,sum(auc_val_list)/5,feature_importance_dict

def get_f1(X_test, y_test, X_val, y_val, feature_subset, lgbm_models, feature_importance_df, column_index_mapping):
    if isinstance(X_test, pd.DataFrame):
        feature_subset = list(feature_subset)

    ############################################################
    
    y_test_true_total = y_test
    
    acc_test_list, pre_test_list, rec_test_list, f1_test_list, auc_test_list = [],[],[],[],[]

    for i, model in enumerate(lgbm_models):
        y_test_pred = model.predict(X_test)
        # test_acc = accuracy_score(y_test_true_total, y_test_pred)
        # test_pre = precision_score(y_test_true_total, y_test_pred)
        # test_rec = recall_score(y_test_true_total, y_test_pred)
        test_f1 = f1_score(y_test_true_total, y_test_pred)
        # test_auc = roc_auc_score(y_test_true_total, y_test_pred)
        
        # acc_test_list.append(test_acc)
        # pre_test_list.append(test_pre)
        # rec_test_list.append(test_rec)
        f1_test_list.append(test_f1)
        # auc_test_list.append(test_auc)
        
        feature_importance_df[f'fold_{i}'] = model.feature_importance()
        # feature_importance_df[f'fold_{i}'] = pd.Series(model.feature_importance(), index=feature_importance_df.index)
        
    feature_importance_df.index = [column_index_mapping.get(index) for index in feature_importance_df.index]
    feature_importance_df['mean'] = feature_importance_df.mean(axis=1)
    # feature_importance_df['index'] = feature_subset
    feature_importance_df['normalised'] = feature_importance_df['mean'] / feature_importance_df['mean'].sum()
    # df_reset = feature_importance_df.reset_index()
    feature_importance_dict = pd.Series(feature_importance_df['normalised'].values, index=feature_importance_df.index).to_dict()

    ############################################################

    y_val_true_total = y_val
    
    acc_val_list, pre_val_list, rec_val_list, f1_val_list, auc_val_list = [],[],[],[],[]
    
    for i, model in enumerate(lgbm_models):
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val_true_total, y_val_pred)
        val_pre = precision_score(y_val_true_total, y_val_pred)
        val_rec = recall_score(y_val_true_total, y_val_pred)
        val_f1 = f1_score(y_val_true_total, y_val_pred)
        val_auc = roc_auc_score(y_val_true_total, y_val_pred)
        
        acc_val_list.append(val_acc)
        pre_val_list.append(val_pre)
        rec_val_list.append(val_rec)
        f1_val_list.append(val_f1)
        auc_val_list.append(val_auc)
        
        cm = confusion_matrix(y_test_true_total, y_test_pred)
        print(cm)
                                            
    ############################################################
    
    return sum(f1_test_list)/5,sum(acc_val_list)/5,sum(pre_val_list)/5,sum(rec_val_list)/5,sum(f1_val_list)/5,sum(auc_val_list)/5,feature_importance_dict

def get_auc(X_test, y_test, X_val, y_val, feature_subset, lgbm_models, feature_importance_df, column_index_mapping):
    if isinstance(X_test, pd.DataFrame):
        feature_subset = list(feature_subset)
    
    ############################################################
    
    y_test_true_total = y_test
    
    acc_test_list, pre_test_list, rec_test_list, f1_test_list, auc_test_list = [],[],[],[],[]

    for i, model in enumerate(lgbm_models):
        y_test_pred = model.predict(X_test)
        # test_acc = accuracy_score(y_test_true_total, y_test_pred)
        # test_pre = precision_score(y_test_true_total, y_test_pred)
        # test_rec = recall_score(y_test_true_total, y_test_pred)
        # test_f1 = f1_score(y_test_true_total, y_test_pred)
        test_auc = roc_auc_score(y_test_true_total, y_test_pred)
        
        # acc_test_list.append(test_acc)
        # pre_test_list.append(test_pre)
        # rec_test_list.append(test_rec)
        # f1_test_list.append(test_f1)
        auc_test_list.append(test_auc)
        
        feature_importance_df[f'fold_{i}'] = model.feature_importance()
        # feature_importance_df[f'fold_{i}'] = pd.Series(model.feature_importance(), index=feature_importance_df.index)
        
    feature_importance_df.index = [column_index_mapping.get(index) for index in feature_importance_df.index]
    feature_importance_df['mean'] = feature_importance_df.mean(axis=1)
    # feature_importance_df['index'] = feature_subset
    feature_importance_df['normalised'] = feature_importance_df['mean'] / feature_importance_df['mean'].sum()
    # df_reset = feature_importance_df.reset_index()
    feature_importance_dict = pd.Series(feature_importance_df['normalised'].values, index=feature_importance_df.index).to_dict()

    ############################################################

    y_val_true_total = y_val
    
    acc_val_list, pre_val_list, rec_val_list, f1_val_list, auc_val_list = [],[],[],[],[]
    
    for i, model in enumerate(lgbm_models):
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val_true_total, y_val_pred)
        val_pre = precision_score(y_val_true_total, y_val_pred)
        val_rec = recall_score(y_val_true_total, y_val_pred)
        val_f1 = f1_score(y_val_true_total, y_val_pred)
        val_auc = roc_auc_score(y_val_true_total, y_val_pred)
        
        acc_val_list.append(val_acc)
        pre_val_list.append(val_pre)
        rec_val_list.append(val_rec)
        f1_val_list.append(val_f1)
        auc_val_list.append(val_auc)
        
        cm = confusion_matrix(y_test_true_total, y_test_pred)
        print(cm)
                                            
    ############################################################
    
    return sum(auc_test_list)/5,sum(acc_val_list)/5,sum(pre_val_list)/5,sum(rec_val_list)/5,sum(f1_val_list)/5,sum(auc_val_list)/5,feature_importance_dict