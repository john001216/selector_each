import pandas as pd
import numpy as np
import random
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
import time

import lgbm
import objective_function
import streamlit as st

# Define paths
DATA_PATH = 'data/'
OUTPUT_PATH = 'log/'

# Load data
# feature_df = pd.read_csv(DATA_PATH + 'keyboard_features_1229.csv')
# feature_df.dropna(axis=0,inplace=True)
# train_label_set = pd.read_csv(DATA_PATH + 'train_label_set.csv', usecols=['student_id', 'is_PHQ-9'])
# validation_label_set = pd.read_csv(DATA_PATH + 'test_label_set.csv', usecols=['student_id', 'is_PHQ-9'])

# train_set = pd.merge(feature_df, train_label_set, on='student_id', how='inner')
# train_set.drop(['student_id','quiz_id','week','try','os'], axis=1, inplace=True)
# X = train_set.copy()
# X.drop(['is_PHQ-9'], axis=1, inplace=True)
# y = train_set['is_PHQ-9']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# validation_set = pd.merge(feature_df, validation_label_set, on='student_id', how='inner')
# validation_set.drop(['student_id','quiz_id','week','try','os'], axis=1, inplace=True)
# X_val = validation_set.copy()
# X_val.drop(['is_PHQ-9'], axis=1, inplace=True)
# y_val = validation_set['is_PHQ-9']

# Load data --> stylus
feature_df = pd.read_csv(DATA_PATH + 'feature_df_231227.csv')
feature_df.dropna(axis=0,inplace=True)
train_label_set = pd.read_csv(DATA_PATH + 'train_label_set.csv', usecols=['student_id', 'is_PHQ-9'])
validation_label_set = pd.read_csv(DATA_PATH + 'test_label_set.csv', usecols=['student_id', 'is_PHQ-9'])

train_set = pd.merge(feature_df, train_label_set, on='student_id', how='inner')
train_set.drop(['student_id','quiz_id','week_id','try_id','device_os'], axis=1, inplace=True)
X = train_set.copy()
X.drop(['is_PHQ-9'], axis=1, inplace=True)
y = train_set['is_PHQ-9']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

validation_set = pd.merge(feature_df, validation_label_set, on='student_id', how='inner')
validation_set.drop(['student_id','quiz_id','week_id','try_id','device_os'], axis=1, inplace=True)
X_val = validation_set.copy()
X_val.drop(['is_PHQ-9'], axis=1, inplace=True)
y_val = validation_set['is_PHQ-9']

# Setup simulated annealing algorithm
def simulated_annealing(X_train,
                        y_train,
                        X_test,
                        y_test,
                        X_val,
                        y_val,
                        maxiters=100,
                        alpha=0.99,
                        beta=100,
                        T_0=1,
                        update_iters=1,
                        temp_reduction='geometric'):
    """
    Function to perform feature selection using simulated annealing
    Inputs:
    X_train: Predictor features for training
    y_train: Train labels
    X_test : Predictor features for training
    y_test : Test labels
    maxiters: Maximum number of iterations
    alpha: Factor to reduce temperature
    beta: Constant in probability estimate 
    T_0: Initial temperature
    update_iters: Number of iterations required to update temperature
    temp_reduction: Strategy for temperature reduction schedule

    Output:
    1) Dataframe of parameters explored and corresponding model performance
    2) Best metric score (i.e. AUC score in this case)
    3) List of subset features that correspond to the best metric
    """
    columns = ['Iteration', 'Feature Count', 'Feature Set', 
               'Metric', 'Best Metric', 'Acceptance Probability', 
               'Random Number', 'Outcome']
    results = pd.DataFrame(index=range(maxiters+1), columns=columns)
    best_subset = None
    hash_values = set()
    T = T_0

    # for plotting
    acc_val_list = []
    pre_val_list = []
    rec_val_list = []
    f1_val_list = []
    auc_val_list = []
    
    # Get ascending range indices of all columns
    full_set = set(np.arange(len(X_train.columns)))

    # Generate initial random subset based on ~50% of columns
    # curr_subset = set(random.sample(list(full_set), round(0.5 * len(full_set))))
    curr_subset = set(random.sample(list(full_set), 1))
    X_train_curr = X_train.iloc[:, list(curr_subset)]
    X_test_curr = X_test.iloc[:, list(curr_subset)]
    X_val_curr = X_val.iloc[:, list(curr_subset)]

    # Get baseline metric score (i.e. AUC) of initial random subset
    lgbm_models = lgbm.get_model(X_train_curr, y_train)
    # xgbst_models = xgbst.get_model(X_train_curr, y_train)
    # acc_test here is 'metric'
    acc_test, acc_val,pre_val,rec_val,f1_val,auc_val = objective_function.get_acc(X_test_curr, y_test, X_val_curr, y_val, curr_subset, lgbm_models) # feature subset initialise한 curr_subset 넘겨줌
    # acc_test, acc_val,pre_val,rec_val = objective_function.get_acc(X_test_curr, y_test, X_val_curr, y_val, curr_subset, xgbst_models)
    # acc_val_list.append(acc_val)
    # pre_val_list.append(pre_val)
    # rec_val_list.append(rec_val)
    # f1_val_list.append(f1_val)
    # auc_val_list.append(auc_val)
    prev_metric = acc_test
    best_metric = prev_metric
    
    progress_chart_acc = st.empty()
    progress_chart_rec = st.empty()
    progress_chart_pre = st.empty()
    progress_chart_f1 = st.empty()
    progress_chart_auc = st.empty()
    
    
    progress_chart = st.empty()
    
    progress_text_val = st.empty()
    progress_text_test = st.empty()

    results.loc[0, 'Iteration'] = 0
    results.loc[0, 'Feature Count'] = len(curr_subset)
    results.loc[0, 'Feature Set'] = sorted(curr_subset)
    results.loc[0, 'Metric'] = acc_test
    results.loc[0, 'Best Metric'] = best_metric
    results.loc[0, 'Acceptance Probability'] = None
    results.loc[0, 'Random Number'] = None
    results.loc[0, 'Outcome'] = None
    
    for i in range(maxiters):
        # Termination conditions
        if T < 0.01:
            print(f'Temperature {T} below threshold. Termination condition met')
            break
        
        print(f'Starting Iteration {i+1}')

        # Execute pertubation (i.e. alter current subset to get new subset)
        while True:
            # Decide what type of pertubation to make
            if len(curr_subset) == len(full_set): 
                move = 'Remove'
            elif len(curr_subset) == 2: # Not to go below 2 features
                move = random.choice(['Add', 'Replace'])
            elif len(curr_subset) == 1: # Not to go below 2 features
                move = 'Add'
            else:
                move = random.choice(['Add', 'Replace', 'Remove'])
            
            # Get columns not yet used in current subset
            pending_cols = full_set.difference(curr_subset) 
            new_subset = curr_subset.copy()   

            if move == 'Add':        
                new_subset.add(random.choice(list(pending_cols)))
            elif move == 'Replace': 
                new_subset.remove(random.choice(list(curr_subset)))
                new_subset.add(random.choice(list(pending_cols)))
            else:
                new_subset.remove(random.choice(list(curr_subset)))
                
            if new_subset in hash_values:
                print('Subset already visited')
            else:
                hash_values.add(frozenset(new_subset))
                break

        # Filter dataframe to current subset
        X_train_new = X_train.iloc[:, list(new_subset)]
        X_test_new = X_test.iloc[:, list(new_subset)]
        X_val_new = X_val.iloc[:, list(new_subset)]
        
        lgbm_models = lgbm.get_model(X_train_new, y_train)
        # xgbst_models = xgbst.get_model(X_train_new, y_train)
        # this metric is 'acc_test'
        metric, acc_val,pre_val,rec_val,f1_val,auc_val = objective_function.get_acc(X_test_new, y_test, X_val_new, y_val, new_subset, lgbm_models)
        # metric, acc_val,pre_val,rec_val,f1_val,auc_val = objective_function.get_acc(X_test_new, y_test, X_val_new, y_val, new_subset, xgbst_models)
        acc_val_list.append(acc_val)
        pre_val_list.append(pre_val)
        rec_val_list.append(rec_val)
        f1_val_list.append(f1_val)
        auc_val_list.append(auc_val)

        # progress_chart.line_chart(acc_val_list)
        # data1 = {'Accuracy': acc_val_list, 'Epoch': list(range(i+2))}
        # fig1 = px.line(data1, x = 'Epoch', y='Accuracy', labels={'Epoch':'Epoch', 'Accuracy':'Accuracy'},
        #               title='Accuracy on validation set')
        # progress_chart_acc.plotly_chart(fig1)
        combined_data = pd.DataFrame({
        'Epoch': list(range(i+1)),
        'Accuracy': acc_val_list[:i+1],
        'Precision': pre_val_list[:i+1],
        'Recall': rec_val_list[:i+1],
        'F1_Score': f1_val_list[:i+1],
        'AUC': auc_val_list[:i+1]
        })

        # Melt the DataFrame
        melted_data = combined_data.melt(id_vars=['Epoch'], 
                                        value_vars=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC'],
                                        var_name='Metric', value_name='Value')

        # Plotting the combined line chart
        fig = px.line(melted_data, x='Epoch', y='Value', color='Metric',
                    labels={'Value': 'Metric Value', 'Epoch': 'Epoch'},
                    title='Performance Metrics Across Epochs on Validation Set')

        # Update the chart in the placeholder
        progress_chart.plotly_chart(fig)
        
        progress_text_val.text(f'Best Accuracy in Val: {max(acc_val_list)}')
        progress_text_test.text(f'Best Accuracy in Test: {best_metric}')
        
        # data2 = {'Precision': pre_val_list, 'Epoch': list(range(i+2))}
        # fig2 = px.line(data2, x = 'Epoch', y='Precision', labels={'Epoch':'Epoch', 'Precision':'Precision'},
        #               title='Precision on validation set')
        # progress_chart_rec.plotly_chart(fig2)
        
        # data3 = {'Recall': rec_val_list, 'Epoch': list(range(i+2))}
        # fig3 = px.line(data3, x = 'Epoch', y='Recall', labels={'Epoch':'Epoch', 'Recall':'Recall'},
        #               title='Recall on validation set')
        # progress_chart_pre.plotly_chart(fig3)
        
        # data4 = {'F1_Score': f1_val_list, 'Epoch': list(range(i+2))}
        # fig4 = px.line(data4, x = 'Epoch', y='F1_Score', labels={'Epoch':'Epoch', 'F1_Score':'F1_Score'},
        #               title='F1_Score on validation set')
        # progress_chart_f1.plotly_chart(fig4)
        
        # data5 = {'AUC': auc_val_list, 'Epoch': list(range(i+2))}
        # fig5 = px.line(data5, x = 'Epoch', y='AUC', labels={'Epoch':'Epoch', 'AUC':'AUC'},
        #               title='AUC on validation set')
        # progress_chart_auc.plotly_chart(fig5)
        
        # epoch_num = st.text_input('Enter a Epoch value')
        # if st.button('info on epoch'):
        #     result_filtered = results[results['iterate'] == int(epoch_num)]
        #     with st.expander("Show Details"):
        #         feature_set = result_filtered.iloc[0]['Feature Set']
        #         feature_set_txt = [list(X_train.columns)[i] for i in list(feature_set)]
        #         feature_cnt = result_filtered.iloc[0]['Feature Count']
        #         one_metric = result_filtered.iloc[0]['Metric']
        #         st.write(f'Feature Set: ',feature_set_txt)
        #         st.write(f'feature Count: ',feature_cnt)
        #         st.write(f'Accuracy: ',one_metric)

        if metric > prev_metric:
            print('Local improvement in metric from {:8.4f} to {:8.4f} ' # test metric // not validation metric
                  .format(prev_metric, metric) + ' - New subset accepted')
            outcome = 'Improved'
            accept_prob, rnd = '-', '-'
            prev_metric = metric
            curr_subset = new_subset.copy()

            # Keep track of overall best metric so far
            if metric > best_metric:
                print('Global improvement in metric from {:8.4f} to {:8.4f} '
                      .format(best_metric, metric) + ' - Best subset updated')
                best_metric = metric
                best_subset = new_subset.copy()
                
        else:
            rnd = np.random.uniform()
            diff = prev_metric - metric
            accept_prob = np.exp(-beta * diff / T) # diff가 클수록 acceptence prob 이 작아짐
                                                   # T가 작아질수록 T가 차가워질수록 acceptence prob 이 작아짐 --> 한 local에 머무를 확률이 높아짐

            if rnd < accept_prob:
                print('New subset has worse performance but still accept. Metric change' +
                      ':{:8.4f}, Acceptance probability:{:6.4f}, Random number:{:6.4f}'
                      .format(diff, accept_prob, rnd))
                outcome = 'Accept'
                prev_metric = metric
                curr_subset = new_subset.copy()
            else:
                print('New subset has worse performance, therefore reject. Metric change' +
                      ':{:8.4f}, Acceptance probability:{:6.4f}, Random number:{:6.4f}'
                      .format(diff, accept_prob, rnd))
                outcome = 'Reject'

        # Update results dataframe
        results.loc[i, 'Iteration'] = i+1
        results.loc[i, 'Feature Count'] = len(curr_subset)
        results.loc[i, 'Feature Set'] = sorted(curr_subset)
        results.loc[i, 'Metric'] = metric
        results.loc[i, 'Best Metric'] = best_metric
        results.loc[i, 'Acceptance Probability'] = accept_prob
        results.loc[i, 'Random Number'] = rnd
        results.loc[i, 'Outcome'] = outcome

        # Temperature cooling schedule
        if i % update_iters == 0:
            if temp_reduction == 'geometric':
                T = alpha * T
            elif temp_reduction == 'linear':
                T -= alpha
            elif temp_reduction == 'slow decrease':
                b = 5 # Arbitrary constant
                T = T / (1 + b * T)
            else:
                raise Exception("Temperature reduction strategy not recognized")

    # Convert column indices of best subset to original names
    best_subset_cols = [list(X_train.columns)[i] for i in list(best_subset)]

    # Drop NaN rows in results
    results = results.dropna(axis=0, how='all')

    # Save results as CSV
    dt_string = dt.now().strftime("%Y%m%d_%H%M%S")
    results.to_csv(f'{OUTPUT_PATH}/sa_output_{dt_string}.csv', index=False)

    return results, best_metric, best_subset_cols, acc_val_list

if __name__ == '__main__':
    results, best_metric, best_subset_cols, acc_val_list = simulated_annealing(X_train, y_train, X_test, y_test, X_val, y_val)
    print('best_metric', best_metric)
    print('best_subset_cols', best_subset_cols)
    print('acc_val_list', acc_val_list)
    # epoch_num = st.text_input('Enter a Epoch value')
    # if st.button('info on epoch'):
    #     result_filtered = results[results['iterate'] == int(epoch_num)]
    #     with st.expander("Show Details"):
    #         feature_set = result_filtered.iloc[0]['Feature Set']
    #         feature_set_txt = [list(X_train.columns)[i] for i in list(feature_set)]
    #         feature_cnt = result_filtered.iloc[0]['Feature Count']
    #         one_metric = result_filtered.iloc[0]['Metric']
    #         st.write(f'Feature Set: ',feature_set_txt)
    #         st.write(f'feature Count: ',feature_cnt)
    #         st.write(f'Accuracy: ',one_metric)
    st.write(f'best_metric: {best_metric}')
    st.write(f'best_subset_cols: {best_subset_cols}')
    st.write(f'acc_val_list: {acc_val_list}')