import pandas as pd
import numpy as np
import random
from datetime import datetime as dt
from sklearn.model_selection import train_test_split

import lgbm
import objective_function

# Define paths
DATA_PATH = 'data/'
OUTPUT_PATH = 'log/'

# Load data
feature_df = pd.read_csv(DATA_PATH + 'keyboard_features_1229.csv')
feature_df.dropna(axis=0,inplace=True)
train_label_set = pd.read_csv(DATA_PATH + 'train_label_set.csv', usecols=['student_id', 'is_PHQ-9'])
validation_label_set = pd.read_csv(DATA_PATH + 'test_label_set.csv', usecols=['student_id', 'is_PHQ-9'])

train_set = pd.merge(feature_df, train_label_set, on='student_id', how='inner')
train_set.drop(['student_id','quiz_id','week','try','os'], axis=1, inplace=True)
X = train_set.copy()
X.drop(['is_PHQ-9'], axis=1, inplace=True)
y = train_set['is_PHQ-9']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

validation_set = pd.merge(feature_df, validation_label_set, on='student_id', how='inner')
validation_set.drop(['student_id','quiz_id','week','try','os'], axis=1, inplace=True)
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
                        beta=1,
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
    results = pd.DataFrame(index=range(maxiters), columns=columns)
    best_subset = None
    hash_values = set()
    T = T_0

    # for plotting
    acc_val_list = []
    
    # Get ascending range indices of all columns
    full_set = set(np.arange(len(X_train.columns)))

    # Generate initial random subset based on ~50% of columns
    curr_subset = set(random.sample(list(full_set), round(0.5 * len(full_set))))
    
    X_train_curr = X_train.iloc[:, list(curr_subset)]
    X_test_curr = X_test.iloc[:, list(curr_subset)]
    X_val_curr = X_val.iloc[:, list(curr_subset)]

    # Get baseline metric score (i.e. AUC) of initial random subset
    lgbm_models = lgbm.get_model(X_train_curr, y_train)
    # acc_test here is 'metric'
    acc_test, acc_val = objective_function.get_acc(X_test_curr, y_test, X_val_curr, y_val, curr_subset, lgbm_models) # feature subset initialise한 curr_subset 넘겨줌
    
    acc_val_list.append(acc_val)
    prev_metric = acc_test
    best_metric = prev_metric

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
        # this metric is 'acc_test'
        metric, acc_val = objective_function.get_acc(X_test_new, y_test, X_val_new, y_val, new_subset, lgbm_models)

        acc_val_list.append(acc_val)

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