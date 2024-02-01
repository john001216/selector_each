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

def data_caller():
    # Define paths
    DATA_PATH = 'data/'
    OUTPUT_PATH = 'log/'

    # Load data
    feature_df_keyboard = pd.read_csv(DATA_PATH + 'keyboard_features_1229.csv')
    feature_df_keyboard.dropna(axis=0,inplace=True)
    train_label_set_keyboard = pd.read_csv(DATA_PATH + 'train_label_set.csv', usecols=['student_id', 'is_PHQ-9'])
    validation_label_set_keyboard = pd.read_csv(DATA_PATH + 'test_label_set.csv', usecols=['student_id', 'is_PHQ-9'])

    # train_set_keyboard = pd.merge(feature_df_keyboard, train_label_set_keyboard, on='student_id', how='inner')
    # train_set_keyboard.drop(['quiz_id','week','try','os'], axis=1, inplace=True)
    # X_keyboard = train_set_keyboard.copy()
    # X_keyboard.drop(['is_PHQ-9'], axis=1, inplace=True)
    # y_keyboard = train_set_keyboard['is_PHQ-9']
    # X_train_keyboard, X_test_keyboard, y_train_keyboard, y_test_keyboard = train_test_split(X_keyboard, y_keyboard, test_size=0.2, random_state=42)

    # validation_set_keyboard = pd.merge(feature_df_keyboard, validation_label_set_keyboard, on='student_id', how='inner')
    # validation_set_keyboard.drop(['quiz_id','week','try','os'], axis=1, inplace=True)
    # X_val_keyboard = validation_set_keyboard.copy()
    # X_val_keyboard.drop(['is_PHQ-9'], axis=1, inplace=True)
    # y_val = validation_set_keyboard['is_PHQ-9']

    # Load data --> stylus
    feature_df_stylus = pd.read_csv(DATA_PATH + 'feature_df_231227.csv')
    feature_df_stylus.dropna(axis=0,inplace=True)
    train_label_set = pd.read_csv(DATA_PATH + 'train_label_set.csv', usecols=['student_id', 'is_PHQ-9'])
    validation_label_set = pd.read_csv(DATA_PATH + 'test_label_set.csv', usecols=['student_id', 'is_PHQ-9'])

    # train_set_stylus = pd.merge(feature_df_stylus, train_label_set, on='student_id', how='inner')
    # train_set_stylus.drop(['quiz_id','week_id','try_id','device_os'], axis=1, inplace=True)
    # X_stylus = train_set_stylus.copy()
    # X_stylus.drop(['is_PHQ-9'], axis=1, inplace=True)
    # y_stylus = train_set_stylus['is_PHQ-9']
    # X_train_stylus, X_test_stylus, y_train_stylus, y_test_stylus = train_test_split(X_stylus, y_stylus, test_size=0.2, random_state=42)

    # validation_set_stylus = pd.merge(feature_df_stylus, validation_label_set, on='student_id', how='inner')
    # validation_set_stylus.drop(['quiz_id','week_id','try_id','device_os'], axis=1, inplace=True)
    # X_val_stylus = validation_set_stylus.copy()
    # X_val_stylus.drop(['is_PHQ-9'], axis=1, inplace=True)
    # y_val_stylus = validation_set_stylus['is_PHQ-9']


    feature_df_keyboard.drop(['quiz_id','week','try','os'], axis=1, inplace=True)
    feature_df_stylus.drop(['quiz_id','week_id','try_id','device_os'], axis=1, inplace=True)
    #########################
    feature_keyboard_train = feature_df_keyboard.copy()
    feature_stylus_train = feature_df_stylus.copy()

    feature_keyboard_test = feature_df_keyboard.copy()
    feature_stylus_test = feature_df_stylus.copy()
    #########################
    train_set_both = pd.merge(feature_keyboard_train, feature_stylus_train, on='student_id', how='inner')

    train_set_both_merge_label = pd.merge(train_set_both, train_label_set, on='student_id', how='inner')
    train_set_both_merge_label_drop = train_set_both_merge_label.copy()
    train_set_both_merge_label_drop.drop(['student_id'], axis=1, inplace=True)

    # print(train_set_both_merge_label_drop)
    # print(train_set_both_merge_label_drop.columns)

    X = train_set_both_merge_label_drop.copy()
    X.drop(['is_PHQ-9'], axis=1, inplace=True)
    y = train_set_both_merge_label_drop['is_PHQ-9']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #########################
    validation_set_both = pd.merge(feature_keyboard_test, feature_stylus_test, on='student_id', how='inner')

    validation_set_both_merge_label = pd.merge(validation_set_both, validation_label_set, on='student_id', how='inner')
    validation_set_both_merge_label_drop = validation_set_both_merge_label.copy()
    validation_set_both_merge_label_drop.drop(['student_id'], axis=1, inplace=True)

    # print(validation_set_both_merge_label_drop)
    # print(validation_set_both_merge_label_drop.columns)

    X_val = validation_set_both_merge_label_drop.copy()
    X_val.drop(['is_PHQ-9'], axis=1, inplace=True)
    y_val = validation_set_both_merge_label_drop['is_PHQ-9']

    return X_train, y_train, X_test, y_test, X_val, y_val