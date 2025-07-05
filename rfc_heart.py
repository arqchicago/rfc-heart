# -*- coding: utf-8 -*-
"""
@author: Ahmad Qadri
Random Forest Classification on Heart Disease dataset

"""

import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection as skms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

random_seed = 295471
target_var = 'target'
test_size = 0.20
train_size = 1-test_size

def read_data(file_path):
    """
    Reads the CSV file and returns a DataFrame.
    """

    # Check if the file exists
    try:
        with open(file_path, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

    # if file exists, read the CSV file
    heart_df = pd.read_csv(file_path)
    rows, cols = heart_df.shape
    print(f'> data rows = {rows}  data cols = {cols}')

    return heart_df


def split_train_test_data(df, target_var, test_size, random_seed=1):
    """
    Splits the dataframe into X_train, X_test, y_train, y_test using sklearn's train_test_split.
    Args:
        df (pd.DataFrame): The input dataframe.
        target_var (str): The name of the target variable column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 1.
    Returns:
        X_train, X_test, y_train, y_test
    """
    y = df[target_var]
    X = df.drop(target_var, axis=1)
    X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=test_size, random_state=random_seed)
    X_train_rows, y_train_rows = X_train.shape[0], y_train.shape[0]
    X_test_rows, y_test_rows = X_test.shape[0], y_test.shape[0]
    rows = X_train_rows + X_test_rows

    print(f'> training set = {X_train_rows} ({round(X_train_rows*1.0/rows,3)})')
    print(f'> testing set = {X_test_rows} ({round(X_test_rows*1.0/rows,3)}) \n')

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, random_seed=1):
    """
    Trains a Random Forest Classifier with hyperparameter tuning using RandomizedSearchCV.
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 1.
    Returns:
        optimized_rfc: The trained Random Forest Classifier with best hyperparameters.
    """

    #----  random forest training with hyperparameter tuning
    random_grid = {'n_estimators': [10, 100, 500, 1000],
                'max_features': [0.25, 0.50, 0.75],
                'max_depth': [5, 10, 20, 25],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [5, 7, 10],
                'bootstrap': [True, False],
                'random_state': [random_seed]}

    print('> Random Forest classifier...')
    optimized_rfc = skms.RandomizedSearchCV(estimator = RandomForestClassifier(), 
                                            param_distributions = random_grid, 
                                            n_iter = 5, 
                                            cv = 3, 
                                            scoring=['roc_auc', 'recall'],
                                            refit ='roc_auc',
                                            verbose=1, 
                                            n_jobs = 1,
                                            random_state = random_seed)

    optimized_rfc.fit(X_train, y_train)
    best_params = optimized_rfc.best_params_
    best_score = optimized_rfc.best_score_
    print('Model training completed!')
    print(f'Best parameters are: {best_params}')
    print(f'Best score (roc_auc) is: {best_score:.3f}')
    print('\n')
    return optimized_rfc


def evaluate_model(model, X_train, X_test, y_train, y_test):
    #----  predicting on the training and testing set
    y_train_pred = model.predict(X_train)
    accuracy_train = round(accuracy_score(y_train, y_train_pred),3)
    roc_auc_train = round(roc_auc_score(y_train, y_train_pred),3)
    recall_train = round(recall_score(y_train, y_train_pred),3)
    precision_train = round(precision_score(y_train, y_train_pred),3)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy_test = round(accuracy_score(y_test, y_pred),3)
    roc_auc_test = round(roc_auc_score(y_test, y_pred),3)
    recall_test = round(recall_score(y_test, y_pred),3)
    precision_test = round(precision_score(y_test, y_pred),3)
    cm = confusion_matrix(y_test, y_pred)

    print('> evaluation metrics \n')
    print('%-10s %20s %10s' % ('metric','training','testing'))
    print('%-10s %20s %10s' % ('roc auc', roc_auc_train, roc_auc_test))
    print('%-10s %20s %10s' % ('accuracy', accuracy_train, accuracy_test))
    print('%-10s %20s %10s' % ('recall', recall_train, recall_test))
    print('%-10s %20s %10s' % ('precision', precision_train, precision_test))
    print('\n')
    print('> confusion matrix \n')
    print(cm)
    print('\n')

    # ROC Plot
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_title('ROC Curve (auc = %0.2f)' % roc_auc, fontsize=22, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(True)
    fig.savefig('output/roc_plot.png')


if __name__ == '__main__':

    random_seed = 295471
    target_var = 'target'
    test_size = 0.20

    # Read the data
    heart_df = read_data('data//heart2.csv')

    if heart_df is not None:
        # Split the data
        X_train, X_test, y_train, y_test = split_train_test_data(heart_df, target_var, test_size, random_seed)

        # Train the model
        optimized_rfc = train_model(X_train, X_test, y_train, y_test, random_seed)

        # Evaluate the model
        evaluate_model(optimized_rfc, X_train, X_test, y_train, y_test)
    
    else:
        print("Model not trained due to data loading failure")

