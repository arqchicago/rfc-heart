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
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

target_var = 'target'
test_size = 0.20
train_size = 1-test_size

#----  reading data
heart_df = pd.read_csv('data\\heart2.csv')
rows, cols = heart_df.shape
target0_rows = heart_df[heart_df[target_var]==0].shape[0]
target1_rows = heart_df[heart_df[target_var]==1].shape[0]
print(f'> data rows = {rows}  data cols = {cols}')
print(f'> {target_var}==0 ({target0_rows})  {target_var}==1 ({target1_rows})')


#----  splitting into training & testing sets
y = heart_df.target
X = heart_df.drop(target_var, axis=1)
features = X.columns.tolist()
X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=test_size)
X_train_rows, y_train_rows = X_train.shape[0], y_train.shape[0]
X_test_rows, y_test_rows = X_test.shape[0], y_test.shape[0]
train_rows, test_rows = -1, -1

if X_train_rows == y_train_rows:
    train_rows = X_train_rows

if X_test_rows == y_test_rows:
    test_rows = X_test_rows
    
print(f'> features = {len(features)}')
print(f'> training set = {train_rows} ({round(train_rows*1.0/rows,3)})')
print(f'> testing set = {test_rows} ({round(test_rows*1.0/rows,3)}) \n')


#----  random forest training with hyperparameter tuning
random_grid = {'n_estimators': [10, 100, 500, 1000],
               'max_features': [0.25, 0.50, 0.75],
               'max_depth': [10, 20, 30, 40, 50, None],
               'min_samples_split': [5, 10, 20],
               'min_samples_leaf': [3, 5, 10],
               'bootstrap': [True, False]}

print('> Random Forest classifier...')
optimized_rfc = skms.RandomizedSearchCV(estimator = RandomForestClassifier(), 
                                        param_distributions = random_grid, 
                                        n_iter = 100, 
                                        cv = 5, 
                                        scoring='roc_auc',
                                        verbose=1, 
                                        random_state=42, 
                                        n_jobs = -1)

optimized_rfc.fit(X_train, y_train)
print('\n')


#----  obtaining results of the grid run
cv_results = optimized_rfc.cv_results_
cv_params = cv_results['params']
cv_mean_scores = optimized_rfc.cv_results_['mean_test_score']
cv_mean_runtimes = cv_results['mean_score_time']

best_params = optimized_rfc.best_params_
best_score = optimized_rfc.best_score_

cv_params_df = pd.DataFrame(cv_params)
cv_params_df['cv_mean_scores'] = pd.Series(cv_mean_scores)
cv_params_df['cv_mean_runtimes'] = pd.Series(cv_mean_runtimes)
print('> hyperparameter tuning results')
print(cv_params_df)

print(f'> best hyperparameters = {best_params}')
print(f'> best cv score = {best_score} \n')


#----  predicting on the training and testing set
y_train_pred = optimized_rfc.predict(X_train)
accuracy_train = round(accuracy_score(y_train, y_train_pred),3)
roc_auc_train = round(roc_auc_score(y_train, y_train_pred),3)
recall_train = round(recall_score(y_train, y_train_pred),3)
precision_train = round(precision_score(y_train, y_train_pred),3)

y_pred = optimized_rfc.predict(X_test)
accuracy_test = round(accuracy_score(y_test, y_pred),3)
roc_auc_test = round(roc_auc_score(y_test, y_pred),3)
recall_test = round(recall_score(y_test, y_pred),3)
precision_test = round(precision_score(y_test, y_pred),3)

print('> evaluation metrics \n')
print('%-10s %20s %10s' % ('metric','training','testing'))
print('%-10s %20s %10s' % ('roc auc', roc_auc_train, roc_auc_test))
print('%-10s %20s %10s' % ('accuracy', accuracy_train, accuracy_test))
print('%-10s %20s %10s' % ('recall', recall_train, recall_test))
print('%-10s %20s %10s' % ('precision', precision_train, precision_test))
print('\n')


#----  getting feature importance
rfc_feature_imp_df = pd.DataFrame(optimized_rfc.best_estimator_.feature_importances_, index=X_train.columns, columns=['importance'])
rfc_feature_imp_df.sort_values(by='importance', ascending=False, inplace=True)
print('> feature importance')
print(rfc_feature_imp_df)


#----  saving model results
cv_params_df.to_csv('output\\cv_params.csv')

best_params_str = ', '.join('{}={}'.format(key, val) for key, val in best_params.items())

with open('output//rfc_results.txt', 'w') as file:
    file.write('best parameters = '+best_params_str+'\n')
    file.write('roc_auc:  '+'(train='+str(roc_auc_train)+')  (test='+str(roc_auc_test)+')'+'\n')
    file.write('accuracy:  '+'(train='+str(accuracy_train)+')  (test='+str(accuracy_test)+')'+'\n')
    file.write('recall:  '+'(train='+str(recall_train)+')  (test='+str(recall_test)+')'+'\n')
    file.write('precision:  '+'(train='+str(precision_train)+')  (test='+str(precision_test)+')'+'\n')