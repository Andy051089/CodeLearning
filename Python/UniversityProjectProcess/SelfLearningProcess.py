import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn import metrics
from pycaret.classification import setup, create_model, tune_model, compare_models, interpret_model
import matplotlib.pyplot as plt

data = pd.read_csv('C:\AI\practice\diabetes.csv')
data.columns
data['Diabetes_binary'].value_counts()

X = data.drop(['Diabetes_binary'], axis=1)
y = data[['Diabetes_binary']]

random_state = 42
test_size = 0.3
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
                                                test_size = test_size, 
                                                random_state = random_state)

diabete_tree = tree.DecisionTreeClassifier(random_state = random_state)
diabete_tree.fit(Xtrain, ytrain)
ytrain_tree_pred = diabete_tree.predict(Xtrain)
ytest_tree_pred = diabete_tree.predict(Xtest)
ytrain_tree_pred_proba = diabete_tree.predict_proba(Xtrain)[:,1]
ytest_tree_pred_proba = diabete_tree.predict_proba(Xtest)[:,1]
print(f' 決策樹訓練資料Accuracy Score: {diabete_tree.score(Xtrain, ytrain)}')
print(f' 決策樹測試資料Accuracy Score: {diabete_tree.score(Xtest, ytest)}')
print(f' 決策樹訓練資料F1 score: {metrics.f1_score(ytrain, ytrain_tree_pred)}')
print(f' 決策樹測試資料F1 score: {metrics.f1_score(ytest, ytest_tree_pred)}')
print(f' 決策樹訓練資料AUC score: {metrics.roc_auc_score(ytrain, ytrain_tree_pred_proba)}')
print(f' 決策樹測試資料AUC score: {metrics.roc_auc_score(ytest, ytest_tree_pred_proba)}')


tree_param = {'criterion' : ['entrophy', 'gini'],
         'max_depth' : [1, 3, 5, 7, 9, 11, 13, 15],
         'splitter' : ['best', 'random'],
         'max_features' : ['auto', 'sqrt', 'log2']} 

grid = GridSearchCV(estimator = tree.DecisionTreeClassifier(random_state = random_state),
                    param_grid = tree_param,
                    cv = 5)
grid.fit(Xtrain, ytrain)
best_tree_grid_params = grid.best_params_
print(f' 最佳參數:{best_tree_grid_params}')
grid_diabete_tree = tree.DecisionTreeClassifier(**best_tree_grid_params, random_state = random_state)
grid_diabete_tree.fit(Xtrain, ytrain)
ytrain_grid_tree_pred = grid_diabete_tree.predict(Xtrain)
ytest_grid_tree_pred = grid_diabete_tree.predict(Xtest)
ytrain_grid_tree_pred_proba = grid_diabete_tree.predict_proba(Xtrain)[:,1]
ytest_grid_tree_pred_proba = grid_diabete_tree.predict_proba(Xtest)[:,1]
print(f' 決策樹GridSearchCV最佳超參數下訓練資料Accuracy Score: {grid_diabete_tree.score(Xtrain, ytrain)}')
print(f' 決策樹GridSearchCV最佳超參數下測試資料Accuracy Score: {grid_diabete_tree.score(Xtest, ytest)}')
print(f' 決策樹GridSearchCV最佳超參數下訓練資料F1 score: {metrics.f1_score(ytrain, ytrain_grid_tree_pred)}')
print(f' 決策樹GridSearchCV最佳超參數下測試資料F1 score: {metrics.f1_score(ytest, ytest_grid_tree_pred)}')
print(f' 決策樹GridSearchCV最佳超參數下訓練資料AUC score: {metrics.roc_auc_score(ytrain, ytrain_grid_tree_pred_proba)}')
print(f' 決策樹GridSearchCV最佳超參數下測試資料AUC score: {metrics.roc_auc_score(ytest, ytest_grid_tree_pred_proba)}')


tree_param_dist = {'criterion' : ['entrophy', 'gini'],
         'max_depth' : range(1, 16),
         'splitter' : ['best', 'random'],
         'max_features' : ['auto', 'sqrt', 'log2']}
    
random_search = RandomizedSearchCV(estimator = diabete_tree, 
                                   param_distributions = tree_param_dist,
                                   n_iter = 96,
                                   random_state = random_state,
                                   n_jobs = -1, 
                                   cv = 5)
random_search.fit(Xtrain, ytrain)
best_tree_random_params = random_search.best_params_
print(f' 最佳參數:{best_tree_random_params}')
random_diabete_tree = tree.DecisionTreeClassifier(**best_tree_random_params, random_state = random_state)
random_diabete_tree.fit(Xtrain, ytrain)
ytrain_random_tree_pred = random_diabete_tree.predict(Xtrain)
ytest_random_tree_pred = random_diabete_tree.predict(Xtest)
ytrain_random_tree_pred_proba = random_diabete_tree.predict_proba(Xtrain)[:,1]
ytest_random_tree_pred_proba = random_diabete_tree.predict_proba(Xtest)[:,1]
print(f' 決策樹RandomizedSearchCV最佳超參數下訓練資料Accuracy Score: {random_diabete_tree.score(Xtrain, ytrain)}')
print(f' 決策樹RandomizedSearchCV最佳超參數下測試資料Accuracy Score: {random_diabete_tree.score(Xtest, ytest)}')
print(f' 決策樹RandomizedSearchCV最佳超參數下訓練資料F1 score: {metrics.f1_score(ytrain, ytrain_random_tree_pred)}')
print(f' 決策樹RandomizedSearchCV最佳超參數下測試資料F1 score: {metrics.f1_score(ytest, ytest_random_tree_pred)}')
print(f' 決策樹RandomizedSearchCV最佳超參數下訓練資料AUC score: {metrics.roc_auc_score(ytrain, ytrain_random_tree_pred_proba)}')
print(f' 決策樹RandomizedSearchCV最佳超參數下測試資料AUC score: {metrics.roc_auc_score(ytest, ytest_random_tree_pred_proba)}')


diabete_xgb = XGBClassifier(objective='binary:logistic', random_state = random_state)
diabete_xgb.fit(Xtrain, ytrain)
ytrain_xgb_pred = diabete_xgb.predict(Xtrain)
ytest_xgb_pred = diabete_xgb.predict(Xtest)
ytrain_xgb_pred_proba = diabete_xgb.predict_proba(Xtrain)[:,1]
ytest_xgb_pred_proba = diabete_xgb.predict_proba(Xtest)[:,1]
print(f' xgb訓練資料Accuracy Score: {diabete_xgb.score(Xtrain, ytrain)}')
print(f' xgb測試資料Accuracy Score: {diabete_xgb.score(Xtest, ytest)}')
print(f' xgb訓練資料F1 score: {metrics.f1_score(ytrain, ytrain_xgb_pred)}')
print(f' xgb測試資料F1 score: {metrics.f1_score(ytest, ytest_xgb_pred)}')
print(f' xgb訓練資料AUC score: {metrics.roc_auc_score(ytrain, ytrain_xgb_pred_proba)}')
print(f' xgb測試資料AUC score: {metrics.roc_auc_score(ytest, ytest_xgb_pred_proba)}')


xgb_param = {'n_estimators' : [200, 350, 500],
         'learning_rate' : [0.05, 0.1, 0.15, 0.2],
         'max_depth' : [1, 5, 10],
         'colsample_bytree' :  [0.5, 0.7, 1]}

grid = GridSearchCV(estimator = XGBClassifier(objective='binary:logistic', random_state = random_state),
                    param_grid = xgb_param,
                    cv = 5,
                    n_jobs = -1)
grid.fit(Xtrain, ytrain)
best_xgb_grid_params = grid.best_params_
print(f' 最佳參數:{best_xgb_grid_params}')
grid_diabete_xgb = XGBClassifier(objective='binary:logistic', **best_xgb_grid_params, random_state = random_state)
grid_diabete_xgb.fit(Xtrain, ytrain)
ytrain_grid_xgb_pred = grid_diabete_xgb.predict(Xtrain)
ytest_grid_xgb_pred = grid_diabete_xgb.predict(Xtest)
ytrain_grid_xgb_pred_proba = grid_diabete_xgb.predict_proba(Xtrain)[:,1]
ytest_grid_xgb_pred_proba = grid_diabete_xgb.predict_proba(Xtest)[:,1]
print(f' xgb GridSearchCV最佳超參數下訓練資料Accuracy Score: {grid_diabete_xgb.score(Xtrain, ytrain)}')
print(f' xgb GridSearchCV最佳超參數下測試資料Accuracy Score: {grid_diabete_xgb.score(Xtest, ytest)}')
print(f' xgb GridSearchCV最佳超參數下訓練資料F1 score: {metrics.f1_score(ytrain, ytrain_grid_xgb_pred)}')
print(f' xgb GridSearchCV最佳超參數下測試資料F1 score: {metrics.f1_score(ytest, ytest_grid_xgb_pred)}')
print(f' xgb GridSearchCV最佳超參數下訓練資料AUC score: {metrics.roc_auc_score(ytrain, ytrain_grid_xgb_pred_proba)}')
print(f' xgb GridSearchCV最佳超參數下測試資料AUC score: {metrics.roc_auc_score(ytest, ytest_grid_xgb_pred_proba)}')


xgb_param_dist = {
    'n_estimators' : range(200, 501),
    'learning_rate' : np.arange(0.01, 0.3, 0.01),
    'max_depth' : range(1, 11),
    'colsample_bytree' :  np.arange(0.5, 1.1, 0.1)}
    
random_search = RandomizedSearchCV(estimator = XGBClassifier(objective='binary:logistic', random_state = random_state), 
                                   param_distributions = xgb_param_dist,
                                   n_iter = 108,
                                   random_state = random_state,
                                   n_jobs = -1, 
                                   cv = 5)
random_search.fit(Xtrain, ytrain)
best_xgb_random_params = random_search.best_params_
print(f' 最佳參數:{best_xgb_random_params}')
random_diabete_xgb = XGBClassifier(objective='binary:logistic', **best_xgb_random_params, random_state = random_state)
random_diabete_xgb.fit(Xtrain, ytrain)
ytrain_random_xgb_pred = random_diabete_xgb.predict(Xtrain)
ytest_random_xgb_pred = random_diabete_xgb.predict(Xtest)
ytrain_random_xgb_pred_proba = random_diabete_xgb.predict_proba(Xtrain)[:,1]
ytest_random_xgb_pred_proba = random_diabete_xgb.predict_proba(Xtest)[:,1]
print(f' xgb RandomizedSearchCV最佳超參數下訓練資料Accuracy Score: {random_diabete_xgb.score(Xtrain, ytrain)}')
print(f' xgb RandomizedSearchCV最佳超參數下測試資料Accuracy Score: {random_diabete_xgb.score(Xtest, ytest)}')
print(f' xgb RandomizedSearchCV最佳超參數下訓練資料F1 score: {metrics.f1_score(ytrain, ytrain_random_xgb_pred)}')
print(f' xgb RandomizedSearchCV最佳超參數下測試資料F1 score: {metrics.f1_score(ytest, ytest_random_xgb_pred)}')
print(f' xgb RandomizedSearchCV最佳超參數下訓練資料AUC score: {metrics.roc_auc_score(ytrain, ytrain_random_xgb_pred_proba)}')
print(f' xgb RandomizedSearchCV最佳超參數下測試資料AUC score: {metrics.roc_auc_score(ytest, ytest_random_xgb_pred_proba)}')


train, test = train_test_split(data, test_size = test_size, random_state = random_state)
setup(data = train, target = 'Diabetes_binary')
dt = create_model('dt', random_state = random_state)
xgboost = create_model('xgboost', random_state = random_state)
dt_tune, tuner = tune_model(estimator = dt,
                               fold = 5,
                               n_iter = 100,
                               return_tuner = True,
                               optimize = 'AUC')
xgboost_tune, tuner = tune_model(estimator = xgboost,
                               fold=5,
                               n_iter=100,
                               return_tuner=True,
                               optimize = 'AUC')
best_model = compare_models([dt_tune, xgboost_tune], sort = "AUC")
pycaret_best_xgb_params = tuner.best_params_
pycaret_best_diabete_xgb = XGBClassifier(objective='binary:logistic', **pycaret_best_xgb_params, random_state = random_state)
pycaret_final = pycaret_best_diabete_xgb.fit(Xtrain, ytrain)
pycaret_best_ytrain_pred = pycaret_best_diabete_xgb.predict(Xtrain)
pycaret_best_ytest_pred = pycaret_best_diabete_xgb.predict(Xtest)
pycaret_ytrain_pred_proba = pycaret_best_diabete_xgb.predict_proba(Xtrain)[:,1]
pycaret_ytest_pred_proba = pycaret_best_diabete_xgb.predict_proba(Xtest)[:,1]
print(f' pycaret下xgb訓練資料Accuracy Score: {pycaret_best_diabete_xgb.score(Xtrain, ytrain)}')
print(f' pycaret下xgb測試資料Accuracy Score: {pycaret_best_diabete_xgb.score(Xtest, ytest)}')
print(f' pycaret下xgb訓練資料F1 score: {metrics.f1_score(ytrain, pycaret_best_ytrain_pred)}')
print(f' pycaret下xgb測試資料F1 score: {metrics.f1_score(ytest, pycaret_best_ytest_pred)}')
print(f' pycaret下xgb訓練資料AUC score: {metrics.roc_auc_score(ytrain, pycaret_ytrain_pred_proba)}')
print(f' pycaret下xgb測試資料AUC score: {metrics.roc_auc_score(ytest, pycaret_ytest_pred_proba)}')

pycaret_final_whole = pycaret_best_diabete_xgb.fit(X, y)
fig = plt.figure()
interpret_model(pycaret_final_whole)
fig
fig.show()