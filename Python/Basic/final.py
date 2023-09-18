import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn import metrics
from pycaret.classification import setup, create_model, tune_model
from pycaret.classification import compare_models, interpret_model
import matplotlib.pyplot as plt

data = pd.read_csv('C:\AI\practice\diabetes.csv')
data.columns
data['Diabetes_binary'].value_counts()
X = data.drop(['Diabetes_binary'], axis=1)
y = data[['Diabetes_binary']]

random_state = 42
test_size = 0.3
n_iter = 500
cv = 5
n_jobs = -1
scoring = 'roc_auc'
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                test_size=test_size,
                                                random_state=random_state)

tree = tree.DecisionTreeClassifier(random_state=random_state)
tree_param_dist = {'criterion': ['entrophy', 'gini'],
                   'max_depth': range(1, 16),
                   'splitter': ['best', 'random'],
                   'max_features': range(1, 22),
                   'min_samples_leaf': range(50, 101)}

random_search = RandomizedSearchCV(estimator=tree,
                                   param_distributions=tree_param_dist,
                                   n_iter=n_iter,
                                   random_state=random_state,
                                   scoring=scoring,
                                   n_jobs=n_jobs,
                                   cv=cv)
random_search.fit(Xtrain, ytrain)
best_tree_random_params = random_search.best_params_
print(f'最佳參數:{best_tree_random_params}')
random_diabete_tree = tree(**best_tree_random_params,
                           random_state=random_state)
random_diabete_tree.fit(Xtrain, ytrain)
ytrain_tree_pred = random_diabete_tree.predict(Xtrain)
ytest_tree_pred = random_diabete_tree.predict(Xtest)
ytrain_tree_proba = random_diabete_tree.predict_proba(Xtrain)[:, 1]
ytest_tree_proba = random_diabete_tree.predict_proba(Xtest)[:, 1]
print(f'決策樹訓練Accuracy Score:{random_diabete_tree.score(Xtrain, ytrain)}')
print(f'決策樹測試Accuracy Score:{random_diabete_tree.score(Xtest, ytest)}')
print(f'決策樹訓練F1 score:{metrics.f1_score(ytrain, ytrain_tree_pred)}')
print(f'決策樹測試F1 score:{metrics.f1_score(ytest, ytest_tree_pred)}')
print(f'決策樹訓練AUC score:{metrics.roc_auc_score(ytrain, ytrain_tree_proba)}')
print(f'決策樹測試AUC score:{metrics.roc_auc_score(ytest, ytest_tree_proba)}')

xgb = XGBClassifier(objective='binary:logistic', random_state=random_state)
xgb_param_dist = {
    'n_estimators': range(100, 501),
    'learning_rate': np.arange(0.01, 0.3, 0.01),
    'max_depth': range(1, 16),
    'colsample_bytree': np.arange(0.1, 1.1, 0.1)}

random_search = RandomizedSearchCV(estimator=xgb,
                                   param_distributions=xgb_param_dist,
                                   n_iter=n_iter,
                                   random_state=random_state,
                                   scoring=scoring,
                                   n_jobs=n_jobs,
                                   cv=cv)
random_search.fit(Xtrain, ytrain)
best_xgb_random_params = random_search.best_params_
print(f'最佳參數:{best_xgb_random_params}')
random_diabete_xgb = xgb(**best_xgb_random_params,
                         random_state=random_state)
random_diabete_xgb.fit(Xtrain, ytrain)
ytrain_xgb_pred = random_diabete_xgb.predict(Xtrain)
ytest_xgb_pred = random_diabete_xgb.predict(Xtest)
ytrain_xgb_proba = random_diabete_xgb.predict_proba(Xtrain)[:, 1]
ytest_xgb_proba = random_diabete_xgb.predict_proba(Xtest)[:, 1]
print(f'xgb訓練資料Accuracy Score:{random_diabete_xgb.score(Xtrain, ytrain)}')
print(f'xgb測試資料Accuracy Score:{random_diabete_xgb.score(Xtest, ytest)}')
print(f'xgb訓練資料F1 score:{metrics.f1_score(ytrain, ytrain_xgb_pred)}')
print(f'xgb測試資料F1 score:{metrics.f1_score(ytest, ytest_xgb_pred)}')
print(f'xgb訓練資料AUC score:{metrics.roc_auc_score(ytrain, ytrain_xgb_proba)}')
print(f'xgb測試資料AUC score:{metrics.roc_auc_score(ytest, ytest_xgb_proba)}')

fold = 5
optimize = 'AUC'
sort = 'AUC'
train, test = train_test_split(data,
                               test_size=test_size,
                               random_state=random_state)

setup(data=train, target='Diabetes_binary')
dt = create_model('dt', random_state=random_state)
xgboost = create_model('xgboost', random_state=random_state)

dt_tune, tuner = tune_model(estimator=dt,
                            fold=fold,
                            n_iter=n_iter,
                            return_tuner=True,
                            optimize=optimize)

xgboost_tune, tuner = tune_model(estimator=xgboost,
                                 fold=fold,
                                 n_iter=n_iter,
                                 return_tuner=True,
                                 optimize=optimize)

best_model = compare_models([dt_tune, xgboost_tune], sort=sort)
py_final = best_model.fit(Xtrain, ytrain)
py_ytrain_pred = py_final.predict(Xtrain)
py_ytest_pred = py_final.predict(Xtest)
py_ytrain_proba = py_final.predict_proba(Xtrain)[:, 1]
py_ytest_proba = py_final.predict_proba(Xtest)[:, 1]
print(f'py訓練資料Accuracy Score:{py_final.score(Xtrain, ytrain)}')
print(f'py測試資料Accuracy Score:{py_final.score(Xtest, ytest)}')
print(f'py訓練資料F1 score:{metrics.f1_score(ytrain, py_ytrain_pred)}')
print(f'py測試資料F1 score:{metrics.f1_score(ytest, py_ytest_pred)}')
print(f'py訓練資料AUC score:{metrics.roc_auc_score(ytrain, py_ytrain_proba)}')
print(f'py測試資料AUC score:{metrics.roc_auc_score(ytest, py_ytest_proba)}')

fig = plt.figure()
interpret_model(random_diabete_xgb)
fig
fig.show()