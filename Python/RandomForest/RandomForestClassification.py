import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV


data = pd.read_csv('C:/AI/practice/final/diabetes.csv')
data.columns
data['Diabetes_binary'].value_counts()

X = data.drop(['Diabetes_binary'], axis=1)
y = data[['Diabetes_binary']]

random_state = 42
test_size = 0.3
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
                                                test_size = test_size, 
                                                random_state = random_state)


diabete_forest = RandomForestClassifier(random_state = random_state)
diabete_forest.fit(Xtrain, ytrain)
ytrain_forest_pred = diabete_forest.predict(Xtrain)
ytest_forest_pred = diabete_forest.predict(Xtest)
ytrain_forest_pred_proba = diabete_forest.predict_proba(Xtrain)[:,1]
ytest_forest_pred_proba = diabete_forest.predict_proba(Xtest)[:,1]
print(f' 隨機森林訓練資料Accuracy Score: {diabete_forest.score(Xtrain, ytrain)}')
print(f' 隨機森林測試資料Accuracy Score: {diabete_forest.score(Xtest, ytest)}')
print(f' 隨機森林訓練資料F1 score: {metrics.f1_score(ytrain, ytrain_forest_pred)}')
print(f' 隨機森林測試資料F1 score: {metrics.f1_score(ytest, ytest_forest_pred)}')
print(f' 隨機森林訓練資料AUC score: {metrics.roc_auc_score(ytrain, ytrain_forest_pred_proba)}')
print(f' 隨機森林測試資料AUC score: {metrics.roc_auc_score(ytest, ytest_forest_pred_proba)}')


forest_param_dist = {'criterion' : ['entrophy', 'gini'],
         'max_depth' : range(1, 16),
         'n_estimators' : range(100, 501),
         'max_features' : range(1, 22),
         'min_samples_leaf' : range(50, 101) }
    
random_search = RandomizedSearchCV(estimator = diabete_forest, 
                                   param_distributions = forest_param_dist,
                                   n_iter = 500,
                                   random_state = random_state,
                                   scoring = 'roc_auc',
                                   n_jobs = -1,
                                   cv = 5)

random_search.fit(Xtrain, ytrain)
best_forest_random_params = random_search.best_params_
print(f' 最佳參數:{best_forest_random_params}')
random_diabete_forest = RandomForestClassifier(**best_forest_random_params, random_state = random_state)
random_diabete_forest.fit(Xtrain, ytrain)
ytrain_random_forest_pred = random_diabete_forest.predict(Xtrain)
ytest_random_forest_pred = random_diabete_forest.predict(Xtest)
ytrain_random_forest_pred_proba = random_diabete_forest.predict_proba(Xtrain)[:,1]
ytest_random_forest_pred_proba = random_diabete_forest.predict_proba(Xtest)[:,1]
print(f' forest RandomizedSearchCV最佳超參數下訓練資料Accuracy Score: {random_diabete_forest.score(Xtrain, ytrain)}')
print(f' forest RandomizedSearchCV最佳超參數下測試資料Accuracy Score: {random_diabete_forest.score(Xtest, ytest)}')
print(f' forest RandomizedSearchCV最佳超參數下訓練資料F1 score: {metrics.f1_score(ytrain, ytrain_random_forest_pred)}')
print(f' forest RandomizedSearchCV最佳超參數下測試資料F1 score: {metrics.f1_score(ytest, ytest_random_forest_pred)}')
print(f' forest RandomizedSearchCV最佳超參數下訓練資料AUC score: {metrics.roc_auc_score(ytrain, ytrain_random_forest_pred_proba)}')
print(f' forest RandomizedSearchCV最佳超參數下測試資料AUC score: {metrics.roc_auc_score(ytest, ytest_random_forest_pred_proba)}')