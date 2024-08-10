#引用模組
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
#%% 
#讀資料
data_file = 'C:/Users/88691/Desktop/自學/AI/practice/Heart Disease.csv'
df = pd.read_csv(data_file)
original_df = df
new_df = df
#%%
#資料預處理
#查看資料型態類別有無缺失值
df.info()
#查看資料平均、標準差四分位
df.describe()
#列出資料中的所有columns
df.columns
#資料是否為平衡資料，No : 283883、Yes : 24971
df['Heart_Disease'].value_counts()

#轉換資料中文至類別數字(使用LabelEncoder)
#初始化LabelEncoder
label_encoder = LabelEncoder()

#Excellent : 0, Fair : 1, Good : 2, Poor : 3, Very Good : 4
new_df['General_Health'] = label_encoder.fit_transform(df['General_Health'])

#5 or more years ago : 0, Never : 1
#Within the past 2 years : 2, Within the past 5 years : 3
#Within the past year : 4
new_df['Checkup'] = label_encoder.fit_transform(df['Checkup'])
#找對應資料
new_df.loc[new_df['Checkup'] == 4]

# No : 0, Yes : 1
new_df['Exercise'] = label_encoder.fit_transform(df['Exercise'])
new_df['Heart_Disease'] = label_encoder.fit_transform(df['Heart_Disease'])
new_df['Skin_Cancer'] = label_encoder.fit_transform(df['Skin_Cancer'])
new_df['Other_Cancer'] = label_encoder.fit_transform(df['Other_Cancer'])
new_df['Depression'] = label_encoder.fit_transform(df['Depression'])
new_df['Diabetes'] = label_encoder.fit_transform(df['Diabetes'])
new_df['Arthritis'] = label_encoder.fit_transform(df['Arthritis'])
new_df['Smoking_History'] = label_encoder.fit_transform(df['Smoking_History'])

# Female : 0, Male : 1
new_df['Sex'] = label_encoder.fit_transform(df['Sex'])

#在Age_Category中如用原方式會分為12類，我想自行設定區間類別
Age_Category_mapping = {'18-24': 0, '25-29': 0, '30-34': 1, '35-39': 1,
                        '40-44': 2, '45-49': 2, '50-54': 3, '55-59': 3,
                        '60-64': 4, '65-69': 4, '70-74': 5, '75-79': 5,
                        '80+': 6}
new_df['Age_Category'] = df['Age_Category'].map(Age_Category_mapping)

#每10歲分一類及80歲以上一類
#原始資料為每月服用總量
#每月喝酒超過7天算有喝酒 沒有 : 0, 有 : 1
new_df['Alcohol_Consumption'] = np.where(df['Alcohol_Consumption'] > 7, 1, 0)
#每天至少2份算有吃水果 沒有 : 0, 有 : 1
new_df['Fruit_Consumption'] = np.where(df['Fruit_Consumption'] > 60, 1, 0)
#每天至少2份算有吃蔬菜 沒有 : 0, 有 : 1
new_df['Green_Vegetables_Consumption'] = np.where(df['Green_Vegetables_Consumption'] > 60, 1, 0)
#每天至少1份算有吃薯條 沒有 : 0, 有 : 1
new_df['FriedPotato_Consumption'] = np.where(df['FriedPotato_Consumption'] > 30, 1, 0)

#修改COLUMNS的名字
new_df.rename(columns={'Alcohol_Consumption': 'Alcohol',
                       'Fruit_Consumption' : 'Fruit',
                       'Green_Vegetables_Consumption' : 'Green_Vegetables',
                       'FriedPotato_Consumption' : 'FriedPotato'},
              inplace=True)
#分割出XY資料
x = new_df.drop(['Heart_Disease'], axis = 1)
y = new_df['Heart_Disease']

#設定常見使用參數
random_state = 42
test_size = 0.3
cv = 10
n_iter = 150
scoring = 'roc_auc'

#分割出續練測試資料資料
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size = test_size, 
                                                random_state = random_state)

#因資料為不平衡資料，需改變權重
weights = compute_class_weight(class_weight = "balanced", 
                               classes = [0, 1], y = ytrain)
class_weight = {0 : weights[0], 
                1 : weights[1]}
#%%
#決策樹
#創建模型
decision_tree = tree.DecisionTreeClassifier(random_state = random_state,
                                            class_weight = {0 : weights[0], 
                                                            1 : weights[1]})
#建立參數範圍
'''
criterion : 通過不同計算方式決定樹分支條件
max_depth : 決定樹最大的層樹
max_features : 決定每次分裂時考慮多少特徵
ccp_alpha : 通過計算如何去修剪樹
min_samples_leaf(每個葉最少需要多少樣本) : 因通過ccp_alpha修剪樹故不設定
min_samples_split(葉必須要有多少樣本才會分) : 因通過ccp_alpha修剪樹故不設定
splitter : 直接使用best，通過criterion所得到的方式做最好的選擇
'''
bayes_tree_param = {'criterion': Categorical(['entropy', 'gini']),
                    'max_depth': Integer(1, 15),  
                    'max_features': Integer(1, 18),
                    #'min_samples_leaf': Integer(1, 19),  
                    #'min_samples_split': Integer(2, 19), 
                    'ccp_alpha': Real(0.0, 0.1, prior = 'uniform')}
#建立BayesSearchCV
bayes_tree_search =  BayesSearchCV(estimator = decision_tree, 
                                   search_spaces = bayes_tree_param,
                                   n_iter = n_iter,
                                   random_state = random_state,
                                   n_jobs = -1, 
                                   cv = cv,
                                   pre_dispatch = 'all',
                                   verbose = True,
                                   scoring = scoring)
#把資料FIT進去找最佳超參數
bayes_tree_search.fit(xtrain, ytrain)
#最佳參數
best_bayes_tree_params = bayes_tree_search.best_params_

#把最佳超參數FIT進決策樹
#bayes_tree = tree.DecisionTreeClassifier(**best_bayes_tree_params, 
#                                         random_state = random_state,
#                                         class_weight = class_weight)
bayes_tree = tree.DecisionTreeClassifier(ccp_alpha = 0.0,
                                         criterion = 'gini',
                                         max_depth = 8,
                                         max_features = 13,
                                         random_state = random_state,
                                         class_weight = class_weight)

#把資料FIT進最佳超參數的決策樹
bayes_tree.fit(xtrain, ytrain)
#把訓練測試資料FIT模型做預測看結果
ytrain_bayes_tree_pred = bayes_tree.predict(xtrain)
ytest_bayes_tree_pred = bayes_tree.predict(xtest)
#把訓練測試資料FIT模型做預測看結果的機率
ytrain_bayes_tree_pred_proba = bayes_tree.predict_proba(xtrain)[:,1]
ytest_bayes_tree_pred_proba = bayes_tree.predict_proba(xtest)[:,1]
print(f' train_tree Accuracy Score: {bayes_tree.score(xtrain, ytrain)}')
print(f' test_tree Accuracy Score: {bayes_tree.score(xtest, ytest)}')
print(f' train_tree F1 score: {metrics.f1_score(ytrain, ytrain_bayes_tree_pred)}')
print(f' test_tree F1 score: {metrics.f1_score(ytest, ytest_bayes_tree_pred)}')
print(f' train_tree AUC score: {metrics.roc_auc_score(ytrain, ytrain_bayes_tree_pred_proba)}')
print(f' test_tree AUC score: {metrics.roc_auc_score(ytest, ytest_bayes_tree_pred_proba)}')
#%%
#XGBoost
#不平衡資料計算
ratio_of_negative_to_positive = ytrain.loc[ytrain == 0].count() / ytrain.loc[ytrain == 1].count()

#創建模型
xgb_tree = XGBClassifier(objective = 'binary:logistic', 
                         random_state = random_state,
                         scale_pos_weight = ratio_of_negative_to_positive,
                         tree_method="gpu_hist",
                         gpu_id=0)
'''
objective : 決定XGBoost執行甚麼任務，最終結果數值轉換
learning_rate : 此超參數於樹的生成及最終做預測計算皆有影響
n_estimators : 總共生幾棵樹
colsample_bytree : 每次樹分支時考慮多少比例的特徵
scale_pos_weight : 不平衡資料需要調整
'''
#參數範圍
bayes_xgb_param = {'learning_rate' : Real(0.0, 0.3, prior = 'uniform'),
                   'max_depth' : Integer(1, 15),
                   'n_estimators' : Integer(100, 500)}
#建立BayesSearchCV
bayes_xgb_search = BayesSearchCV(estimator = xgb_tree, 
                                 search_spaces = bayes_xgb_param,
                                 n_iter = n_iter,
                                 random_state = random_state,
                                 n_jobs = -1, 
                                 cv = cv,
                                 pre_dispatch = 'all',
                                 verbose = True,
                                 scoring = scoring)
#把資料FIT進去找最佳超參數
bayes_xgb_search.fit(xtrain, ytrain)


#最佳參數
best_bayes_xgb_params = bayes_xgb_search.best_params_

#把最佳超參數FIT進決策樹
bayes_xgb = XGBClassifier(objective='binary:logistic', 
                          **best_bayes_xgb_params, 
                          random_state = random_state,
                          scale_pos_weight = ratio_of_negative_to_positive)
bayes_xgb = XGBClassifier(objective='binary:logistic', 
                          random_state = random_state,
                          scale_pos_weight = ratio_of_negative_to_positive)

#把資料FIT進最佳超參數的決策樹
bayes_xgb.fit(xtrain, ytrain)
#把訓練測試資料FIT模型做預測看結果
ytrain_bayes_xgb_pred = bayes_xgb.predict(xtrain)
ytest_bayes_xgb_pred = bayes_xgb.predict(xtest)
#把訓練測試資料FIT模型做預測看結果的機率
ytrain_bayes_xgb_pred_proba = bayes_xgb.predict_proba(xtrain)[:,1]
ytest_bayes_xgb_pred_proba = bayes_xgb.predict_proba(xtest)[:,1]
print(f' train_xgb Accuracy Score: {bayes_xgb.score(xtrain, ytrain)}')
print(f' test_xgb Accuracy Score: {bayes_xgb.score(xtest, ytest)}')
print(f' train_xgb F1 score: {metrics.f1_score(ytrain, ytrain_bayes_xgb_pred)}')
print(f' test_xgb F1 score: {metrics.f1_score(ytest, ytest_bayes_xgb_pred)}')
print(f' train_xgb AUC score: {metrics.roc_auc_score(ytrain, ytrain_bayes_xgb_pred_proba)}')
print(f' test_xgb AUC score: {metrics.roc_auc_score(ytest, ytest_bayes_xgb_pred_proba)}')
#%%