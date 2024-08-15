#%% 引用模組
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, TargetEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, AdamW
import pickle
from kerastuner import tuners, Objective
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#%%讀資料
data_file = 'C:/Users/88691/Desktop/自學/AI/practice/Heart Disease.csv'
df = pd.read_csv(data_file)
original_df = df.copy()
new_df = df.copy()
#%%資料檢視
# 查看前5筆
df.head()
#查看資料型態類別有無缺失值
df.info()
#查看資料平均、標準差四分位
df.describe()
#列出資料中的所有columns
df.columns
#%%設定使用參數
random_state = 42
test_size = 0.3
cv = 5
n_iter = 100
scoring = 'f1'
threshold = 0.5
#%%分割出特徵及目標變數資料
x = new_df.drop(['Heart_Disease'], axis = 1)
y = new_df['Heart_Disease']
#%%分割出續練測試資料資料
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, 
    test_size = test_size, 
    random_state = random_state)
#%%資料預處理
'''
模型無法直接用文字做訓練，類別資料雖對樹模型不影響結果，但如LINER REGRESSION、
    NN等需要把資料標準化
轉換資料中文字、無序資料至數值數字
TargetEncoder : EX:A,B,C三類分別有或無，計算有 : A,B,C，無 : A,B,C的比例
在做資料數值轉換時，要先切割好訓練、測試資料
'''
# 創建targetencoder
encoder = TargetEncoder()
# 要轉換的列
columns_to_encode = [
    'General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 
    'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 
    'Sex', 'Age_Category', 'Smoking_History',
    'Alcohol_Consumption', 'Fruit_Consumption',
    'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
for col in columns_to_encode:
    xtrain[col] = encoder.fit_transform(xtrain[[col]], ytrain)
    xtest[col] = encoder.transform(xtest[[col]])
# 把數值資料轉換至0-1之間(最小最大值標準化)
minmax = MinMaxScaler()
x_train_test = [xtrain, xtest]
for i in x_train_test:
    i['Height_(cm)'] = minmax.fit_transform(i[['Height_(cm)']])
    i['Weight_(kg)'] = minmax.fit_transform(i[['Weight_(kg)']])
    i['BMI'] = minmax.fit_transform(i[['BMI']])
    i.rename(columns={
        'Height_(cm)': 'Height',
        'Weight_(kg)' : 'Weight'},
        inplace=True)
# 把目標變數轉為0,1
label_encoder = LabelEncoder()
ytrain = label_encoder.fit_transform(ytrain)
ytest = label_encoder.fit_transform(ytest)
# 查看資料是否為平衡資料
'''
不平衡資料經調整權重之後，可以更好的訓練模型，泛化外來資料。
在最後結果指標的評估，假如全部預測為多數樣本0，那Accuracy很高，但沒有甚麼用。
F1 Score對少數樣本的預測效果敏感，當少數樣本預測不好時會顯著下降。
'''
[sum(ytrain == 0), sum(ytrain == 1)]
# 因資料為不平衡資料，需改變權重
weights = compute_class_weight(class_weight = "balanced", 
                               classes = np.array(
                                   [0, 1]), 
                               y = ytrain)
class_weight = {
    0 : weights[0], 
    1 : weights[1]}
#%%決策樹
'''
1.監督式學習。把資料透過不同特徵條件做分類，在做每一次分類分支決定時，把所有特徵
    條件都做一次分支，並計算每個特徵條件分支後criterion(gini, entropy, 
    information gain)，由計算結果去決定哪個條件分類的比較好。但如果把資料全部分
    完，雖然訓練資料有很好的結果，但沒辦法泛化，會有overfitting的問題，可以透過
    調整max_depth、min_samples_leaf、min_samples_split，去決定數的發展，或是分
    到最後之後，通過計算每個葉的alpha(ccp_alpha)，去做修剪樹的動作。
2.
criterion : 通過不同計算方式決定樹分支條件
max_depth : 決定樹最大的層樹
max_features : 決定每次分裂時考慮多少特徵
ccp_alpha : 通過計算如何去修剪樹
min_samples_leaf : 每個葉最少需要多少樣本
min_samples_split : 葉必須要有多少樣本才會分
splitter : 直接使用best，通過criterion所得到的方式做最好的選擇
n_jobs = -1 : 把所有可用的CPU都用
pre_dispatch : 把2倍CPU的工作量，分給所有CPU
'''
# 創建模型
decision_tree = tree.DecisionTreeClassifier(
    random_state = random_state,
    class_weight = class_weight)
# 建立參數範圍
bayes_tree_param = {
    'criterion': Categorical(['entropy', 'gini']),
    'max_depth': Integer(1, 15),  
    'max_features': Integer(1, 18),
    'min_samples_leaf': Integer(1, 19),  
    'min_samples_split': Integer(2, 19), 
    'ccp_alpha': Real(0.0, 0.1, prior = 'uniform')}
# 建立BayesSearchCV
bayes_tree_search =  BayesSearchCV(
    estimator = decision_tree, 
    search_spaces = bayes_tree_param,
    n_iter = n_iter,
    random_state = random_state,
    n_jobs = -1,     
    cv = cv,
    pre_dispatch = '2*n_jobs',   
    verbose = 2,
    scoring = scoring)
# 把資料FIT進去找最佳超參數
bayes_tree_search.fit(xtrain, ytrain)
# 最佳超參數
best_bayes_tree_params = bayes_tree_search.best_params_
# 把最佳超參數存下來
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_tree_params.pkl"
pickle.dump(best_bayes_tree_params, open(file_name, "wb"))
best_bayes_tree_params = pickle.load(open(file_name, "rb"))
# 把最佳超參數FIT進決策樹
bayes_tree = tree.DecisionTreeClassifier(
    **best_bayes_tree_params, 
    random_state = random_state,
    class_weight = class_weight)
# bayes_tree = tree.DecisionTreeClassifier(
#     ccp_alpha = 0.0,
#     criterion = 'entropy',
#     max_depth = 8,
#     max_features = 14,
#     min_samples_leaf = 7,
#     min_samples_split = 5,
#     random_state = random_state,
#     class_weight = class_weight)
# 把資料FIT進最佳超參數的決策樹
best_tree_model = bayes_tree.fit(xtrain, ytrain)
#把訓練測試資料FIT模型做預測看結果
ytrain_best_tree_pred = best_tree_model.predict(xtrain)
ytest_best_tree_pred = best_tree_model.predict(xtest)
#把訓練測試資料FIT模型做預測看結果的機率
ytrain_best_tree_pred_proba = best_tree_model.predict_proba(xtrain)[:,1]
ytest_best_tree_pred_proba = best_tree_model.predict_proba(xtest)[:,1]
print(f' train_tree Accuracy Score: {best_tree_model.score(xtrain, ytrain)}')
print(f' test_tree Accuracy Score: {best_tree_model.score(xtest, ytest)}')
print(f' train_tree F1 score: {metrics.f1_score(ytrain, ytrain_best_tree_pred)}')
print(f' test_tree F1 score: {metrics.f1_score(ytest, ytest_best_tree_pred)}')
print(f' train_tree AUC score: {metrics.roc_auc_score(ytrain, ytrain_best_tree_pred_proba)}')
print(f' test_tree AUC score: {metrics.roc_auc_score(ytest, ytest_best_tree_pred_proba)}')
#%%隨機森林
'''
1.監督式學習。當使用決策樹仍有overfitting、較大的variance時，可以透過bagging的方
    式解決，random forest就是常見的bagging方式。透過生成很多顆各自獨立的決策樹，
    最終結果是計算所有樹分類結果較多者。在每顆樹的生成開始時，都會把訊練資料做
    自抽法至與訓練資料大小相同的資料筆數，隨機抽取N個特徵，並在樹的每個分支使用
    不同特徵條件，並重複n_estimators設定的次數。
2.
criterion : 通過不同計算方式決定樹分支條件
max_depth : 決定樹最大的層樹
max_features : 決定每次分裂時考慮多少特徵
n_estimators : 總共建多少棵樹
min_samples_leaf : 每個葉最少需要多少樣本
min_samples_split : 葉必須要有多少樣本才會分
n_jobs = -1 : 把所有可用的CPU都用
pre_dispatch : 把2倍CPU的工作量，分給所有CPU
'''
# 建模型
forest = RandomForestClassifier(
    random_state = random_state,
    class_weight = class_weight)
# 建超參數範圍
forest_param_dist = {
    'criterion' : Categorical(['entropy', 'gini']),
    'max_depth' : Integer(1, 30),
    'n_estimators' : Integer(100, 1000),
    'max_features' : Integer(1, 18),
    'min_samples_leaf' : Integer(1, 10),
    'min_samples_split' : Integer(1,100)}
# 建立BayesSearchCV
bayes_forest_search =  BayesSearchCV(
    estimator = forest, 
    search_spaces = forest_param_dist,
    n_iter = n_iter,
    random_state = random_state,
    n_jobs = -1,     
    cv = cv,
    pre_dispatch = '2*n_jobs',   
    verbose = 2,
    scoring = scoring)    
#把資料FIT進去找最佳超參數
bayes_forest_search.fit(xtrain, ytrain)
#最佳參數
best_bayes_forest_params = bayes_forest_search.best_params_
# 把最佳超參數存下來
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_forest_params.pkl"
pickle.dump(best_bayes_forest_params, open(file_name, "wb"))
best_bayes_forest_params = pickle.load(open(file_name, "rb"))
# 把最佳超參數FIT進隨機森林
bayes_forest = RandomForestClassifier(
    **best_bayes_forest_params, 
    random_state = random_state,
    class_weight = class_weight)
# bayes_forest = RandomForestClassifier(
    # criterion = 'gini',
    # max_depth = 27,
    # max_features = 6,
    # min_samples_leaf = 10,
    # min_samples_split = 87,
    # random_state = random_state,
    # class_weight = class_weight,
    # n_estimators = 156)
# 把資料FIT進最佳超參數的隨機森林
best_forest_model = bayes_forest.fit(xtrain, ytrain)
# 把訓練測試資料FIT模型做預測看結果
ytrain_best_forest_pred = best_forest_model.predict(xtrain)
ytest_best_forest_pred = best_forest_model.predict(xtest)
# 把訓練測試資料FIT模型做預測看結果的機率
ytrain_best_forest_pred_proba = best_forest_model.predict_proba(xtrain)[:,1]
ytest_best_forest_pred_proba = best_forest_model.predict_proba(xtest)[:,1]
print(f' train_forest Accuracy Score: {best_forest_model.score(xtrain, ytrain)}')
print(f' test_forest Accuracy Score: {best_forest_model.score(xtest, ytest)}')
print(f' train_forest F1 score: {metrics.f1_score(ytrain, ytrain_best_forest_pred)}')
print(f' test_forest F1 score: {metrics.f1_score(ytest, ytest_best_forest_pred)}')
print(f' train_forest AUC score: {metrics.roc_auc_score(ytrain, ytrain_best_forest_pred_proba)}')
print(f' test_forest AUC score: {metrics.roc_auc_score(ytest, ytest_best_forest_pred_proba)}')
#%%XGBoost
'''
1.監督式學習。當有較大的bias時，boosting則會是其中一個方式，與random forest的差
    別為下一顆生產的樹，會更加專注上一顆樹分類不好的資料，並加重權重，使得這一
    顆生產的樹可以改善之前分類不好的資料。
2.在每一顆樹的生成時，只使用subsample設定的比例做為生成樹的資料。也只使用
    colsample_bytree設定比例，作為可以被選為作為分支的特徵條件。在一顆樹的生成
    時，每一次的分支，會計算每個不同特徵條件下，實際與預測結果計算得出的一個值
    (gain值)，來決定使用哪個特徵條件做分支，直至max_depth設定的最大層數，完成一
    顆完整樹的生成。下一顆樹會計算上一顆完整樹的實際與預測差距，加重分類不好資
    料權重，並專注於分類不好的資料。直至n_estimators設定的顆樹為止。最終預測結
    果由初始值與learning rate及每顆樹所分類到的葉上值計算所得。另外在預防
    overfitting上，可以通過計算min_child_weight(每個葉最小的樣本數)、
    gamma(設定葉上最小的GAIN值)來限制葉是否繼續分支。或是透過正規化reg_alpha(L1)
    、reg_lambda(L2)降低一點bias來大大提高variance。
3.EARLY STOPPING(**除非不能等時間，有可能你設定10次沒改善就結束，只是這10次超參
                 數選到爛的，第11次剛好可以讓結果進步):
    為了節省時間，且可能已經找到最佳的n_estimators，使用EarlyStopping。把原本
    訓練資料再次切格成訓練驗證資料，當生成的樹在驗證資料中設定
    early_stopping_rounds的次數，並沒改善的評估分數，就會停止生成樹，決定最
    佳n_estimators。
  
4.
objective : 決定XGBoost執行甚麼任務，最終結果數值轉換
learning_rate : 此超參數於樹的生成及最終做預測計算皆有影響
n_estimators : 總共產生幾棵樹
colsample_bytree : 生成每顆樹使用多少比例的特徵
scale_pos_weight : 不平衡資料需要調整
reg_alpha : 正規劃L1
reg_lambda : 正規劃L2
gamma : 決定一個葉是否繼續做分支
min_child_weight : 決定一個葉是否繼續做分支
subsample : 使用多少比例的資料生成每顆樹
scale_pos_weight : 把樣本數多/樣本數少 = 權重小/權重大
tree_method = "hist"、device = "cuda" : 使用GPU做運算
設定在建XGB n_estimators : 最大生成幾顆
early_stopping_rounds : 多少次之後仍無改善就停止
FIT處eval_set : 設定EARLY STOPPING的評估驗證資料資料
.get_booster().best_iteration : 把最終的n_estimators拿出
'''
# 把原本的訓練資料再分一次成訓練驗證資料
X_train, X_val, y_train, y_val = train_test_split(
    xtrain, ytrain, 
    test_size = 0.1, 
    random_state = random_state)
# 把分割再分割的訓練驗證資料重新算不平衡比例
weights_for_es = compute_class_weight(
    class_weight = "balanced", 
    classes = np.array([0, 1]), 
    y = y_train)
scale_pos_weight = weights_for_es[1] / weights_for_es[0]
scale_pos_weight1 = sum(y_train == 0) / sum(y_train == 1)
# 創建模型
xgb_tree = xgb.XGBClassifier(
    objective = 'binary:logistic', 
    random_state = random_state,
    scale_pos_weight = scale_pos_weight,
    tree_method = "hist", 
    device = "cuda",     
    n_estimators = 1000,   
    early_stopping_rounds = 10)  
# 參數範圍
bayes_xgb_param = {
    'learning_rate' : Real(0.01, 0.3, prior = 'uniform'),
    'max_depth' : Integer(3, 10),        
    'subsample' : Real(0.5, 1, prior = 'uniform'),
    'colsample_bytree' : Real(0.5, 1, prior = 'uniform'),
    'gamma' : Real(0, 10, prior = 'uniform'),
    'min_child_weight' : Integer(0, 10),
    'reg_lambda' : Real(0, 1, prior = 'uniform'),
    'reg_alpha' : Real(0, 1, prior = 'uniform')}
# 建立BayesSearchCV
bayes_xgb_search = BayesSearchCV(
    estimator = xgb_tree, 
    search_spaces = bayes_xgb_param,
    n_iter = 10,
    random_state = random_state,
    n_jobs = -1, 
    cv = 5,
    pre_dispatch = '2*n_jobs',
    verbose = 2,
    scoring = scoring)
# 把資料FIT進去找最佳超參數
bayes_xgb_search.fit(
    X_train, y_train,
    eval_set = [(X_val, y_val)])
# 最佳參數
best_bayes_xgb_params = bayes_xgb_search.best_params_
best_bayes_xgb_estimator = bayes_xgb_search.best_estimator_
best_xgb_nestimator = best_bayes_xgb_estimator.get_booster().best_iteration
# 把最佳超參數存下來
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_xgb_params.pkl"
pickle.dump(best_bayes_xgb_params, open(file_name, "wb"))
best_bayes_xgb_params = pickle.load(open(file_name, "rb"))
file_name = "C:/Users/88691/Desktop/自學/AI/practice/es_xgb_estimator.pkl"
pickle.dump(best_bayes_xgb_estimator, open(file_name, "wb"))
best_bayes_xgb_estimator = pickle.load(open(file_name, "rb"))
# 最終MODEL用原本的訓練測試資料，計算scale_pos_weight
all_pos_weight = weights[1] / weights[0]
# 把最佳超參數FIT進XGB
bayes_xgb = xgb.XGBClassifier(
    objective='binary:logistic', 
    **best_bayes_xgb_params,
    n_estimators = best_xgb_nestimator, 
    random_state = random_state,
    scale_pos_weight = all_pos_weight)
# bayes_xgb = xgb.XGBClassifier(
#     objective = 'binary:logistic', 
#     random_state = random_state,
#     scale_pos_weight = all_pos_weight,
#     colsample_bytree = 0.8670140089927842,
#     gamma = 9.393697376027717,
#     learning_rate = 0.05744608180517957,
#     max_depth = 4,
#     min_child_weight = 8,
#     reg_alpha = 0.37257977798325786,
#     reg_lambda = 0.4590245141508057,
#     subsample = 0.7673825800605678,
#     n_estimators = best_xgb_nestimator)

# 把資料FIT進最佳超參數的xgb
best_xgb_model = bayes_xgb.fit(X_train, y_train)
# 把訓練測試資料FIT模型做預測看結果
ytrain_best_xgb_pred = best_xgb_model.predict(X_train)
ytest_best_xgb_pred = best_xgb_model.predict(xtest)
# 把訓練測試資料FIT模型做預測看結果的機率
ytrain_best_xgb_pred_proba = best_xgb_model.predict_proba(X_train)[:,1]
ytest_best_xgb_pred_proba = best_xgb_model.predict_proba(xtest)[:,1]
print(f' train_xgb Accuracy Score: {best_xgb_model.score(X_train, y_train)}')
print(f' test_xgb Accuracy Score: {best_xgb_model.score(xtest, ytest)}')
print(f' train_xgb F1 score: {metrics.f1_score(y_train, ytrain_best_xgb_pred)}')
print(f' test_xgb F1 score: {metrics.f1_score(ytest, ytest_best_xgb_pred)}')
print(f' train_xgb AUC score: {metrics.roc_auc_score(y_train, ytrain_best_xgb_pred_proba)}')
print(f' test_xgb AUC score: {metrics.roc_auc_score(ytest, ytest_best_xgb_pred_proba)}')
'''
1.一定要切割在切割，沒有驗證集不能做
2.只要不寫early stopping，寫上其他都不會執行，結果都一樣
3.有沒有寫EVAL沒差，EVAL寫logloss，結果不變
4.有沒有寫n estermate，有差
5.寫在XGB還是BSCV，有差
'''
#%%類神經ANN
'''
1.在NN中包含了Input Layer，中間的Hidden Layer，及輸出的Output Layer。Input Layer
及Output Layer可以有很多個神經元，但只會有一層，Hidden Layer可以有很多層。每一
個Input Layer都會與下一層的所有Hidden Layer連接，最後一層的Hidden Layer會與所有
Output Layer連接，這被稱做Fully Connected。在每一個Input Layer連結每一個Hidden 
Layer會通過Weight及Bias計算，再進入Active Function，例如:Relu、Tanh、Sigmoid，
對應出Active Function中的某部分，並畫出多條線。之後每一個Active Function連結至
每一個Output Layer都會再經過一個Weight及Active Function，這時多條線透過計算及
對應，合併並且調整、轉換成一條最Fit Data的線。最終Output Layer呈現的結果通常是
任何數，如是Regression可直接使用，如是Classification會經過SoftMax、Sigmoid轉換
成0至1之間的數。過程中所有的Weight及Bias，都是通過Backpropagation決定，第一
個Weight及Bias由隨機生成做計算，得出結果後在Regression問題中可以使用MSE等，
Classification可使用Cross Entropy，看預測出的結果跟實際差多少，在反過來調整
Weight及Bias去縮小差距。為了找到最佳的Weight及Bias，可使用的方式是Gradient 
Decent，通過不同Weight及Bias所計算出資料的MSE或Cross Entropy，可以畫出一個點及
此點的梯度線，向梯度線的下方移動，希望可以獲得一組全局最佳參數，一條梯度為0的線
，但我們無法得知全局最佳故通常為局部最佳參數。Gradient Decent中又可以分為三種
Batch Gradient Decent、Stochastic Gradient Decent、Mini Batch Gradient Decent。
Batch Gradient Decent是會用所有資料去計算MSE或Cross Entropy，優點是比較精確，
缺點是太耗時。Stochastic Gradient Decent會隨機所有資料只用一筆資料，優點是速度
快，缺點是不精準。而Mini Batch Gradient Decent則是會隨機一小批，結合了Batch 
Gradient Decent、Stochastic Gradient Decent優缺點。

2.keras.Sequential([
    keras.layers.Dense(30, input_shape = [30], activation = 'relu'),
    keras.layers.Dense(3, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')])
Sequential:一堆層、Dense:每個神經元與下一層所有神經元連結(fully connected)
Dense(第一個HIDDEN LAYER有幾個神經元, input_shape = [input形狀]，
Dense(第二個HIDDEN LAYER有幾個神經元, activation = 使用的激活函數),                                                   
最後一筆Dense(幾個OUTPUT LAYER，轉換結果的ACTIVE FUNCTION)]

3.hp : 定義超參數的類型INT、FLOAT
    num_hidden_layers = hp.Int(
        'num_hidden_layers', 
        min_value=1, 
        max_value=5, step=1) : 名字為num_hidden_layers的參數，從最小1至最大5，
    每次選擇步長1，最終傳一個整數給num_hidden_layers 
    hp.Float() : 浮點數
    hp.Choice('activation', values=['relu', 'tanh', 'sigmoid']) : 
        名字叫activation的參數，由values中選一個
        
4.在FOR迴圈中，當我num_hidden_layers決定有幾個隱藏層後，就會跑幾次迴圈，每次回圈都會
    重新選擇units、activation、dropout
    
5.metrics  
[keras.metrics.BinaryCrossentropy(name='cross entropy'),
keras.metrics.MeanSquaredError(name='Brier score'),
keras.metrics.TruePositives(name='tp'),
keras.metrics.FalsePositives(name='fp'),
keras.metrics.TrueNegatives(name='tn'),
keras.metrics.FalseNegatives(name='fn'), 
keras.metrics.BinaryAccuracy(name='accuracy'),
keras.metrics.Precision(name='precision'),
keras.metrics.Recall(name='recall'),
keras.metrics.AUC(name='auc'),
keras.metrics.AUC(name='prc', curve='PR'),
keras.metrics.F1Score(name='f1_score')]

6.
model.compile:
    optimizer : 像是用甚麼方式找最佳參數(EX:SGD)
    loss : 可以理解為MSE、cross entropy，透過優化LOSS找到更好的WEIGHT、BIAS
    metrics : 當模型在訓練時，可以隨時監控的指標
BayesianOptimization:
    objective = 'accuracy' : 比較哪組參數比較好的標準(val_accuracy有做ES可改) 
    max_trials = 50 : 總共會找幾次參數
    num_initial_points = 10 : 初始參數筆數
    directory = '...' : 把最佳參數及模型儲存在主目錄
    project_name = '...' : 把最佳參數及模型儲存在子目錄
tuner.search
    epochs : 做幾次的BACK PROPAGATION
    validation_split : 把資料切出多少比例做為驗證集(因ANN中無K-FOLD的參數，可以
                       設定split當每組參數的比較)
    batch_size : 把所有資料丟進去訓練類神經網路，每次用batch_size設定的筆資料計
        算LOSS，約做 資料數/BATCH SIZE次 的BACKPROPAGATION優化WEIGHT BIAS，循環
        Epochs次
Early Stopping:
    monitor : 通過甚麼指標監測是否執行ES    
    patience : 經過幾次沒有改善執行ES           
    restore_best_weights :把最加的結果存下來 

7.總共會進行max_trials設定的50次，第一次會先選出num_initial_points設定得10筆所
有的參數，每筆參數會進行epochs設定的100次的BACK PROPAGATION，透過計算選到的loss
並使用選到的optimizer去優化WEIGHT、BIASES。最後把10筆計算出每筆objective
設定的結果，建立代理模型，第11筆參數會基於代理模型中所有筆資料去找更好的參數，
計算結果加入代理模型，直到max_trials設定的50次。總共產生50個參數模型，
.get_best_models(num_models = 1)[0]、.get_best_hyperparameters(num_trials = 1)[0]
從50個參數模型選出最好的。

8.預防OVERFITTING
Batch Normalization : 每一層輸入前都做一次特徵標準化，不但可以加快梯度下降
    的求解速度，而且在一定程度緩解了深層網絡中梯度消失的問題，
    也很好解決OVERFITTING。(甚至可以取代DROPOUT)
    
9.classification:
    output layer active function : 二元sigmoid、多元softmax
    hidden layer active function : 不可sigmoid
'''
# tf建立全局隨機種子，後續不用在設定
tf.random.set_seed(random_state)
# 建模型
def ann_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(
        units = hp.Int('first_hidden_unit', 
            min_value = 1, max_value = 1000, step = 100),
        activation = hp.Choice(
            'activation_input', 
            values = ['relu', 'LeakyReLU', 'tanh']),
        input_shape = (xtrain.shape[1],),
        kernel_regularizer = regularizers.l1_l2(
            l1 = hp.Float('l1', 1e-6, 1e-2, sampling = 'log'), 
            l2 = hp.Float('l2', 1e-6, 1e-2, sampling = 'log'))))  
    model.add(layers.BatchNormalization())
    num_hidden_layers = hp.Int('num_hidden_layers', 1, 4, step = 1)
    
    for i in range(num_hidden_layers):
        model.add(layers.Dense(
            units = hp.Int(f'units_{i}', 2, 100, step = 4),
            activation = hp.Choice(f'activation_{i}', 
                values = ['relu', 'LeakyReLU', 'tanh']),
            kernel_regularizer = regularizers.l1_l2(
                l1 = hp.Float('l1', 1e-6, 1e-2, sampling = 'log'), 
                l2 = hp.Float('l2', 1e-6, 1e-2, sampling = 'log'))))
        model.add(layers.BatchNormalization()) 
        
    model.add(layers.Dense(
        1, activation='sigmoid'))
    choiced_optimizer = hp.Choice(
        'optimizer', values=['adam', 'sgd', 'rmsprop', 'adamw'])
    choiced_learning_rate = hp.Float(
        'learning_rate', 1e-4, 1e-2, sampling='log')
    
    if choiced_optimizer == 'adam':
        optimizer = Adam(learning_rate = choiced_learning_rate)
    elif choiced_optimizer == 'sgd':
        optimizer = SGD(learning_rate = choiced_learning_rate)
    elif choiced_optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate = choiced_learning_rate)
    elif choiced_optimizer == 'adamw':
        optimizer = AdamW(learning_rate = choiced_learning_rate)
        
    model.compile(  
        optimizer = optimizer,
        loss = 'binary_crossentropy',
        metrics = keras.metrics.AUC(name ='auc'))
    return model

# early_stopping = callbacks.EarlyStopping(
#     monitor = 'val_accuracy',    
#     patience = 2,           
#     restore_best_weights = True)

# 創建BayesianOptimization
ann_tuner = tuners.BayesianOptimization(
    ann_model,
    objective = Objective(
        'val_auc', direction = 'max'),
    max_trials = 100,
    num_initial_points = 50)  
# 把資料FIT進去找
ann_tuner.search(
    xtrain, ytrain, 
    epochs = 100,
    batch_size = 64,
    class_weight = class_weight,
    validation_split = 0.1)          
# callbacks = [early_stopping])            
# 最佳模型(不需再重新FIT一次資料)num_models = 1 : 所有裡面最佳的
best_ann_model = ann_tuner.get_best_models(num_models = 1)[0]
# 最佳超參數
best_ann_hp = ann_tuner.get_best_hyperparameters(num_trials = 1)[0]
# 把模型及參數存起來
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_ann_model.pkl"
pickle.dump(best_ann_model, open(file_name, "wb"))
best_ann_model = pickle.load(open(file_name, "rb"))
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_ann_hp.pkl"
pickle.dump(best_ann_hp, open(file_name, "wb"))
best_ann_hp = pickle.load(open(file_name, "rb"))
final_best_hp = best_ann_hp.values.items()
ytrain_best_ann_pred = best_ann_model.predict(xtrain)
ytest_best_ann_pred = best_ann_model.predict(xtest)
# >0.5 : 1, <0.5 : 0
ytrain_ann_predicted_labels = (ytrain_best_ann_pred > threshold).astype(int)
ytest_ann_predicted_labels = (ytest_best_ann_pred > threshold).astype(int)
print(f' train_ann Accuracy Score: {metrics.accuracy_score(ytrain, ytrain_ann_predicted_labels)}')
print(f' test_ann Accuracy Score: {metrics.accuracy_score(ytest, ytest_ann_predicted_labels)}')
print(f' train_ann F1 score: {metrics.f1_score(ytrain, ytrain_ann_predicted_labels)}')
print(f' test_ann F1 score: {metrics.f1_score(ytest, ytest_ann_predicted_labels)}')
print(f' train_ann AUC score: {metrics.roc_auc_score(ytrain, ytrain_best_ann_pred)}')
print(f' test_ann AUC score: {metrics.roc_auc_score(ytest, ytest_best_ann_pred)}')
#%%CNN
'''
1.CNN主要是做圖片的預測分類辨識。如果把圖片放大後，可以是一小格一小格的像素點，
每格包含R、G、B的3組數字組成。當在訓練CNN模型時，主要分為Convolution Layer和
Pooling Layer。Convolution Layer中模型會通過計算，把整張圖辨識拆成各項特徵，擷取
出圖中各項某些部分可以辨識的特徵。並且為了泛化會把各項特徵周圍一起進行學習訓練。
Convolution Layer後接著Pooling Layer，主要是把上一層Convolution Layer擷取的各項
特徵，進行降低緯度，讓圖變模糊及縮小，更方便計算及泛化未來資料。再傳入下一個
Convolution Layer，把各項特徵結合在一起，組合成各項特徵的結合，最終把各項特徵壓扁結合
，作為ANN的訓練資料。在特徵偵測擷取時，以幾*幾的方格作為每個特徵的大小，以strides設定
的移動格數把圖的每個地方都至少輪過一次，但較靠近圖中央會被方格反覆偵測到，而周圍次數少
，padding是在原圖的周圍加上空白的格子，讓周圍的圖增加偵測到的次數。做不做Pooling在後
面特徵擷取計算時都會有相同大小為了增加圖像的預測能力，訓練資料的圖片是可以平移、旋轉、
縮放、亮度調整等。

2.
rescale : 把所有ImageDataGenerator讀進來的圖都做標準化/255
.flow_from_directory : 為數據生成器
target_size : 圖像尺寸，大多模型是讀224,244
class_mode : binary為二分類，categorical為多分類
shuffle : 是否打散資料
filter : 要幾個特徵 
kernel_size : 每個特徵的大小(2*2、3*3)
其餘可見ANN
'''
import os

data_dir = 'C:/Users/88691/Desktop/自學/AI/practice/圖/xray_dataset_covid19_prac/pic'
# 原始文件地方
class_dirs = os.listdir(data_dir)
# 列出資料夾中資料夾名稱

# 把data_dir及class_dir結合成完整路徑，把分別路徑下的圖讀出來
all_data = []
for class_dir in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_path):
        images = [img for img in os.listdir(class_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
        all_data.extend([(os.path.join(class_path, img), class_dir) for img in images])
# for class_dir in class_dirs:
#     class_path = os.path.join(data_dir, class_dir)
#     images = os.listdir(class_path)

train_val_imgs, test_imgs = train_test_split(
    all_data, 
    test_size = 0.2, 
    random_state = random_state)

train_imgs, val_imgs = train_test_split(
    train_val_imgs, 
    test_size = 0.1,
    random_state = random_state)

train_data = []
test_data = []
val_data = []

train_data.extend(train_imgs)
test_data.extend(test_imgs)
val_data.extend(val_imgs)
# 把圖和對應資料夾名結合並加到剛剛創的LIST
# train_data.extend([(os.path.join(class_path, img), class_dir) for img in train_imgs])
# test_data.extend([(os.path.join(class_path, img), class_dir) for img in test_imgs])
# val_data.extend([(os.path.join(class_path, img), class_dir) for img in val_imgs])

# 建立FOR訓練資料的生成器並初始化功能
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    brightness_range = [0.8, 1.2],
    fill_mode = 'nearest')
# 建立FOR驗證資料的生成器並初始化功能
val_datagen = ImageDataGenerator(rescale = 1. / 255)
# 建立FOR測試資料的生成器並初始化功能
test_datagen = ImageDataGenerator(rescale=1./255)
# 建立訓練生成器
train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame(train_data, columns = ['filename', 'class']),
    x_col = 'filename',
    y_col = 'class',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True)
# 建立測試生成器
test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame(test_data, columns = ['filename', 'class']),
    x_col = 'filename',
    y_col = 'class',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle=False)
# 建立驗證生成器
val_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame(val_data, columns = ['filename', 'class']),
    x_col = 'filename',
    y_col = 'class',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle=False)
# x_cnn = []
# y_cnn = []
# # 把讀取的圖片及標籤加到LIST裡
# for i in range(len(data_generator.filenames)):  
#     x_batch, y_batch = data_generator.next()
#     x_cnn.append(x_batch[0])  
#     y_cnn.append(y_batch[0])
# # 轉換格式
# x_cnn = np.array(x_cnn)
# y_cnn = np.array(y_cnn)
# # 分資料
# xtrain_cnn, xtest_cnn, ytrain_cnn, ytest_cnn = train_test_split(
#     x_cnn, y_cnn, 
#     test_size = test_size, 
#     random_state = random_state)

# 建模型
def cnn_model(hp):
    model = keras.Sequential()
    nums_set_layers = hp.Int('nums_set_layers', 1, 5, 1)    
    for i in range(nums_set_layers):
        if i == 0:
            model.add(layers.Conv2D(
                filters = hp.Int('start_filter', 32, 256, 32),
                kernel_size = hp.Choice('start_kernel_size', 
                                        values = [3, 5, 7]),
                strides = (1, 1),
                padding = 'same',
                activation = hp.Choice('start_activation', 
                                       values = ['relu', 'LeakyReLU', 'tanh']),
                input_shape = train_generator.image_shape,
                kernel_regularizer = regularizers.l1_l2(
                    l1 = hp.Float(f'l1_1_{i}', 1e-6, 1e-2, 
                                  sampling = 'log'), 
                    l2 = hp.Float(f'l2_1_{i}', 1e-6, 1e-2, 
                                  sampling = 'log'))))    
        else:
            model.add(layers.Conv2D(
                filters = hp.Int('start_filter', 32, 256, 32),
                kernel_size = hp.Choice('start_kernel_size', 
                                        values = [3, 5, 7]),
                strides = (1, 1),
                padding = 'same',
                activation = hp.Choice('start_activation', 
                                       values = ['relu', 'LeakyReLU', 'tanh']),
                kernel_regularizer = regularizers.l1_l2(
                    l1 = hp.Float(f'l1_1_{i}', 1e-6, 1e-2, 
                                  sampling = 'log'), 
                    l2 = hp.Float(f'l2_1_{i}', 1e-6, 1e-2, 
                                  sampling = 'log'))))       
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(
            pool_size = hp.Choice(f'pool_size_{i}', 
                                  values = [2, 3])))
        
    nums_ann_layers = hp.Int('nums_ann_layers', 1, 3, step = 1)
    model.add(layers.Flatten())        
    for i in range(nums_ann_layers):
        model.add(layers.Dense(
            units = hp.Int(f'units_{i}', 2, 100, step = 4),
            activation = hp.Choice(f'activation_{i}', 
                                   values = ['relu', 'LeakyReLU', 'tanh']),
            kernel_regularizer = regularizers.l1_l2(
                l1 = hp.Float(f'l1_2_{i}', 1e-6, 1e-2, sampling = 'log'), 
                l2 = hp.Float(f'l2_2_{i}', 1e-6, 1e-2, sampling = 'log'))))
        model.add(layers.BatchNormalization())
            
    model.add(layers.Dense(1, activation='sigmoid'))
    choiced_optimizer = hp.Choice('optimizer', 
                                  values = ['adam', 'sgd', 'rmsprop', 'adamw'])
    choiced_learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, 
                                     sampling = 'log')
        
    if choiced_optimizer == 'adam':
        optimizer = Adam(learning_rate = choiced_learning_rate)
    elif choiced_optimizer == 'sgd':
        optimizer = SGD(learning_rate = choiced_learning_rate)
    elif choiced_optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate = choiced_learning_rate)
    elif choiced_optimizer == 'adamw':
        optimizer = AdamW(learning_rate = choiced_learning_rate)
            
    model.compile(  
        optimizer = optimizer,
        loss = 'binary_crossentropy',
        metrics = keras.metrics.BinaryAccuracy(name = 'accuracy'))
    return model
# early_stopping = callbacks.EarlyStopping(
#     monitor = 'val_accuracy',    
#     patience = 2,           
#     restore_best_weights = True)
# 創建BayesianOptimization
cnn_tuner = tuners.BayesianOptimization(
    cnn_model,
    objective = Objective('val_accuracy', direction = 'max'),
    max_trials = 50,
    num_initial_points = 25)
# 把資料FIT進去找

'''
一次epochs，會從train_generator拿設定的batch_size張圖，進行訓練後再繼續拿設定的
batch_size張圖，直到張數達原本資料及中圖片的數目。所拿的圖不一定是原本資料夾的圖。訓
練完後進入驗證過程，從test_generator拿設定的batch_size張圖，直到資料夾中的張數。再進
入到下一次epochs


'''
cnn_tuner.search(train_generator,
                 steps_per_epoch=len(train_generator),
                 epochs = 50,
                 validation_data = val_generator,
                 validation_steps = len(val_generator))
# callbacks = [early_stopping])
# 最佳模型
best_cnn_model = cnn_tuner.get_best_models(num_models = 1)[0]
# 最佳超參數
best_cnn_hp = cnn_tuner.get_best_hyperparameters(num_trials = 1)[0]
# 存檔
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_cnn_model.pkl"
pickle.dump(best_cnn_model, open(file_name, "wb"))
best_cnn_model = pickle.load(open(file_name, "rb"))
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_cnn_hp.pkl"
pickle.dump(best_cnn_hp, open(file_name, "wb"))
best_cnn_hp = pickle.load(open(file_name, "rb"))

cnn_train_images, cnn_train_labels = [], []
for i in range(len(train_generator)):
    x_batch, y_batch = train_generator[i]
    cnn_train_images.extend(x_batch)
    cnn_train_labels.extend(y_batch)
cnn_train_images = np.array(cnn_train_images)
cnn_train_labels = np.array(cnn_train_labels)

cnn_test_images, cnn_test_labels = [], []
for i in range(len(test_generator)):
    x_batch, y_batch = test_generator[i]
    cnn_test_images.extend(x_batch)
    cnn_test_labels.extend(y_batch)
cnn_test_images = np.array(cnn_test_images)
cnn_test_labels = np.array(cnn_test_labels)

ytrain_cnn_pred = best_cnn_model.predict(cnn_train_images)
ytest_cnn_pred = best_cnn_model.predict(cnn_test_images)

ytrain_cnn_label = []
for i in range(len(cnn_train_labels)):
    if cnn_train_labels[i,0] == 0:
        ytrain_cnn_label.append(1)
    else:
        ytrain_cnn_label.append(0)
ytrain_cnn_label = np.array(ytrain_cnn_label)

ytest_cnn_label = []
for i in range(len(cnn_test_labels)):
    if cnn_test_labels[i,0] == 0:
        ytest_cnn_label.append(1)
    else:
        ytest_cnn_label.append(0)
ytest_cnn_label = np.array(ytest_cnn_label)
# >0.5 : 1, <0.5 : 0
ytrain_cnn_predicted_labels = (ytrain_cnn_pred > threshold).astype(int)
ytest_cnn_predicted_labels = (ytest_cnn_pred  > threshold).astype(int)

train_accuracy = metrics.accuracy_score(label, ytrain_cnn_predicted_labels)
train_f1 = metrics.f1_score(label, ytrain_cnn_predicted_labels)
train_auc = metrics.roc_auc_score(label, y_train_pred)


print(f' train_cnn Accuracy Score: {metrics.accuracy_score(ytrain_cnn_label, ytrain_cnn_predicted_labels)}')
print(f' test_cnn Accuracy Score: {metrics.accuracy_score(ytest_cnn_label , ytest_cnn_predicted_labels)}')
print(f' train_cnn F1 score: {metrics.f1_score(ytrain_cnn_label, ytrain_cnn_predicted_labels)}')
print(f' test_cnn F1 score: {metrics.f1_score(ytest_cnn_label ,ytest_cnn_predicted_labels)}')
print(f' train_cnn AUC score: {metrics.roc_auc_score(ytrain_cnn_label, ytrain_cnn_pred)}')
print(f' test_cnn AUC score: {metrics.roc_auc_score(ytest_cnn_label , ytest_cnn_pred)}')
#%%問題
'''
單純onehot encoder、label encoder不用先切吧?沒有差吧?
'''
