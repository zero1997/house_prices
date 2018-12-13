# -*- coding: UTF-8 -*-
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing    
import warnings
from sklearn.linear_model import Ridge 
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor


warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)

# 估价平滑
y_train = np.log1p(train.pop('SalePrice'))
all_data = pd.concat([train, test], ignore_index = False)



# MSSubClass编码
# print all_data[['MSSubClass']].dtypes
# all_data[['MSSubClass']].apply(lambda x: x = str(x))
#print all_data.index
# for i in all_data.index:
#     all_data.loc[i, 'MSSubClass'] = str(all_data.loc[i, 'MSSubClass'])
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
# all_data = pd.get_dummies(all_data.MSSubClass, prefix='MSSubClass')
# 将所有字符型的进行编码
all_data_dummies = pd.get_dummies(all_data)
# print all_data_dummies.isnull().sum().sort_values()

# 数值型缺失值处理
mean_cols = all_data_dummies.mean()
all_data_dummies = all_data_dummies.fillna(mean_cols)

# print all_data_dummies['SalePrice']
# 将数值型数据进行标准化
for col in all_data.columns[all_data.dtypes != 'object']:
    all_data_dummies[col] = preprocessing.scale(all_data_dummies[col])
# print all_data_dummies.head()



# print all_data_dummies.index
# 将数据重新分为训练集和测试集
train_dummies = all_data_dummies.loc[train.index]   
test_dummies = all_data_dummies.loc[test.index]
# print train_dummies.shape, test_dummies.shape
y_train = train_dummies[['TrainPrice']].values
tmp = train_dummies
tmp = tmp.drop(['TrainPrice', 'SalePrice'], axis = 1)
X_train = tmp.values
tmp = test_dummies
tmp = tmp.drop(['TrainPrice', 'SalePrice'], axis = 1)
X_test = tmp



#岭回归交叉验证
# X_train = train_dummies.loc[:, (train_dummies.columns != 'TrainPrice') & (train_dummies.columns != 'SalePrice')].values
# print X_train.shape, y_train.shape, train_dummies.shape
test_scores = list()
alphas = np.logspace(-3, 2, 50)
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(alphas, test_scores)
plt.title("Ridge")
plt.savefig("Ridge.png")
plt.show()

'''
# 随机森林（基分类器是回归决策树）交叉验证
max_features = [.1, .3, .5, .7, .9]
test_scores = list()
for max_feat in max_features:
    print '1'
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    #print '1'
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    #test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
print '2'
plt.plot(max_features, test_scores)
plt.title("Random Frest")
plt.savefig("Random Forest.png")
'''
'''
# Bagging, 把alpha=0.5作为基分类器
params = [1, 10, 15, 20, 25, 30, 40]
test_scores = list()
for param in params:
    clf = BaggingRegressor(n_estimators=param, base_estimator=Ridge(15))
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv = 20, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(params, test_scores)
plt.title("Bagging")
plt.savefig("Bagging.png")
plt.show()
'''

# bg = BaggingRegressor(n_estimators=25, base_estimator=Ridge(15))
# bg.fit(X_train, y_train)
# print bg.predict(X_test)

# y_final = np.expm1(bg.predict(X_test))
# print y_final
# # submission = pd.DataFrame(data = {'ID': test.index, 'SalePrice': y_final})

# #submission.to_csv("submission.csv")
