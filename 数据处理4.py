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

# 给字符型编码
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data_dummies = pd.get_dummies(all_data)

#数值型填充
mean_cols = all_data_dummies.mean()
all_data_dummies = all_data_dummies.fillna(mean_cols)

# 将数值型数据进行标准化
for col in all_data.columns[all_data.dtypes != 'object']:
    all_data_dummies[col] = preprocessing.scale(all_data_dummies[col])

# 将数据重新分为训练集和测试集
train_dummies = all_data_dummies.loc[train.index]   
test_dummies = all_data_dummies.loc[test.index]
X_train = train_dummies.values
X_test = test_dummies.values

'''
#岭回归交叉验证
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
'''
# 随机森林（基分类器是回归决策树）交叉验证
max_features = [.1, .3, .5, .7, .9, .99]
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
plt.show()
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
)
plt.savefig("Bagging.png")
plt.show()
'''
bg = BaggingRegressor(n_estimators=25, base_estimator=Ridge(15))
bg.fit(X_train, y_train)

y_final = np.expm1(bg.predict(X_test))

submission = pd.DataFrame(data = {'Id': test.index, 'SalePrice': y_final})

submission.to_csv("submission.csv", index=False)