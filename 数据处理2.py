# -*- coding: UTF-8 -*-
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 估价平滑
all_data = pd.concat([train, test], ignore_index = False)
all_data['TrainPrice'] = np.log1p(all_data.SalePrice)
# all_data.loc[all_data['SalePrice'].notnull() , ['SalePrice', 'TrainPrice']].hist()
# plt.show()
# all_data.loc[['SalePrice', 'TrainPrice']].hist()

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
mean_cols = all_data_dummies.mean()
all_data_dummies = all_data_dummies.fillna(mean_cols)
