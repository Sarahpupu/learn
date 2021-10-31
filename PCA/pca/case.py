# /Users/chuxiuru/anaconda3/bin/python
# _*_coding=utf-8 _*_
# @author sarah_chu
# @date 2021-10-06 20:25

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN

data = pd.read_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/PCA/data/digit recognizor.csv')
X = data.iloc[:,1:]
y = data.iloc[:,0]
print(X.shape)

# 探索PCA的n_components的个数
# 画累计方差贡献率曲线，找最佳降维后降维的范围
# pca_line = PCA().fit(X)
# print(pca_line.explained_variance_ratio_)
# plt.figure(figsize=[20,5])
# plt.plot(np.cumsum(pca_line.explained_variance_ratio_))  # plt.plot()X轴可以省略，默认从0开始增加
# plt.xlabel('number of components after dimension reduction')
# plt.ylabel('cumulative explained variance')
# plt.show()

# 降维后纬度的学习曲线，继续虽小最佳纬度的范围
# score = []
# for i in range(1,101,10):
#     X_dr = PCA(i).fit_transform(X)
#     once = cross_val_score(RFC(n_estimators=10,random_state=0)
#                            ,X_dr,y,cv=5).mean()
#     score.append(once)
# plt.figure(figsize=[20,5])
# plt.plot(range(1,101,10),score)
# plt.show()
# # 选20左右效果比较好
#
# score = []
# for i in range(10,25):
#     X_dr = PCA(i).fit_transform(X)
#     once = cross_val_score(RFC(n_estimators=10,random_state=0)
#                            ,X_dr,y,cv=5).mean()
#     score.append(once)
# plt.figure(figsize=[20,5])
# plt.plot(range(10,25),score)
# plt.show()

# 看图选取21最优
X_dr = PCA(21).fit_transform(X)
print(X_dr.shape)
print(cross_val_score(RFC(n_estimators=10,random_state=0),X_dr,y,cv=5).mean())
print(cross_val_score(RFC(n_estimators=100,random_state=0),X_dr,y,cv=5).mean())

# 对降维后的21个特征换KNN模型再试一次
print(cross_val_score(KNN(),X_dr,y,cv=5).mean())

# 画KNN的学习曲线
score = []
for i in range(10):
    X_dr = PCA(21).fit_transform(X)
    once = cross_val_score(KNN(i+1),X_dr,y,cv=5).mean()
    score.append(once)

plt.figure(figsize=[20,5])
plt.plot(range(10),score)
plt.show()

# KNN的最优是3
print(cross_val_score(KNN(3),X_dr,y,cv=5).mean())
