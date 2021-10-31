# /Users/chuxiuru/anaconda3/bin/python
# _*_coding=utf-8 _*_
# @author sarah_chu
# @date 2021-07-24 21:39

# B站菜菜老师的机器学习系列课程——逻辑回归（1）
# https://www.bilibili.com/video/BV1MA411J7wm?p=55
# 1、正则化：L1范式、L2范式，设置参数penalty、C值；
# 2、特征选择：embedded嵌入式——SelectFromModel函数

# 逻辑回归——正则化
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分测试集和训练集
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

# data = load_breast_cancer()
# X = data.data
# y = data.target
# print(type(data))  # bunch类型，本质上字典类型
# print(data.values())
# print(data.keys())
# print(X.shape)
# print('X type:',type(X))
#
# # 正则化分别用l1,l2
# lrl1 = LR(penalty='l1', solver='liblinear', C=0.5, max_iter=1000)
# lrl2 = LR(penalty='l2', solver='liblinear', C=0.5, max_iter=1000)
# # 逻辑回归重要属性coef_, 查看每个特征所对应的参数
# lrl1 = lrl1.fit(X, y)
# print(lrl1.coef_)
# print((lrl1.coef_ !=0).sum(axis=1))  # 系数不等于0的系数个数统计
#
# lrl2 = lrl2.fit(X, y)
# print(lrl2.coef_)
# print((lrl2.coef_ !=0).sum(axis=1))
#
# l1 = []
# l2 = []
# l1test = []
# l2test = []
#
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
#
# # C的大小决定惩罚项的大小，影响拟合效果的表现
# for i in np.linspace(0.05,1,19):  # linspace表示在0。05-1之间取19个数
#     lrl1 = LR(penalty='l1', solver='liblinear', C=i, max_iter=1000)
#     lrl2 = LR(penalty='l2', solver='liblinear', C=i, max_iter=1000)
#
#     lrl1 = lrl1.fit(Xtrain, Ytrain)
#     l1.append(accuracy_score(lrl1.predict(Xtrain), Ytrain))
#     l1test.append(accuracy_score(lrl1.predict(Xtest), Ytest))
#
#     lrl2 = lrl2.fit(Xtrain, Ytrain)
#     l2.append(accuracy_score(lrl2.predict(Xtrain), Ytrain))
#     l2test.append(accuracy_score(lrl2.predict(Xtest), Ytest))
#
# graph = [l1, l2, l1test, l2test]
# color = ['green', 'black', 'lightgreen', 'gray']
# label = ['L1', 'L2', 'L1test', 'L2test']
# print("graph start")
# plt.figure(figsize=(6, 6))
#
# # 找到适合的c的取值
# for i in range(len(graph)):
#     plt.plot(np.linspace(0.05,1,19), graph[i], color[i], label=label[i])
#     # np.linspace()表示X轴的取值11
# print("graph end")
# plt.legend(loc=4)
# plt.savefig("/Users/chuxiuru/PycharmProjects/machinelearning/logistic_regression/bb.png")
# plt.show()

# 实际使用时，默认使用l2正则化，如果感觉模型效果不好，可以换l1

# 逻辑回归-特征工程
# 1、业务选择
# 2、PCA、SVD一般不用，降维结果不可解释
# 3、统计方法：（方差、卡方、互信息），方差过滤、方差膨胀因子（VIF，variance inflation factor)

# R：统计学思维，更加缜密，先验思维
# python：计算机思维，简单明了，后验思维


# 高效的嵌入法：尽量保留愿数据信息，让模型特征减少。
data = load_breast_cancer()
LR_ = LR(solver='liblinear', C=0.8, random_state=420)  # 默认正则化是l2
# print(cross_val_score(LR_, data.data, data.target, cv=10).mean())  # 交叉验证，cv表示交叉验证10次
# X_embeded = SelectFromModel(LR_, norm_order=1).fit_transform(data.data, data.target)
# selectfrommodel(LR_, threshold, nor_order)三个参数，threshold表示阈值，低于这个阈值的特征值就会被删除

# print(X_embeded.shape)
# print(cross_val_score(LR_, X_embeded, data.target, cv=10).mean())

# 两种调优方式

# 1、threshold
# fullx = []
# fsx = []
# # 使用判断指标，不是L1范式，而是逻辑回归的系数
# # 查看所有特征值的系数
# print(LR_.fit(data.data, data.target).coef_)
# # 取特征值系数的最大值、
# print(abs(LR_.fit(data.data, data.target).coef_).max())
# # threshold在0-特征值系数中取值20个
# threshold = np.linspace(0, abs(LR_.fit(data.data,data.target).coef_).max(), 20)
#
# k = 0
# for i in threshold:
#     X_embeded = SelectFromModel(LR_, threshold=i).fit_transform(data.data, data.target)
#     fullx.append(cross_val_score(LR_, data.data, data.target, cv=5).mean())
#     fsx.append(cross_val_score(LR_, X_embeded,data.target, cv=5).mean())
#     print((threshold[k], X_embeded.shape[1]))
#     k += 1
#
# plt.figure(figsize=(20, 9))
# plt.plot(threshold, fullx, label='full')
# plt.plot(threshold, fsx, label='feature selection')
# plt.xticks(threshold)
# plt.legend()
# plt.show()

# 2、调整逻辑回归本身的参数：c值

# fullx = []
# fsx = []
#
# # c一般是0-1，但是这里面取到相对较大的值效果较好
# c = np.arange(0.01,10.01,0.5)
# for i in c:
#     LR_ = LR(solver='liblinear', C=i, random_state=420)
#     fullx.append(cross_val_score(LR_,data.data,data.target,cv=10).mean())
#     X_embeded = SelectFromModel(LR_, norm_order=1).fit_transform(data.data,data.target)
#     fsx.append(cross_val_score(LR_,X_embeded,data.target,cv=10).mean())
#
# print(max(fsx),c[fsx.Rindex(max(fsx))])  # fsx.index()返回索引
#
# plt.figure(figsize=(20, 5))
# plt.plot(c, fullx, label='full')
# plt.plot(c, fsx, label='feature selection')
# plt.xticks(c)
# plt.legend()
# plt.show()


# 每次滚动的方向就是梯度向量的方向；max_iter:能走的步数，最大的人迭代次数
# 梯度：多元函数，对各个自变量求偏导数，然后用向量表示出来，这个向量就是梯度向量。
# 梯度的方向：是损失函数的值增长最快的方向。
# 梯度的反方向：是损失函数的值减少最快的方向。
# 逻辑回归的损失函数：基于极大似然估计得出。（没有求解参数需求的模型是没有损失函数的，比如KNN、决策树）
# 步长-阿尔法：学习率：

# 参数： max_iter设置迭代次数，如果不到设置的最大迭代次数就取到最优解则停止；
# 看迭代的次数；lr.n_iter_

# 多分类问题：1对多——one-vs-rest，OvR；多对多，many-vs-many,sklearn中表示为Multinominal

# solver参数：
# liblinear：二分类专用，默认求解器；支持二分类，支持ovr，但不支持多分类；

