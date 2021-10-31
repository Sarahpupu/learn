# /Users/chuxiuru/anaconda3/bin/python
# _*_coding=utf-8 _*_
# @author sarah_chu
# @date 2021-09-17 21:54

# 高阶的降维算法是：LDA、NMF（高阶矩阵分解，类似于神经网络的级别，效果非常好）

# pca：主成分分析，利用方差的原理
# 迷你案例
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

iris = load_iris()
print(type(iris))  #类型是<class 'sklearn.utils.Bunch'>
print(iris.target_names)
print(iris)


y = iris.target
X = iris.data  # nd.array类型
print(X)
# 查看X是几维数组：2维数组；
print(X.shape)
# 但作为特征矩阵，是4维特征矩阵（有4个特征）
# X = pd.DataFrame(X)
# print(type(X))
# print(X.head())


# 调用pca，降维到二维
pca = PCA(n_components=2)  # 实例化，n_components表示希望获取的纬度
pca = pca.fit(X)  # 拟合模型
X_dr = pca.transform(X)  #获取新矩阵，dr表示dismotion reduction纬度降级
print(X_dr)
# #  一步到位的写法
# X_dr = PCA(2).fit_transform(X)   # n_components可以省略不写

# # 使用布尔索引取出三种鸢尾花（布尔：true/false）
# plt.figure()  #现在我要画图了，请给我一个画布吧
# # X_dr[y==0,0]表示取y=0的行，及第一个特征列
# plt.scatter(X_dr[y==0,0], X_dr[y==0,1], c='red', label=iris.target_names[0])
# plt.scatter(X_dr[y==1,0], X_dr[y==1,1], c='black', label=iris.target_names[1])
# plt.scatter(X_dr[y==2,0], X_dr[y==2,1], c='orange', label=iris.target_names[2])
# plt.legend()
# plt.title('PCA of IRIS dataset')
# plt.show()

# 以上画图代码可以优化为：
colors= ['red','black','orange']
plt.figure()
for i in [0,1,2]:
    plt.scatter(X_dr[y==i,0]
               ,X_dr[y==i,1]
               ,alpha=.7
               ,c=colors[i]
               ,label=iris.target_names[i])
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()

# 查看降维后的特征向量所携带的信息量（可解释性方差的大小）
print(pca.explained_variance_)  # 第一个特征所携带信息多
# 查看降维后每个特征向量所占的信息量占原始信息量的百分比
print(pca.explained_variance_ratio_)
# 降维后所有特征向量所携带的信息占原始数据总信息量的比率
print(pca.explained_variance_ratio_.sum())

# 累计可解释性方差贡献率
import numpy as np
pca_line = PCA().fit(X)
plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))   # np.cumsum是累加，比如1+2，1+2+3，1+2+3+¥
plt.xticks([1,2,3,4])   # 指定坐标轴刻度，如果不指定，X轴会自定义刻度，出现小数
# plt.yticks([0,1,2,3])
plt.xlabel('number of components after dimension reduction')
plt.ylabel('cumulative explained variance ratio')
plt.show()
# 根据画图结果，特征值选择2或3，就可以保留愿数据集95%以上的信息量

# 最大似然估计自选超参数（计算量会很大，谨慎使用）
pca_mle = PCA(n_components='mle')
pca_mle = pca_mle.fit(X)
X_mle = pca_mle.transform(X)
print(X_mle)
print(pca_mle.explained_variance_ratio_.sum())  # 对所有降维后特征值的信息量占比求和


# 按照信息量占比选超参数
pca_f = PCA(n_components=0.97,svd_solver='full')
pca_f = pca_f.fit(X)
X_f = pca_f.transform(X)
print(X_f)
print(pca_f.explained_variance_ratio_)

# 人脸识别中n_components的运用
