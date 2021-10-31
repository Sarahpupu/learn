# /Users/chuxiuru/anaconda3/bin/python
# _*_coding=utf-8 _*_
# @author sarah_chu
# @date 2021-09-28 22:47

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

faces = fetch_lfw_people(min_faces_per_person=60)  #实例化

print(faces.data.shape)  # 行是样本数，列是样本相关的所有特征数
print(faces.images.shape)  # 1277是矩阵中图像的个数，62是每个图像的特征矩阵的行，47是每个图像特征矩阵的列

X = faces.data

fig,axes = plt.subplots(4,5
                        ,figsize=(8,4)
                        ,subplot_kw={'xticks':[],'yticks':[]})

# axes[0][0].imshow(faces.images[0,:,:])

# axes是二维数组，利用flat降维,len([*axes.flat])查看长度,
# [*enumerate(axes.flat)]添加了索引的组成了元组
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i,:,:],cmap='gray')
plt.show()

# 降维（只接受二维）
pca = PCA(150).fit(X)  # 150表示最后想要保留的特征的个数
V = pca.components_
print(V.shape)

# 把V可视化
fig, axes = plt.subplots(4,5
                        ,figsize=(8,4)
                        ,subplot_kw={'xticks':[],'yticks':[]})
for i, ax in enumerate(axes.flat):
    ax.imshow(V[i,:].reshape(62,47),cmap='gray')
plt.show()


# inverse_transform参数：降维后的结果再进行升维，得到与原数据一样的数据量的特征值
# pca实例化
pca = PCA(150)
X_dr = pca.fit_transform(X)
X_inverse = pca.inverse_transform(X_dr)

# 画图验证X与X_inverse的差异：X_inverse得到的结果更模糊，说明再次升维后不能完全一致，降维丢失了一部分信息
# X_inverse是从降维后的150个特征再次升维到原来的一千多个特征值，会丢失掉信息，没有实现数据的逆转，不会包含100%信息
# 所以降维是不可逆的
fig, ax = plt.subplots(2,10,figsize=(10,2.5)
                        ,subplot_kw={"xticks":[],"yticks":[]}
                        )
for i in range(10):
    ax[0,i].imshow(faces.images[i,:,:],cmap='binary_r')
    ax[1,i].imshow(X_inverse[i].reshape(62,47),cmap='binary_r')
plt.show()

