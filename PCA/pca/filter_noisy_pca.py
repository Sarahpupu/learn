# /Users/chuxiuru/anaconda3/bin/python
# _*_coding=utf-8 _*_
# @author sarah_chu
# @date 2021-10-06 19:48

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import numpy as np

digits = load_digits()

print(digits.data.shape)
print(digits.images.shape)
# 查看target的值：利用set实现去重
print(set(digits.target.tolist()))

# 作图看图像
def plot_digits(data):
    # data的结构必须是(m,n)，并且n能够被分成（8，8）的结构
    fig, axes = plt.subplots(4,10,figsize=(10,4)
                             ,subplot_kw={'xticks':[],'yticks':[]}
                             )
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8),cmap='binary')
    plt.show()

plot_digits(digits.data)

# 原数据比较完美，人为的添加的噪音
rng = np.random.RandomState(42)  #规定numpy中的随机模式
noisy = rng.normal(digits.data,2)  #表示从第一个参数中取出服从正太分布的数据，且方差为第二个参数
# 因为是服从正态分布，从原始数据抽取的结果中，有部分数据是重复出现的，还有部分数据是没有出现过的
plot_digits(noisy)

# 利用PCA去除噪音
pca = PCA(0.5, svd_solver='full').fit(noisy)  # 0.5表示降维后的特征保留原本信息的50%
X_dr = pca.transform(noisy)
print(X_dr.shape)

# 利用inverse_transform逆转数据
without_noise = pca.inverse_transform(X_dr)
print(without_noise)
plot_digits(without_noise)
# 感想:这种方法逆转后的结果并不是完全回复原始数据，并不是很好的方法



