# /Users/chuxiuru/anaconda3/bin/python
# _*_coding=utf-8 _*_
# @author sarah_chu
# @date 2021-07-18 14:27


# B站：唐宇迪，
# https://www.bilibili.com/video/BV16k4y1q7Bd?from=search&seid=5560562709149028960
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# 读取数据

# 定义sigmoid函数:np.exp代表e的次幂
def sigmoid(z):
    return 1/(1+np.exp(-z))


# nums = np.arange(-10,10,step=1)
# fig, ax = plt.subplots(figsize=(12,4))
# ax.plot(nums, sigmoid(nums), 'r')
# plt.show()


#
def model(X, theta):
    return sigmoid(np.dot(X, theta.T))  # np.dot表示矩阵的乘法





if __name__ == '__main__':
    pdData = pd.read_csv('data.csv')
    # 增加为1的一列，在测算Thera0
    pdData.insert(0, 'Ones', 1)  # 0代表在第0列插入
    # print(pdData)

    orig_data = pdData.as_matrix()  # 数组转化为数组模式
    # print(orig_data)
    # print(type(orig_data))
    cols = orig_data.shape[1]  # 取列数
    # print(cols)
    X = orig_data[:, 0:cols-1]  # 取前三列特征值
    y = orig_data[:,cols-1:cols]  # 取最后一列结果判定值
    # print(X)
    # print(y)

    # 构造1行三列的theta值进行占位
    theta = np.zeros([1,3])
    print(theta)

    # 定义损失函数
    def cost(X, y, theta):
        left = np.multiply(-y,np.log(model(X, theta)))  # model是计算sigmoid值
        right = np.multiply(1-y,np.log(1-model(X,theta)))
        return np.sum(left-right)/(len(X))
    print(cost(X,y,theta))

    # 计算梯度
    def gradient(X, y, theta):
        grad = np.zeros(theta.shape)
        error = (model(X, theta)-y).ravel() #.ravle将多维数组转化为一维数组
        for j in range(len(theta.ravel())):
            term = np.multiply((error, X[:,j]))
            grad[0,j] = np.sum(term)/len(X)

        return grad

    # 比较三种不同梯度下降方法
    STOP_ITER = 0
    STOP_COST = 1
    STOP_GRAD = 2

    def stopCriterion(type, value, threshold):  # threshold表示阈值
        # 设置三种 不同的设置策略
        if type == STOP_ITER:
            return value>threshold  # 根据迭代次数
        elif type == TOP_COST :
            return abs(value[-1]-value[-2])<threshold  # 根据目标值变化
        elif type == STOP_GRAD:
            return np.linalg.norm(value)<threshold  # 根据梯度

    # 洗牌，把有序的数据变成无序
    def shuffleData(data):
        np.random.shuffle(data)  # shuffle洗牌
        cols = data.shape[1]
        X = data[:, 0:cols-1]
        y = data[:, cols-1:]
        return X, y

    def descent(data, theta, batchSize, stopType, thresh, alpha):

        init_time = time.time()
        i = 0  # 迭代次数
        k = 0  # batch
        X, y = shuffleData(data)
        grad = np.zeros(theta.shape)  # 计算的梯度
        costs = [cost(X, y, theta)]  # 损失值

        while True:
            grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
            k += batchSize  # 取batch数量个数据
            if k >= n:
                k = 0
                X, y = shuffleData(data)  #重新洗牌
            theta = theta - alpha*grad  #参数更新
            costs.append(cost(X, y, theta))  # 计算新的损失
            i += 1

            if stopType == STOP_ITER:
                value = i
            elif stopType ==STOP_COST:
                value == costs
            elif stopType == STOP_GRAD:
                value = grad
            if stopCriterion((stopType, value, thresh)): break

        return theta, i-1, costs, grad, time.time()-init_time

    def runExpe(data, theta, batchSize, stopType, thresh, alpha):
        # import pdb; pdb.set_traace();
        theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh,alpha)
        name = "Original" if (data[:, 1]>2).sum() >1 else "Scaled"
        name += "data - learning rate: {}-".format(alpha)
        if batchSize==n:
            strDescType = "Gradient"
        elif batchSize==1:
            strDescType = "Stochastic"
        else:
            strDescType = "Mini-batch({})".format(batchSize)
        name += strDescType + "descent - Stop:"
        if stopType == STOP_ITER:
            strStop = "{} iterations".format(thresh)
        elif stopType == STOP_COST:
            strStop = "costs change < {}".format(thresh)
        else:
            strStop = "gradient norm < {}".format(thresh)
        name += strStop
        print("***{}\nTheta: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
            name, theta, iter, cost[-1], dur
        ))
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(np.arange(len(costs)), costs, 'r')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.set_title(name.upper() + '- Error vs. Iteration')
        return theta

    n = 100
    runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)
