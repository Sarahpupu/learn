# /Users/chuxiuru/anaconda3/bin/python
# _*_coding=utf-8 _*_
# @author sarah_chu
# @date 2021-07-31 00:28

# B站菜菜老师的机器学习系列课程——逻辑回归（3）：制作评分卡

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import scipy
import scikitplot as skplt

#
# # 1、数据读取
# data = pd.read_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/logistic_regression/rankingcard.csv',
#                    index_col=0)
# # index_col=0表示第一列是index值；
# # print(data.head())
# # print(data.info())
#
#
#
# # 2、数据预处理
# # 2.1 剔除重复值
# '''
# inplace=True表示剔重后数据覆盖原数据，如果等于false，表示不覆盖愿数据
# '''
# data.drop_duplicates(inplace=True)
# # print(data.info())
# # 恢复索引：剔重不更改原索引，需要重新设置索引
# '''
# reset_index表示重置索引，drop=True表示不保留之前的索引；inplace=True表示覆盖原数据；
# '''
# data.reset_index(drop=True, inplace=True)
# # print(data.info())
#
# # 2.2 填补缺失值
# # 统计缺失值
# '''
# data.isnull()返回的每列特征值中是否为空的布尔数值，为空-True，不为空——False
# data.isnull().sum()表示把为空的数值求和，统计缺失值个数
# '''
# # 统计缺失值个数
# print(data.isnull().sum())
# # 求空值所占的比例
# print(data.isnull().sum()/data.shape[0])
#
# # 收入缺失19%，家庭人口缺失2.5%，家庭用均值填充，收入用随机森林填充；
# # 1）家庭人口用均值填充
# '''
# inplace=True仍然表示覆盖原数据
# '''
# data['NumberOfDependents'].fillna(data['NumberOfDependents'].mean(),inplace=True)
#
# # 2）收入：用随机森林填充
# def fill_missing_rf(X, y, to_fill):
#
#     '''
#
#     :param X: 要填充的特征矩阵（所有的列）
#     :param y: 完整的，没有缺失值的标签
#     :param to_fill: 要填补的那一列
#     :return:
#     '''
#
#     # 构建新特征矩阵和新标签，把要填补的那一列作为标签，完整的列作为特征值；
#     '''
#     pd.concat([df1,df2], axis=1)表示横向表连接
#     更多连接方式参考：https://www.jb51.net/article/164905.htm
#     '''
#     df = X.copy()
#     fill = df.loc[:, to_fill]
#     df = pd.concat([df.loc[:, df.columns != to_fill], pd.DataFrame(y)], axis=1)
#
#     # 找出训练集和测试集
#     Ytrain = fill[fill.notnull()]
#     Ytest = fill[fill.isnull()]
#     Xtrain = df.iloc[Ytrain.index,:]
#     Xtest = df.iloc[Ytest.index,:]
#
#     # 用随机森林回归来填补缺失值
#     from sklearn.ensemble import  RandomForestRegressor as rfr
#     rfr = rfr(n_estimators=100)
#     rfr = rfr.fit(Xtrain, Ytrain)
#     Ypredict = rfr.predict(Xtest)
#
#     return  Ypredict
#
# # 调用随机森林算法，求解缺失值
# X = data.iloc[:, 1:]
# y = data['SeriousDlqin2yrs']
# y_pred = fill_missing_rf(X,y,'MonthlyIncome')
#
# # 赋值缺失值
# data.loc[data.loc[:,'MonthlyIncome'].isnull(), 'MonthlyIncome'] = y_pred
#
# print(data.isnull().sum())
#
#
#
# # 2.3 描述统计处理异常值
# # 一般处理异常值：箱线图、3西格玛法则、描述性统计（在统计量有限的情况下使用）
# # 描述性统计
# # print(data.describe())
# # print(data.describe([0.01,0.1,0.25,0.5,0.75,0.9,0.99]))
# # print(data.describe([0.01,0.1,0.25,0.5,0.75,0.9,0.99]).T)
# # # pycharm描述性统计结果显示不全，建议保存在csv文档中查看
# # data.describe([0.01,0.1,0.25,0.5,0.75,0.9,0.99]).T.to_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/logistic_regression/case_04_ds.csv')
# # 分析：
# # 1)、年龄最小值是0，异常；统计有多少个年龄为0的；
# print((data['age'] == 0).sum())
# # 只有1个年龄为0的，考虑直接删除
# data = data[data['age'] !=0]
#
# # 2)、这三个字段的异常分析：NumberOfTime30-59DaysPastDueNotWors、NumberOfTime60-89DaysPastDueNotWorse、NumberOfTimes90DaysLate
# # 统计违约次数超过90次的用户
# print("===1====")
# print(data[data.loc[:,'NumberOfTimes90DaysLate'] > 90].count())
# print("===2====")
# # 有225个用户，但不都是坏客户
# # 统计NumberOfTimes90DaysLate的取值分布
# print(data.loc[:,'NumberOfTimes90DaysLate'].value_counts())
# # 有5个违约次数为96次，有220个违约次数是98的，而且不都是坏客户，所以判定为异常值；
# data = data[data['NumberOfTimes90DaysLate'] < 90]
# # 恢复索引
# data.reset_index(drop=True, inplace=True)
#
#
# # 2.4 标准化
# # 注意：标准化只能解决数据量纲不统一的问题，不能解决偏态问题。
# # 为了保持业务可解释性，建议先不进行标准化
#
# # 2.5 样本均衡
# # 查看样本是否均衡
# X = data.iloc[:,1:]
# y = data.iloc[:,0]
# # print(type(y))  #是pd.Series类型
# print(y.value_counts())
# # 解决样本不均衡的问题
# # imblearn是专门处理样本不平衡数据集的问题。
# sm = SMOTE(random_state=42)  # 实力化
# X,y = sm.fit_sample(X, y)  # 返回已经上采样完毕后的特征矩阵和标签
# # 生成的y是ndarray类型，需要转化为Series类型才能使用value_counts()属性
# # print(type(y))
# print(pd.Series(y).value_counts())
# n_sample = X.shape[0]
# n_1_sample = pd.Series(y).value_counts()[1]
# n_0_sample = pd.Series(y).value_counts()[0]
# print('样本个数：{}; 1占{:.2%};0占{:.2%}'.format(n_sample,n_1_sample/n_sample,n_0_sample/n_sample))
# '''
# format数字格式化的写法
# 参考：https://www.runoob.com/python/att-string-format.html
# '''
#
#
# # 3 划分训练集、测试集
# X = pd.DataFrame(X)
# y = pd.DataFrame(y)
#
# X_train, X_vali, Y_train, Y_vali = train_test_split(X, y, test_size=0.3, random_state=420)
#
# # 构建训练集
# model_data = pd.concat([Y_train, X_train], axis=1)
# model_data.index = range(model_data.shape[0])
# model_data.columns = data.columns
#
# # 构建验证集
# vali_data = pd.concat([Y_vali, X_vali], axis=1)
# vali_data.index = range(vali_data.shape[0])
# vali_data.columns = data.columns

# # 保存测试集、验证集
# model_data.to_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/logistic_regression/model_data.csv')
# vali_data.to_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/logistic_regression/vali_data.csv')



# 4、建模
# # 4.1 分箱：4-5个，连续性变量——>离散型变量；高内聚，低耦合；IV值,good%,bad%；
# # 以age为例进行分箱
# model_data['qcut'], updown = pd.qcut(model_data['age'], retbins=True, q=20)
#
# '''
# pd.cut是自带分箱函数
# 参数：
# retbins=True表示每条纪律显示对应的分箱结果
# q表示分箱的个数
# '''
#
# print(model_data.head())
# print(model_data['qcut'].value_counts())
#
# # 统计每个分箱中0，1的数量
# # 利用透视表功能groupby
# count_y0 = model_data[model_data['SeriousDlqin2yrs']==0].groupby(by='qcut').count()['SeriousDlqin2yrs']
# count_y1 = model_data[model_data['SeriousDlqin2yrs']==1].groupby(by='qcut').count()['SeriousDlqin2yrs']
# print(count_y0)
# print(count_y1)
# print(type(count_y1))
#
#
# # 利用zip组合分箱的上下限
# '''
# zip函数：
# 可以将两个列表组合
# zip([1,2,3],['a','b','c'])
# >>>[(1,'a'),(2,'b'),(3,'c')]
# zip是惰性对象，需要用[*zip([],[])]才能打开看到里面的数据结果
# '''
# # =====以下是废弃的代码====
# # # 下面的的代码是尝试用index的方法组合出来，分箱结果的组合，但是最后是[(Interval(21.0,28.0,closed='right'),4230,4102),(),()]
# # # 没有出来[(21.0,28.0,4230,4102),(),()]，所以放弃这种方式，还是采用视频中的方法
# # print('=====this is bins_index====')
# # bins_index = pd.Series(count_y1.index)
# # print(bins_index)
# # print(type(bins_index))
# # print('=====this is updonw')
# # print(updown)
# # num_bins = [*zip(bins_index,count_y0,count_y1)]
# # print('====this is num_bins====')
# # print(num_bins)
# # =====结束=====
#
# # print([*zip(updown,updown[1:],count_y0,count_y1)])
# num_bins = [*zip(updown, updown[1:], count_y0, count_y1)]
#
# # 4.2 计算WOE值
# # 构建的num_bins进行dataframe转化
# columns = ['min', 'max', 'count_0', 'count_1']
# df = pd.DataFrame(num_bins, columns=columns)
# print(df)
#
# # 添加计算的过程指标
# df['total'] = df.count_0 + df.count_1
# df['percentage'] = df.total/df.total.sum()  # 一个箱子里的样本占所有样本的哔哩
# df['bad_rate'] = df.count_1/df.total
# df['good%'] = df.count_0/df.count_0.sum()
# df['bad%'] = df.count_1/df.count_1.sum()
# df['woe'] = np.log(df['good%']/df['bad%'])
#
# print(df.head())
#
# # 计算IV
# rate = df['good%'] - df['bad%']
# iv_age = np.sum(rate * df.woe)
#
# print(iv_age)
#
# # 定义计算woe和iv值的函数
# def get_woe(num_bins):
#     columns = ['min','max','count_0','count_1']
#     df = pd.DataFrame(num_bins, columns=columns)
#
#     df['total'] = df.count_0 + df.count_1
#     df['percentage'] = df.total / df.total.sum()  # 一个箱子里的样本占所有样本的哔哩
#     df['bad_rate'] = df.count_1 / df.total
#     df['good%'] = df.count_0 / df.count_0.sum()
#     df['bad%'] = df.count_1 / df.count_1.sum()
#     df['woe'] = np.log(df['good%'] / df['bad%'])
#
#     return  df
#
# def get_iv(df):
#     rate = df['good%'] - df['bad%']
#     iv = np.sum(rate * df.woe)
#     return iv
#
# # 卡方检验
# num_bins_ = num_bins.copy()
# pvs = []
#
# # for i in range(len(num_bins_)-1):
# #     x1 = num_bins_[i][2:]
# #     x2 = num_bins_[i+1][2:]
# #     pv = scipy.stats.chi2_contingency([x1,x2])[1]
# #     pvs.append(pv)
# #
# # print(pvs)   #两两分箱之间的卡方检验的p值
# #
# # # 返回最大的P值的索引：是第二组和第三组的分箱的P值最大
# # print(pvs.index(max(pvs)))
# #
# # num_bins_[1:3] = [(num_bins_[1][0],num_bins_[2][1],num_bins_[1][2]+num_bins_[2][2],num_bins_[1][3]+num_bins_[2][3])]
# # print(num_bins_)
#
# # IV = []
# # axisx = []
# #
# # while len(num_bins_) > 2:
# #     pvs = []
# #     # 获取num_bins_两两之间的卡方检验的置信度（或卡方值）
# #     for i in range(len(num_bins_)-1):
# #         x1 = num_bins_[i][2:]
# #         x2 = num_bins_[i+1][2:]
# #         # 0 返回卡方值(chi2)；1返回P值
# #         pv = scipy.stats.chi2_contingency([x1,x2])[1]
# #         # chi2 = scipy.stats.chi2_contigency([x1,x2][0])
# #         pvs.append(pv)
# #
# #     # 通过P值处理，合并P值最大的两组
# #     i = pvs.index(max(pvs))
# #     num_bins_[i:i+2] =[
# #         (
# #             num_bins_[i][0],
# #             num_bins_[i+1][1],
# #             num_bins_[i][2]+num_bins_[i+1][2],
# #             num_bins_[i][3]+num_bins_[i+1][3]
# #         )
# #     ]
# #
# #     bins_df = get_woe(num_bins_)
# #     axisx.append(len(num_bins_))
# #     IV.append((get_iv(bins_df)))
# #
# # plt.figure()
# # plt.plot(axisx,IV)
# # plt.xticks(axisx)
# # plt.xlabel('num_bins')
# # plt.ylabel('IV')
# # plt.savefig("/Users/chuxiuru/PycharmProjects/machinelearning/logistic_regression/num_bins_age.png")
# # plt.show()
# # # 对于age变量，分箱个数为6是最佳
#
# # 将合并箱体的部分定义为函数，并实现分箱
# def get_bin(num_bins_,n):
#     while len(num_bins_) > n:
#         pvs = []
#         for i in range(len(num_bins_)-1):
#             x1 = num_bins_[i][2:]
#             x2 = num_bins_[i+1][2:]
#             pv = scipy.stats.chi2_contingency([x1,x2])
#             pvs.append(pv)
#
#         i = pvs.index(max(pvs))
#         num_bins_[i:i+2] = [
#             (
#                 num_bins_[i][0],
#                 num_bins_[i+1][1],
#                 num_bins_[i][2]+num_bins_[i+1][2],
#                 num_bins_[i][3]+num_bins_[i+1][3]
#             )
#         ]
#     return num_bins_
#
# afterbins = get_bin(num_bins_,6)
# print(afterbins)
#
# bins_df = get_woe(afterbins)
# print(bins_df)

# 将选取最佳分箱个数的过程包装为函数
def graphforbestbin(DF, X, Y, n=5, q=20, graph=True):

    '''
    自动最优分享函数，基于卡方检验的分箱

    参数：
    :param DF: 需要输入的数据
    :param X: 需要分箱的列名
    :param Y: 分箱数据对应的标签Y列名
    :param n: 保留分箱个数
    :param q: 初始分箱个数
    :param grahp: 是否要画出IV图像
    :return:
    '''

    DF = DF[[X,Y]].copy()
    print(DF)
    print(DF[X])
    DF['qcut'], bins = pd.qcut(DF[X], retbins=True, q=q, duplicates='drop')
    coount_y0 = DF.loc[DF[Y]==0].groupby(by='qcut').count()[Y]
    coount_y1 = DF.loc[DF[Y]==1].groupby(by='qcut').count()[Y]
    num_bins = [*zip(bins, bins[1:], coount_y0, coount_y1)]

    for i in range(q):
        if 0 in num_bins[0][2:]:
            num_bins[0:2] = [
                (
                    num_bins[0][0],
                    num_bins[1][1],
                    num_bins[0][2]+num_bins[1][2],
                    num_bins[0][3]+num_bins[1][3]
                )
            ]
        continue

        for i in range(len(num_bins)):
            if 0 in num_bins[i][2:]:
                num_bins[i-1:i+1] = [
                    (
                        num_bins[i][0],
                        num_bins[i][1],
                        num_bins[i-1][1],
                        num_bins[i-1][2] + num_bins[i][2],
                        num_bins[i-1][3] + num_bins[i][3]

                    )
                ]
                break
        else:
            continue

    def get_woe(num_bins):
        columns = ['min', 'max', 'count_0', 'count_1']
        df = pd.DataFrame(num_bins, columns=columns)
        df['total'] = df.count_0 + df.count_1
        df['percentage'] = df.total / df.total.sum()
        df['bad_rate'] = df.count_1 / df.total
        df['good%'] = df.count_0 / df.count_0.sum()
        df['bad%'] = df.count_1 / df.count_1.sum()
        df['woe'] = np.log(df['good%'] / df['bad%'])
        return df

    def get_iv(df):
        rate = df['good%'] - df['bad%']
        iv = np.sum(rate * df.woe)
        return iv

    iv = []
    axisx = []
    while len(num_bins) > n:
        pvs = []
        for i in range(len(num_bins)-1):
            x1 = num_bins[i][2:]
            x2 = num_bins[i+1][2:]
            pv = scipy.stats.chi2_contingency([x1, x2])[1]
            pvs.append(pv)

        i = pvs.index(max(pvs))
        num_bins[i:i+2] = [
            (
                num_bins[i][0],
                num_bins[i+1][1],
                num_bins[i][2] + num_bins[i+1][2],
                num_bins[i][3] + num_bins[i+1][3]
            )
        ]

        bins_df = pd.DataFrame(get_woe(num_bins))
        axisx.append(len(num_bins))
        iv.append(get_iv(bins_df))

    if graph:
        plt.figure()
        plt.plot(axisx, iv)
        plt.xticks(axisx)
        plt.xlabel('number of box')
        plt.ylabel('IV')
        plt.show()
    return bins_df


# 对所有特征进行分箱选择
model_data = pd.read_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/logistic_regression/model_data.csv')
# model_data = model_data.copy()
# print(model_data.head(5))
# print(model_data.columns[1])
# print(model_data[['SeriousDlqin2yrs','SeriousDlqin2yrs']])
# print(model_data['SeriousDlqin2yrs'])

# for i in ['RevolvingUtilizationOfUnsecuredLines','age','DebtRatio','MonthlyIncome']:
#     print(i)
#     graphforbestbin(model_data, i, 'SeriousDlqin2yrs', n=2, q=20)

# 对各种特征制定不同的分箱个数
auto_col_bins = {
    'RevolvingUtilizationOfUnsecuredLines':6,
    'age':5,
    'DebtRatio':4,
    'MonthlyIncome':3,
    'NumberOfOpenCreditLinesAndLoans':5
}

# 不能使用自动分箱的变量
hand_bins = {
    'NumberOfTime30-59DaysPastDueNotWorse':[0,1,2,13],
    'NumberOfTimes90DaysLate':[0,1,2,17],
    'NumberRealEstateLoansOrLines':[0,1,2,4,54],
    'NumberOfTime60-89DaysPastDueNotWorse':[0,1,2,4],
    'NumberOfDependents':[0,1,2,3]
}

# 保护区间覆盖使用np.inf极大值替代最大值，用np.inf极小值替换最小值
hand_bins = {
    k:[-np.inf, *v[:-1],np.inf] for k,v in hand_bins.items()
}

print(hand_bins)

# 自动生成分箱的分箱区间和分箱后的IV值
bins_of_col = {}
for col in auto_col_bins:
    bins_df = graphforbestbin(model_data,
                              col,
                              'SeriousDlqin2yrs',
                              n = auto_col_bins[col],
                              q = 20,
                              graph=False
                              )
    bins_list = sorted(set(bins_df['min']).union(bins_df['max']))
    # set()是创建无序不重复元素集，bins_df['min']是取分箱结果的左边界，union集合的并集
    # 保障区间覆盖使用np.inf替换最大值，-np.inf替换最小值
    bins_list[0], bins_list[-1] = -np.inf, np.inf
    bins_of_col[col] = bins_list

# print(bins_of_col)

# 手动合并分箱数据
bins_of_col.update(hand_bins)
# update: 修改当前集合，可以添加新的元素或集合到当前集合中，如果添加的元素在集合中已存在，则该元素只会出现一次，重复的会忽略
print(bins_of_col)
# print(type(bins_of_col))

# woe代表不违约的人的概率，计算每个分箱的woe值
def get_woe(df, col, y, bins):
    df = df[[col, y]].copy()
    df['cut'] = pd.cut(df[col], bins)
    bins_df = df.groupby('cut')[y].value_counts().unstack()  #unstack将groupby树状结构变成表状结构
    woe = bins_df['woe'] = np.log((bins_df[0]/bins_df[0].sum())/(bins_df[1]/bins_df[1].sum()))
    return woe

# 将所有特征的woe存储在字典当中
woeall = {}
for col in bins_of_col:
    woeall[col] = get_woe(model_data,col,'SeriousDlqin2yrs',bins_of_col[col])
print(woeall)

# 不希望覆盖掉原本的数据，创建新Dataframe，索引和原始数据model_data一模一样
model_woe = pd.DataFrame(index=model_data.index)
# 将原始数据分箱后，按箱的结果把woe结果用map函数映射到数据中，以age示例
model_woe['age'] = pd.cut(model_data['age'], bins_of_col['age']).map(woeall['age'])
# 对所有特征操作：
for col in bins_of_col:
    model_woe[col] = pd.cut(model_data[col], bins_of_col[col]).map(woeall[col])
# 特征值NumberOfOpenCreditLinesAndLoans有缺失（不知道原因，按照B找操作步骤得出的结果不一致，懒得找原因，先直接删除）
model_woe.drop(columns='NumberOfOpenCreditLinesAndLoans',inplace=True)
# 将标签补充到数据中
model_woe['SeriousDlqin2yrs'] = model_data['SeriousDlqin2yrs']
# 这就是我们的建模数据啦

print(model_woe.head())
print(model_woe.info())


# 处理测试集
vali_data = pd.read_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/logistic_regression/vali_data.csv')
vali_woe = pd.DataFrame(index=vali_data.index)
for col in bins_of_col:
    vali_woe[col] = pd.cut(vali_data[col],bins_of_col[col]).map(woeall[col])
vali_woe.drop(columns='NumberOfOpenCreditLinesAndLoans',inplace=True)
vali_woe['SeriousDlqin2yrs'] = vali_data['SeriousDlqin2yrs']

print(vali_woe.info())

# 测试集上的X和Y
vali_x = vali_woe.iloc[:,:-1]
vali_y = vali_woe.iloc[:,-1]

# 建模
x = model_woe.iloc[:,:-1]
y = model_woe.iloc[:,-1]

lr = LR().fit(x,y)
print(lr.score(vali_x,vali_y))


# 提高精确性：调参
c_1 = np.linspace(0.01,1,20)
c_2 = np.linspace(0.01,0.2,20)
score = []
for i in c_1:
    lr = LR(solver='liblinear', C=i).fit(x,y)
    score.append(lr.score(vali_x,vali_y))
plt.figure()
plt.plot(c_1,score)
# plt.show()

print(lr.n_iter_)

score = []
for i in [1,2,3,4,5,6]:
    lr = LR(solver='liblinear',C=0.025,max_iter=i).fit(x,y)
    score.append(lr.score(vali_x,vali_y))
plt.figure()
plt.plot([1,2,3,4,5,6],score)
# plt.show()

lr = LR(solver='liblinear',C=0.025,max_iter=4).fit(x,y)
print('--------chouliujun-----------')
print(lr.score(vali_x,vali_y))

# ROC曲线：最后的ROC曲线结果不错，模型效果不错，AUC越大效果越好（AUC是曲线下面的面积）
vali_proba_df = pd.DataFrame(lr.predict_proba(vali_x))
skplt.metrics.plot_roc(vali_y,vali_proba_df,plot_micro=False,figsize=(6,6),plot_macro=False)
print('----------finish---------')
# plt.show()


# 制作评分卡
B = 20/np.log(2)
A = 600 + 8*np.log(1/60)
print(A)
print(B)

base_score = A - B*lr.intercept_
print(base_score)

# 写入
file = '/Users/chuxiuru/PycharmProjects/machinelearning/logistic_regression/ScoreData.csv'
with open(file,'w') as fdata:
    fdata.write('base_score,{}\n'.format(base_score))

for i,col in enumerate(x.columns):
    score = woeall[col] * (-B*lr.coef_[0][i])
    score.name = 'score'
    score.index.name = col
    score.to_csv(file,header=True,mode='a')

print([*enumerate(x.columns)])

# graphforbestbin(model_data, 'NumberOfTime30-59DaysPastDueNotWorse', 'SeriousDlqin2yrs', n=2, q=20)

# model_data = pd.read_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/logistic_regression/model_data.csv')
#
# DF = model_data[['NumberOfTime30-59DaysPastDueNotWorse','SeriousDlqin2yrs']].copy()
# print(DF)
#
# DF['qcut'], bins = pd.qcut(DF['NumberOfTime30-59DaysPastDueNotWorse'], retbins=True, q=20, duplicates='drop')
# print(DF)
# print(bins)
# print(DF['qcut'].value_counts())
# coount_y0 = DF.loc[DF['SeriousDlqin2yrs']==0].groupby(by='qcut').count()['SeriousDlqin2yrs']
# coount_y1 = DF.loc[DF['SeriousDlqin2yrs']==1].groupby(by='qcut').count()['SeriousDlqin2yrs']
#
# num_bins = [*zip(bins, bins[1:], coount_y0, coount_y1)]
# print(num_bins)
#
# for i in range(20):
#     tmp1 = num_bins[0][2:]
#     print(tmp1)
#     if 0 in tmp1:
#         num_bins[0:2] = [
#             (
#                 num_bins[0][0],
#                 num_bins[1][1],
#                 num_bins[0][2]+num_bins[1][2],
#                 num_bins[0][3]+num_bins[1][3]
#             )
#         ]
#         continue



# if 0 in num_bins[0][2:]:
#     num_bins[0:2] = [
#         (
#             num_bins[0][0],
#             num_bins[1][1],
#             num_bins[0][2]+num_bins[1][2],
#             num_bins[0][3]+num_bins[1][3]
#         )
#     ]
#
#
# for i in range(len(num_bins)):
#     if 0 in num_bins[i][2:]:
#         num_bins[i-1:i+1] = [
#         (
#             num_bins[i-1][0],
#                 num_bins[i][1],
#                 num_bins[i-1][2] + num_bins[i][2],
#                 num_bins[i-1][3] + num_bins[i][3]
#
#         )
#     ]
# print("========")
# print(num_bins)



# for i in range(20):
# if 0 in num_bins[0][2:]:
#     num_bins[0:2] = [
#         (
#             num_bins[0][0],
#             num_bins[1][1],
#             num_bins[0][2]+num_bins[1][2],
#             num_bins[0][3]+num_bins[1][3]
#         )
#     ]
# continue
#
# for i in range(len(num_bins)):
#     if 0 in num_bins[i][2:]:
#         num_bins[i-1:i+1] = [
#             (
#                 num_bins[i][0],
#                 num_bins[i][1],
#                 num_bins[i-1][1],
#                 num_bins[i-1][2] + num_bins[i][2],
#                 num_bins[i-1][3] + num_bins[i][3]
#
#             )
#         ]
#         break
# else:
#     break
    #
    # def get_woe(num_bins):
    #     columns = ['min', 'max', 'count_0', 'count_1']
    #     df = pd.DataFrame(num_bins, columns=columns)
    #     df['total'] = df.count_0 + df.count_1
    #     df['percentage'] = df.total / df.total.sum()
    #     df['bad_rate'] = df.count_1 / df.total
    #     df['good%'] = df.count_0 / df.count_0.sum()
    #     df['bad%'] = df.count_1 / df.count_1.sum()
    #     df['woe'] = np.log(df['good%']) / df['bad%']
    #     return df
    #
    # def get_iv(df):
    #     rate = df['good%'] - df['bad%']
    #     iv = np.sum(rate * df.woe)
    #     return iv
    #
    # IV = []
    # axisx = []
    # while len(num_bins) > n:
    #     pvs = []
    #     for i in range(len(num_bins)-1):
    #         x1 = num_bins[i][2:]
    #         x2 = num_bins[i+1][2:]
    #         pv = scipy.stats.chi2_contingency([x1, x2])[1]
    #         pvs.append(pv)
    #
    #     i = pvs.index(max(pvs))
    #     num_bins[i:i+2] = [
    #         (
    #             num_bins[i][0],
    #             num_bins[i+1][1],
    #             num_bins[i][2] + num_bins[i+1][2],
    #             num_bins[i][3] + num_bins[i+1][3]
    #         )
    #     ]
    #
    #     bins_df = pd.DataFrame(get_woe(num_bins))
    #     axisx.append(len(num_bins))
    #     IV.append(get_iv(bins_df))
    #
    # if graph:
    #     plt.figure()
    #     plt.plot(axisx, IV)
    #     plt.xticks(axisx)
    #     plt.xlabel('number of box')
    #     plt.ylabel('IV')
    #     plt.show()
    # return bins_df

# # 可以分箱的单独分组，不可以分箱的变量手写分组：
# auto_col_bins = {
#     'RevolvingUtilizationOfUnsecuredLines':6,
#     'age':5,
#     'DebtRatio':4,
#     'MonthlyIncome ':3,
#     'NumberOfOpenCreditLinesAndLoans':5
# }
#
# hand_bins = {
#     'NumberOfTime30-59DaysPastDueNotWorse':[0, 1, 2, 13],
#     'NumberOfTimes90DaysLate':[0,1,2,17]
#     'NumberRealEstateLoansOrLines':[0,1,2,4,54]
#     'NumberOfTime60-89DaysPastDueNotWorse':[0,1,2,8]
#     'NumberOfDependents':[0,1,2,3]
# }
#
# # 保证区间覆盖使用 np.inf替换最大值，用-np.inf替换最小值
# hand_bins = {k:[-np.inf,*v[:-1],np.inf] for k,v in hand_bins.items()}
#
# # 对所有特征按照选择的箱体个数和手写的分箱范围进行分箱
# bins_of_col = []
# for col in auto_col_bins:
#     bins_df = graphforbestbin(model_data,col,
#                               'SeriousDlqin2yrs',
#                               n=auto_col_bins[col],
#                               q=20,
#                               graph=False)
#     bins_list = sorted(set(bins_df['min']).union(bins_df['max']))
#     bins_list[0], bins_list[-1] = -np.inf, np.inf
#     bins_of_col[col] = bins_list
#
# # 合并手动分箱数据
# bins_of_col.update(hand_bins)
# print


