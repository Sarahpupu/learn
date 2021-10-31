# /Users/chuxiuru/anaconda3/bin/python
# _*_coding=utf-8 _*_
# @author sarah_chu
# @date 2021-08-18 00:22

import pandas as pd
import time
import numpy as np
import os


data_07 = pd.read_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/work_test/new_sjd_user_hj_202107.csv',encoding = 'gb18030')
# # print(data_07.head())
# 按照操作时间降序排列
data_07_sorted = data_07.sort_values(by=['操作时间'], ascending=False)
# data_07_sorted_head50 = data_07_sorted.iloc[0:50,:]
# # data_07_sorted_head50.to_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/work_test/data_07_sorted_head50.csv', encoding="utf_8_sig", index=False)
# # print(data_07_sorted_head50.info())
# print(data_07_sorted_head50.iloc[1,0])
# print(type(data_07_sorted_head50.iloc[1,0]))
# print(data_07_sorted_head50['手机号码'])
# print(type(data_07_sorted_head50['手机号码']))
# if data_07_sorted_head50.iloc[1,0].tolist() in data_07_sorted_head50['手机号码'].tolist():
#     print("True")
# else:
#     print("Error")

# data_07_sorted_head50 = pd.read_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/work_test/data_07_sorted_head50.csv')


# 提取重复的纪录
data_07_sorted['duplicate'] = data_07_sorted.duplicated(subset=['手机号码'], keep=False)
# print(data_07_sorted.info())
print(data_07_sorted['duplicate'].value_counts())
data_07_sorted_duplicateT = data_07_sorted[data_07_sorted['duplicate'] == True]
# print(data_07_sorted_duplicateT.info())

# 提取只有一条办理的明细
data_07_sorted_duplicateF = data_07_sorted[data_07_sorted['duplicate'] == False]

# 2、建立一个空Dataframe存储筛选后的数据集，并插入第一条排序后的原数据
columns = data_07_sorted_duplicateT.columns.values.tolist()
data_07_pure = pd.DataFrame(columns=columns)
data_07_pure = data_07_pure.append(data_07_sorted_duplicateT.iloc[0,:])
map = {}
# 3、按照筛选逻辑进行for循环遍历
for i in range(len(data_07_sorted_duplicateT)):
    start = time.time()
    tel = data_07_sorted_duplicateT.iloc[i, 0].tolist()
    if map.get(tel) == 1:
        data_07_pure_tel = data_07_pure[data_07_pure['手机号码'] == tel]
        if data_07_pure_tel['升降档标志'].tolist()[0] in [1,]:
            end = time.time()
            print("执行第 %s 次，用时 %s s" % (i, end - start))
            continue
        elif data_07_sorted_duplicateT.iloc[i,:]['升降档标志'].tolist() in [1,]:
            data_07_pure.drop(index=data_07_pure_tel.index, inplace=True)
            data_07_pure = data_07_pure.append(data_07_sorted_duplicateT.iloc[i,:])
    else:
        data_07_pure = data_07_pure.append(data_07_sorted_duplicateT.iloc[i,:])
        map[tel]=1
    end = time.time()
    print("执行第 %s 次，用时 %s s" %(i, end - start))

data_07_pure.to_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/work_test/data_07_pure.csv', encoding="utf_8_sig", index=False)






#
# columns = data_07_sorted.columns.values.tolist()
# data_07_pure = pd.DataFrame(columns=columns)
#
#
# data_07_pure = data_07_pure.append(data_07_sorted.iloc[0,:])
# print(data_07_pure)
# print(type(data_07_pure['升降档标志']))
#
# # print(len(data_07_sorted_head50))
# for i in range(len(data_07_sorted_head50)):
#     start = time.time()
#     if data_07_sorted_head50.iloc[i,0].tolist() in data_07_pure['手机号码'].tolist():
#         tel = data_07_sorted_head50.iloc[i,0].tolist()
#         data_07_pure_tel = data_07_pure[data_07_pure['手机号码'] == tel]
#         j = data_07_pure[data_07_pure['手机号码'] == tel].index[0]
#         if data_07_pure.iloc[j,:]['升降档标志'] == 1:
#             end = time.time()
#             print("执行第 %s 次，用时 %s s" % (i, end - start))
#             continue
#         elif data_07_sorted_head50.iloc[i,:]['升降档标志'].tolist() == 1:
#             data_07_pure.drop(index=data_07_pure_tel.index, inplace=True)
#             data_07_pure = data_07_pure.append(data_07_sorted_head50.iloc[i,:])
#
#     else:
#         data_07_pure = data_07_pure.append(data_07_sorted_head50.iloc[i,:])
#     end = time.time()
#     print("执行第 %s 次，用时 %s s" %(i, end - start))
#
# # data_07_pure_head10 = data_07_pure.iloc[0:10,:]
# data_07_pure.to_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/work_test/data_07_pure.csv', encoding="utf_8_sig", index=False)