# /Users/chuxiuru/anaconda3/bin/python
# _*_coding=utf-8 _*_
# @author sarah_chu
# @date 2021-08-18 21:37

import pandas as pd
import time
import numpy as np
import os

# 读取明细
data_07 = pd.read_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/work_test/new_sjd_user_hj_202107.csv',encoding = 'gb18030')
# print(data_07.info())

# 按照操作时间降序排列
data_07_sorted = data_07.sort_values(by=['操作时间'], ascending=False)

# 提取重复的纪录
data_07_sorted['duplicate'] = data_07_sorted.duplicated(subset=['手机号码'], keep=False)
# print(data_07_sorted.info())
print(data_07_sorted['duplicate'].value_counts())
data_07_sorted_duplicateT = data_07_sorted[data_07_sorted['duplicate'] == True]
# print(data_07_sorted_duplicateT.info())

# 提取只有一条办理的明细
data_07_sorted_duplicateF = data_07_sorted[data_07_sorted['duplicate'] == False]

# 定义去重规则
def get_quchong(x):
    # print(type(x))
    # print(x)
    start = time.time()
    if len(x) == 1:
        end = time.time()
        print("用时 %s s" % (end - start))
        return x
    filter_value = x['升降档标志'].tolist()
    # print(filter_value)
    if 1 in filter_value:
        index = filter_value.index(1)
        # print(index)
        # print(x[index:index+1])
        # print(type(x[index:index+1]))
        end = time.time()
        print("用时 %s s" % (end - start))
        return x[index:index+1]
    else:
        # print(x[0:1])
        # print(type(x[0:1]))
        end = time.time()
        print("用时 %s s" % (end - start))
        return x[0:1]

# 按照号码分组并执行剔重规则
data_07_sorted_drop_duplicateT = data_07_sorted_duplicateT.groupby('手机号码',as_index=False).apply(get_quchong)

# 分组后号码是索引，把号码作为特征列，使用reset_index
data_07_sorted_drop_duplicateT = data_07_sorted_drop_duplicateT.reset_index(drop=True)
# print(data_07_head50_sorted_group)
# print(data_07_head50_sorted_group.info())
# print(data_07_sorted_duplicateT_group.info())

# 拼接剔重后纪录和只有一条纪录的明细：用concat函数
data_07_sorted_drop_duplicate_all = pd.concat([data_07_sorted_drop_duplicateT,data_07_sorted_duplicateF])

# 保存剔重后明细结果
data_07_sorted_drop_duplicate_all.to_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/work_test/data_07_sorted_drop_duplicate_all.csv',encoding="utf_8_sig", index=False)



# 历史代码：
# data_07_sorted_head50 = data_07_sorted.iloc[0:50,:]
# data_07_sorted_head50['duplicate'] = data_07_sorted_head50.duplicated(subset=['手机号码'],keep=False)
# # duplicate用法：https://blog.csdn.net/lost0910/article/details/109724264
#
# print(data_07_sorted_head50.head())
# # print(data_07_sorted.duplicated(subset=['手机号码']).sum())
# # 定义去重函数