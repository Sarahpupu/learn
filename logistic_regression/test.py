# /Users/chuxiuru/anaconda3/bin/python
# _*_coding=utf-8 _*_
# @author sarah_chu
# @date 2021-08-07 16:38

import pandas as pd
import numpy as np

# data = pd.read_csv(r'/Users/chuxiuru/PycharmProjects/machinelearning/logistic_regression/test')
# print(data)
# # data['qcut'], bins = pd.qcut(data['age'], retbins=True, q=10, duplicates='drop')
# # print(data)
# # print(bins)
# # print(data['qcut'].value_counts())
# # print(type(data))
#
# data = data.values.tolist()
# print("========")
# print(data)
# print(type(data))
#
# print(data[1][1:])
# if 22 in data[1][1:]:
#     print('True')
# else:
#     print('nan')
#
# data[1:3] = [
#             (
#                 data[1][0],
#                 data[2][1],
#                 data[1][2]
#             )
#         ]
# print(data)

# bins= pd.qcut(range(10),q=10)
# print(bins)

a = []
for i in range(len(num_bins)):
    tmp = num_bins[i][2:]
    if 0 in tmp:
        a[i] = i;

for i in a:
    j = a[i];
    num_bins[j  - 1:j  + 1] = [
        (
            num_bins[j  - 1][0],
            num_bins[j ][1],
            num_bins[j  - 1][2] + num_bins[j ][2],
            num_bins[j  - 1][3] + num_bins[j ][3]

        )
    ]