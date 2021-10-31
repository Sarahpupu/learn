# /Users/chuxiuru/anaconda3/bin/python
# _*_coding=utf-8 _*_
# @author sarah_chu
# @date 2021-07-29 23:26

# B站菜菜老师的机器学习系列课程——逻辑回归（2）
# 二元回归和多元回归：重要参数solver和multi_class

from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import  load_iris

iris = load_iris()
# print(type(iris))
# print(iris)
# print(iris.target)

for multi_class in ('multinomial','ovr'):
    lr = LR(solver='sag', max_iter=100, random_state=42,multi_class=multi_class).fit(iris.data,iris.target)
    print('traning score: %.3f (%s)' %(lr.score(iris.data,iris.target),multi_class))

# multi_class默认是ovr，表示分类问题是二分类问题，或者用一对多的形式来处理多分类问题；

