# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import math
from sklearn.model_selection import train_test_split
import os


def judge(a, b):
    if a in userRecommendMat[b]:
        return 1
    else:
        return 0


# 读取文件
ratingsDF = pd.read_table("./u.data", header=None, names=['userId', 'movieId', 'rating', 'timestamp'], index_col=None,
                          engine='python')
# ratingsDF = pd.read_csv("./ratings.csv")
# 隐式化数据
ratingsDF = ratingsDF[3 <= ratingsDF['rating']]
# 划分训练集 测试集 9:1
trainRatingsDF, testRatingsDF = train_test_split(ratingsDF, test_size=0.1, random_state=1)
print("total_movie_count:" + str(len(set(ratingsDF['movieId'].values.tolist()))))
print("total_user_count:" + str(len(set(ratingsDF['userId'].values.tolist()))))
print("train_movie_count:" + str(len(set(trainRatingsDF['movieId'].values.tolist()))))
print("train_user_count:" + str(len(set(trainRatingsDF['userId'].values.tolist()))))
print("test_movie_count:" + str(len(set(testRatingsDF['movieId'].values.tolist()))))
print("test_user_count:" + str(len(set(testRatingsDF['userId'].values.tolist()))))

trainRatingsPivotDF = pd.pivot_table(trainRatingsDF[['userId', 'movieId', 'rating']], columns=['movieId'],
                                     index=['userId'], values='rating', fill_value=0)

moviesMap = dict(enumerate(list(trainRatingsPivotDF.columns)))
usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))
ratingValues = trainRatingsPivotDF.values.tolist()

# 用户的物品分数
resource = np.zeros((len(ratingValues), len(ratingValues[0])))
for i in range(len(ratingValues)):
    for j in range(len(ratingValues[0])):
        if ratingValues[i][j] != 0:
            resource[i][j] = 1

# 物品的度
degree_i = np.zeros(len(ratingValues[0]))
for i in range(len(ratingValues[0])):
    for j in range(len(ratingValues)):
        if ratingValues[j][i] != 0:
            degree_i[i] += ratingValues[j][i]

# 物品到用户的传播矩阵
translation1 = np.zeros((len(ratingValues[0]), len(ratingValues)))
for i in range(len(ratingValues[0])):
    for j in range(len(ratingValues)):
        if ratingValues[j][i] != 0:
            translation1[i][j] = ratingValues[j][i]/degree_i[i]


# 物品向用户的传播

# #用户的度
degree_u = np.zeros(len(ratingValues))
for i in range(len(ratingValues)):
    for j in range(len(ratingValues[0])):
        if ratingValues[i][j] != 0:
            degree_u[i] += 1


translation2 = np.zeros((len(ratingValues), len(ratingValues[0])))
for i in range(len(ratingValues)):
    for j in range(len(ratingValues[0])):
        if ratingValues[i][j] != 0:
            translation2[i][j] = 1/(degree_u[i])


resource_mat = np.mat(resource)
translation1_mat = np.mat(translation1)
resource_u_mat = resource_mat * translation1_mat

# resource_u_list = resource_u_mat.tolist()
# for i in range(len(ratingValues)):
#     for j in range(len(ratingValues)):
#         resource_u_list[i][j] = resource_u_list[i][j] * degree_u[j] / (abs(degree_u[i] - degree_u[j]) + 1)
# resource_u_mat = np.mat(resource_u_list)

translation2_mat = np.mat(translation2)
resource_i_1 = resource_u_mat * translation2_mat
resource_i_1 = resource_i_1.tolist()


userRecommendDict = dict()
for i in range(len(ratingValues)):
    userRecommendDict[i] = sorted(enumerate(resource_i_1[i]), key=lambda x: x[1], reverse=True)

# 删除已经交互的物品
for key, value in userRecommendDict.items():
    list_pub = list()
    for (movieId, val) in value:
        if ratingValues[key][movieId] == 0:
            list_pub.append((movieId, val))
    userRecommendDict[key] = list_pub
resource = np.array(resource)
print((len(ratingValues[0]) - degree_u[10]))
print(len(userRecommendDict[10]))

# 将一开始的索引转换为原来用户id与电影id

# userRecommendMat = np.zeros((611, 8130), dtype=np.float32)
# userRecommendMat1 = np.zeros((len(ratingValues), 8130), dtype=np.float32)
userRecommendMat = np.zeros((944, 1567), dtype=np.float32)
userRecommendMat1 = np.zeros((len(ratingValues), 1567), dtype=np.float32)
moviesrec = list()
# moviesRecVal = np.zeros(len(ratingValues[0]))
# moviesRecVal[movieId] += 1

for key, value in userRecommendDict.items():
    i = 0
    for (movieId, val) in value:
        userRecommendMat[usersMap[key], i] = moviesMap[movieId]
        userRecommendMat1[key, i] = moviesMap[movieId]
        i += 1


# 测试集
testRatingsPivotDF = pd.pivot_table(testRatingsDF[['userId', 'movieId', 'rating']], columns=['movieId'],
                                    index=['userId'], values='rating', fill_value=0)

moviesMap1 = dict(enumerate(list(testRatingsPivotDF.columns)))
usersMap1 = dict(enumerate(list(testRatingsPivotDF.index)))
testValues = testRatingsPivotDF.values.tolist()

# r
sum_r = 0
num_r = 0
for i in range(len(testValues)):
    for j in range(len(testValues[0])):
        if testValues[i][j] > 0:
            if moviesMap1[j] in userRecommendMat[usersMap1[i]]:
                sum_r += ((np.argwhere(userRecommendMat[usersMap1[i]] == moviesMap1[j])[0][0] + 1)/(len(
                    userRecommendDict[i])))
                num_r += 1
print("r：")
print(sum_r/num_r)

# hitting rate
num_test = 0
num_right = 0

for i in range(len(testValues)):
    for j in range(len(testValues[0])):
        if testValues[i][j] > 0:
            num_test += 1
            if judge(moviesMap1[j], usersMap1[i]) == 1:
                if np.argwhere(userRecommendMat[usersMap1[i]] == moviesMap1[j])[0][0] < 50:
                    num_right += 1

print("hitting rate：")
print((num_right/num_test))

# 准确率
