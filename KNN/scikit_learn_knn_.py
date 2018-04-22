import numpy as np
from collections import Counter
# from sklearn.neighbors import KNeighborsClassifier

class KNN:

    def __init__(self,k):
        assert k>0 ,"k must be a value"
        self.k = k
        self._X = None
        self._y = None

    def fit(self,X,y):
        assert X.shape[0]==y.shape[0],'X must be equal to y'
        assert self.k <=X.shape[0],'X must be bigger than k'
        self._X = X
        self._y = y
        return  "KNN(k={})".format(self.k)

    def predict(self,x_pred):
        assert self._X is not None and self._y is not None ,"must fit before predict"
        assert self._X.shape[1]==x_pred.shape[1],"x_pred must be equal to X"
        y_pred = [self._predict(x) for x in x_pred]
        return np.array(y_pred)

    def _predict(self,x):
        distance = np.sqrt(np.sum((self._X-x)**2,axis=1))
        arg = np.argsort(distance)
        tops = [self._y[i] for i in arg[:self.k]]
        votes = Counter(tops)
        return votes.most_common(1)[0][0]   #最优秀的一个


if __name__ == "__main__":
    X = np.random.rand(10,2)*10
    y = np.array([0,0,0,0,0,1,1,1,1,1])

    x = np.random.rand(1,2)*10
    x_pred = x.reshape(1,-1)
    knn = KNN(6)
    knn.fit(X,y)
    last = knn.predict(x_pred)
    print(last)

# import numpy as np
# from cmath import *
# from collections import Counter
# from sklearn.neighbors import KNeighborsClassifier
#
# X = np.random.rand(10,2)*10 #训练集
# y = np.array([0,0,0,0,0,1,1,1,1,1]) #标签集,一维数组矩阵是列的
# print(X.shape,y.shape)
#
# x = np.array([1.234324,2.23525])    #待预测数据
# print(X,x)
# print(X-x)
# print((X-x)**2)
# print(np.sum((X-x)**2,axis=1))
# print(np.argsort(np.sqrt(np.sum((X-x)**2,axis=1))))
# KNN_calssifier = KNeighborsClassifier(n_neighbors=6)
# KNN_calssifier.fit(X,y) #训练的数据集与标记
# x_predict =x.reshape(1,-1)  #-1自动决定第二维度的大小
# predict = KNN_calssifier.predict(x_predict)   #预测x的标记，x最好是矩阵形式
# print(predict[0],1)
#
#








# #########自己构造一个KNN的类

# import numpy as np
# from cmath import *
# from collections import Counter
#
# class KNNClassfier:
#
#     def __init__(self,k):
#         #初始化
#         assert k>=1,"k must be valid"
#         self.k = k
#         self._X = None
#         self._y = None
#
#     def fit(self,X,y):
#         #放入训练集和标签集
#         assert X.shape[0] == y.shape[0],"size of X must be equal to size of y"
#         assert self.k<=X.shape[0],"the size of X must be bigger k"
#         self._X = X
#         self._y = y
#         return "KNN(k={})".format(self.k)
#
#     def predict(self,x_pred):
#         assert self._X is not None and self._y is not None ,"must fit before predict"
#         assert x_pred.shape[1]==self._X.shape[1],"the number of x_pred must be equal to X"
#         y_predict = [self._predict(x) for x in x_pred]
#         return np.array(y_predict)
#
#     def _predict(self,x):
#         distance = np.sqrt(np.sum((self._X-x)**2,axis=1))
#         nearest = np.argsort(distance)
#         top_y = [self._y[i] for i in nearest[:self.k]]
#         votes = Counter(top_y)
#         return votes.most_common(1)[0][0]
#
#
# X = np.random.rand(10,2)*10 #训练集
# y = np.array([0,0,0,0,0,1,1,1,1,1]) #标签集,一维数组矩阵是列的
#
# x = np.array([1.234324,2.23525])
# x = x.reshape(1,-1)     #这一步很关键，要化成矩阵的形式
# knn = KNNClassfier(4)
# knn.fit(X,y)
# print(knn.predict(x)[0])
