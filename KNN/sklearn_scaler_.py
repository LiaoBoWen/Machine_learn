import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#数据归一化的模块
from sklearn.preprocessing import StandardScaler


iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=666)
standardscaler = StandardScaler()
standardscaler.fit(X_train)
meanis = standardscaler.mean_   #由用户传进去的数据得到的数据后面带_     这是代码规范
stdis = standardscaler.scale_   #std_这种查询方差的方法不合理，用scale_
print(meanis)
print(stdis)
transformis = standardscaler.transform(X_train)
X_train = transformis   #得到归一化后的数据
print(X_train)
X_test = standardscaler.transform(X_test)   #得到归一化后的数据
print(X_test)
knn_cif = KNeighborsClassifier(n_neighbors=3)
knn_cif.fit(X_train,y_train)
pred = knn_cif.predict(X_test)
print(pred)
print(knn_cif.score(X_test,y_test))



#自定义StandardScaler
# class StandardScaler:
#
#     def __init__(self):
#         self.mean_ = None
#         self.scale_ = None
#     def fit(self,X):
#         "暂时处理二维的数据"
#         assert X.ndim == 2,"the dimension of X must be 2"
#         self.mean_ = np.mean(X,axis=0)
#         self.scale_ = np.std(X,axis=0)
#         return self
#     def transform(self,X):
#         "也暂时处理二维数据"
#         assert self.mean_ is not None and self.scale_ is not None,'must  fit before transform'
#         assert X.shape[1] == len(self.mean_) ,"the feature number of X must be equal to mean_ and  scale_ "
#         assert X.ndim == 2 ,"the dimension of X must be 2"
#         #要把用户的数据确定为浮点型
#         resX = np.array(X,dtype=float)
#         #这两种方法好像有相同的作用
#         # resX = np.empty(shape=X.shape,dtype=float)
#         result = (resX-self.mean_)/self.scale_
#         return result
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=666)
# standardscaler = StandardScaler()
# standardscaler.fit(X_train)
# print(standardscaler.mean_)
# print(standardscaler.scale_)
# X_train = standardscaler.transform(X_train)
# X_test = standardscaler.transform(X_test)
# knn_cif = KNeighborsClassifier(n_neighbors=3)
# knn_cif.fit(X_train,y_train)
# print(knn_cif.score(X_test,y_test))

'''
[5.83416667 3.0825     3.70916667 1.16916667]
[0.81019502 0.44076874 1.76295187 0.75429833]
1.0
'''
