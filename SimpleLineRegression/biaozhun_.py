import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from 向量化运算 import SimpleLineRegression1
from sklearn.metrics import mean_squared_error,mean_absolute_error  #差异模块


boston = datasets.load_boston()
# print(boston.DESCR)
# print(boston.feature_names)     #获取特征列表
X = boston.data[:,5]
y = boston.target
X = X[y<50] #除去不准确的数据
y = y[y<50] #除去不准确的数据
print(X)
print(y)
print(X.shape)
print(y.shape)
# plt.plot(X,y,"ro")
# plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=666)
reg = SimpleLineRegression1()
reg.fit(X_train,y_train)
print(reg.a_)
print(reg.b_)
# plt.plot(X_train,y_train,"ro")
# plt.plot(X_train,reg.predict(X_train))
# plt.plot(X_test,reg.predict(X_test),"go")
# plt.show()

#MSE
y_predict = reg.predict(X_test)

mse = np.sum((y_predict-y_test)**2)/len(y_test)
mse = mean_squared_error(y_predict,y_test)

rmse = math.sqrt(mse)

mae = np.sum(np.absolute(y_predict-y_test))/len(y_test)
mae = mean_absolute_error(y_predict,y_test)

#R Square
r_square =  1-mean_squared_error(y_predict,y_test)/np.var(y_test)   #np.var方差

print(mse,rmse,mae,r_square)

# plt.plot(X_test,y_predict,"yo")
# plt.show()
