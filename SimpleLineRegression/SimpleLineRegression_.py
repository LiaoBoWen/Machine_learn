import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5])
y = np.array([1,3,2,3,5])
plt.plot(x,y,"ro")
plt.axis([0,6,0,6])        #规定坐标轴 的大小
# plt.show()

x_mean = np.mean(x)
y_mean = np.mean(y)
a = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - a * x_mean
y_hat = a * x + b
plt.plot(x,y_hat,color="y")
x_predict = 6
y_predict = a * x_predict + b
plt.plot(x_predict,y_predict,"o")
# plt.show()


#封装算法
class SimpleLineRegression1:

    def __init__(self):
        self.a_ = None
        self.b_ = None
    def fit(self,x_train,y_train):
        assert len(x_train) == len(y_train),"the size of x_train must be equal to size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        self.a_ = np.sum((x_train - x_mean) * (y_train - y_mean))/np.sum((x_train - x_mean) ** 2)
        self.b_ = y_mean - self.a_ * x_mean
    def predict(self,x_predict):
        assert self.a_ is not None and self.b_ is not None,'must fit before predict'
        y_predict = self.a_ * x_predict + self.b_
        return y_predict

slr = SimpleLineRegression1()
slr.fit(x,y)
li = np.array([1,2,3,4,5,6])
pred = slr.predict(li)
print(pred)
plt.plot(li,pred,"g*--")
plt.show()
print(slr.a_,slr.b_)
