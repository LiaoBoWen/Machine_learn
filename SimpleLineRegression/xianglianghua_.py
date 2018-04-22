import numpy as np
import matplotlib.pyplot as plt


class SimpleLineRegression1:

    def __init__(self):
        self.a_ = None
        self.b_ = None
    def fit(self,x_train,y_train):
        assert len(x_train) == len(y_train),"the size of x_train must be equal to size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        # self.a_ = ((x_train - x_mean).dot(y_train - y_mean))/(x_train - x_mean).dot(x_train - x_mean)
        self.a_ = np.sum((x_train - x_mean) * (y_train - y_mean))/np.sum((x_train - x_mean) ** 2)
        self.b_ = y_mean - self.a_ * x_mean
    def predict(self,x_predict):
        assert self.a_ is not None and self.b_ is not None,'must fit before predict'
        y_predict = self.a_ * x_predict + self.b_
        return y_predict
if __name__ == "__main__":
    x = np.array([1,2,3,4,5])
    y = np.array([1,3,2,3,5])
    slr = SimpleLineRegression1()
    slr.fit(x,y)
    li = np.array([1,2,3,4,5,6])
    pred = slr.predict(li)
    print(pred)
    print(slr.a_,slr.b_)
    # plt.plot(li,pred,"g*--")
    # plt.axis([0,6,0,6])
    # plt.show()
    m = 1000000
    big_x = np.random.random(size=m)
    big_y = big_x *2.0 +3.0+np.random.normal(size =m)
    sl = SimpleLineRegression1()
    sl.fit(big_x,big_y)
    x = np.arange(100001)
    y_hat = sl.a_ * x +sl.b_
    plt.plot(big_x,big_y,"*")
    plt.plot(x,y_hat)
    plt.show()
