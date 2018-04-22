import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler        #归一化模块

class LinearRegression:

    def __init__(self):
        #设定系数
        self.coef_ = None
        #设定截距
        self.interception_ = None
        #整体的 （西塔）
        self._theta = None

    def fit_normal(self,X_train,y_train):
        assert X_train.shape[0] == y_train.shape[0],'the size of X_train must be equal to the size of y_train'
        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)  #此处原理是求  theta  的公式
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    #多维向量的处理：
    def fit_gd(self,X_train,y_train,eta=0.01,n_iters=1e4,epsilon=1e8):
        assert X_train.shape[0] == y_train.shape[0] ,\
        'the size of X_train must be equal to size of y_train'

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float("inf")

        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))  #开空间
            X_b = np.hstack([np.ones((len(X_b), 1)), X_b])  # 使用的括号个数......

            res = X_b.T.dot(X_b.dot(theta) - y)
            # print(X_b)
            '''[3.00517447]
            [4.02369667 3.00517447]'''
            return res * 2 / len(X_b)

        def gradient_descents(X_b, y,init_theta, eta, n_iters=n_iters, epsilon=epsilon):
            theta = init_theta
            i_iter = 0

            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon:
                    break

                i_iter += 1

            return theta

        init_theta = np.zeros(X.shape[1] + 1)
        self._theta  = gradient_descents(X_train,y_train,init_theta,eta,)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    #随机梯度下降法
    def fit_sgd(self,X_train,y_train,n_iters=5,t0=5,t1=50):
        assert X_train.shape[0] == y_train.shape[0],\
        'the size of X_train must be equal to the size of y_train'
        assert  n_iters >0,'n_iters musst'

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iters,t0=5,t1=50):

            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)

            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)  #把m以内的数字乱序处理
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter*m+i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        init_iters = np.random.randn(X_b.shape[1])
        self._theta = sgd(X_b, y_train, init_iters, n_iters=len(X_train))
        self.interception_ = self._theta[0]
        self.coef_= self._theta[1:]

    def predict(self,X_predict):
        assert self.interception_ is not None and self.coef_ is not None ,'must fit_normal before predict'
        assert X_predict.shape[1] == len(self.coef_),'the size of X_predict must be equal to X_train'

        X_b = np.hstack(([np.ones((len(X_predict),1)),X_predict]))
        return X_b.dot(self._theta)

    def score(self,X_test,y_test):
        y_predict = self.predict(X_test)
        r_square = 1 - mean_squared_error(y_predict,y_test)/np.var(y_test)
        return r_square

    def __repr__(self):
        return "LinearRegression()"




if __name__ == "__main__":
    # np.random.seed(666) #设置随机种子


    # x = 2 * np.random.random(size=100)
    # y = x * 3. + 4. +np.random.normal(size=100)
    # X = x.reshape(-1,1)


    # print(gradient_descents(X,y,init_theta,eta))        #[4.02369667 3.00517447]   ->[截距,斜率...]
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X = X[y < 50.0]
    y = y[y < 50.0]
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=666)
    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    X_train_standard = standardScaler.transform(X_train)
    X_test_standard = standardScaler.transform(X_test)
    lin_reg = LinearRegression()
    lin_reg.fit_sgd(X_train_standard,y_train,n_iters=2)
    print(lin_reg.score(X_test_standard,y_test))
    print(lin_reg._theta)
    print(lin_reg.coef_)
    print(lin_reg.interception_)

###############################################    lin_reg.fit_gd(X,y,eta=0.000001,n_iters=1e6)
    # print(X[:10,:])     #发现数据的规模差异太大，缩小eta尝试
    # print(lin_reg.coef_)
    # print(lin_reg._theta)
    # print(lin_reg.score(X_test,y_test))
    # m = 10000
    # X = np.random.normal(size=m)
    # y = 4.0 *X + 3. +np.random.normal(0,3,size=m)       ####这两行不要换行，因为np.normal的原因会产生一个数组
    # X = X.reshape(-1,1)                                 ##############################################
    # X_b = np.hstack([np.ones((len(X),1)),X])









    # a = np.random.randint(1,10,size=5)
    # a = a.reshape(-1,1)
    # b = np.random.randint(2,5,size=[4,5])
    # print(a)
    # print(a.T)
    # print(b)
    # print(b.dot(a))
    # # print(a.T.dot(b))
    # a = np.random.randint(100,size=101)
    # a = a.reshape(-1,1)
    # c = np.zeros(a.shape[1]+1).reshape(1,2) #矩阵相乘每一行乘上每一列
    # print(c)
    # print(a.dot(c))



# a = np.random.randint(2,10,size=[4,5])
# print(a)
# b = np.random.randint(2,4,size=[5,2])
# print(b)
# print(a.dot(b))
