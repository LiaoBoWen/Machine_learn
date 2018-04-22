import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import datasets


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
    reg = LinearRegression()
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X = X[y<50.0]
    y = y[y<50.0]
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=666)
    reg.fit_normal(X_train,y_train)
    # print(reg.coef_)
    # print(reg.interception_)
    score = reg.score(X_test,y_test)
    print(score)
