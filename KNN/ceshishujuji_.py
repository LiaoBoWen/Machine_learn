import numpy as np
from sklearn import datasets
from scikit_learn实现knn import KNN
from sklearn.model_selection import train_test_split    #顾名思义

iris = datasets.load_iris()
X = iris.data
y = iris.target
            # print(x.shape,y.shape)
            # print(x,y)


def train_test_split(X,y,test_ratio=0.2,seed=None):
    assert X.shape[0]==y.shape[0], \
        "the size of x must be equal to size of y"
    assert 0<=test_ratio<=1, \
        "test_retio must be valid"
    if seed:
        np.random.seed(seed)

    shuffle_index = np.random.permutation(len(X))
    test_size = int(len(X)*test_ratio)
    test_index = shuffle_index[:test_size]
    train_index = shuffle_index[:test_size]

    X_train = X[train_index]
    y_train = y[train_index]

    X_test = X[test_index]
    y_test = y[test_index]

    return X_train,X_test,y_train,y_test

    #np.random.shuffle(y)    #打乱y数组的方法
    #print(y)


if __name__ == "__main__":
    # x_train, y_train, x_test, y_test = train_test_split(X,y,test_size=0.2,random_state=666) #这是模块当中的用法           random_state是随机种子
    x_train,  x_test, y_train, y_test = train_test_split(X,y,test_ratio=0.2,seed=666) #这是模块当中的用法           random_state是随机种子

    knn = KNN(4)
    knn.fit(x_train,y_train)
    pred = knn.predict(x_test)
    print(pred)
    print(y_test)
    print("拟合度：{:.2%}".format(np.sum(pred==y_test)/len(y_test)))
