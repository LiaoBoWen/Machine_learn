import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100,2))
X[:,0] = np.random.uniform(0.,100.,size =100)     #uniform 随机取样
X[:,1] = 0.75 * X[:,0] + 3.0 + np.random.normal(0,10,size=100)


#减去mean ,归零处理
def demean(X):
    return X - np.mean(X,axis=0)
X_demean = demean(X)


#梯度上升法
def f(w,X):
    return np.sum(X.dot(w)**2) / len(X)

def df(w,X):
    return X.T.dot(X.dot(w)) *2 / len(X)

#让w变成单位向量
def direction(w):
    return w / np.linalg.norm(w)    #w除以向量的模得到单位向量

#梯度上升主体部分
def first_component(X,initical_w,eta,n_iter= 1e4,epsilon=1e-8):

    w = direction(initical_w)
    cur_iter = 0

    while cur_iter < n_iter:
        gradient = df(w,X)
        last_w =  w
        w = w + eta * gradient
        w = direction(w)
        if abs(f(w,X) - f(last_w,X)) <epsilon:
            break

        cur_iter += 1
    return w

initicial_w = np.random.random(X.shape[1])
eta = 0.01
w = first_component(X_demean,initicial_w,eta)
print(w)


#去除第一主成分相应的分量
# X2 = np.empty(X.shape)
# for i in range(len(X)):
#     X2[i] = X[i]- X[i].dot(w)* w
print(X.dot(w).shape)           #注意：X.dot(w)的shape是(100,)  X.dot(w)的shape是(100,1)，不一样的
print(X.dot(w).reshape(-1,1).shape)
X2 = X - X.dot(w).reshape(-1,1)* w
w2 = first_component(X2,initicial_w,eta)
# print(w2)
print(w.dot(w2))    #向量w,w2相乘的0，说明X2与X相垂直


def first_n_components(n,X,eta=0.01,n_iters=1e4,epsilon=1e-8):
    X_pca = X.copy()
    X_pca = demean(X_pca)
    res =[]
    for i in range(n):
        initicial_w = np.random.random(X_pca.shape[1])
        w = first_component(X_pca,initicial_w,eta)
        res.append(w)

        X_pca = X_pca - X_pca.dot(w).reshape(-1,1) * w

    return res

print(first_n_components(2,X))





#小草稿
a = np.random.randint(1,20,size =[4,5])
b = np.random.randint(2,10,size=5)
c = a.dot(b).reshape(-1,1)
# print(a)
# print(b)
# print(c)
# print(c.shape)
