import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100,2))
X[:,0] = np.random.uniform(0.,100.,size =100)     #uniform 随机取样
X[:,1] = 0.75 * X[:,0] + 3.0 + np.random.normal(0,10,size=100)
# plt.scatter(X[:,0],X[:,1])
# plt.show()


#减去mean ,归零处理
def demean(X):
    return X - np.mean(X,axis=0)
X_demean = demean(X)
# plt.scatter(X_demean[:,0],X_demean[:,1])
# plt.show()


#梯度上升法
def f(w,X):
    return np.sum(X.dot(w)**2) / len(X)

def df_math(w,X):
    return X.T.dot(X.dot(w)) *2 / len(X)

#让w变成单位向量
def direction(w):
    return w / np.linalg.norm(w)    #w除以向量的模得到单位向量

#梯度上升主体部分
def gradient_ascent(df,X,initical_w,eta,n_iter= 1e4,epsilon=1e-8):

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

initicial_w = np.random.random(X.shape[1])            #不可以代0向量
eta = 0.001
print(initicial_w)
end_w = gradient_ascent(df_math,X_demean,initicial_w,eta)
print(end_w)
# plt.scatter(X_demean[:,0],X_demean[:,1])
# plt.plot([0,end_w[0]*40],[0,end_w[1]*40],color = 'r')
# plt.show()


#简易测试
X2 = np.empty((100,2))
print(X2.shape)
X2[:,0] = np.random.uniform(0,100,size =100)
X2[:,1] = 0.75 *X2[:,0] + 3
w = np.random.random(X2.shape[1])
eta = 0.001
X2_demean = demean(X2)
plt.scatter(X2_demean[:,0],X2_demean[:,1])
w = gradient_ascent(df_math,X2_demean,w,eta)
print(w)
plt.plot([0,w[0] * 30],[0,w[1] * 30],color='r')
plt.show()
