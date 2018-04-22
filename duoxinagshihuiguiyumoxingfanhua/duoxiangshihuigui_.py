import numpy as np
import matplotlib.pyplot as plt

X = np.random.uniform(-3,3,size=100)
y = 0.5 * X**2 + X +2 + np.random.normal(0,1,size=100)
X = X.reshape(-1,1)
print(X)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
print(X.shape,y.shape)
lin_reg.fit(X,y)
y_pre = lin_reg.predict(X)

X2 = np.hstack([X,X**2])
lin_reg2 = LinearRegression()
lin_reg2.fit(X2,y)
y_pre2 = lin_reg2.predict(X2)
print(lin_reg2.coef_)   #系数

print(lin_reg2.intercept_)  #截距

print(X)
m = np.random.randint(1,10,size=12)
plt.plot(np.sort(X,axis=0),y_pre2[np.argsort(X,axis=0)],color ="r")       #这里要注意，列向量的arg，和物理使用axis
plt.scatter(X,y_pre)
plt.scatter(X,y)
plt.show()
