import numpy as np
import matplotlib.pyplot as plt

X = np.random.uniform(-5,5,size=100)
y = 0.5 * X**2 + 3*X +10 +np.random.normal(0,4,100)
X = X.reshape(-1,1)

#sklearn 中的多项式回归方法
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)   #degree 最多几次幂
poly.fit(X)
X2 = poly.transform(X)
print(X2.shape)
print(X2[:5])   #[x的0次方，x是我1次方，x的2次方]

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X2,y)
y_pre = lin_reg.predict(X2)
print(y_pre)
print(lin_reg.coef_)
print(lin_reg.intercept_)
plt.plot(np.sort(X,axis=0),y_pre[np.argsort(X,axis=0)],"r")
plt.scatter(X,y)
# plt.show()

X = np.random.randint(2,10,size=[5,2])
poly = PolynomialFeatures(degree=2)
poly.fit(X)
X3 = poly.transform(X)  #变成多项式
print(X3)

#Pipeline           把线性回归，多项式的特征，归一化    合为一体
X = np.random.uniform(-3,3,size=100)
y = 0.5 * X**2 + X + 2 + np.random.normal(0,1,100)
X = X.reshape(-1,1)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
poly_reg = Pipeline([
    ("poly",PolynomialFeatures(degree=2)),
    ("std_scaler",StandardScaler()),
    ("li_reg",LinearRegression())
])

poly_reg.fit(X,y)   #一次进行以上三部
y_predict = poly_reg.predict(X)
print(y_predict)
plt.scatter(X,y)
plt.plot(np.sort(X,axis=0),y_predict[np.argsort(X,axis=0)],color="r")
plt.show()
