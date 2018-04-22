import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
X = np.random.uniform(-3,3,size=100)
y = 0.5 * X**2 + X + 2 +np.random.normal(0,1,size=100)
X = X.reshape(-1,1)
plt.scatter(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
poly.fit(X)
X2 = poly.transform(X)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X2,y)
y_pre = lin_reg.predict(X2)
plt.plot(np.sort(X,axis=0),y_pre[np.argsort(X,axis=0)],color="r")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
poly_reg = Pipeline([
    ("poly",PolynomialFeatures(degree=2)),
    ("std_scaler",StandardScaler()),
    ("li_reg",LinearRegression())
])
poly_reg.fit(X,y)
y_pre = poly_reg.predict(X)
plt.scatter(X,y)
plt.plot(np.sort(X,axis=0),y_pre[np.argsort(X,axis=0)])
plt.show()
#未完待续……
