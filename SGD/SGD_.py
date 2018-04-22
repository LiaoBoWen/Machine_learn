from sklearn.linear_model import SGDRegressor       #只能解决线性问题中的梯度下降问题
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y<50]
y = y[y<50]
X_train,X_test,y_train,y_test = train_test_split(X,y)
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform((X_test))
sgd_reg = SGDRegressor(n_iter=100)
sgd_reg.fit(X_train_standard,y_train)
print(sgd_reg.score(X_test_standard,y_test))

