import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":          #由于 n_jobs的干扰，000000000
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    X = X[y<50.0]
    y = y[y<50.0]

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=666)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train,y_train)
    print(lin_reg.coef_)
    print(lin_reg.intercept_)
    print(lin_reg.score(X_test,y_test))


    knn_reg = KNeighborsRegressor()
    print("=========================================================================================")
    print(X_train.shape,y_train.shape)
    knn_reg.fit(X_train,y_train)
    knn_reg_score = knn_reg.score(X_test,y_test)
    print(knn_reg.predict(X_test))
    print(knn_reg_score)
    print("==========================")

grid = [
    {
        "weights":['uniform'],
        "n_neighbors":[i for i in range(1,11)]
    },
    {
        "weights":["distance"],
        "n_neighbors":[i for i in range(1,11)],
        'p':[i for i in range(1,6)]
    }
]
if __name__ == "__main__":
    knn_reg = KNeighborsRegressor()
    grid_search = GridSearchCV(knn_reg,grid,n_jobs=-1,verbose=1)
    grid_search.fit(X_train,y_train)
    print(grid_search.best_estimator_)
    knn_reg = grid_search.best_estimator_
    print(knn_reg.score(X_test,y_test))
