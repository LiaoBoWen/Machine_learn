if __name__ =="   ":
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(666)
    X = np.random.random(size = (1000,10))
    true_theta = np.arange(1,12,dtype = float)
    X_b = np.hstack([np.ones((len(X),1)),X])
    y = X_b.dot(true_theta) + np.random.normal(size = 1000)
    print(X_b.shape)
    print(y.shape)
    print(true_theta)

    def J(theta,X_b,y):
        try:
            return np.sum((y-X_b.dot(theta))**2)
        except:
            return float("inf")

    def dJ_math(theta,X_b,y):
        return X_b.T.dot(X_b.dot(theta)-y) *2 /len(y)

    def dJ_debug(theta,X_b,y,epsilon=0.01):
        res = np.empty(len(theta))
        for i in range(len(theta)):
            theta_1 = theta.copy()
            theta_1[i] += epsilon
            theta_2 = theta.copy()
            theta_2[i] -= epsilon
            res[i] = (J(theta_1,X_b,y) - J(theta_2,X_b,y)) / (2 * epsilon)
        return res

    def gradinent_descent(dJ,X_b,y,initial_theta,eta,n_iter = 1e4,epsilon=0.01):
        theta = initial_theta
        cur_iter = 0

        while cur_iter < n_iter:
            gradient = dJ(theta,X_b,y)
            last_theta = theta
            theta = theta - gradient *eta
            if abs(J(theta,X_b,y) - J(last_theta,X_b,y)) < epsilon:
                break
            cur_iter += 1

        return theta

    X = np.random.random(size = [1000,10])
    X_b = np.hstack([np.ones((len(X),1)),X])
    thetas = np.zeros(X_b.shape[1])
    y = X_b.dot(thetas) + np.random.normal(size= 1000)
    eta = 0.01
    print(X_b.shape)
    print(y)
    print(thetas)
    # theta2 = dJ_math(thetas,X_b,y)  #找到theta值
    # print(theta2)
    theta1 = gradinent_descent(dJ_debug,X_b,y,thetas,eta=0.01)
    print(theta1)

