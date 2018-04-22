import numpy as np

class PCA:
    def __init__(self,n_components):
        """初始化PCA"""
        assert n_components >= 1,\
        'n_component must be valid'
        self.n_components = n_components
        self.components_ = None

    def fit(self,X,eta=0.01,n_iters=1e4):
        assert self.n_components <= X.shape[1],\
        'n_components must not be greater than the feature number of X'
        def demean(X):
            return X - np.mean(X,axis=0)

        def f(w,X):
            return np.sum(X.dot(w)**2) /len(X)

        def df(w,X):
            return X.T.dot(X.dot(w)) *2/len(X)

        def direction(X):
            return X / np.linalg.norm(X)

        def first_component(X,initicial_w,eta=0.01,n_iters=1e4,epsilon=1e-8):
            w = direction(X)
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = df(w,X)
                last_w = w

                w = direction(X)
                if abs(f(w,X) - f(last_w,X)) < epsilon:
                    break
                cur_iter += 1
            return w

        def first_n_components(n,X,eta):
            X_pca = X.copy()
            X_pca = demean(X_pca)
            self.components_ = np.empty(self.n_components,X.shape[1])
            for i in range(n):
                initicial_w = np.random.random(X_pca.shape[1])
                w = first_component(X,initicial_w,eta)
                self.components_[i,:] = w
                X_pca = X_pca - X_pca.dot(w).reshape(-1.1) * w
        first_n_components(self.n_components,X,eta)
        return self

    def __repr__(self):
        return "PCA(n_components=%d"%self.n_components

if __name__ == " __main__":
    X = np.empty((100,2))
    X[:,0] = np.random.uniform(0.,100.,size =100)     #uniform 随机取样
    X[:,1] = 0.75 * X[:,0] + 3.0 + np.random.normal(0,10,size=100)
    pca = PCA(2)
    pca.fit(X)
