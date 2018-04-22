from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
# from 高位数据向低微数据的映射 import PCA


digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=666)
print(X_train.shape)

#knn算法
from sklearn.neighbors import KNeighborsClassifier
knn_cif = KNeighborsClassifier()
knn_cif.fit(X_train,y_train)
print(knn_cif.score(X_test,y_test))


#PCA算法
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
knn_cif = KNeighborsClassifier()
knn_cif.fit(X_train_reduction,y_train)
print(knn_cif.score(X_test_reduction,y_test))   #发现降维的话精度太低了
print(pca.explained_variance_ratio_)    #PCA解释的方差相应的的比例

#选择降到最合适的维度
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
print(pca.explained_variance_ratio_)        #通过观察数据来观察可以去除的维度
plt.plot([i for i in range(X.shape[1])],[np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
# plt.show()

#选择可以解释原来95%以上的方差
pca = PCA(0.95)
pca.fit(X_train)
print(pca.n_components_)    #查看降维后的维数
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
knn_cif =  KNeighborsClassifier()
knn_cif.fit(X_train_reduction,y_train)
print(knn_cif.score(X_test_reduction,y_test))

#降维到2维的意义：可视化
pca = PCA(n_components=2)
pca.fit(X)
X_reduction = pca.transform(X)
for i in range(10):
    plt.scatter(X_reduction[y == i,0],X_reduction[y==i,1],alpha=0.8)
plt.show()
