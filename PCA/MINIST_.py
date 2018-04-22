import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original")  #下载数据MNIST集
print(mnist)

X,y = mnist['data'],mnist['target']
print(X.shape)

#数据本来就分开来的，所i有不用train_test_split,同时由于存放的是int类型，所以先转换成float类型
X_train = np.array(X[:60000],dtype=float)
X_test = np.array(X[60000:],dtype=float)
y_train = np.array(y[:60000],dtype=float)
y_test = np.array(y[60000:],dtype=float)


#用knn 来做对比      由于对于图片的像素点来说，他的尺度在同一个层面上，所以不需要归一化
from sklearn.neighbors import KNeighborsClassifier
# knn_cif = KNeighborsClassifier()
# knn_cif.fit(X_train,y_train)      #相当的耗时
# print(knn_cif.score(X_test,y_test))   #更加耗时  0.9688


pca = PCA(0.9)
pca.fit(X_train)    #由于32位的python只能容纳2G的内存，所以此处回报memorryerror的错，可以忽略
X_train_reduction = pca.transform(X_train)  #降维
X_test_reduction= pca.transform(X_test)
print(pca.n_components_)

knn_cif = KNeighborsClassifier()
knn_cif.fit(X_train_reduction,y_train)
print(knn_cif.score(X_test_reduction,y_test))       #0.9728  精度提高了，由于降噪的原因，但并不是所有的降维都会提高精度8
