import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from  sklearn.model_selection import train_test_split
from  sklearn.neighbors import KNeighborsClassifier


digits = datasets.load_digits()
# print(digits)   #字典格式
# print(digits.keys())    #查看键
# print(digits.DESCR)
x = digits.data
print(x.shape)
y = digits.target
print(y,len(y))
print(digits.target_names)
# some_digit = x[666]
# some_digit_image = some_digit.reshape(8,-1)
# plt.imshow(some_digit_image,cmap=matplotlib.cm.binary)
# plt.show()
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2) #注意这4个的顺序
mathods = ["uniform",'distance']
best_mathod = None
best_p = None
stp = 0.0
for mathod in mathods:  #调参过程
    for p in range(1,11):
        Knn = KNeighborsClassifier(n_neighbors=3,weights=mathod,p=p)
        Knn.fit(X_train,y_train)
        pred = Knn.predict(X_test)
        temp = np.sum(pred == y_test) / len(y_test)
        if temp>stp:
            stp = temp
            best_mathod = mathod
            best_p = p
print("拟合度:{:.2%}".format(np.sum(pred==y_test)/len(y_test)))
print("最佳p:{}，最佳mathod：{} 最佳拟合度:{:.2%}".format(best_p,best_mathod,stp))
