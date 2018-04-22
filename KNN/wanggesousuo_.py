# #
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import datasets
#
# digits = datasets.load_digits()
# X = digits.data
# y = digits.target
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=666)
#
#
# #网格搜索部分
# param_grid = [
#     {
#         "weights":["uniform"],
#         "n_neighbors":[i for i in range(1,11)]
#     },
#     {
#         "weights":["distance"],
#         "n_neighbors":[i for i in range(1,11)],
#         "p":[i for i in range(1,6)]
#     }
# ]
# knn = KNeighborsClassifier()
# grid_search = GridSearchCV(knn,param_grid,verbose=2)    #网格搜索  (分类器,搜索列表,n_jobs=并行个数,verbose=输出详细程度)
# grid_search.fit(X_train,y_train)
# bests = grid_search.best_estimator_
# bestli = grid_search.best_params_
# print(bests,bestli)
# knn = grid_search.best_estimator_   #返回网格搜索到的最佳结果
# pred = knn.predict(X_test)
# print(pred)
# print(grid_search.best_score_)  #最高的拟合度
# print(knn.score(X_test,y_test))#查看准确率


#####特别奇怪，两个的时间花费不同




from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import GridSearchCV    #网格搜索模块         交叉验证
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.2,random_state=666)
# knn = KNeighborsClassifier(n_neighbors=4,weights="uniform")
# knn.fit(X_train,y_train)
# knn.score(X_test,y_test)    #拟合度方法

param_grid = [
    {
        "weights":["uniform"],
        "n_neighbors":[i for i in range(1,11)]
    },
    {
        "weights":["distance"],
        "n_neighbors":[i for i in range(1,11)],
        "p":[i for i in range(1,6)]
    }
]


knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn,param_grid,verbose=2)
grid_search.fit(X_train,y_train)
bests = grid_search.best_estimator_
best_li = grid_search.best_params_
print(bests)
print(best_li)
knn = grid_search.best_estimator_
predict = knn.predict(X_test)
print(predict)
print(knn.score(X_test,y_test))



