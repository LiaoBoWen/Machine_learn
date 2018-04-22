import numpy as np
import matplotlib.pyplot as plt
from cmath import *
from collections  import Counter
X = np.random.rand(10,2)*10
print('X:')
print(X)
Y = np.array([0,0,0,0,0,1,1,1,1,1])
print("Y:")
print(Y)
x = np.array([1.22123,3.2141221])
plt.scatter(X[Y==0,0],X[Y==0,1],color='g')
plt.scatter(X[Y==1,0],X[Y==1,1],color="r")
plt.scatter(x[0],x[1],color='b')
distance  =  [sqrt(np.sum((X_-x)**2)) for X_ in X]
print(distance)
K = 6
top = np.argsort(distance)
top_y = [Y[i] for i in top[:K]]
votes = Counter(top_y)  #统计每个元素有多少个
print(votes)
last = votes.most_common(1) #找出票数最多的一个数据
print(last[0][0])
print(last)


import numpy as np
import matplotlib.pyplot as plt

seed = np.random.seed(666)
X = np.linspace(1,10,100,True)
print(X)
print(np.random.choice(X,(5,3)))



