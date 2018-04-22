#背景原因：
# 可能会被其中某一个元素主导
#解决方法：
#1、 吧所有的数据映射到0-1上
#适用于有明显边界的情况缺点：受边界的影响比较大
#2、吧数据归一到均指为0方差为1的分布中
#适用于没有特别边界性的数据，但像分数成绩，图像像素之类有明显边界的不适合
import numpy as np
import  matplotlib.pyplot as plt

#解决方法1：
x = np.random.randint(0,100,size=100)
print(x)
a = (x - np.min(x))/(np.max(x)-np.min(x))
print(a)

x = np.random.randint(0,100,size=[50,2])
x = np.array(x,dtype=float)
print(x)
x[:,0] = (x[:,0]-np.min(x[:,0]))/(np.max(x[:,0])-np.min(x[:,0]))
x[:,1] = (x[:,1]-np.min(x[:,1]))/(np.max(x[:,1])-np.min(x[:,1]))
print(x[:10])
print(x[:10]==x[:10,:])
# plt.scatter(x[:,0],x[:,1])
# plt.show()
print(np.mean(x[:,0]))#查看均值
print(np.std(x[:,1]))#查看方差

#解决方法2：
x2 = np.random.randint(0,100,size=[50,2])
x2 = np.array(x2,dtype= float)
x2[:,0] = (x2[:,0]-np.mean(x2[:,0]))/np.std(x2[:,0])
x2[:,1] = (x2[:,1]-np.mean(x2[:,1]))/np.std(x2[:,1])
plt.scatter(x2[:,0],x2[:,1])
plt.show()
print(np.mean(x2))
print(np.std(x2))
print(np.mean(x2[:,0]))
print(np.std(x2[:,0]))
