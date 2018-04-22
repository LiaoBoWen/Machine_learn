import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100,2))
X[:,0] = np.random.uniform(0,100,size =100)
X[:,1] = 0.75*X[:,0] + 3. + np.random.normal(0,3,size=100)
plt.scatter(X[:,0],X[:,1])
# plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(X)
X_reduction = pca.transform(X)
X_restore = pca.inverse_transform(X_reduction)
plt.scatter(X_restore[:,0],X_restore[:,1])
plt.show()

#再来一个测试
from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target
print(X.shape)
print(y.shape)
noisy_digits = X + np.random.normal(0,4,size=X.shape)
example_digits = noisy_digits[y==0,:][:10]
for num in range(1,10):
    X_num = noisy_digits[y==num,:][:10]
    example_digits = np.vstack([example_digits,X_num])
print(example_digits.shape)

def plot_digits(data):
    fig , axes =plt.subplots(10,10,figsize=(10,10),
                             subplot_kw={'xticks':[],'yticks':[]},
                             gridspec_kw=dict(hspace=0.1,wspace=0.1))
    for i,ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8),
                  cmap='binary',interpolation='nearest',
                  clim= (0,16))
    plt.show()
plot_digits(example_digits)

pca =PCA(0.5)
pca.fit(example_digits)
print(pca.n_components_)
component = pca.transform(example_digits)
invser = pca.inverse_transform(component)
plot_digits(invser)
