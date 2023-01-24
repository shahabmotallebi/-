from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
np.random.seed(1)
class LOG_reg:
    def __init__(self,n,lr):
        self.n = n
        self.lr = lr
    def transpoz(self,x):
        return np.transpose(x)
    def dot(self,x):
        ob = LOG_reg(self.n,self.lr)
        return np.dot(ob.transpoz(x),x)
    def inv(self,x):
        ob = LOG_reg(self.n,self.lr)
        return np.linalg.inv(ob.dot(x))
    def zarib(self,x,y):
        ob = LOG_reg(self.n,self.lr)
        return np.dot(ob.inv(x),np.dot(ob.transpoz(x),y))
    def pridect(self,x_prid,X_train,y_train):
        ob = LOG_reg(self.n,self.lr)
        return 1/(1+np.exp(-np.dot(x_prid,ob.zarib(X_train,y_train))))
    def fit(self,xtr,ytr):
        ob = LOG_reg(self.n,self.lr)
        teta = ob.zarib(xtr,ytr)
        for i in range(self.n):
            yprid = ob.pridect(xtr,xtr,ytr)
            err = mse(ytr,yprid)
            teta = teta-self.lr*err
        return np.transpose(teta)
    def Prid_data(self,xts):
        ob = LOG_reg(self.n,self.lr)
        return 1/(1+np.exp(-np.dot(xts,ob.fit(xtr,ytr))))
def cir(a,b):
    t = np.linspace(0,2*np.pi,100)
    x = np.cos(t) + a
    y = np.sin(t) + b
    plt.plot(x,y,'.')
    plt.scatter(a,b)#c = np.random.choice(['r','b','c','g','p','k','y'])
centers = [[2,2], [0, 2], [2, 0]]
X, labels_true = make_blobs(
    n_samples=1000, centers=centers, cluster_std=0.4, random_state=0
)
plt.scatter(X[:, 0], X[:, 1],c=labels_true,marker='p',alpha=0.7)
cir(centers[0][0],centers[0][1])
cir(centers[1][0],centers[1][1])
cir(centers[2][0],centers[2][1])
plt.show()
np.shape(X)
xtr,xts,ytr,yts = train_test_split(X,labels_true,train_size=0.7,random_state=0)
model  = LogisticRegression()
model.fit(xtr,ytr)
model.predict([[1,-1]]) 
model = LOG_reg(10,0.1)
model.fit(xtr,ytr)
model.Prid_data(xts)
