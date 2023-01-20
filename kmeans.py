import PLOT
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
np.random.seed(2)
'''
objects 

'''
ob = PLOT.Plot()
scale = StandardScaler()
Data = load_iris()
# -------------------------
'''
functions

'''
def find_closest_center(x, centroids):
    return np.argmin(np.linalg.norm(x - centroids, axis=1))
def maker(x,k):
    l = [ random.choice(x) for i in range(k)]
    return  np.array(l) 
def kMeans(X, initial_centroids, max_iters):
    K = initial_centroids.shape[0]
    centroids = np.array(initial_centroids)
    cluster_ids = None
    for i in range(max_iters):
        cluster_ids = np.array([find_closest_center(X[i], centroids) for i in range(X.shape[0])])
        for k in range(K):
            centroids[k] = np.mean(X[cluster_ids==k], axis=0)
    return cluster_ids, centroids
# --------------------------------------------------------------------
'''
data set (pre pross)

'''
Data_key = Data.keys()
# print(Data_key) #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
X = np.array(Data['data'])
Y = np.array(Data['target'])
# ------------------------------------------
# print(X[0:10])
# dict_ERRO = {'k3':None,'K5':None,'K7':None}
x = scale.fit_transform(X)
# y = scale.fit_transform(Y)
# print(x[0:10])
xtr,xts,ytr,yts = train_test_split(x,Y,train_size=0.7,random_state=3)
k= [3,5,7]
err_l=[]
y_pred = []
y_PRED = []
for k_cluster in k:
    c1=maker(X,k_cluster)
    max_iter = 100
    my_model = kMeans(X,c1,max_iter)
    skl_model = KMeans(n_clusters=k_cluster)
    y = skl_model.fit_predict(X)
    # print(y)
    # print('.'*60)
    # print(my_model[0])
    y_pred.append(my_model[0])
    y_PRED.append(y)
    err_l.append(msle(y,my_model[0]))
ob.plot(k,err_l,'mean_squared_log_error','k','mean_squared_log_error(skl_model,my_model[0])')
ob.SHOW()
for i in range(len(k)):
    plt.subplot(3,1,i+1)
    plt.scatter(np.array(X[:,0]),np.array(y_pred[i]),c=y_pred[i])
    plt.legend(str(k[i]))
plt.xlabel('x 0 input of iris data ')
plt.show()
'''
 silhouette_score\n
 
'''
se = []
for i in range(3,20):
    skl_model = KMeans(n_clusters=i)
    y = skl_model.fit_predict(X)
    score = silhouette_score(X,y);
    se.append(score)
plt.plot(list(range(3,20)),se,'r-p')
plt.title('silhouette_score(X,y)')
plt.grid()
plt.show()
