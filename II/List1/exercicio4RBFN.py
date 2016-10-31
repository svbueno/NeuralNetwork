import scipy.io
import numpy as np

from scipy import *
from scipy.linalg import norm, pinv
from scipy.cluster.vq import vq, kmeans

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

class RBF:
     
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.beta = 8
        self.W = np.random.random((self.numCenters, self.outdim))
         
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return np.exp(-self.beta * norm(c-d)**2)
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
     
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
         
        # choose random center vectors from training set
        # by k-means
        self.centers = kmeans(X,self.numCenters)[0] 
        # by random
        #rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        #self.centers = [X[i,:] for i in rnd_idx]
        
        #print ("center", self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        #print (G)
         
        # calculate output weights (pseudoinverse)
        self.W = np.dot(pinv(G), Y)
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """
         
        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y
 
      
if __name__ == '__main__':
    
    #mostra dados originais
    mat = scipy.io.loadmat('dados_map.mat')
    xyz = np.transpose(mat['dados_rbf'])
    ############## plot dados #############
    fig = plt.figure(11)
    ax = fig.add_subplot(111, projection='3d')
        
    count = 0
    for d in mat['dados_rbf']:
        xs = xyz[0][count]
        ys = xyz[1][count]
        zs = xyz[2][count]
        count = count+1
        ax.scatter(xs, ys, zs, c=plt.cm.coolwarm(zs), alpha=.4)
    plt.show()
         
    #treino
    xy = mat['dados_rbf'][:,:2]
    z = mat['dados_rbf'][:,2].reshape(-1,1)
    ntd = xy.shape[0]   #qtd dados
    #nf = x.shape[1]   #qtd features
    
    #metaparametros
    n = 8 #qtd neuronios
    kf = KFold(ntd, n_folds=3)
    
    i = 1
    #validacao cruzada
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = xy[train_index], xy[test_index]
        y_train, y_test = z[train_index], z[test_index]    
    
        # rbf regression
        rbf = RBF(2, n, 1)
        rbf.train(X_train, y_train)
        zest = rbf.test(X_test)
        
            
        MSE = mean_squared_error(y_test, zest)
        print(MSE)

        
        X = np.transpose(X_test)
        #mostra dados estimados
        fig = plt.figure(i)
        ax = fig.add_subplot(111, projection='3d')
        count = 0
        for d in zest:
            xs = X[0][count]
            ys = X[1][count]
            zs = zest[count]
            count = count+1
            ax.scatter(xs, ys, zs, c=plt.cm.coolwarm(zs), alpha=.4)
        plt.show()
        
        #mostra dados estimados
        fig = plt.figure(i+3)
        ax = fig.add_subplot(111, projection='3d')
        count = 0
        for d in zest:
            xs = X[0][count]
            ys = X[1][count]
            zs = y_test[count]
            count = count+1
            ax.scatter(xs, ys, zs, c=plt.cm.coolwarm(zs), alpha=.4)
        plt.show()
        i = i+1
    
    # rbf regression
    rbf = RBF(2, n, 1)
    rbf.train(xy, z)
    zest = rbf.test(xy)
    
    #mostra dados estimados
    fig = plt.figure(12)
    ax = fig.add_subplot(111, projection='3d')
    count = 0
    for d in zest:
        xs = xyz[0][count]
        ys = xyz[1][count]
        zs = zest[count]
        count = count+1
        ax.scatter(xs, ys, zs, c=plt.cm.coolwarm(zs), alpha=.4)
    plt.show()
    
     
