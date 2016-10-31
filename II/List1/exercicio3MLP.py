import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

#importa sinais
mat = scipy.io.loadmat('dados2.mat')

x = mat['ponto'][:,:2]
y = mat['ponto'][:,2].reshape(-1,1)
y = np.tanh(y)
ntd = x.shape[0]   #qtd dados
nf = x.shape[1]   #qtd features

############## plot dados #############
#mostra dados originais
xyz = np.transpose(mat['ponto'])
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
count = 0
for d in mat['ponto']:
    xs = xyz[0][count]
    ys = xyz[1][count]
    zs = xyz[2][count]
    count = count+1
    ax.scatter(xs, ys, zs, c=plt.cm.coolwarm(zs), alpha=.4)
plt.title('mapeamento dados originais desejado')
plt.show()

# embaralha dados
c = list(zip(x, y))
random.shuffle(c)
x, y = zip(*c)
x = np.asanyarray(x)
y = np.asanyarray(y)

#metaparametros
n = 16 #qtd neuronios
alpha = 0.1
max_it = 500
min_err = 0.01
kf = KFold(ntd, n_folds=3)
i = 1

#validacao cruzada
for train_index, test_index in kf:

    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
 
    #Inicialize o vetor de pesos w e o bias b
    #np.random.seed(42)
    w1 = np.random.rand(nf+1,n) #distribuicao uniforme [0, 1)
    w2 = np.random.rand(n+1,1) #distribuicao uniforme [0, 1)

    bt = np.ones((X_train.shape[0],1)) #bias trainning
    bp = np.ones((X_test.shape[0],1)) #bias prediction
    b1,b2 = 1,1
    
    t = 0
    MSE = 1
    MSEot = 42
    MSEval, MSEtrain = [],[]
    
    while t < max_it and MSE > min_err:
       
        #foward
        y0 = np.concatenate((bt*b1,X_train),axis=1)
        u = np.dot(y0,w1)
        y1 = np.concatenate((bt*b2,np.tanh(u)),axis=1)
        y2 = np.tanh(np.dot(y1,w2))
        MSEtrain.append(mean_squared_error(y_train, y2))
           
        #backpropagation
        dEdOut = -1*(y_train-y2)
        du = np.tanh(y2)
        dOutdNet = np.asmatrix(1-np.multiply(du,du))
        delta2 = np.transpose(np.multiply(dEdOut,dOutdNet))
        dNetdW2 = y1
        
        dEdOut1 = w2[1:]*delta2
        du = np.tanh(y1[:,1:])
        dOutdNet1 = np.asmatrix(1-np.multiply(du,du))
        delta1 = np.multiply(dEdOut1,np.transpose(dOutdNet1))
        dNetdW1 = y0
        
        #atualization
        w1 = w1 - alpha*np.transpose(sum(delta1[:,i]*y0[i] for i in range(X_train.shape[0])))/X_train.shape[0]
        w2 = w2 - alpha*np.transpose(delta2*dNetdW2)/X_train.shape[0]
        b1 = b1 - alpha*np.mean(np.sum(sum(delta1[:,i] for i in range(X_train.shape[0])))/X_train.shape[0])
        b2 = b2 - alpha*np.mean(np.sum(np.transpose(delta2)))
        
        #validation
        y0 = np.concatenate((bp*b1,X_test),axis=1)
        u = np.dot(y0,w1)
        y1 = np.tanh(np.concatenate((bp*b2,u),axis=1))
        y2 = np.tanh(np.dot(y1,w2))
        
        #salve optimus
        MSE = mean_squared_error(y_test, y2)
        MSEval.append(MSE)
        if MSE < MSEot:
            MSEot = MSE
            w1ot = w1
            w2ot = w2
            b1ot = b1
            b2ot = b2
        t = t+1

    print(MSEot)
  
    plt.figure(2)
    plt.subplot(3,1,i)
    plt.plot(np.asanyarray(MSEval), marker='*', linestyle='--', color='r', label="validacao")
    plt.title('erros de validacao')  
    i = i+1
    
#mostra dados com outras amostras
#importa sinais
mat = scipy.io.loadmat('dados3.mat')
x = mat['ponto'][:,:2]
y = mat['ponto'][:,2].reshape(-1,1)
y = np.tanh(y)
ntd = x.shape[0]   #qtd dados
nf = x.shape[1]   #qtd features

xyz = np.transpose(mat['ponto'])
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
count = 0
for d in mat['ponto']:
    xs = xyz[0][count]
    ys = xyz[1][count]
    zs = xyz[2][count]
    count = count+1
    ax.scatter(xs, ys, zs, c=plt.cm.coolwarm(zs), alpha=.4)
plt.title('mapeamento dados3 desejado')
plt.show()

b = np.ones((x.shape[0],1))
y0 = np.concatenate((b1ot*b,x),axis=1)
u = np.dot(y0,w1ot)
y1 = np.tanh(np.concatenate((b2ot*b,u),axis=1))
y2 = np.tanh(np.dot(y1,w2ot))

MSE = mean_squared_error(y, y2)
print(MSE)
fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
count = 0
for d in mat['ponto']:
    xs = xyz[0][count]
    ys = xyz[1][count]
    zs = np.float(y2[count])
    count = count+1
    ax.scatter(xs, ys, zs, c=plt.cm.coolwarm(zs), alpha=.4)
plt.title('mapeamento dados3 estimado')
plt.show()

#
#
#
#
