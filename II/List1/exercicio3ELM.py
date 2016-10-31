import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

#importa sinais
mat = scipy.io.loadmat('dados2.mat')
x = mat['ponto'][:,:2]
y = mat['ponto'][:,2].reshape(-1,1)
ntd = x.shape[0]   #qtd dados
nf = x.shape[1]   #qtd features

#metaparametros
n = 10 #qtd neuronios
kf = KFold(ntd, n_folds=3)

MSEot = 42
#validacao cruzada
for train_index, test_index in kf:
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #Inicialize o vetor de pesos w e o bias b
    #np.random.seed(42)
    for i in range(100):
        w1 = np.random.rand(nf+1,n) #distribuicao uniforme [0, 1)
        w2 = np.random.rand(n+1,1) #distribuicao uniforme [0, 1)
    
        b = np.ones((X_train.shape[0],1))
       
        y0 = np.concatenate((b,X_train),axis=1)
        u = np.dot(y0,w1)
        y1 = np.concatenate((b,np.tanh(u)),axis=1)
        #y2 = np.dot(y1,w2)
        
        H, HT = y1,np.transpose(y1)
        ##regularizacao
        expoente = np.linspace(-24,25,50)
        cespace = np.exp2(expoente)
        I = np.eye(H.shape[1]);
        mseot = 1
        for c in cespace:
            W = np.dot(np.dot(np.linalg.pinv(np.dot(HT,H)+c*I),HT),y_train.reshape(X_train.shape[0],1))
            # Calculate the validating accuracy
            yestreg = np.dot(H,W)
            mse = mean_squared_error(y_train, yestreg)
            if mse < mseot:
                mseot = mse
                Wot = W
                cot = c
    #trainning
    print(cot)
    y2 = np.dot(H,Wot)
    
    #prediction
    b = np.ones((X_test.shape[0],1))
    y0 = np.concatenate((b,X_test),axis=1)
    u = np.dot(y0,w1)
    y1 = np.tanh(np.concatenate((b,u),axis=1))
    y2 = np.dot(y1,Wot)
    
    MSE = mean_squared_error(y_test, y2)
    print(MSE)
    
    #salve optimus
    if MSE < MSEot:
        MSEot = MSE
        w1ot = w1
        w2ot = Wot

#teste em dados com outras amostras
mat = scipy.io.loadmat('dados3.mat')
x = mat['ponto'][:,:2]
y = mat['ponto'][:,2].reshape(-1,1)
ntd = x.shape[0]   #qtd dados
nf = x.shape[1]   #qtd features

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
plt.title('mapeamento desejado') 
plt.show()

b = np.ones((x.shape[0],1))
y0 = np.concatenate((b,x),axis=1)
u = np.dot(y0,w1ot)
y1 = np.tanh(np.concatenate((b,u),axis=1))
y2 = np.dot(y1,w2ot)

MSE = mean_squared_error(y, y2)
print(MSE)
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
count = 0
for d in mat['ponto']:
    xs = xyz[0][count]
    ys = xyz[1][count]
    zs = np.float(y2[count])
    count = count+1
    ax.scatter(xs, ys, zs, c=plt.cm.coolwarm(zs), alpha=.4)
plt.title('mapeamento estimado')
plt.show()









