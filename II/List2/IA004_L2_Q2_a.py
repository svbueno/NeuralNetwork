import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import csv
from sklearn.preprocessing import StandardScaler
import scipy.io

import random

tipovinho = 'red'
#tipovinho = 'white'
########################### atributos selecionados #######################
if (tipovinho == 'red'):
    included_cols = [1, 2, 6, 7, 9, 10, 11] #11 eh o y
    #included_cols = [1, 9, 10, 11] #11 eh o y
    ruim = [3,4,5]
    medio = [6]
    bom = [7,8]
elif (tipovinho == 'white'):
    included_cols = [0, 1, 4, 6, 7, 10, 11] #11 eh o y
    #included_cols = [ 4, 7, 10, 11] #11 eh o y
    ruim = [3,4,5]
    medio = [6]
    bom = [7,8,9]
   
############################# importa sinais ############################# 
def get_data(filename,included_cols):
    data = []
    with open(filename,'r') as dest_f:
        data_iter = csv.reader(dest_f, 
                               delimiter = ',')
        for row in data_iter:
                content = list(row[i] for i in included_cols)
                data.append(content)
    
    ########## amostragem para os graficos ############
    amostras = int(len(data)/10)
    data = random.sample(data, amostras)
    ########## amostragem para os graficos ############

    data_array = np.asarray(data, dtype = 'float32')  
    y = data_array[:,len(included_cols)-1]
    X = data_array[:,:len(included_cols)-1]

    Xnew, Ynew = [],[]
    for classe in range(10):
        ind = np.where(y == classe)
        for i in ind[0]:
            Xnew.append(X[i,:])
            if (classe in ruim):
                Ynew.append(0)
            elif (classe in medio):
                Ynew.append(1)
            elif (classe in bom):
                Ynew.append(2)

    return(np.array(Xnew),np.array(Ynew))

    
#get data
X_train, y_train = get_data('wine'+tipovinho+'_train.csv', included_cols)
X_valid, y_valid = get_data('wine'+tipovinho+'_valid.csv', included_cols)
X_test, y_test = get_data('wine'+tipovinho+'_test.csv', included_cols)

Xaux = np.concatenate((X_train,X_valid))
X    = np.concatenate((Xaux,X_test))
#Xt   = np.transpose(X)
yaux = np.concatenate((y_train,y_valid))
y    = np.concatenate((yaux,y_test))

##Inicialize o vetor de pesos w e o bias b
#m = X.shape[1]
#n = X.shape[0]
#
##np.random.seed(42)
#w = np.random.rand(m,)*2-1 #distribuicao uniforme [-1, 1)
#b = np.ones((n,))
##passo de adaptacao
#alpha = 0.01
#while_erro = 1.0
#erro_evol = []
#
#count = 0
#while (while_erro > 0.01 ) and (count<10):
#    
#    yest = np.sign(np.dot(X,w)+b)
#    erro = y-yest
#    w    = w+alpha*np.dot(erro,X)
#    while_erro = np.sum(erro**2)
#    
#    erro_evol.append(while_erro/len(yest)) 
#    count = count+1 
#
#yest = np.sign(np.dot(X,w)+b)
#
#plt.plot(np.asanyarray(erro_evol))
#plt.xlabel('epoca')
#plt.ylabel('erro')

############## plot dados #############

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color = [('g'),('b'),('r')]

count = 0
for d in y_train:

    xs = X[count][2]
    ys = X[count][4]
    zs = X[count][5]
    count = count+1

    ax.scatter(xs, ys, zs, c=color[d], alpha=.4)

#plt.show()

############### plot hiperplano #############
#
#point  = np.array([1, 2, 3])
#normal = w
#
## a plane is a*x+b*y+c*z+d=0
## [a,b,c] is the normal. Thus, we have to calculate
## d and we're set
#d = -point.dot(normal)
## create x,y
#x = np.linspace(np.floor(np.min(X[0]))-2, np.ceil(np.max(X[1]))+2, n)
#y = np.linspace(np.floor(np.min(X[1]))-2, np.ceil(np.max(X[0]))+2, n) 
#xx, yy = np.meshgrid(x, y)
#
## calculate corresponding z
#z = (-normal[0] * xx - normal[1] * yy - 1) * 1. /normal[2]
## plot the surface
##plt3d = plt.figure().gca(projection='3d')
#ax.plot_surface(xx, yy, z, cmap=cm.hot, alpha=0.2)
#plt.show()



#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#color = [('g'),('b'),('r')]
#
#count = 0
#for d in mat['desejado']:
#
#    xs = xyz[0][count]
#    ys = xyz[1][count]
#    zs = xyz[2][count]
#    count = count+1
#
#    ax.scatter(xs, ys, zs, c=color[d], alpha=.4)
#
#plt.show()
#
