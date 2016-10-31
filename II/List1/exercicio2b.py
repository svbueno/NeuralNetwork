# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:41:07 2016

@author: sbueno
"""

import scipy.io
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#read data
mat = scipy.io.loadmat('dados1.mat')
xyz = np.transpose(mat['x'])

m,n = xyz.shape[0],xyz.shape[1]
#passo de adaptacao
alpha_vet = [0.01, 0.05, 0.09 ]
for alpha in alpha_vet:
    
    #np.random.seed(42)
    #Inicialize o vetor de pesos w e o bias b
    w = np.random.rand(m,1) #distribuicao uniforme [0, 1)
    b = np.ones((1,n))
    erro_evol = []
    while_erro = 1.0
    while while_erro > 0.01:
        yest = np.sign(np.dot(np.transpose(w),xyz)+b)
        erro = np.transpose(mat['desejado'])-yest
        w = w+alpha*np.transpose(np.dot(erro,mat['x']))
        while_erro = np.sum(erro**2)
        erro_evol.append(while_erro) 
    yest = np.sign(np.dot(np.transpose(w),xyz)+b)

    plt.plot(np.asanyarray(erro_evol), label='alpha '+str(alpha))
    plt.xlabel('epoca')
    plt.ylabel('erro')
    plt.legend( loc='best', numpoints = 1 )

############## plot dados #############

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color = [('g'),('b'),('r')]

count = 0
for d in mat['desejado']:

    xs = xyz[0][count]
    ys = xyz[1][count]
    zs = xyz[2][count]
    count = count+1

    ax.scatter(xs, ys, zs, c=color[d], alpha=.4)

#plt.show()

############## plot hiperplano #############

point  = np.array([1, 2, 3])
normal = w

# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)
# create x,y
x = np.linspace(np.floor(np.min(xyz[0]))-2, np.ceil(np.max(xyz[0]))+2, n)
y = np.linspace(np.floor(np.min(xyz[1]))-2, np.ceil(np.max(xyz[1]))+2, n) 
xx, yy = np.meshgrid(x, y)

# calculate corresponding z
z = (-normal[0] * xx - normal[1] * yy - 1) * 1. /normal[2]
# plot the surface
#plt3d = plt.figure().gca(projection='3d')
ax.plot_surface(xx, yy, z, cmap=cm.hot, alpha=0.2)
plt.show()



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
