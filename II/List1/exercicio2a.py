# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:41:07 2016

@author: sbueno
"""

import scipy.io
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mat = scipy.io.loadmat('dados1.mat')
xyz = np.transpose(mat['x'])

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

plt.show()

