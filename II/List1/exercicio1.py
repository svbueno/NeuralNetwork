# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:55:45 2016
@author: sbueno
Neurais II
"""
import numpy as np
import matplotlib.pyplot as plt



x_data = np.array([ 0, 1, 2, 5, 7, 9])
y_data = np.array([ 0, 2, 3, 8, 7, 10])

n =  len(x_data)

A = np.vstack([x_data, np.ones(n)]).T

a, b = np.linalg.lstsq(A, y_data )[0]

print(a,b)

y_est = x_data*a+b

plt.plot(x_data, y_data, 'o', label='Pontos dados', markersize=10)
plt.plot(x_data, y_est, 'r', label='reta encontrada')
plt.legend()
plt.show()

EQM = (1/n)*sum((y_data-y_est)**2)

print(EQM)
