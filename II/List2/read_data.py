# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import csv
import numpy as np
        
from numpy import genfromtxt
my_data = genfromtxt('winered_train.csv', delimiter=';')
        
with open('winered_train.csv','r') as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = ',')
    data = [data for data in data_iter]
data_array = np.asarray(data, dtype = 'float32')  

