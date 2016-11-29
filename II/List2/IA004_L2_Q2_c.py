import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

tipovinho = 'red'
#tipovinho = 'white'      
with open('wine'+tipovinho+'_train.csv','r') as dest_f:
    data_iter = csv.reader(dest_f, 
                           delimiter = ',')
    data = [data for data in data_iter]
data_array = np.asarray(data, dtype = 'float32')  

#Normalizar os dados
saida = data_array[:,-1]
X = data_array[:,:-1]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#Pearson
pearson = dict()
name_list = ['acidez_fixa',
  'acidez_volatil',
  'acido_citrico',
  'açucar_residual',
  'cloretos',
  'dioxido_de_enxofre_livre',
  'dioxido_de_enxofre_total',
  'densidade',
  'pH',
  'sulfatos',
  'alcool'
  ]

atributos = []
for attribute in range(X.shape[1]):
    P = np.corrcoef(X[:,attribute], saida, rowvar=0)[0, 1]
    modP = (P*P)**(1/2)
    pearson[str(name_list[attribute])] = modP 
    #P, p_value = pearsonr(data_array_norm[:,i], saida)
    atributos.append(modP )
indices_crescentes = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(atributos))]
              
for key, value in pearson.items():
  print(key, value)
                      