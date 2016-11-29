import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,normalize
import csv
import pickle
from random import shuffle

tipovinho = 'red'
#tipovinho = 'white'
########### atributos selecionados em IA004_L2_Q2_c.py #######################
if (tipovinho == 'red'):
    included_cols = [1, 2, 6, 7, 9, 10, 11] #11 eh o y
    ruim = [3,4,5]
    medio = [6]
    bom = [7,8]
elif (tipovinho == 'white'):
    included_cols = [0, 1, 4, 6, 7, 10, 11] #11 eh o y
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
    data_array = np.asarray(data, dtype = 'float32')  
    y = data_array[:,len(included_cols)-1]
    X = data_array[:,:len(included_cols)-1]

    Xnew, Ynew = [],[]
    for classe in range(10):
        ind = np.where(y == classe)
        for i in ind[0]:
            Xnew.append(X[i,:])
            if (classe in ruim):
                Ynew.append([1,0,0])
            elif (classe in medio):
                Ynew.append([0,1,0])
            elif (classe in bom):
                Ynew.append([0,0,1])

    shuffle(Xnew)
    shuffle(Ynew)
    return(np.array(Xnew),np.array(Ynew))

X_train, y_train = get_data('wine'+tipovinho+'_train.csv', included_cols)
X_valid, y_valid = get_data('wine'+tipovinho+'_valid.csv', included_cols)
X_test, y_test = get_data('wine'+tipovinho+'_test.csv', included_cols)

#normaliza
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test  = scaler.transform(X_test)

#metaparametros
nf = X_valid.shape[1]
expoente = np.linspace(-24,25,50)
cespace = np.exp2(expoente)
nn = np.exp2(np.linspace(4,11,8)) #qtd neuronios

ACCotFinal = 0
for n in nn:
    #Inicialize o vetor de pesos w e o bias b
    #np.random.seed(42)

    #treinamento
    for i in range(100):
        w1 = np.random.rand(nf+1,n)*2-1 #distribuicao uniforme [-1, 1)
        b = np.ones((X_train.shape[0],1))
        y0 = np.concatenate((b,X_train),axis=1)
        u = np.dot(y0,w1)
        y1 = np.concatenate((b,np.tanh(u)),axis=1)
        
        H, HT = y1,np.transpose(y1)
        #regularizacao
        I = np.eye(H.shape[1]);
        
        for c in cespace:
            W = np.dot(np.dot(np.linalg.pinv(np.dot(HT,H)+c*I),HT),y_train.reshape(X_train.shape[0],y_train.shape[1]))

            #validacao
            b = np.ones((X_valid.shape[0],1))
            y0 = np.concatenate((b,X_valid),axis=1)
            u = np.dot(y0,w1)
            y1 = np.tanh(np.concatenate((b,u),axis=1))
            y2 = np.dot(y1,W)
            
            y2 = np.argmax(y2, axis=1) #pass only index one per row (bias :/)
            y_validaux = np.argmax(y_valid, axis=1)
            ACC = accuracy_score(y2,y_validaux)
        
            if (ACC > ACCotFinal):
                print('acuracia validacao')
                print(ACC)
                ACCotFinal = ACC
                w1otFinal = w1
                w2otFinal = W
                cotFinal = c
                notFinal = n
    print('neuronios')
    print(n)
print('melhor acuracia validacao')
print(ACCotFinal)
print('regularizaco utilizada')
print(cotFinal)    
print('numero otimo de neuronios')
print(notFinal)

##teste em dados com outras amostras
#b = np.ones((X_test.shape[0],1))
#y0 = np.concatenate((b,X_test),axis=1)
#u = np.dot(y0,w1otFinal)
#y1 = np.tanh(np.concatenate((b,u),axis=1))
#y2 = np.dot(y1,w2otFinal)
#y2 = np.argmax(y2, axis=1) #pass only index one per row (bias :/)
#y_aux = np.argmax(y_test, axis=1)
#ACC = accuracy_score(y2,y_aux)
#print('acuracia teste')
#print(ACCotFinal)
#
#network = {'w1otFinal': w1otFinal, 'w2otFinal': w2otFinal, 'cotFinal': c }
#network_name = 'ELM_weight_'+tipovinho
#with open('/home/sergio/Downloads/RedesNeuraisII_L2Q2/'+network_name + '.pkl', 'wb') as f:
#    pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)
