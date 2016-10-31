import numpy as np
import scipy.io
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, max_it, learning_rate, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        self.epochs = max_it
        self.learning_rate = learning_rate
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X_train, X_test, y_train, y_test):

        ones = np.atleast_2d(np.ones(X_train.shape[0]))
        X = np.concatenate((ones.T, X_train), axis=1)
        
        MSEval = []
        MSEtrain = []
        for k in range(self.epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = y_train[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))
            # reverse
            deltas.reverse()
            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += self.learning_rate * layer.T.dot(delta)

            #if ((k % 100) == 0): 
             #   print ('epochs:', k)
                
            y2 = []
            for e in X_train:
                pred = self.predict(e)
                y2.append(pred)
            MSE = mean_squared_error(np.asanyarray(y2), y_train)
            MSEtrain.append(MSE)
                
            y2 = []
            for e in X_test:
                pred = self.predict(e)
                y2.append(pred)
            MSE = mean_squared_error(np.asanyarray(y2), y_test)  
            MSEval.append(MSE)
             
            MSEot = 42
            #salve optimus
            if MSE < MSEot:
                MSEot = MSE
                self.weightsopt = self.weights
                
        return self.weightsopt, MSEtrain, MSEval, MSEot

    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':

    #importa sinais
    mat = scipy.io.loadmat('dados2.mat')
    
    x = mat['ponto'][:,:2]
    y = tanh(mat['ponto'][:,2]).reshape(-1,1)
    ntd = x.shape[0]   #qtd dados
    nf = x.shape[1]   #qtd features
    
    # embaralha dados
    c = list(zip(x, y))
    random.shuffle(c)
    x, y = zip(*c)
    X = np.asanyarray(x)
    y = np.asanyarray(y)

    #metaparametros
    n = 32 #qtd neuronios
    alpha = 0.01
    max_it = 5000
    min_err = 0.01
    kf = KFold(ntd, n_folds=3)
    i = 1
    MSEot = 42
    nn = NeuralNetwork([nf,n,1],max_it,alpha)

    #validacao cruzada
    for train_index, test_index in kf:
    
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
     
        weights, MSEtrain, MSEval, MSE = nn.fit(X_train, X_test, y_train, y_test)

        plt.figure(1)
        plt.subplot(3,1,i)
        line1, = plt.plot(np.asanyarray(MSEtrain), marker='*', linestyle='--', color='r', label="treinamento")
        line2, = plt.plot(np.asanyarray(MSEval), marker='x', linestyle='-', color='b', label="validacao")
        plt.legend(handles=[line1, line2])
        plt.title('erros')  
        i = i+1

        #salve optimus
        if MSE < MSEot:
            MSEot = MSE
            weightsopt = weights
        print(MSEot)
        
    #teste em dados com outras amostras
    #importa sinais
    mat = scipy.io.loadmat('dados3.mat')
    x = mat['ponto'][:,:2]
    y = tanh(mat['ponto'][:,2]).reshape(-1,1)
    xyz = np.transpose(mat['ponto'])
    
    fig = plt.figure(2)
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
    y2 = []
    for e in x:
        pred = nn.predict(e)
        y2.append(pred)
    MSE = mean_squared_error(np.asanyarray(y2), y)  
    print(MSE)

    fig = plt.figure(3)
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