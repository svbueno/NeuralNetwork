import numpy as np
import csv
import itertools
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

def relax_accuracy_score(y,yest):
    
    dif0   = yest-y
    dif1   = np.multiply(dif0,dif0)
    dif2   = np.where(dif1<2,0,1)    
    acc    = 1-np.sum(dif2)/(len(yest))
    
    return acc
    
############################# obtem melhor svm ############################# 
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

    return(X,y)
############################# obtem melhor svm ############################# 
def get_machine(Cs,gammas,X_train,y_train,y_valid,relax,cw):
    accTrainot, accValidot = 0,0
    acc_trainList = []
    acc_validList = []
    for C in Cs:
        for g in gammas:
            clf = svm.SVC(C=C, cache_size=200, class_weight=cw, coef0=0.0,
                decision_function_shape=None, degree=3, gamma=g, kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False).fit(X_train, y_train)
            
            yestTRAIN = clf.predict(X_train)
            yestVALID = clf.predict(X_valid)
           
            if(relax=='true'):
                accTrain = relax_accuracy_score(y_train, yestTRAIN)
                accValid = relax_accuracy_score(y_valid, yestVALID)

            elif(relax=='false'):
                accTrain = accuracy_score(y_train, yestTRAIN)
                accValid = accuracy_score(y_valid, yestVALID)

            
            if accTrain > accTrainot:
                accTrainot = accTrain
                print('acuracia de treinamento')
                print(accTrain)
                #ptrain = clf.get_params()
            if accValid > accValidot:
                accValidot = accValid
                print('acuracia de validacao')
                print(accValid)
                machine = clf
                
            acc_trainList.append(accTrain)
            acc_validList.append(accValid)
                                    
    return (machine, np.asarray(acc_trainList),np.asarray(acc_validList))
###############################################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
###############################################################################

tipovinho = 'red'
tipovinho = 'white'
class_names = ['3','4','5','6','7','8','9']    
#relaxing
relax = 'false'
relax = 'true'
#class_weight
cw = 'balanced'
cw = None

Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
refinamento = np.array([1,1/8,1/4,1/2,2,4,8])

########################### atributos selecionados #######################
if (tipovinho == 'red'):
    included_cols = [1, 2, 6, 7, 9, 10, 11] #11 eh o y
elif (tipovinho == 'white'):
    included_cols = [0, 1, 4, 6, 7, 10, 11] #11 eh o y

#get data
X_train, y_train = get_data('wine'+tipovinho+'_train.csv', included_cols)
X_valid, y_valid = get_data('wine'+tipovinho+'_valid.csv', included_cols)
X_test, y_test = get_data('wine'+tipovinho+'_test.csv', included_cols)

#normalize
X = np.concatenate((X_train,X_valid))
y = np.concatenate((y_train,y_valid))

scaler = StandardScaler()
scaler.fit(X)
X       = scaler.transform(X)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test  = scaler.transform(X_test)

######### trainning ########
machine,acc_train,acc_valid = get_machine(Cs,gammas,X_train,y_train,y_valid,relax,cw)
print( machine.get_params() )
#save machine
joblib.dump(machine, 'svm.pkl') 

#results
classifier = svm.SVC(kernel=machine.kernel, C=machine.C, gamma=machine.gamma)
y_pred = classifier.fit(X_train, y_train).predict(X_valid)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_valid, y_pred)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure(1)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
## Plot normalized confusion matrix
#plt.figure(2)
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')
plt.show()

#plot accuracy
plt.figure(3)
t = np.arange(0,len(acc_train),1)
plt.plot(t, acc_train, 'rs--', label="trainning")
plt.plot(t, acc_valid, 'b*', label="validation")
plt.legend(bbox_to_anchor=(0.05, 1), loc=2, borderaxespad=0.)
plt.show()
   
######### refinement ########
#pvalid = machine.get_params()
#print('refining')
#gammas      = pvalid['gamma']*refinamento
#Cs          = pvalid['C']*refinamento     
#
#machine,acc_train,acc_valid = get_machine(Cs,gammas,X_train,y_train,y_valid,relax,cw)
#print( machine.get_params() )
#
##save machine
#joblib.dump(machine, 'svm_refined.pkl') 
#
##results
#classifier = svm.SVC(kernel=machine.kernel, C=machine.C, gamma=machine.gamma)
#y_pred = classifier.fit(X_train, y_train).predict(X_valid)
## Compute confusion matrix
#cnf_matrix = confusion_matrix(y_valid, y_pred)
#np.set_printoptions(precision=2)
## Plot non-normalized confusion matrix
#plt.figure(4)
#plot_confusion_matrix(cnf_matrix, classes=class_names,
#                      title='Refined Confusion matrix, without normalization')
### Plot normalized confusion matrix
##plt.figure(5)
##plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
##                      title='Refined Normalized confusion matrix')
#plt.show()
#
##plot accuracy
#plt.figure(6)
#t = np.arange(0,len(acc_train),1)
#plt.plot(t, acc_train, 'gs--', label="refined trainning")
#plt.plot(t, acc_valid, 'ms--', label="refinied validation")
#plt.legend(bbox_to_anchor=(0.05, .5), loc=2, borderaxespad=0.)
#plt.show()


print(classification_report(y_valid, y_pred, target_names=class_names))
######################################## teste ################################
#load machine
trainedmachine = joblib.load('svm.pkl') 
yest = trainedmachine.predict(X_test)
if(relax=='true'):
    accTest = relax_accuracy_score(y_test, yest)
elif(relax=='false'):
    accTest = accuracy_score(y_test, yest)
print('acuracia de teste')
print(accTest)


#load machine
trainedmachine = joblib.load('svm_refined.pkl') 
yest = trainedmachine.predict(X_test)
if(relax=='true'):
    accTest = relax_accuracy_score(y_test, yest)
elif(relax=='false'):
    accTest = accuracy_score(y_test, yest)
print('acuracia de teste')
print(accTest)