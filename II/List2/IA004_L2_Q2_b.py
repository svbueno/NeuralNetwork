# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import csv
from sklearn.model_selection import train_test_split

#"fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"
winered = []
with open('winered.csv', 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in spamreader:
        winered.append(row)
        #print (', '.join(row))

train_aux, test = train_test_split(winered, test_size = 0.2)
train, valid = train_test_split(train_aux, test_size = 0.2)

with open('winered_train.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(train)
    
with open('winered_test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(test)

with open('winered_valid.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(valid)
    
#"fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"
winewhite = []
with open('winewhite.csv', 'rt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in spamreader:
        winewhite.append(row)
        #print (', '.join(row))

train_aux, test = train_test_split(winewhite, test_size = 0.2)
train, valid = train_test_split(train_aux, test_size = 0.2)

with open('winewhite_train.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(train)
    
with open('winewhite_test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(test)

with open('winewhite_valid.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(valid)
    