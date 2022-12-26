import csv
import numpy as np
import pandas as pnd
import pylab
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def sdgFunction(x, y) -> None:

    print("SDG")
    xtrain, xtest, ytrain, ytest1 = train_test_split(x, y, test_size = 0.1)
    sgdc = SGDClassifier(max_iter=450, tol=0.01)
    print(sgdc)
    sgdc.fit(xtrain, ytrain)
    score = sgdc.score(xtrain, ytrain)
    print("Training score: ", score)
    ypred1 = sgdc.predict(xtest)
    cm = confusion_matrix(ytest1, ypred1)
    print(cm)
    cr = classification_report(ytest1, ypred1)
    print(cr)

    iris = load_iris()
    x, y = iris.data, iris.target
    x = scale(x)
    xtrain, xtest, ytrain, ytest2=train_test_split(x, y, test_size=0.1)
    sgdc = SGDClassifier(max_iter=450, tol=0.01)
    print(sgdc)
    sgdc.fit(xtrain, ytrain)
    score = sgdc.score(xtrain, ytrain)
    print("Score: ", score)
    ypred2 = sgdc.predict(xtest)
    cm = confusion_matrix(ytest2, ypred2)
    print(cm)
    cr = classification_report(ytest2, ypred2)
    print(cr)

    plt.subplot(1, 2, 1)
    plt.scatter(ypred1,ytest1,color="green")
    plt.subplot(1, 2, 2)
    plt.scatter(ypred2,ytest2,color="orange")
    plt.show()

def svmFunction(x,y) -> None:

    print("SVM")    
    xtrain, xtest, ytrain, ytest1 = train_test_split(x, y, test_size = 0.1)
    classifier = SVC(kernel='linear')
    classifier.fit(xtrain, ytrain)
    ypred1 = classifier.predict(xtest)
    print("accuracy of the model 1: ")
    print(accuracy_score(ytest1, ypred1))

    iris = load_iris()
    x, y = iris.data, iris.target
    x = scale(x)
    xtrain, xtest, ytrain, ytest2=train_test_split(x, y, test_size=0.1)
    classifier = SVC(kernel='linear')
    classifier.fit(xtrain, ytrain)
    ypred2 = classifier.predict(xtest)
    print("accuracy of the model 2: ")
    print(accuracy_score(ytest2, ypred2))

    plt.subplot(1, 2, 1)
    plt.scatter(ypred1,ytest1,color="green")
    plt.subplot(1, 2, 2)
    plt.scatter(ypred2,ytest2,color="orange")
    plt.show()

def lgFunction(x,y) -> None:
    print("Logistic Regression")
    xtrain, xtest, ytrain, ytest1 = train_test_split(x, y, test_size = 0.1)
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(xtrain, ytrain)
    ypred1 = model.predict(xtest)    
    print("score 1:")
    print(model.score(xtrain, ytrain))
    print(classification_report(ytest1, ypred1))

    xtrain, xtest, ytrain, ytest2 = train_test_split(x, y, test_size = 0.1)
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(xtrain, ytrain)
    ypred2 = model.predict(xtest)    
    print("score 2:")
    print(model.score(xtrain, ytrain))
    print(classification_report(ytest2, ypred2))

    plt.subplot(1, 2, 1)
    plt.scatter(ypred1,ytest1,color="green")
    plt.subplot(1, 2, 2)
    plt.scatter(ypred2,ytest2,color="orange")
    plt.show()


dataset = pnd.read_csv('nutrition_values.csv', delimiter='\t', lineterminator='\r' )
dataset = dataset.drop(['Chain','Item','Type'], axis=1)
dataset = dataset.head(450)

print ("Столбец Калории")
column = dataset['Calories']
column = column.head(450)
print(column)

dataset = dataset.iloc[:,:2].values

sdgFunction(dataset, column)
svmFunction(dataset, column)
lgFunction(dataset, column)


    





