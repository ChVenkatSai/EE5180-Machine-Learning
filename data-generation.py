import random
import numpy as np
import math

Xtrain = []
ytrain = []
Xtest = []
ytest = []
XTrain = []
yTrain = []


def func(rad):
    Xtrain = []
    ytrain = []
    Xtest = []
    ytest = []
    XTrain = []
    yTrain = []
    num = 1000
    for i in range (num):
        #Muller's method: randomly sampling from d-ball
        arr = np.random.normal(0,1,30)
        norm = np.sum(arr**2)**(0.5)
        r = random.random()**(1.0/30)
        val = r*arr/norm
        elem = []

        #positive or negative point with prob 0.5
        prob = 0.5
        result = np.random.binomial(1,prob)

        #centre for positive ball is (1,1,..)
        #centre for negative ball is (-1,-1,..)
        for x in val:

            if (result == 1):
                x = (x*rad) + 1

            else:
                x = (x*rad) - 1
                result = -1

            elem.append(x)

        yTrain.append(result)
        XTrain.append(elem)

    val = 700

    #Training data
    for k in range(val):
        Xtrain.append(XTrain[k])
        ytrain.append(yTrain[k])

    #Testing data
    for j in range(k,num,1):
        Xtest.append(XTrain[j])
        ytest.append(yTrain[j])

    return (Xtrain,ytrain,Xtest,ytest)

