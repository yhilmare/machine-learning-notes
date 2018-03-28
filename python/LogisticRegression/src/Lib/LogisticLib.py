'''
Created on 2018年3月28日

@author: IL MARE
'''
import numpy as np
from matplotlib import pyplot as plt

def sigmod(intX):
    return 1.0 / (1 + np.exp(-intX))

def gradDescent(dataMatrix, classLabels, maxCycle=10000):#原始梯度下降算法
    dataMatrix = np.matrix(dataMatrix, dtype=np.float)
    classLabels = np.matrix(classLabels, dtype=np.float).transpose()
    alpha = 0.001
    m, n = dataMatrix.shape
    weight = np.ones((n, 1))
    res = np.zeros((maxCycle, n))
    for i in range(maxCycle):
        h = sigmod(dataMatrix * weight)
        error = h - classLabels
        res[i] = weight.transpose()
        weight = weight - alpha * dataMatrix.transpose() * error
    return np.matrix(weight, dtype=np.float), res

def stocGradDescent(dataSetIn, labels, numIter=150):#改进版随机梯度下降算法
    dataSetIn = np.matrix(dataSetIn, dtype=np.float)
    labels = np.matrix(labels, dtype=np.float).transpose()
    m, n = dataSetIn.shape
    weight = np.ones((n, 1))
    res = np.ones((numIter * m, n))
    for j in range(numIter):
        dataIndex = np.random.randint(0, m, m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            h = sigmod(dataSetIn[dataIndex[i]] * weight)
            error = h - labels[dataIndex[i]]
            res[j * m + i] = weight.transpose()
            weight = weight - alpha * dataSetIn[dataIndex[i]].transpose() * error
    return np.matrix(weight, dtype=np.float), res

def plotWeightFig(res, ranges):
    if len(ranges) > 6:
        return None
    fig = plt.figure("Test")
    x = np.arange(0, res.shape[0])
    for i in range(len(ranges)):
        ax = fig.add_subplot(321 + i)
        ax.set_ylabel("w%d" % (ranges[i]))
        ax.plot(x, res[:, ranges[i]])
    plt.show()

def classifyVector(intX, weight):
    intX = np.matrix(intX, dtype=np.float)
    prob = sigmod(float(intX * weight))
    if prob > 0.5:
        return 1
    else:
        return 0