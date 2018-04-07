'''
Created on 2018年3月4日

@author: IL MARE
'''
import time
from Lib import SVMLib as SVMLib
from Util import DataUtil as DataUtil
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

path=r'G:\机器学习实战源代码\Ch06\testSet.txt'

def BankDataSetTest():
    start = time.clock()
    # dataSet, labelSet = DataUtil.loadDataForSVMOrLRModel("bank-additional", "svm")#正统方法
    dataSet, labelSet = DataUtil.loadTempDataForSVMOrLRModel("bank-addtional-format-svm")
    dataSet, labelSet = DataUtil.underSampling(dataSet, labelSet, 1, -1)
    trainSet, trainLabel, testSet, testLabel = DataUtil.generateTrainSet(dataSet, labelSet)
    kTup = ("lin", 1.3)
    alphas, b = SVMLib.realSMO(trainSet, trainLabel, 0.6, 0.01, kTup, 10)
    errorCount = 0
    sv, svl = SVMLib.getSupportVectorandSupportLabel(trainSet, trainLabel, alphas)
    for data, label in zip(testSet, testLabel):
        predict_label = SVMLib.predictLabel(data, *[sv, svl, alphas, b, kTup])
        if predict_label != label:
            errorCount += 1
    ratio = errorCount / len(testLabel)
    print("the error ratio is %.3f, the correct ratio is %.3f -- %.3fs" % (ratio, 1 - ratio, time.clock() - start))

if __name__ == "__main__":
    try:
        fp = open(path, "r")
        positive = []
        negative = []
        trainSet = []
        labelSet = []
        for line in fp.readlines():
            tmp = line.rstrip().split("\t")
            trainSet.append(tmp[0:2])
            labelSet.append(tmp[-1])
            if tmp[-1] == "-1":
                negative.append(tmp[0:2])
            else:
                positive.append(tmp[0:2])
        positive = np.matrix(positive, dtype=np.float)
        negative = np.matrix(negative, dtype=np.float)
        kTup = ("lin", 1.3)
        alphas, b = SVMLib.realSMO(trainSet, labelSet, 0.9, 0.01, kTup, 10)
        sv, svl = SVMLib.getSupportVectorandSupportLabel(trainSet, labelSet, alphas)
        weight = np.multiply(svl.T, alphas[alphas != 0]) * sv
        fig = plt.figure("SVM")
        ax = fig.add_subplot(111)
        ax.plot(positive[:,0], positive[:,1], 'b_')
        ax.plot(negative[:,0], negative[:,1], "g+")
        x = np.arange(2.5, 7, 1)
        y = -(weight[0, 0] * x + b) / weight[0, 1]
        ax.plot(x, y, 'r')
        ax.plot(sv[:, 0], sv[:, 1], "r.")
        plt.show()
    except Exception as e:
        print(e)
    finally:
        fp.close();
    