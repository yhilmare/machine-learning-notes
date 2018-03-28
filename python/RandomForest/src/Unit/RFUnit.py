'''
Created on 2018年3月4日

@author: IL MARE
'''
import Util.DataUtil as DataUtil
import Lib.RFLib as RFLib
import time
from matplotlib import pyplot as plt
import numpy as np

def loadDataSet(filename):
    print("Loading data...")
    dataSet, labelSet = DataUtil.loadDataForRMOrDTModel(filename)
    print("Loaded data!")
    print("Undersampling data...")
    dataSet, labelSet = DataUtil.underSampling(dataSet, labelSet, "yes", "no")
    print("Undersampled data!")
    return dataSet, labelSet

def testRFModel(dataSet, labelSet, T=20):
    trainSet, trainLabel, testSet, testLabel = DataUtil.generateTrainSet(dataSet, labelSet)
    forest = RFLib.generateRandomForest(trainSet, trainLabel, T)
    errorCount = 0
    for data, label in zip(testSet, testLabel):
        predict_label = RFLib.predictByRandomForest(forest, data)
        if predict_label != label:
            errorCount += 1
    RFratio = float(errorCount) / len(testLabel)
    print("RF:total error ratio is %.3f, correct ratio is %.3f" % (RFratio, 1 - RFratio))
    return RFratio

if __name__ == "__main__":
    start = time.clock()
    dataSet, labelSet = loadDataSet("bank-additional")
    tmp_lst = []
    for T in range(20, 0, -1):
        totalError = 0
        errorList = []
        for i in range(5):
            errorRatio = testRFModel(dataSet, labelSet, T)
            errorList.append("%.3f" % (1 - errorRatio))
            totalError += errorRatio
        print(errorList, "%.3f -- %.3fs" % (1 - totalError / 5.0, time.clock() - start))
        tmp_lst.append((T, errorList, 1 - totalError / 5.0))
    for item in tmp_lst:
        print(item)
    y = np.array([item[2] for item in tmp_lst], dtype=np.float)
    x = np.arange(y.shape[0] + 1, 1, -1)
    fig = plt.figure("test")
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_ylabel("correct ratio of RF")
    ax.set_xlabel("count of basic leaner")
    plt.show()