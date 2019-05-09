'''
Created on 2018年3月7日

@author: IL MARE
'''
import Util.DataUtil as DataUtil
import Lib.DecisionTreeLib as DTLib
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

if __name__ == "__main__":
    start = time.clock()
    dataSet, labelSet = loadDataSet("bank-additional")
    tmp_lst = []
    for i in range(100):
        trainSet, trainLabel, testSet, testLabel = DataUtil.generateTrainSet(dataSet, labelSet)
        model = DTLib.createDecisionTree(trainSet, trainLabel)
        errorRatio = DTLib.testDTModel(testSet, testLabel, model)
        tmp_lst.append(1 - errorRatio)
    y = np.array(tmp_lst, dtype=np.float)
    print("the avg correct ratio is %.3f, the std is %.3f" % (y.mean(), y.std()))
    x = np.arange(0, len(tmp_lst))
    fig = plt.figure("test")
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_ylim([0, 1])
    ax.set_ylabel("correct ratio of DT")
    ax.set_xlabel("count of exp")
    plt.show()