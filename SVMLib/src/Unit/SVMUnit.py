'''
Created on 2018年3月4日

@author: IL MARE
'''
import time
from Lib import SVMLib as SVMLib
from Util import DataUtil as DataUtil    

if __name__ == "__main__":
    start = time.clock()
    # dataSet, labelSet = DataUtil.loadDataForSVMOrLRModel("bank-additional", "svm")#正统方法
    dataSet, labelSet = DataUtil.loadTempDataForSVMOrLRModel("bank-addtional-format-svm")
    dataSet, labelSet = DataUtil.underSampling(dataSet, labelSet, 1, -1)
    trainSet, trainLabel, testSet, testLabel = DataUtil.generateTrainSet(dataSet, labelSet)
    kTup = ("lin", 1.2)
    alphas, b = SVMLib.realSMO(trainSet, trainLabel, 0.6, 0.01, kTup, 10)
    errorCount = 0
    sv, svl = SVMLib.getSupportVectorandSupportLabel(trainSet, trainLabel, alphas)
    for data, label in zip(testSet, testLabel):
        predict_label = SVMLib.predictLabel(data, *[sv, svl, alphas, b, kTup])
        if predict_label != label:
            errorCount += 1
    ratio = errorCount / len(testLabel)
    print("the error ratio is %.3f, the correct ratio is %.3f -- %.3fs" % (ratio, 1 - ratio, time.clock() - start))
    