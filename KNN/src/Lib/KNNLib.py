'''
Created on 2018年3月28日

@author: IL MARE
'''
import numpy as np
import os

def classify0(intX:"需要被分类的数据", dataSet:"数据集", labels:"数据集标签", k:"k值")->tuple:
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(intX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistanceIndex = distance.argsort()
    classCount = {}
    for i in range(k):
        votelabel = labels[sortedDistanceIndex[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    tmp_count = -1
    tmp_flag = "-1"
    for item in classCount.items():
        if tmp_count < item[1]:
            tmp_flag = item[0]
            tmp_count = item[1]
    return tmp_flag if tmp_flag != "-1" else None

def file2matrix(filename):#从文件中读取数据
    try:
        fp = open(filename, "r")
        arrayLine = fp.readlines()
        numberOfLine = len(arrayLine)
        returnMat = np.zeros((numberOfLine, 3))
        classLabelVector = []
        index = 0
        for line in arrayLine:
            line = line.strip()
            listFromLine = line.split("\t")
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector
    except Exception as e:
        print(e)
    finally:
        fp.close()
        
def autoNormal(dataSet):#正规化数据
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    rangeVal = maxVal - minVal
    normalDataSet = np.zeros(dataSet.shape, dtype = np.float)
    m = dataSet.shape[0]
    normalDataSet = dataSet - np.tile(minVal, (m, 1))
    normalDataSet /= np.tile(rangeVal, (m, 1))
    return normalDataSet, rangeVal, minVal