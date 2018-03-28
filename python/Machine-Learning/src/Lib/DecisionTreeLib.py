'''
Created on 2018年3月28日

@author: IL MARE
'''
import numpy as np

def calShannonEnt(trainLabel):
    m = len(trainLabel)
    uniqueVal = set(trainLabel)
    countDict = {}
    shannonNum = 0.0
    for label in trainLabel:
        countDict[label] = countDict.get(label, 0) + 1
    for label in uniqueVal:
        p = (countDict[label] / m)
        shannonNum -= p * np.log2(p)
    return shannonNum

def splitDataMatrix(dataMatrix, label, axis, value):
    returnMat = []
    labelMat = []
    for row, row1 in zip(dataMatrix, label):
        if row[axis] == value:
            tmp_lst = row[0: axis]
            tmp_lst.extend(row[axis + 1:])
            returnMat.append(tmp_lst)
            labelMat.append(row1)
    return returnMat, labelMat

def chooseBestFeature(trainSet, label):
    m = len(trainSet)
    maxGain = -1
    baseShannonEnt = calShannonEnt(label)
    index = -1
    for i in range(len(trainSet[0])):
        uniqueAttr = set([example[i] for example in trainSet])
        tmp_Ent = 0
        for attr in uniqueAttr:
            subSet, labelMat = splitDataMatrix(trainSet, label, i, attr)
            newShannonEnt = calShannonEnt(labelMat)
            tmp_Ent += float(len(subSet) / m) * newShannonEnt
        gain = baseShannonEnt - tmp_Ent
        if gain > maxGain:
            maxGain = gain
            index = i
    return index

def createDecisionTree(trainSet, trainLabel):
    if trainLabel.count(trainLabel[0]) == len(trainLabel):
        return trainLabel[0]
    if len(trainSet[0]) == 0:
        return "no" if trainLabel.count("no") > trainLabel.count("yes") else "yes"
    index = chooseBestFeature(trainSet, trainLabel)
    Tree = {index:{}}
    uniqueVal = set([elt[index] for elt in trainSet])
    for value in uniqueVal:
        subSet, label = splitDataMatrix(trainSet, trainLabel, index, value)
        Tree[index][value] = createDecisionTree(subSet, label)
    return Tree

def predictByDTModel(data, model):
    if type(model) == str:
        return model
    key = iter(model.keys()).__next__()
    value = data[key]
    res = model[key].get(value, None)
    if res != None:
        return predictByDTModel(data, res)
    else:
        tmp_lst = [item for item in model[key].keys()]
        return predictByDTModel(data, model[key][np.random.choice(tmp_lst, 1)[0]])

def testDTModel(testData, testLabel, model):
    predictLabel = []
    for row in testData:
        predictLabel.append(predictByDTModel(row, model))
    errorCount = 0
    for val, val1 in zip(predictLabel, testLabel):
        if val != val1:
            errorCount += 1
    ratio = float(errorCount) / len(testLabel)
    print("DT:total error ratio is %.3f, correct ratio is %.3f" % (ratio, 1 - ratio))
    return ratio