'''
Created on 2018年3月28日

@author: IL MARE
'''
import numpy as np
import Util.RandomUtil as RandomUtil

'''
计算香农墒
'''
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
'''
切分数据集
'''
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
'''
由信息增益最大化计算出需要切分的属性索引值
'''
def chooseBestFeature(trainSet, label):
    tmp = int(np.log2(len(trainSet[0])))
    k = 1 if tmp == 0 else tmp
    indexSet = RandomUtil.generateRandom(0, len(trainSet[0]), k)
    m = len(trainSet)
    maxGain = -1
    baseShannonEnt = calShannonEnt(label)
    index = -1
    for i in indexSet:
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
'''
训练随机森林所需要的弱分类器
'''
def generateWeakLearner(trainSet, trainLabel):
    if trainLabel.count(trainLabel[0]) == len(trainLabel):
        return trainLabel[0]
    if len(trainSet[0]) == 0:
        return "no" if trainLabel.count("no") > trainLabel.count("yes") else "yes"
    index = chooseBestFeature(trainSet, trainLabel)
    Tree = {index:{}}
    uniqueVal = set([elt[index] for elt in trainSet])
    for value in uniqueVal:
        subSet, label = splitDataMatrix(trainSet, trainLabel, index, value)
        Tree[index][value] = generateWeakLearner(subSet, label)
    return Tree

def generateRandomForest(trainSet, trainLabel, T):
    forest = []
    for i in range(T):
        model = generateWeakLearner(trainSet, trainLabel)
        forest.append(model)
    return forest

def classfyData(data, model):
    if type(model) == str:
        return model
    key = iter(model.keys()).__next__()
    value = data[key]
    res = model[key].get(value, None)
    if res != None:
        return classfyData(data, res)
    else:
        tmp_lst = [item for item in model[key].keys()]
        return classfyData(data, model[key][np.random.choice(tmp_lst, 1)[0]])

def predictByRandomForest(models, data):
    tmp_lst = []
    for model in models:
        predict_label = classfyData(data, model)
        tmp_lst.append(predict_label)
    tmp_set = set(tmp_lst)
    res_lst = []
    for res in tmp_set:
        res_lst.append((tmp_lst.count(res), res))
    res_lst = sorted(res_lst, key=lambda index:index[0], reverse=True)
    if len(res_lst) == 1:
        return res_lst[0][1]
    else:
        tmp_res = res_lst[0][0]
        return_lst = [res_lst[0][1]]
        for i in range(1, len(res_lst)):
            if res_lst[i][0] == tmp_res:
                return_lst.append(res_lst[i][1])
        if len(return_lst) == 1:
            return return_lst[0]
        else:
            return np.random.choice(return_lst, 1)[0]