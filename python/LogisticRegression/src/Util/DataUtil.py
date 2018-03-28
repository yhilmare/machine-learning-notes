'''
Created on 2018年3月28日

@author: IL MARE
'''
import csv
import numpy as np
import re
import Util.RandomUtil as RandomUtil
from Lib import RFLib as RFLib

filePath = r"G:\研究生课件\数据挖掘\实验数据"

'''
==========================与随机森林及决策树有关的工具类==============================
'''
'''
为随机森林模型及决策树模型产生数据，函数接受一个csv文件的文件名，具体路径在filePath中写明
'''
def loadDataForRMOrDTModel(filename):
    try:
        fp = open("{0}/{1}.csv".format(filePath, filename), "r")
        reader = csv.reader(fp)
        trainSet = []
        trainLabel = []
        reader.__next__()
        for line in reader:
            tmp_lst = []
            for msg in line[0].split(";"):
                tmp_lst.append(re.search(r"[0-9a-zA-Z.-]+", msg)[0])
            trainSet.append(tmp_lst[0: -1])
            trainLabel.append(tmp_lst[-1])
        return processData(trainSet), trainLabel
    except Exception as e:
        print(e)
    finally:
        fp.close()
'''
该函数为随机森林服务，将原始数据集中的连续值离散化，便于随机森林处理
'''
def processData(trainSet):
    for row in trainSet:
        if float(row[0]) < 20:
            row[0] = "1"
        elif float(row[0]) >= 20 and float(row[0]) < 30:
            row[0] = "2"
        elif float(row[0]) >= 30 and float(row[0]) < 40:
            row[0] = "3"
        elif float(row[0]) >= 40 and float(row[0]) < 50:
            row[0] = "4"
        elif float(row[0]) >= 50 and float(row[0]) < 60:
            row[0] = "5"
        elif float(row[0]) >= 60 and float(row[0]) < 70:
            row[0] = "6"
        else:
            row[0] = "7"
        row[10] = str(float(row[10]) // 30 + 1)
        row[-2] = str(float(row[-2]) // 0.1 + 1)
    return trainSet

'''
==========================与SVM及对数回归有关的工具类==============================
'''
'''
为SVM及对数回归模型产生数据，函数接受一个csv文件的文件名，具体路径在filePath中写明
'''
global_var = {1:['blue-collar', 'entrepreneur', 'unemployed', 'admin.', 'retired', 'services', \
                 'technician', 'self-employed', 'management', 'housemaid', 'student'],
              2:['single', 'married', 'divorced'],
              4:['yes', 'no'],
              5:['yes', 'no'],
              6:['yes', 'no'],
              7:['cellular', 'telephone'],
              14:['failure', 'success', 'nonexistent']}

global_var_order = {3:['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree'],
                    8:['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                    9:['mon', 'tue', 'wed', 'thu', 'fri']}
'''
读取处理好的数据，加快模型测试速度
'''
def loadTempDataForSVMOrLRModel(filename):
    try:
        fp = open("{0}/{1}.csv".format(filePath, filename), "r")
        reader = csv.reader(fp)
        trainSet = []
        trainLabel = []
        for line in reader:
            trainSet.append(line[0: -1])
            trainLabel.append(int(line[-1]))
        return trainSet, trainLabel
    except Exception as e:
        print(e)
    finally:
        fp.close()
'''
为SVM和对数回归模型生成数据，重点的功能是将数据量化处理
'''
def loadDataForSVMOrLRModel(filename, modelType="lr"):
    try:
        fp = open("{0}/{1}.csv".format(filePath, filename), "r")
        reader = csv.reader(fp)
        trainSet = []
        trainLabel = []
        reader.__next__()
        for line in reader:
            tmp_lst = []
            for msg in line[0].split(";"):
                tmp_lst.append(re.search(r"[0-9a-zA-Z.-]+", msg)[0])
            trainSet.append(tmp_lst[0: -1])
            trainLabel.append(tmp_lst[-1])
        fullfilltheUnknownValue(trainSet, trainLabel)
        quantizedData(trainSet, trainLabel, modelType)
        normalData(trainSet, modelType)
        return trainSet, trainLabel
    except Exception as e:
        print(e)
    finally:
        fp.close()
'''
正规化数据
'''
def normalData(trainSet, modelType):
    tmp_lst = []
    for i in range(len(trainSet[0])):
        tmp_lst.append(np.array([item[i] for item in trainSet], dtype=np.float))
    for i in range(len(tmp_lst)):
        item = tmp_lst[i]
        tmp_lst[i] = (item.min(), item.max(), item.mean(), item.std())
    for i in range(len(trainSet)):
        for j in range(len(trainSet[i])):
            val = tmp_lst[j]
            if modelType == "lr":
                trainSet[i][j] = (float(trainSet[i][j]) - val[0]) / (val[1] - val[0])
            else:
                trainSet[i][j] = (float(trainSet[i][j]) - val[2]) / val[3]
'''
为随机森林预测模型产生数据，该函数的作用是删除数据集中unknown的数据
'''
def formatTrainSet(trainSet, trainLabel, axis):
    dataSet = []
    for item in trainSet:
        dataSet.append(item.copy())
    labelSet = trainLabel.copy()
    value_set = set()
    del_index = []
    for i in range(len(dataSet)):
        temp = dataSet[i][axis]
        if temp == "unknown":
            del_index.append(i)
        else:
            value_set.add(temp)
        dataSet[i][axis] = labelSet[i]
        labelSet[i] = temp
    for i in range(len(del_index)):
        index = del_index[i] - i
        del dataSet[index]
        del labelSet[index]
    return dataSet, labelSet, value_set
'''
训练随即森林模型用于预测缺失值
'''
def trainPredictRandomForest(trainSet, trainLabel, axis):
    dataSet, labelSet, value_set = formatTrainSet(trainSet, trainLabel, axis)
    dataSet1, labelSet1 = underSampling(dataSet, labelSet, *value_set)
    forest = RFLib.generateRandomForest(dataSet1, labelSet1, 19)
    return forest
'''
遍历数据集，将原始数据集中的缺失值补上
'''
def predictValue(dataSet, labelSet, axis):
    forest = trainPredictRandomForest(dataSet, labelSet, axis)
    for item in zip(dataSet, labelSet):
        if item[0][axis] == "unknown":
            tmp_lst = item[0][0:axis]
            tmp_lst.extend([item[1]])
            tmp_lst.extend(item[0][axis + 1:])
            predict = RFLib.predictByRandomForest(forest, tmp_lst)
            item[0][axis] = predict
'''
该函数用于将数据集中为unknown的属性值都用随机森林预测值来补上
'''
def fullfilltheUnknownValue(dataSet, labelSet):
    predict_set = set()
    for data, label in zip(dataSet, labelSet):
        for i in range(len(data)):
            if data[i] == "unknown":
                predict_set.add(i)
    for index in predict_set:
        predictValue(dataSet, labelSet, index)
'''
将原始数据集中的离散值量化
'''
def quantizedData(dataSet, labelSet, modelType="lr"):
    if modelType == "lr":
        for i in range(len(labelSet)):
            if labelSet[i] == "no":
                labelSet[i] = 0
            else:
                labelSet[i] = 1
    else:
        for i in range(len(labelSet)):
            if labelSet[i] == "no":
                labelSet[i] = -1
            else:
                labelSet[i] = 1
    global global_var_order, global_var
    index_lst = [index for index in global_var_order.keys()]
    index_lst.extend([index for index in global_var.keys()])
    index_lst = sorted(index_lst)
    for i in range(len(dataSet)):
        item = dataSet[i]
        tmp_lst = []
        for index in index_lst:
            variable = generateDummyVar(item[index], index) if generateDummyVar(item[index], index) \
                else generateOrderVar(item[index], index)
            if variable == None:
                raise NameError("变量量化失败")
            tmp_lst.append((index, variable))
        dataSet[i] = generateNewList(item, tmp_lst)
'''
根据量化值扩展远列表
'''
def generateNewList(oldList, tmp_lst):
    return_mat = []
    index_set = list()
    for item in tmp_lst:
        index_set.append(item[0])
    for i in range(len(oldList)):
        if i in index_set:
            for item in tmp_lst[0][1]:
                return_mat.append(item)
            del tmp_lst[0]
        else:
            return_mat.append(oldList[i])
    return return_mat
'''
对无序离散值生成哑变量
'''
def generateDummyVar(variable, index):
    global global_var
    var_list = global_var.get(index, None)
    if var_list == None:
        return None
    num_dumm = len(var_list) - 1
    retrun_mat = [0] * num_dumm
    for i in range(num_dumm):
        var = var_list[i]
        if var == variable:
            retrun_mat[i] = 1
            return retrun_mat
    return retrun_mat
'''
对有序离散值生成连续变量
'''
def generateOrderVar(variable, index):
    global global_var_order
    var_list = global_var_order.get(index, None)
    if var_list == None:
        return None
    for i in range(len(var_list)):
        if variable == var_list[i]:
            return [i + 1]
    return None
'''
=================================通用工具函数==========================================
'''
'''
该函数用语欠抽样原始数据集，由于原始数据集中类别不平衡，正例只有反例的十分之一
为了模型的泛化能力，需要欠抽样来保证正例和反例数目相同
'''

def underSampling(dataSet, labelSet, *args):
    trainSet = dataSet.copy()
    trainLabel = labelSet.copy()
    labelcount_lst = []
    for label in args:
        labelcount_lst.append((trainLabel.count(label), label))
    labelcount_lst = sorted(labelcount_lst, key=lambda item:item[0])
    min_val, labelName = labelcount_lst[0]
    label_set = set(args) - set([labelName])
    for label in label_set:
        tmp_set = set()
        for item in enumerate(trainLabel):
            if item[1] == label:
                tmp_set.add(item[0])
        indexSet = RandomUtil.generateRandomIndex(tmp_set, min_val)
        del_set = tmp_set - indexSet
        del_set = sorted(list(del_set))
        for i in range(len(del_set)):
            index = del_set[i] - i
            del trainSet[index]
            del trainLabel[index]
    return trainSet, trainLabel

'''
该方法在欠抽样后的数据集上工作，使用自助法产生训练集和测试集，训练集大小为愿数据集大小的62%
测试集大小为原始数据集大小的32%
'''
def generateTrainSet(dataSet, labelSet):
    trainSet = []
    trainLabel = []
    testSet = []
    testLabel = []
    m = len(labelSet)
    trainIndex = set()
    totalIndex = set()
    for i in range(m):
        index = np.random.randint(0, m, 1)[0]
        trainIndex.add(index)
        totalIndex.add(i)
        trainSet.append(dataSet[index])
        trainLabel.append(labelSet[index])
    for item in totalIndex - trainIndex:
        testSet.append(dataSet[item])
        testLabel.append(labelSet[item])
    return trainSet, trainLabel, testSet, testLabel