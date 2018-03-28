'''
Created on 2018年3月28日

@author: IL MARE
'''
import numpy as np
'''
该函数用来在一个集合中随机抽取size个互不相同的随机值
'''
def generateRandomIndex(a, size):
    if len(a) < size:
        return None
    elif len(a) == size:
        return set(a)
    returnMat = set()
    while True:
        returnMat.add(np.random.choice(list(a), 1)[0])
        if len(returnMat) == size:
            break
    return returnMat
'''
在指定范围内产生指定数目的不重复的随机数
'''
def generateRandom(low, high, size):
    returnSet = set()
    while True:
        returnSet.add(np.random.randint(low, high, 1)[0])
        if len(returnSet) == size:
            break
    return returnSet