'''
Created on 2018年3月28日

@author: IL MARE
'''
import numpy as np

def kernalTransfrom(dataMatrix, vector, kTup):
    if kTup[0] == "lin":
        return vector * dataMatrix.transpose()
    elif kTup[0] == "rbf":
        delta = dataMatrix - vector
        K = np.matrix(np.diag(delta * delta.transpose()), dtype=np.float)
        K = np.exp(K / (-2 * kTup[1] ** 2))
        return K
    else:
        raise NameError("Kernal Name Error")

class osStruct:
    def __init__(self, dataMatIn, classlabels, C , toler, kTup):
        self.dataMatrix = np.matrix(dataMatIn, dtype=np.float)
        self.labelMatrix = np.matrix(classlabels, dtype=np.float).transpose()
        self.C = C
        self.toler = toler
        self.m = self.dataMatrix.shape[0]
        self.b = 0
        self.alphas = np.matrix(np.zeros((self.m, 1)), dtype=np.float)
        self.eCache = np.matrix(np.zeros((self.m, 2)), dtype=np.float)
        self.K = np.matrix(np.zeros((self.m, self.m)), dtype=np.float)
        for i in range(self.m):
            self.K[i] = kernalTransfrom(self.dataMatrix, self.dataMatrix[i, :], kTup)

def selectJRand(i, m):
    j = i
    while j == i:
        j = np.random.randint(0, m, 1)[0]
    return j

def clipAlpha(alpha, L, H):
    if alpha >= H:
        return H
    elif alpha <= L:
        return L
    else:
        return alpha

def calEi(obj, i):
    fxi = float(np.multiply(obj.alphas, obj.labelMatrix).transpose() * \
                obj.K[:, i]) + obj.b
    Ek = fxi - obj.labelMatrix[i, 0]
    return float(Ek)

def updateEi(obj, i):
    Ei = calEi(obj, i)
    obj.eCache[i] = [1, Ei]

def selectJIndex(obj, i, Ei):
    maxJ = -1
    maxdelta = -1
    Ek = -1
    obj.eCache[i] = [1, Ei]
    vaildEiList = np.nonzero(obj.eCache[:, 0].A)[0]
    if len(vaildEiList) > 1:
        for j in vaildEiList:
            if j == i:
                continue
            Ej = calEi(obj, j)
            delta = np.abs(Ei - Ej)
            if delta > maxdelta:
                maxdelta = delta
                maxJ = j
                Ek = Ej
    else:
        maxJ = selectJRand(i, obj.m)
        Ek = calEi(obj, maxJ)
    return Ek, maxJ

def innerLoop(obj, i):
    Ei = calEi(obj, i)
    if (obj.labelMatrix[i, 0] * Ei < -obj.toler and obj.alphas[i, 0] < obj.C) or \
            (obj.labelMatrix[i, 0] * Ei > obj.toler and obj.alphas[i, 0] > 0):
        Ej, j = selectJIndex(obj, i, Ei)
        alphaIold = obj.alphas[i, 0].copy()
        alphaJold = obj.alphas[j, 0].copy()
        if obj.labelMatrix[i, 0] == obj.labelMatrix[j, 0]:
            L = max(0, obj.alphas[i, 0] + obj.alphas[j, 0] - obj.C)
            H = min(obj.C , obj.alphas[i, 0] + obj.alphas[j, 0])
        else:
            L = max(0, obj.alphas[j, 0] - obj.alphas[i, 0])
            H = min(obj.C, obj.C - obj.alphas[i, 0] + obj.alphas[j, 0])
        if L == H:
            return 0
        eta = obj.K[i, i] + obj.K[j, j] - 2 * obj.K[i, j]
        if eta <= 0:
            return 0
        obj.alphas[j, 0] += obj.labelMatrix[j, 0] * (Ei - Ej) / eta
        obj.alphas[j, 0] = clipAlpha(obj.alphas[j, 0], L, H)
        updateEi(obj, j)
        if np.abs(obj.alphas[j, 0] - alphaJold) < 0.00001:
            return 0
        obj.alphas[i, 0] += obj.labelMatrix[i, 0] * obj.labelMatrix[j, 0] * (alphaJold - obj.alphas[j, 0])
        updateEi(obj, i)
        b1 = -Ei - obj.labelMatrix[i, 0] * obj.K[i, i] * (obj.alphas[i, 0] - alphaIold) \
             - obj.labelMatrix[j, 0] * obj.K[i, j] * (obj.alphas[j, 0] - alphaJold) + obj.b
        b2 = -Ej - obj.labelMatrix[i, 0] * obj.K[i, j] * (obj.alphas[i, 0] - alphaIold) \
             - obj.labelMatrix[j, 0] * obj.K[j, j] * (obj.alphas[j, 0] - alphaJold) + obj.b
        if obj.alphas[i, 0] > 0 and obj.alphas[i, 0] < obj.C:
            obj.b = b1
        elif obj.alphas[j, 0] > 0 and obj.alphas[j, 0] < obj.C:
            obj.b = b2
        else:
            obj.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def realSMO(trainSet, trainLabels, C, toler, kTup=('lin', 1.3), maxIter=40):
    obj = osStruct(trainSet, trainLabels, C, toler, kTup)
    entrySet = True
    iterNum = 0
    alphapairschanged = 0
    while (iterNum < maxIter) and (alphapairschanged > 0 or entrySet):
        print(iterNum)
        alphapairschanged = 0
        if entrySet:
            for i in range(obj.m):
                alphapairschanged += innerLoop(obj, i)
                if i % 100 == 0:
                    print("full set loop, iter: %d, alphapairschanged: %d, iterNum: %d" % (i, alphapairschanged, iterNum))
            iterNum += 1
        else:
            vaildalphsaList = np.nonzero((obj.alphas.A > 0) * (obj.alphas.A < C))[0]
            for i in vaildalphsaList:
                alphapairschanged += innerLoop(obj, i)
                if i % 100 == 0:
                    print("non-bound set loop, iter: %d, alphapairschanged: %d, iterNum: %d" % (i, alphapairschanged, iterNum))
            iterNum += 1
        if entrySet:
            entrySet = False
        elif alphapairschanged == 0:
            entrySet = True
            print("iter num: %d" % (iterNum))
    return obj.alphas, obj.b

def getSupportVectorandSupportLabel(trainSet, trainLabel, alphas):
    vaildalphaList = np.nonzero(alphas.A)[0]
    dataMatrix = np.matrix(trainSet, dtype=np.float)
    labelMatrix = np.matrix(trainLabel, dtype=np.float).transpose()
    sv = dataMatrix[vaildalphaList]#得到支持向量
    svl = labelMatrix[vaildalphaList]
    return sv, svl

def predictLabel(data, sv, svl, alphas, b, kTup):
    kernal = kernalTransfrom(sv, np.matrix(data, dtype=np.float), kTup).transpose()
    fxi = np.multiply(svl.T, alphas[alphas != 0]) * kernal + b
    return np.sign(float(fxi))