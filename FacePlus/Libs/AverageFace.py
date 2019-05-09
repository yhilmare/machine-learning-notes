'''
Created on 2018年7月8日

@author: IL MARE
'''

import numpy as np
import os
import cv2

class AverageFace:
    def __init__(self, filePath, shape=(600, 600), keyPoint=(26, 14)):
        if filePath.endswith("/") or filePath.endswith("\\"):
            self._filePath = filePath
        else:
            self._filePath = "{0}/".format(filePath)
        self._keyPoint = keyPoint
        self._shape = shape
        self.read_landmark()
        self.readImage()
        self.generateImage()
    def read_landmark(self):
        self._pointsArray = []
        for file in os.listdir(self._filePath):
            if file.endswith("txt"):
                with open(r"{0}\{1}".format(self._filePath, file)) as fp:
                    landMark = fp.read().split()
                    landMark = np.reshape(np.array(landMark, dtype=np.int32), [-1, 2])
                    self._pointsArray.append(landMark)
    def readImage(self):
        self._imageList = []
        for file in os.listdir(self._filePath):
            if file.endswith("jpg"):
                image = cv2.imread(r"{0}\{1}".format(self._filePath, file))
                image = image / 255.0
                self._imageList.append(image)
    @staticmethod
    def similarityTransform(input, output):
        s60 = np.sin(60 * np.pi / 180.0)
        c60 = np.cos(60 * np.pi / 180.0)
        inPts = np.copy(input).tolist()
        outPts = np.copy(output).tolist()
        xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60 * (inPts[0][0] - inPts[1][0]) - c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]
        inPts.append([np.int(xin), np.int(yin)])
        xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60 * (outPts[0][0] - outPts[1][0]) - c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]
        outPts.append([np.int(xout), np.int(yout)])
        return cv2.estimateRigidTransform(np.array([inPts], dtype=np.int32), 
                                          np.array([outPts], dtype=np.int32), True)
    @staticmethod
    def calculateDelaunayTriangles(rect, points):
        def rectContains(point, rect):
            if point[0] < rect[0] or point[0] > rect[2]:
                return False
            elif point[1] < rect[1] or point[1] > rect[3]:
                return False
            return True  
        subDiv = cv2.Subdiv2D(rect)
        for point in points:
            subDiv.insert((point[0], point[1]))
        triangleList = subDiv.getTriangleList()
        return_mat = []
        for triangle in triangleList:
            pt = []
            pt.append((triangle[0], triangle[1]))
            pt.append((triangle[2], triangle[3]))
            pt.append((triangle[4], triangle[5])) 
            if rectContains(pt[0], rect) and rectContains(pt[1], rect) and rectContains(pt[2], rect):
                ind = []
                for i in range(3):
                    for j in range(len(points)):
                        if np.abs(pt[i][0] - points[j][0]) < 1.0 and np.abs(pt[i][1] - points[j][1]) < 1.0:
                            ind.append(j)
                if len(ind) == 3:
                    return_mat.append(ind)
        return return_mat
    @staticmethod
    def constrainPoint(p, w, h):
        p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
        return p
    @staticmethod
    def warpTriangle(img1, img2, t1, t2):
        def applyAffineTransform(src, srcTri, dstTri, size) :
            warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
            dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, 
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            return dst
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        t1Rect = [] 
        t2Rect = []
        t2RectInt = []
        for i in range(0, 3):
            t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
            t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1, 1, 1));
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        size = (r2[2], r2[3])
        img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
        img2Rect = img2Rect * mask
        img2[r2[1]: r2[1] + r2[3], r2[0]: r2[0] + r2[2]] = img2[r2[1]: r2[1] + r2[3], 
                                                                r2[0]: r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
        img2[r2[1]: r2[1] + r2[3], r2[0]: r2[0] + r2[2]] = img2[r2[1]: r2[1] + r2[3],
                                                                 r2[0]: r2[0] + r2[2]] + img2Rect
    def generateImage(self):
        width = self._shape[0]
        height = self._shape[1]
        eyePoint = [(0.34 * width, height / 2.2), (0.66 * width, height / 2.2)]
        boundPoint = [(0, 0), (width / 2.0, 0), (width - 1, 0), (width - 1, height / 2.0), (width - 1, height - 1),
                      (width / 2.0, height - 1), (0, height - 1), (0, height / 2.0)]
        pointsAvg = np.array([(0, 0)] * (len(self._pointsArray[0]) + len(boundPoint)), np.float32)
        numImages = len(self._imageList)
        pointsNorm = []
        imagesNorm = []
        for point, image in zip(self._pointsArray, self._imageList):
            eyePointSrc = [point[self._keyPoint[0]], point[self._keyPoint[1]]]
            transform = AverageFace.similarityTransform(eyePointSrc, eyePoint)
            img = cv2.warpAffine(image, transform, (width, height))
            points = np.reshape(point, [len(self._pointsArray[0]), 1, 2])
            points = np.reshape(cv2.transform(points, transform), [len(self._pointsArray[0]), 2])
            points = np.append(points, boundPoint, 0)
            pointsAvg = pointsAvg + points / numImages
            pointsNorm.append(points)
            imagesNorm.append(img)
        rect = (0, 0, width, height)
        triangleList = AverageFace.calculateDelaunayTriangles(rect, pointsAvg)
        output = np.zeros((width, height, 3), dtype=np.float32)
        for i in range(len(imagesNorm)):
            img = np.zeros([width, height, 3], dtype=np.float32)
            for j in range(len(triangleList)):
                tin = []
                tout = []
                for k in range(3):
                    pIn = pointsNorm[i][triangleList[j][k]]
                    pIn = AverageFace.constrainPoint(pIn, width, height)
                    pOut = pointsAvg[triangleList[j][k]]
                    pOut = AverageFace.constrainPoint(pOut, width, height)
                    tin.append(pIn)
                    tout.append(pOut)
                AverageFace.warpTriangle(imagesNorm[i], img, tin, tout)
            output = output + img
        self._output = output / len(imagesNorm)
    @property
    def averageImage(self):
        return self._output
    def showImage(self):
        cv2.imshow("image", self._output)
        cv2.waitKey(0)
    def saveImage(self, path):
        cv2.imwrite(path, np.int32(self._output * 255.0))
          

if __name__ == "__main__":
    obj = AverageFace(r"G:\python\sources\nwpu\dectImage")
    obj.showImage()
    obj.saveImage(r"g:/aaa.jpg")
