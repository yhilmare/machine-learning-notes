'''
Created By ILMARE
@Date 2019-3-2
'''

# from dlib import shape_predictor as predictor
# from dlib import get_frontal_face_detector as detector
import cv2
import os
from  matplotlib import pyplot as plt
import math
import numpy as np
import re
import datetime

modelFile = r"/home/ilmare/Desktop/FaceReplace/shape_predictor_68_face_landmarks.dat"

def transformationFormPoints(sourcePoints, destPoints):
    sourcePoints = np.asmatrix(sourcePoints, dtype=np.float32)
    destPoints = np.asmatrix(destPoints, dtype=np.float32)
    sourceMean = np.mean(sourcePoints, 0)
    destMean = np.mean(destPoints, 0)
    sourcePoints -= sourceMean
    destPoints -= destMean
    sourceStd = np.std(sourcePoints)
    destStd = np.std(destPoints)
    sourcePoints /= sourceStd
    destPoints /= destStd
    U, S, Vt = np.linalg.svd(destPoints.T * sourcePoints)
    R = (U * Vt).T
    return np.vstack([np.hstack(((sourceStd / destStd) * R,
                                 sourceMean.T - (sourceStd / destStd) * R * destMean.T)),
                      np.matrix([0., 0., 1.])])

def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60 * math.pi / 180);
    c60 = math.cos(60 * math.pi / 180);

    inPts = np.copy(inPoints).tolist();
    outPts = np.copy(outPoints).tolist();

    xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0];
    yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1];

    inPts.append([np.int(xin), np.int(yin)]);

    xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0];
    yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1];

    outPts.append([np.int(xout), np.int(yout)]);

    return cv2.getAffineTransform(np.array(inPts, dtype=np.float32), np.array(outPts, dtype=np.float32))

class PhotoParser:
    def __init__(self, videoPath, modelFile, destShape):
        self._modelFile = modelFile
        self._videoPath = videoPath
        res = re.match(r"^/.+/\w+\.\w+$", self._videoPath)
        if res is None:
            raise Exception("video path is invalid")
        res = re.search(r"/.+/(?=\w+\.\w+)", self._videoPath)
        self._savePath = "{0}{1}/".format(res.group(), "parseImg")
        if not os.path.exists(self._savePath):
            os.mkdir(self._savePath)
        self._trainPath = "{0}{1}/".format(res.group(), "trainImg")
        if not os.path.exists(self._trainPath):
            os.mkdir(self._trainPath)
        self._destShape = destShape
        self._photoCount = 0
    @property
    def trainImagePath(self):
        return self._trainPath
    def getPhotoFromVideo(self):
        vc = cv2.VideoCapture(self._videoPath)
        while True:
            rval, frame = vc.read()
            if rval:
                cv2.imwrite("{0}{1}.jpg".format(self._savePath, self._photoCount), frame)
                self._photoCount += 1
                if (self._photoCount % 100) == 0:
                    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- Total Parsed Photo:", self._photoCount)
            else:
                break
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- Total Parsed Photo: ", self._photoCount)
        vc.release()
    # def detectorPhotoFace(self):
    #     if len(os.listdir(self._savePath)) == 0:
    #         raise Exception("There is no photo at {0}".format(self._savePath))
    #     detectorObj = detector()
    #     predictorObj = predictor(self._modelFile)
    #     imageShape = (640, 360)
    #     destEyePoint = [(316, 92), (385, 92)]
    #     try:
    #         fileList = os.listdir(self._savePath)
    #         for file, idx in zip(fileList, range(len(fileList))):
    #             filePath = "{0}{1}".format(self._savePath, file)
    #             img = cv2.imread(filePath)
    #             rects = detectorObj(img, 1)
    #             if len(rects) > 1 or len(rects) == 0:
    #                 print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "-", filePath)
    #                 continue
    #             img = self.__warpPhoto(predictorObj, destEyePoint, imageShape, img, rects[0])
    #             rects = detectorObj(img, 1)
    #             if len(rects) > 1 or len(rects) == 0:
    #                 print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "-", filePath)
    #                 continue
    #             rect = rects[0]
    #             left, top, width, height = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
    #             img = img[top:top + height, left:left + width, :]
    #             img = self.__resizePhoto(img)
    #             cv2.imwrite("{0}{1}.jpg".format(self._trainPath, idx), img)
    #             if (idx % 100) == 0:
    #                 print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- current index: ", idx)
    #     except Exception as e:
    #         print(e)
    def __warpPhoto(self, predictorObj , destEyePoint, imageShape, img, rect):
        points = predictorObj(img, rect)
        inPts = [(points.parts()[36].x, points.parts()[36].y), (points.parts()[45].x, points.parts()[45].y)]
        warpMatrix = similarityTransform(inPts, destEyePoint)
        return cv2.warpAffine(img, warpMatrix, dsize=imageShape)
    def __resizePhoto(self, img):
        try:
            height = img.shape[0]
            width = img.shape[1]
            interval = abs(width - height)
            margin = interval // 2
            if width > height:
                if (interval % 2) == 0:
                    img = img[:, margin: width - margin, :]
                else:
                    img = img[:, margin + 1: width - margin, :]
            elif height > width:
                if (interval % 2) == 0:
                    img = img[margin: height - margin, :, :]
                else:
                    img = img[margin + 1: height - margin, :, :]
            return cv2.resize(img, self._destShape)
        except Exception as e:
            print(e)
            return None



if __name__ == "__main__":
    videoPath = r"F:/tensorflow/automodel/scrawler/video-1/dest2.mp4"
    obj = PhotoParser(videoPath, modelFile, (128, 128))
    obj.getPhotoFromVideo()

