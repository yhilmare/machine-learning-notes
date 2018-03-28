'''
Created on 2018年3月28日

@author: IL MARE
'''
from urllib.request import urlopen
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import Lib.KNNLib as kNN

default_path = r"g:/machine/kNN_dataSet/"
dest_vertical_dim = 18
def getToken():
    filePath = default_path + "token"
    for i in range(100):
        try:
            resp = urlopen("http://jiaowu.swjtu.edu.cn/servlet/GetRandomNumberToJPEG", timeout=10)
            fp = open("{0}/{1:d}.jpg".format(filePath, i), "wb")
            fp.write(resp.read())
        except Exception as e:
            print(e)
        finally:
            fp.close()

def parseImg():
    filePath = default_path + "token"
    for filename in os.listdir(filePath):
        print("processing the {0}".format(filename))
        img = Image.open(filePath + "/" + filename)
        img = img.convert("L")
        img = img.point(lambda i : 255 if i > 110 else 0)
        img = img.crop((1, 1, 55, 22))
        img.save("{0}new/{1}".format(default_path, filename.split(".")[0] + ".png"))

def crop(img, rect):#剪裁图片
    size = img.shape
    left = rect[0]
    top = rect[1]
    width = rect[2]
    height = rect[3]
    if (left + width) > size[1]:
        return None
    elif (top + height) > size[0]:
        return None;
    return img[top:(top + height), left:(left + width)]

def getAlphabet():
    filePath = default_path + "new"
    fileNames = os.listdir(filePath)
    print(len(fileNames))
    for j in range(len(fileNames)):
        name = fileNames[j]
        if name == ".DS_Store":
            continue
        img = Image.open(filePath + "/" + name)
        img = np.asarray(img)
        size = img.shape
        print("processing the image {0}".format(name))
        del_lst = []
        for i in range(size[1]):
            tmp_lst = img[:, i]
            if tmp_lst.sum() / (size[0] * 255) >= 0.9:
                if len(del_lst) != 0 and i == (del_lst[-1] + 1):
                    del_lst.remove(i - 1)
                    del_lst.append(i)
                else:
                    del_lst.append(i)
                    if len(del_lst) == 5:
                        break
        for i in range(len(del_lst) - 1):
            index = del_lst[i]
            width = del_lst[i + 1] - index
            sub_img = crop(img, (index, 0, width, size[0]))
            sub_img = Image.fromarray(resize_pic(sub_img, dest_vertical_dim))
            sub_img = sub_img.convert("L")
            sub_img.save(r"{0}/{2}_{1}_{3}.png".format(default_path + "alphabet", name.split(".")[0], name.split(".")[0][i], i))

def resize_pic(img, dim):
    size = img.shape
    if size[1] < dim:
        range_val = dim - size[1]
        left = 0
        right = 0
        if range_val % 2 == 0:
            left = right = range_val // 2
        else:
            left = range_val // 2 + 1
            right = left - 1
        tmp_matrix = np.zeros((21, left + right + size[1]))
        for i in range(tmp_matrix.shape[1]):
            if i <= (left - 1):
                tmp_matrix[:, i] = np.tile(255, size[0])
            elif i > (left - 1) and i < (left + size[1]):
                tmp_matrix[:, i] = img[:, i - left]
            else:
                tmp_matrix[:, i] = np.tile(255, size[0])   
        return tmp_matrix
    else:
        range_val = size[1] - dim
        left = 0
        right = 0
        if range_val % 2 == 0:
            left = right = range_val // 2
        else:
            left = range_val // 2 + 1
            right = left - 1
        tmp_matrix = crop(img, (left, 0, dim, 21))
        return tmp_matrix

def analyzeImg(filename=default_path + "new\ACSS.png"):
    img = Image.open(filename)
    img = np.asarray(img)
    print(img.shape)
    size = img.shape
    tmp = []
    del_lst = []
    for i in range(size[1]):
        tmp_lst = img[:, i]
        tmp.append(tmp_lst.sum() / size[0])
        if tmp_lst.sum() / (size[0] * 255) >= 0.9:
            if len(del_lst) != 0 and i == (del_lst[-1] + 1):
                del_lst.remove(i - 1)
                del_lst.append(i)
            else:
                del_lst.append(i)
                if len(del_lst) == 5:
                    break
    fig = plt.figure("test")
    ax = fig.add_subplot(321)
    ax.imshow(img)
    for item in del_lst:
        ax.plot([item,item], [0, size[0]])
    cx = fig.add_subplot(322)
    x = np.arange(0, len(tmp))
    cx.plot(x, tmp)
    for i in range(len(del_lst) - 1):
        index = del_lst[i]
        width = del_lst[i + 1] - index
        temp = 323 + i
        dx = fig.add_subplot(temp)
        sub_img = crop(img, (index, 0, width, size[0]))
        dx.imshow(sub_img)
    plt.show()

def resizeTheImg():#该方法废弃不用，被resize_pic代替
    fileName = default_path + "alphabet"
    for name in os.listdir(fileName):
        filePath = fileName + "/" + name
        img = Image.open(filePath)
        img = np.asarray(img)
        print("resizing the picture {0}, ".format(name), end="")
        size = img.shape
        if size[1] < dest_vertical_dim:
            range_val = dest_vertical_dim - size[1]
            left = 0
            right = 0
            if range_val % 2 == 0:
                left = right = range_val // 2
            else:
                left = range_val // 2 + 1
                right = left - 1
            tmp_matrix = np.zeros((21, left + right + size[1]))
            for i in range(tmp_matrix.shape[1]):
                if i <= (left - 1):
                    tmp_matrix[:, i] = np.tile(255, size[0])
                elif i > (left - 1) and i < (left + size[1]):
                    tmp_matrix[:, i] = img[:, i - left]
                else:
                    tmp_matrix[:, i] = np.tile(255, size[0])   
            img = Image.fromarray(tmp_matrix)
            img = img.convert("L")
            print("the picture size is {0}, {1}".format(img.size[0], img.size[1]))
            img.save(filePath)
        else:
            range_val = size[1] - dest_vertical_dim
            left = 0
            right = 0
            if range_val % 2 == 0:
                left = right = range_val // 2
            else:
                left = range_val // 2 + 1
                right = left - 1
            tmp_matrix = crop(img, (left, 0, dest_vertical_dim, 21))
            img = Image.fromarray(tmp_matrix)
            img = img.convert("L")
            print("the picture size is {0}, {1}".format(img.size[0], img.size[1]))
            img.save(filePath)

def readDataSet():#从本地读出数据集
    filePath = default_path + "alphabet"
    dirPath = os.listdir(filePath)
    m = len(dirPath)
    returnMat = np.zeros((m, dest_vertical_dim * 21))
    labels = []
    for i in range(m):
        name = dirPath[i]
        img = Image.open(filePath + "/" + name)
        matrix = np.asarray(img)
        tmp_matrix = np.zeros((1, dest_vertical_dim * 21))
        for j in range(matrix.shape[0]):
            for k in range(matrix.shape[1]):
                tmp_matrix[0, j * dest_vertical_dim + k] = matrix[j, k]
        returnMat[i, :] = tmp_matrix
        labels.append(name.split("_")[0])
    return returnMat, labels

def kNNidentify(dataSet, labels, filename=default_path + "834.jpg", k=10):#传入数据集，标签以及图片名称
    img = Image.open(filename)
    img = img.convert("L")
    img = img.point(lambda i : 255 if i > 110 else 0)
    img = img.crop((1, 1, 55, 22))
    img = np.asarray(img)
    img = resize_pic(img, 57)
    size = img.shape
    tmp = []
    del_lst = []
    for i in range(size[1]):
        tmp_lst = img[:, i]
        tmp.append(tmp_lst.sum() / size[0])
        if tmp_lst.sum() / (size[0] * 255) >= 0.9:
            if len(del_lst) != 0 and i == (del_lst[-1] + 1):
                del_lst.remove(i - 1)
                del_lst.append(i)
            else:
                del_lst.append(i)
                if len(del_lst) == 5:
                    break
    resultMat = np.zeros((len(del_lst) - 1, dest_vertical_dim * 21))
    for i in range(len(del_lst) - 1):
        index = del_lst[i]
        width = del_lst[i + 1] - index
        sub_img = crop(img, (index, 0, width, size[0]))
        sub_img = resize_pic(sub_img, dest_vertical_dim)
        tmp_matrix = np.zeros((1, dest_vertical_dim * 21))
        for j in range(sub_img.shape[0]):
            for k in range(sub_img.shape[1]):
                tmp_matrix[0, j * dest_vertical_dim + k] = sub_img[j, k]
        resultMat[i, :] = tmp_matrix
    pattern = []
    for item in resultMat:
        pattern.append(kNN.classify0(item, dataSet, labels, 10))
    return "".join(pattern)

if __name__ == "__main__":
    dataSet, labels = readDataSet()
#     print(kNNidentify(dataSet, labels, default_path + "33.jpg"))
    path = default_path + "testData"
    error = 0
    dirPath = os.listdir(path)
    m = len(dirPath)
    for name in dirPath:
        filePath = path + "\\" + name
        pattern = kNNidentify(dataSet, labels, filePath)
        name = name.split(".")[0]
        print("the classfier came back with {0}, the real answer is {1}".format(pattern, name))
        if pattern != name:
            error += 1
    print("the total error ratio is {0:.3f}".format(error / m))
#==================准备数据集用到的代码=============================================
#     getToken()#从远端获得验证码
#     parseImg()#处理图片，将图片转化二值化，转化成灰度图
#     getAlphabet()#将图片转化成字母图片，注意这一步之前要将上一步得到的图片名称改为验证码中的所代表的字符
#     analyzeImg()#分析具体的一张图