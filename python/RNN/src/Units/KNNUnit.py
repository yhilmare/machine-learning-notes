'''
Created on 2018年4月25日

@author: IL MARE
'''
from urllib.request import urlretrieve
import os
import zipfile
import tensorflow as tf
import collections
import numpy as np
import random
import math

url = "http://mattmahoney.net/dc/"
vocabularySize = 5000

def download_file(fileName, expectedSize):
    if not os.path.exists(fileName):
        try:
            urlretrieve(url + fileName, fileName)
        except Exception as e:
            return
    statinfo = os.stat(fileName)
    if statinfo.st_size == expectedSize:
        print("Found and Verified", fileName)
    else:
        print(statinfo.st_size)
        raise Exception("Failed to verify " + fileName)
    return fileName

def read_data(fileName):
    with zipfile.ZipFile(fileName) as fp:
        data = tf.compat.as_str(fp.read(fp.namelist()[0])).split()
    return data

def build_dataSet(words):
    '''
            该函数返回四个值，第一个值data用来表示该篇文章中所有词出现的频度，所一个词没有排在最常出现的前5000名则该位上置0，否则
            置这个词出现频度的排位，排位越靠前说明出现的频度越大。
    count用来表示出现频度前5000名的词的出现次数。
    dic表示出现频度最高的前5000个词的排序，排序序号越小则出现频度越高，以单词为索引
    reverse_dic是dic的键值倒置字典，以出现频度为索引
    '''
    count = [["UNK", -1]]
    count.extend(collections.Counter(words).most_common(vocabularySize - 1))
    dic = dict()
    for word, num in count:
        dic[word] = len(dic)
    data = []
    unk_count = 0
    for word in words:
        index = dic.get(word, 0)
        unk_count += 1 if index == 0 else 0
        data.append(index)
    count[0][1] = unk_count
    reverse_dic = dict(zip(dic.values(), dic.keys()))
    return data, count, dic, reverse_dic

dataIndex = 0
def generate_batch(batchSize, skipWindow, numSkip, data):
    global dataIndex
    assert not batchSize % numSkip, "样本规模大小必须为numSkip的整数倍"
    assert numSkip <= skipWindow * 2, "numSkip的大小必须不大于skipWindow的两倍"
    batch = np.ndarray(shape=(batchSize), dtype=np.int32)
    labels = np.ndarray(shape=(batchSize, 1), dtype=np.int32)
    span = 2 * skipWindow + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[dataIndex])
        dataIndex = (dataIndex + 1) % len(data)
    for i in range(batchSize // numSkip):
        target = skipWindow
        target_to_void = [target]
        for j in range(numSkip):
            while target in target_to_void:
                target = random.randint(0, span - 1)
            target_to_void.append(target)
            batch[i * numSkip + j] = buffer[skipWindow]
            labels[i * numSkip + j, 0] = buffer[target]
        buffer.append(dataIndex)
        dataIndex = (dataIndex + 1) % len(data)
    return batch, labels

batchSize = 128#选取的样本规模大小为128
embeddingSize = 128#生成稠密向量的维度为128
skipWindow = 1#单词关联程度为1
numSkip = 2#与目标单词关联的单词数

vaildSize = 16#用来测试的单词规模
vaildWindow = 100#抽取前100个出现频率最高的词汇
vaildExamples = np.random.choice(vaildWindow, vaildSize, replace=False)#随机抽取vaildSize个单词索引
numSampled = 64#噪声词汇的数目

if __name__ == "__main__":
#     fileName = "text8.zip"
#     download_file(fileName, 31344016)
#     words = read_data(fileName)
#     data, count, dic, reverse_dic = build_dataSet(words)
#     print("Most Common Words (+UNK): ", count[:5])
#     print("Sample data: ", data[: 10], [reverse_dic[i] for i in data[: 10]])
#     del words
#     batch, labels = generate_batch(batchSize, skipWindow, numSkip, data)
    graph = tf.Graph()
    with graph.as_default():
        trainInputs = tf.placeholder(tf.int32, [batchSize])
        trainLabels = tf.placeholder(tf.int32, [batchSize, 1])
        vaildDataSet = tf.constant(vaildExamples, tf.int32)
        with tf.device("/cpu:0"):
            embeddings = tf.Variable(
                tf.random_uniform((vocabularySize, embeddingSize), -1, 1, tf.float32))
            embed = tf.nn.embedding_lookup(embeddings, trainInputs)
            nceWeight = tf.Variable(tf.truncated_normal
                        ([vocabularySize, embeddingSize], stddev =  1.0 / math.sqrt(embeddingSize)))
            nceBiases = tf.Variable(tf.zeros([vocabularySize]))            
        
          