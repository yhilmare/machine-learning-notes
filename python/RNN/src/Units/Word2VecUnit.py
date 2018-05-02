'''
Created on 2018年4月28日

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
import shelve

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
            该函数返回四个值，第一个值data用来表示该篇文章中所有词出现的频度，
            所一个词没有排在最常出现的前5000名则该位上置0，否则
            置这个词出现频度的排位，排位越靠前说明出现的频度越大。
    count用来表示出现频度前5000名的词的出现次数。
    dic表示出现频度最高的前5000个词的排序，排序序号越小则出现频度越高，
            以单词为索引reverse_dic是dic的键值倒置字典，以出现频度为索引
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
        buffer.append(data[dataIndex])
        dataIndex = (dataIndex + 1) % len(data)
    return batch, labels

batchSize = 128#选取的样本规模大小为128
embeddingSize = 128#生成稠密向量的维度为128
skipWindow = 2#单词关联程度为1
numSkip = 4#与目标单词关联的单词数

vaildSize = 16#用来测试的单词规模
vaildWindow = 100#抽取前100个出现频率最高的词汇
vaildExamples = np.random.choice(vaildWindow, vaildSize, replace=False)#随机抽取vaildSize个单词索引
numSampled = 64#噪声词汇的数目

if __name__ == "__main__":
    fileName = "text8.zip"
    download_file(fileName, 31344016)
    words = read_data(fileName)
    data, count, dic, reverse_dic = build_dataSet(words)
    print("Most Common Words (+UNK): ", count[:5])
    print("Sample data: ", data[: 10], [reverse_dic[i] for i in data[: 10]])
    del words
    graph = tf.Graph()
    with graph.as_default():
        trainInputs = tf.placeholder(tf.int32, [batchSize])
        trainLabels = tf.placeholder(tf.int32, [batchSize, 1])
        vaildDataSet = tf.constant(vaildExamples, tf.int32)
        with tf.device("/cpu:0"):
            embeddings = tf.Variable(
                tf.random_uniform([vocabularySize, embeddingSize], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, trainInputs)
            nceWeight = tf.Variable(tf.truncated_normal
                        ([vocabularySize, embeddingSize], stddev=1.0 / math.sqrt(embeddingSize)))
            nceBiases = tf.Variable(tf.zeros([vocabularySize]))
            nceLoss = tf.reduce_mean(tf.nn.nce_loss(nceWeight, 
                                     nceBiases, 
                                     trainLabels, 
                                     embed, 
                                     numSampled, 
                                     vocabularySize))
            optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(nceLoss)
            normal = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            normalized_embeddings = embeddings / normal
            vaild_embeddings = tf.nn.embedding_lookup(normalized_embeddings, vaildDataSet)
            similarity = tf.matmul(vaild_embeddings, tf.transpose(normalized_embeddings))
            init = tf.global_variables_initializer()
            num_step = 100000
            with tf.Session(graph=graph) as sess:
                init.run()
                print("initialized...")
                average_loss = 0
                for step in range(num_step):
                    batch, labels = generate_batch(batchSize, skipWindow, numSkip, data)
                    feed_dict = {trainInputs: batch, trainLabels: labels}
                    _, loss_val = sess.run([optimizer, nceLoss], feed_dict=feed_dict)
                    average_loss += loss_val
                    if step % 2000 == 0:
                        average_loss /= 2000.0
                        print("Average loss at step ", step, " is " , average_loss)
                        average_loss = 0
                    if step % 10000 == 0:
                        sim = similarity.eval()
                        for i in range(vaildSize):
                            vaildWord = reverse_dic[vaildExamples[i]]
                            top_k = 8
                            nearst = (-sim[i, :]).argsort()[1: top_k + 1]
                            log_str = "Nearst to %s: " % (vaildWord)
                            for j in range(top_k):
                                close_word = reverse_dic[nearst[j]]
                                possible = -sim[i, :][nearst[j]]
                                log_str = "%s %s - <%.3f>," % (log_str, close_word, -possible)
                            print(log_str)
                final_embeddings = normalized_embeddings.eval()
                print(final_embeddings, final_embeddings.shape)
                sim = similarity.eval()
                for i in range(vaildSize):
                    row = -sim[i, :]
                    log_str = "Nearst to %s :" % (reverse_dic[vaildExamples[i]])
                    for index in row.argsort()[1: 9]:
                        possible = row[index]
                        str = reverse_dic[index]
                        log_str = "%s %s - <%.3f>," % (log_str, str, -possible)
                    print(log_str)
                with shelve.open("parameter") as fp:
                    fp["word2vec"] = final_embeddings
                    fp["reverse_dic"] = reverse_dic
                    fp["dic"] = dic