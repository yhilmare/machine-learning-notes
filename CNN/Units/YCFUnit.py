'''
Created on 2018年7月30日

@author: IL MARE
'''
import tensorflow as tf
import os
import numpy as np
import re
import time
import random
from PIL import Image

save_path = r"G:/logs/"
file_path = r"G:/研究生课件/人工神经网络/神经网络/dataset_cat_dog_classification/dataset/"

BATCH_SIZE = 16

class YCFNet:
    def __init__(self, lr, k, classify, maxIter, imageObject):
        self._imageObject = imageObject
        self._maxIter = maxIter
        self._k = k
        self._lr = lr
        self._classify = classify
        self.defineNetwork()
        self.defineLoss()
    @property
    def classify(self):
        return self._classify
    @property
    def keep_prob(self):
        return self._keep_prob
    @property
    def lr(self):
        return self._lr
    def defineNetwork(self):
        with tf.variable_scope('conv1') as scope:
            self.images = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 208, 208, self._k],name='input')
            self.labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, self._classify],name='labels')
            weights = tf.get_variable('weights',
                                     shape = [3,3,self._k,16],
                                     dtype = tf.float32,
                                     initializer = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape = [16],
                                     dtype = tf.float32,
                                    initializer = tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(self.images, weights, strides=[1,1,1,1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name = scope.name)
        with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize = [1,3,3,1], strides = [1,2,2,1],
                                  padding = 'SAME', name = 'pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias = 1.0, alpha=0.001/9.0,
                             beta=0.75,name='norm1')
        with tf.variable_scope('conv2') as scope:
            weights = tf.get_variable('weights',
                                     shape=[3,3,16,16],
                                     dtype=tf.float32,
                                     initializer = tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            biases = tf.get_variable('biases',
                                    shape=[16],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1], padding = 'SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name='conv2')
        with tf.variable_scope('pooling2_lrn') as scope:
            norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha = 0.001/9.0,
                             beta=0.75,name='norm2')
            pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,1,1,1],
                                  padding='SAME',name='pooling2')
        with tf.variable_scope('local3') as scope:
            reshape = tf.reshape(pool2,shape=[16, -1])
            dim = reshape.get_shape()[1].value
            weights = tf.get_variable('weights',
                                     shape=[dim,128],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                    shape=[128],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name=scope.name)
        with tf.variable_scope('local4') as scope:
            weights = tf.get_variable('weights',
                                     shape=[128,128],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                    shape=[128],
                                    dtype = tf.float32,
                                    initializer = tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3,weights)+biases, name='local4')
        with tf.variable_scope('softmax_layer') as scope:
            weights = tf.get_variable('softmax_linear',
                                     shape=[128,self._classify],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
            biases = tf.get_variable('biases',
                                    shape=[self._classify],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.1))
            softmax_liner=tf.add(tf.matmul(local4, weights),biases,name='softmax_linear')
            out = tf.matmul(local4, weights) + biases
            self.pre = tf.nn.softmax(out)
    def defineLoss(self):
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.pre, labels=self.labels)
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            tf.summary.scalar(scope.name+'/loss', self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
            self.train_op = optimizer.minimize(self.loss)
            self._accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels, 1), 
                                                             tf.argmax(self.pre, 1)), dtype=tf.float32))
    def train(self):
        try:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(self._maxIter):
                    train, label = self._imageObject.nextBatch(BATCH_SIZE)
                    _, accuracy, loss = sess.run([self.train_op, self._accuracy, self.loss], feed_dict={self.images: train, 
                                                     self.labels: label})
                    if i % 10 == 0:
                        print("step {0:d}/{1:d},accuracy: {2:.3f}, loss: {3:.3f}".format(i, self._maxIter, accuracy, loss))
                    if i % 250 == 0:    
                        tf.train.Saver().save(sess, "{0}model".format(save_path), global_step=i)
        except Exception as e:
            print(e)
    def loadModel(self):
        self._sess = tf.Session()
        print(tf.train.latest_checkpoint(save_path))
        tf.train.Saver().restore(self._sess, tf.train.latest_checkpoint(save_path))
    def testCatAndDog(self):
        result = []
        for img, label in self._imageObject.generateTestBatch(BATCH_SIZE):
            accuracy, _, loss = self._sess.run([self._accuracy, self.pre, self.loss], 
                                 feed_dict={self.images: img, self.labels: label})
            result.append(accuracy)
            print("step:{0:d}, accuracy: {1:.3f}, loss: {2: .3f}".format(len(result), accuracy, loss))
        print("average accuracy: {0:.3f}".format(np.mean(np.array(result))))


class ImageObject:
    def __init__(self, filePath, shape=(224, 224)):
        self._shape = shape
        self._filePath = filePath
        self.generateDataSet()
    def generateDataSet(self):
        list = os.listdir(self._filePath)
        self._train_path = "{0}{1}".format(self._filePath, "train/")
        self._test_path = "{0}{1}".format(self._filePath, "test/")
        if not os.path.exists(self._train_path) or not os.path.exists(self._test_path):
            os.mkdir(self._train_path)
            os.mkdir(self._test_path)
        if os.listdir(self._train_path) and os.listdir(self._test_path):
            self._trainSet = set(os.listdir(self._train_path))
            self._testSet = set(os.listdir(self._test_path))
            return
        print("正在初始化训练集和测试集。。。")
        self._trainSet = set()
        self._testSet = set(list) - set(["train", "test"])
        for i in range(len(list)):
            if i % 500 == 0:
                print(i)
            index = np.random.randint(0, len(list), 1)[0]
            item = list[index]
            if item == "test" or item == "train":
                continue
            if item not in self._trainSet:
                self._trainSet.add(item)
                image = Image.open("{0}{1}".format(self._filePath, item))
                image = image.resize(self._shape)
                image.save("{0}{1}".format(self._train_path, item))
        self._testSet = self._testSet - self._trainSet
        i = 0
        for name in self._testSet:
            i += 1
            if i % 500 == 0:
                print(i)
            image = Image.open("{0}{1}".format(self._filePath, name))
            image = image.resize(self._shape)
            image.save("{0}{1}".format(self._test_path, name))
    def nextBatch(self, num=50):
        random.seed(time.time())
        list = random.sample(self._trainSet, num)
        train = []
        label = []
        for name in list:
            image = Image.open("{0}{1}".format(self._train_path, name))
            train.append(np.asarray(image))
            if re.match(r"^cat.*$", name):
                label.append(np.array([1, 0]))
            else:
                label.append(np.array([0, 1]))
        return np.array(train), np.array(label)
    def generateTestBatch(self, num=100):
        test = []
        label = []
        for name in self._testSet:
            image = Image.open("{0}{1}".format(self._test_path, name))
            test.append(np.asarray(image))
            if re.match(r"^cat.*$", name):
                label.append(np.array([1, 0]))
            else:
                label.append(np.array([0, 1]))
            if len(test) % num == 0:
                yield np.array(test), np.array(label)
                test = []
                label = []
        yield np.array(test), np.array(label)

if __name__ == "__main__":
    obj = ImageObject(file_path, shape=(208, 208))
    alex = YCFNet(0.0001, 3, 2, 20000, obj)
    alex.train()
#     alex.loadModel()
#     alex.testCatAndDog()