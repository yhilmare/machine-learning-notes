'''
Created on 2018年7月2日

@author: IL MARE
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from Utils.DataUtil import ImageObject
from matplotlib import pyplot as plt
import matplotlib as mpl
from PIL import Image

save_path = r"G:/Machine-Learning/python/CNN/modelFile/AlexNet/dogandcat/"
file_path = r"G:/研究生课件/人工神经网络/神经网络/dataset_cat_dog_classification/dataset/"

def weight_variable(shape):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
    return tf.Variable(initial)

def conv2d(x, W, strides, padding="VALID"):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

class AlexNet:
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
        self._x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        image = self._x / 255.0
#         image = tf.reshape(self._x, [-1, 28, 28, self._k])
        self._y = tf.placeholder(tf.float32, [None, self._classify])
        self._keep_prob = tf.placeholder(dtype=tf.float32)
        with tf.name_scope("conv1") as scope:    
            kernel = tf.Variable(tf.truncated_normal([5, 5, self._k, 32], 
                                                     stddev=0.1, dtype=tf.float32))
            h_conv1 = conv2d(image, kernel, [1, 4, 4, 1])
#             biases = tf.Variable(tf.constant(0.1, shape=[32], dtype=tf.float32))
            biases = tf.Variable(tf.random_normal(shape=[32], stddev=0.1, dtype=tf.float32))
            conv1 = tf.nn.relu(h_conv1 + biases)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], 
                                   strides=[1, 2, 2, 1], padding="SAME")
        with tf.name_scope("conv2") as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], 
                                                     dtype=tf.float32, stddev=0.1))
            h_conv5 = conv2d(pool1, kernel, [1, 1, 1, 1], "SAME")
            biases = tf.Variable(tf.random_normal(shape=[64], stddev=0.1, dtype=tf.float32))
#             biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64]))
            conv5 = tf.nn.relu(h_conv5 + biases)
            pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], 
                                   strides=[1, 2, 2, 1], padding="SAME")
            self._dim = 1
            var = pool5.get_shape().as_list()
            for i in range(len(var) - 1):
                self._dim *= var[i + 1]
            pool5 = tf.reshape(pool5, [-1, self._dim])
        with tf.name_scope("link1") as scope:
            kernel = tf.Variable(tf.truncated_normal([self._dim, 1024], stddev=0.1, dtype=tf.float32))
            biases = tf.Variable(tf.random_normal(shape=[1024], stddev=0.1, dtype=tf.float32))
#             biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[1024]))
            h_fc = tf.nn.dropout(tf.matmul(pool5, kernel) + biases, keep_prob=self._keep_prob)
        with tf.name_scope("link3") as scope:
            kernel = tf.Variable(tf.truncated_normal([1024, self._classify], stddev=0.1, dtype=tf.float32))
            biases = tf.Variable(tf.random_normal(shape=[self._classify], stddev=0.1, dtype=tf.float32))
            self._out = tf.matmul(h_fc, kernel) + biases
            self._pre = tf.nn.softmax(self._out)
    def defineLoss(self):
        self._cross_entry = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self._out, labels=self._y))
        self._accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self._y, 1), 
                                                             tf.argmax(self._pre, 1)), dtype=tf.float32))
#         vars = tf.trainable_variables()
#         grads, _ = tf.clip_by_global_norm(tf.gradients(self._cross_entry, vars), 5)
#         optimizer = tf.train.AdamOptimizer(self._lr)
#         self._train = optimizer.apply_gradients(zip(grads, vars))
        self._train = tf.train.AdamOptimizer(self._lr).minimize(self._cross_entry)
    def train_1(self):
        try:
            fig = plt.figure("cross-entropy")
            mpl.rcParams['xtick.labelsize'] = 8
            mpl.rcParams['ytick.labelsize'] = 8
            ax = fig.add_subplot(111)
            ax.grid(True)
            ac = []
            aac = []
            for i in range(self._maxIter):
                train, label = self._imageObject.nextBatch(24)
                _, accuracy, loss = self._sess.run([self._train, self._accuracy, self._cross_entry], feed_dict={self._x: train, 
                                                     self._y: label, self._keep_prob: 0.5})
                ac.append(accuracy)
                aac.append(np.mean(np.array(ac)))
                ax.plot(np.arange(len(ac)), np.array(ac), linewidth=0.8, color="b")
                ax.plot(np.arange(len(aac)), np.array(aac), linewidth=0.8, color="r")
                plt.pause(0.1)
                if i % 10 == 0:
                    print("step {0:d}/{1:d},accuracy: {2:.3f}, loss: {3:.3f}".format(i, self._maxIter, accuracy, loss))
                if i % 250 == 0:    
                    tf.train.Saver().save(self._sess, "{0}model".format(save_path), global_step=i)
        except Exception as e:
            print(e)
        finally:
            plt.show()
    def train(self):
        try:
#             mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#             fig = plt.figure("cross-entropy")
#             mpl.rcParams['xtick.labelsize'] = 8
#             mpl.rcParams['ytick.labelsize'] = 8
#             ax = fig.add_subplot(111)
#             ax.grid(True)
            ac = []
            aac = []
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(self._maxIter):
#                     train, label = mnist.train.next_batch(50)
                    train, label = self._imageObject.nextBatch(24)
                    _, accuracy, loss = sess.run([self._train, self._accuracy, self._cross_entry], feed_dict={self._x: train, 
                                                     self._y: label, self._keep_prob: 0.5})
                    ac.append(accuracy)
                    aac.append(np.mean(np.array(ac)))
#                     ax.plot(np.arange(len(ac)), np.array(ac), linewidth=0.8, color="b")
#                     ax.plot(np.arange(len(aac)), np.array(aac), linewidth=0.8, color="r")
#                     plt.pause(0.1)
                    if i % 10 == 0:
                        print("step {0:d}/{1:d},accuracy: {2:.3f}, loss: {3:.3f}".format(i, self._maxIter, accuracy, loss))
                    if i % 250 == 0:    
                        tf.train.Saver().save(sess, "{0}model".format(save_path), global_step=i)
        except Exception as e:
            print(e)
        finally:
            fig = plt.figure("cross-entropy")
            mpl.rcParams['xtick.labelsize'] = 8
            mpl.rcParams['ytick.labelsize'] = 8
            ax = fig.add_subplot(111)
            ax.plot(np.arange(len(ac)), np.array(ac), linewidth=0.8, color="b")
            ax.plot(np.arange(len(aac)), np.array(aac), linewidth=0.8, color="r")
            plt.show()
    def loadModel(self):
        self._sess = tf.Session()
        print(save_path)
        print(tf.train.latest_checkpoint(save_path))
        tf.train.Saver().restore(self._sess, tf.train.latest_checkpoint(save_path))
    def testCatAndDog(self):
        result = []
        for img, label in self._imageObject.generateTestBatch(200):
            accuracy, pre, loss = self._sess.run([self._accuracy, self._pre, self._cross_entry], 
                                 feed_dict={self._x: img, self._y: label, self._keep_prob:1.0})
#             for i in range(len(label)):
#                 lab = label[i]
#                 predict = pre[i]
#                 image = img[i]
#                 if np.argmax(predict) == 0:
#                     tmp = Image.fromarray(image)
#                     if np.argmax(lab) == 0:
#                         tmp.save("g:/dogandcat/cat/cat-{0}-{1}.jpg".format(len(result), i))
#                     else:
#                         tmp.save("g:/dogandcat/cat/dog-{0}-{1}.jpg".format(len(result), i))
#                 else:
#                     tmp = Image.fromarray(image)
#                     if np.argmax(lab) == 0:
#                         tmp.save("g:/dogandcat/dog/cat-{0}-{1}.jpg".format(len(result), i))
#                     else:
#                         tmp.save("g:/dogandcat/dog/dog-{0}-{1}.jpg".format(len(result), i))
            
            result.append(accuracy)
            print("step:{0:d}, accuracy: {1:.3f}, loss: {2: .3f}".format(len(result), accuracy, loss))
        print("average accuracy: {0:.3f}".format(np.mean(np.array(result))))
    def test(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        count = 0
        i = 0
        for img, label in zip(mnist.test.images, mnist.test.labels):
            img = np.reshape(img, [1, 784])
            label = np.reshape(label, [1, 10])
            pre = self._sess.run(self._pre, 
                                 feed_dict={self._x: img, self._y: label, self._keep_prob:1.0})
            if np.equal(np.argmax(pre, 1), np.argmax(label, 1)):
                count += 1
            i += 1
            if i % 100 == 0:
                print("step: {0:d}/{1:d}, accuracy: {2:.3f}".format(i, len(mnist.test.images), count / i))
        print("accuracy: ", (count / i))

if __name__ == "__main__":
    obj = ImageObject(file_path)
    alex = AlexNet(0.0001, 3, 2, 20000, obj)
    alex.train()
#     alex.loadModel()
#     alex.testCatAndDog()