'''
Created on 2018年7月2日

@author: IL MARE
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from Utils.DataUtil import ImageObject
from matplotlib import pyplot as plt
import matplotlib as mpl

save_path = r"G:/Machine-Learning/python/CNN/modelFile/AlexNet/dogandcat/"

def weight_variable(shape):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, dtype=tf.float32)

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
            kernel = tf.Variable(tf.truncated_normal([11, 11, self._k, 96], 
                                                     stddev=0.1, dtype=tf.float32))
            h_conv1 = conv2d(image, kernel, [1, 4, 4, 1])
            biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32))
            conv1 = tf.nn.relu(h_conv1 + biases)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], 
                                   strides=[1, 2, 2, 1], padding="SAME")
        with tf.name_scope("conv2") as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], 
                                                     stddev=0.1, dtype=tf.float32))
            h_conv2 = conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding="SAME")
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32))
            conv2 = tf.nn.relu(h_conv2 + biases)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], 
                                   strides=[1, 2, 2, 1], padding="SAME")
        with tf.name_scope("conv3") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384], 
                                                      stddev=0.1, dtype=tf.float32))
            h_conv3 = conv2d(pool2, kernel, [1, 1, 1, 1], "SAME")
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[384]))
            conv3 = tf.nn.relu(h_conv3 + biases)
        with tf.name_scope("conv4") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], 
                                                     dtype=tf.float32, stddev=0.1))
            h_conv4 = conv2d(conv3, kernel, [1, 1, 1, 1], "SAME")
            biases= tf.Variable(tf.constant(0.0, tf.float32, shape=[384]), dtype=tf.float32)
            conv4 = tf.nn.relu(h_conv4 + biases)
        with tf.name_scope("conv5") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], 
                                                     dtype=tf.float32, stddev=0.1))
            h_conv5 = conv2d(conv4, kernel, [1, 1, 1, 1], "SAME")
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]))
            conv5 = tf.nn.relu(h_conv5 + biases)
            pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], 
                                   strides=[1, 2, 2, 1], padding="SAME")
            self._dim = 1
            var = pool5.get_shape().as_list()
            for i in range(len(var) - 1):
                self._dim *= var[i + 1]
            pool5 = tf.reshape(pool5, [-1, self._dim])
        with tf.name_scope("link1") as scope:
            kernel = tf.Variable(tf.truncated_normal([self._dim, 4096], stddev=0.1, dtype=tf.float32))
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[4096]))
            h_fc = tf.nn.dropout(tf.matmul(pool5, kernel) + biases, keep_prob=self._keep_prob)
        with tf.name_scope("link2") as scope:
            kernel = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.1, dtype=tf.float32))
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[4096]))
            h_fc1 = tf.nn.dropout(tf.matmul(h_fc, kernel) + biases, keep_prob=self._keep_prob)
        with tf.name_scope("link3") as scope:
            kernel = tf.Variable(tf.truncated_normal([4096, self._classify], stddev=0.1, dtype=tf.float32))
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self._classify]))
            self._out = tf.matmul(h_fc1, kernel) + biases
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
    def train(self):
        try:
#             mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            fig = plt.figure("cross-entropy")
            mpl.rcParams['xtick.labelsize'] = 8
            mpl.rcParams['ytick.labelsize'] = 8
            ax = fig.add_subplot(111)
            ax.grid(True)
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
                    ax.plot(np.arange(len(ac)), np.array(ac), linewidth=0.8, color="b")
                    ax.plot(np.arange(len(aac)), np.array(aac), linewidth=0.8, color="r")
                    plt.pause(0.1)
                    if i % 10 == 0:
                        print("step {0:d}/{1:d},accuracy: {2:.3f}, loss: {3:.3f}".format(i, self._maxIter, accuracy, loss))
                    if i % 100 == 0:    
                        tf.train.Saver().save(sess, "{0}model".format(save_path), global_step=i)
        except Exception as e:
            print(e)
        finally:
            plt.show()
    def loadModel(self):
        self._sess = tf.Session()
        tf.train.Saver().restore(self._sess, tf.train.latest_checkpoint(save_path))
    def testCatAndDog(self):
        result = []
        for img, label in self._imageObject.generateTestBatch(50):
            accuracy = self._sess.run(self._accuracy, 
                                 feed_dict={self._x: img, self._y: label, self._keep_prob:1.0})
            result.append(accuracy)
            print("step:{0:d}, accuracy: {1:.3f}".format(len(result), accuracy))
        print("average accuracy:", np.mean(np.array(result)))
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

file_path = r"G:/研究生课件/人工神经网络/神经网络/dataset_cat_dog_classification/dataset/"

if __name__ == "__main__":
    obj = ImageObject(file_path)
    alex = AlexNet(0.0001, 3, 2, 2000, obj)
    alex.train()
#     alex.loadModel()
#     alex.testCatAndDog()