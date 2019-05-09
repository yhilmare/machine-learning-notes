'''
Created on 2018年7月1日

@author: IL MARE
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

save_path = r"G:/Machine-Learning/python/CNN/modelFile/SimpleCNN/"

def weight_variable(shape):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, dtype=tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

class SimpleCNN:
    def __init__(self, lr, maxIter):
        self._maxIter = maxIter
        self._lr = lr
        self.defineNetWork()
        self.defineLoss()
    @property
    def lr(self):
        return self._lr
    @property
    def keep_prob(self):
        return self._keep_prob
    def defineNetWork(self):
        self._x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self._y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        x_image = tf.reshape(self._x, [-1, 28, 28, 1])
        with tf.name_scope("conv1") as scope:
            kernal = weight_variable([5, 5, 1, 32])
            biases = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(x_image, kernal) + biases)
            h_pool1 = max_pool_2x2(h_conv1)
        with tf.name_scope("conv2") as scope:
            kernal = weight_variable([5, 5, 32, 64])
            biases = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, kernal) + biases)
            h_pool2 = max_pool_2x2(h_conv2)
        with tf.name_scope("link1") as scope:
            kernal = weight_variable([7 * 7 * 64, 1024])
            biases = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, kernal) + biases)
            self._keep_prob = tf.placeholder(dtype=tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=self._keep_prob)
        with tf.name_scope("link2") as scope:
            kernal = weight_variable([1024, 10])
            biases = bias_variable([10])
            self._out = tf.matmul(h_fc1_drop, kernal) + biases
            self._y_conv = tf.nn.softmax(self._out)
    def defineLoss(self):
        self._cross_entry = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=self._out))
#         self._cross_entry = tf.reduce_mean(-tf.reduce_sum(self._y * tf.log(self._y_conv), 
#                                                     reduction_indices=[1]))
#         vars = tf.trainable_variables()
#         grads, _ = tf.clip_by_global_norm(tf.gradients(self._cross_entry, vars), 5)
#         optimizer = tf.train.AdamOptimizer(self._lr)
#         self._train_step = optimizer.apply_gradients(zip(grads, vars))
        self._train_step = tf.train.AdamOptimizer(self._lr).minimize(self._cross_entry)
        correct_prediction = tf.equal(tf.argmax(self._y_conv, 1), tf.argmax(self._y, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    def train(self):
        try:
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
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
                    train, label = mnist.train.next_batch(50)
                    _, accuracy, loss = sess.run([self._train_step, self._accuracy, self._cross_entry], feed_dict={self._x: train, 
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
    def test(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        count = 0
        i = 0
        for img, label in zip(mnist.test.images, mnist.test.labels):
            img = np.reshape(img, [1, 784])
            label = np.reshape(label, [1, 10])
            pre = self._sess.run(self._y_conv, 
                                 feed_dict={self._x: img, self._y: label, self._keep_prob:1.0})
            if np.equal(np.argmax(pre, 1), np.argmax(label, 1)):
                count += 1
            i += 1
            if i % 100 == 0:
                print("step: {0:d}/{1:d}, accuracy: {2:.3f}".format(i, len(mnist.test.images), count / i))
        print("accuracy: ", (count / i))

file_path = r"G:/研究生课件/人工神经网络/神经网络/dataset_cat_dog_classification/dataset/"

if __name__ == "__main__":
    cnn = SimpleCNN(0.001, 2000)
    cnn.loadModel()
    cnn.test()