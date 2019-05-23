'''
Created By ILMARE
@Date 2019-2-27
'''
import sys
import os

sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
from Tools.DataObject import ImageTrainObject
import datetime
import re
import matplotlib as mpl
import os
from Tools.DataObject import get_training_data

class AutoEncoder:
    def __init__(self, learning_rate, max_step,
                 batch_size, modelPath, channel=3):
        self._modelPath = modelPath
        # if re.match(r"^/.+/[^.]+$", self._modelPath) is None:
        #     raise Exception("filePath is invalid")
        if self._modelPath[len(self._modelPath) - 1] != '/':
            self._modelPath += '/'
        self._channel = channel
        self._learning_rate = learning_rate
        self._max_step = max_step
        self._batch_size = batch_size
        self._sess = tf.Session()
        self.define_network()
        self.define_loss()
    def define_network(self):
        self._x = tf.placeholder(shape=[None, 64, 64, self._channel], dtype=tf.float32)
        self._input = tf.placeholder(shape=[None, 64, 64, self._channel], dtype=tf.float32)
        def define_encoder():
            with tf.name_scope("conv1"):
                conv1_kernal = tf.get_variable(name="conv1_kernal", initializer=tf.glorot_uniform_initializer,
                                               shape=[5,5,self._channel,128], dtype=tf.float32)
                h_conv1 = tf.nn.conv2d(self._input, conv1_kernal, strides=[1,2,2,1], padding="SAME")
                conv1_b = tf.Variable(initial_value=tf.zeros([128], dtype=tf.float32))
                conv1 = tf.nn.leaky_relu(tf.nn.bias_add(h_conv1, conv1_b), alpha=0.1)
            with tf.name_scope("conv2"):
                conv2_kernal = tf.get_variable(name="conv2_kernal", initializer=tf.glorot_uniform_initializer,
                                               shape=[5,5,128,256], dtype=tf.float32)
                h_conv2 = tf.nn.conv2d(conv1, conv2_kernal, strides=[1,2,2,1], padding="SAME")
                conv2_b = tf.Variable(initial_value=tf.zeros([256], dtype=tf.float32))
                conv2 = tf.nn.leaky_relu(tf.nn.bias_add(h_conv2, conv2_b), alpha=0.1)
            with tf.name_scope("conv3"):
                conv3_kernal = tf.get_variable(name="conv3_kernal", initializer=tf.glorot_uniform_initializer,
                                               shape=[5,5,256,512], dtype=tf.float32)
                h_conv3 = tf.nn.conv2d(conv2, conv3_kernal, strides=[1,2,2,1], padding="SAME")
                conv3_b = tf.Variable(initial_value=tf.zeros([512], dtype=tf.float32))
                conv3 = tf.nn.leaky_relu(tf.nn.bias_add(h_conv3, conv3_b), alpha=0.1)
            with tf.name_scope("conv4"):
                conv4_kernal = tf.get_variable(name="conv4_kernal", initializer=tf.glorot_uniform_initializer,
                                               shape=[5,5,512,1024], dtype=tf.float32)
                h_conv4 = tf.nn.conv2d(conv3, conv4_kernal, strides=[1,2,2,1], padding="SAME")
                conv4_b = tf.Variable(initial_value=tf.zeros([1024], dtype=tf.float32))
                conv4 = tf.nn.leaky_relu(tf.nn.bias_add(h_conv4, conv4_b), alpha=0.1)
                flatten1 = tf.layers.Flatten()(conv4)
                tmp_dim = int(flatten1.get_shape()[1])
            with tf.name_scope("dense1"):
                w1 = tf.get_variable(name="w1", initializer=tf.glorot_uniform_initializer,
                                     shape=[tmp_dim, 1024])
                b1 = tf.Variable(initial_value=tf.zeros([1024], dtype=tf.float32))
                link1 = tf.nn.bias_add(tf.matmul(flatten1, w1), b1)
            with tf.name_scope("dense2"):
                w2 = tf.get_variable(name="w2", initializer=tf.glorot_uniform_initializer,
                                     shape=[1024, tmp_dim])
                b2 = tf.Variable(initial_value=tf.zeros([tmp_dim], dtype=tf.float32))
                link2 = tf.nn.bias_add(tf.matmul(link1, w2), b2)
                link2 = tf.reshape(link2, [-1, 4, 4, 1024])
                self._tmp = tf.reduce_mean(link1)
            with tf.name_scope("upscale1"):
                upscale1_kernal = tf.Variable(initial_value=tf.truncated_normal(
                    [3, 3, 1024, 2048], stddev=0.1, dtype=tf.float32))
                upscale1_b = tf.Variable(initial_value=tf.zeros([2048], dtype=tf.float32))
                h_upscale1 = tf.nn.conv2d(link2, upscale1_kernal, strides=[1, 1, 1, 1], padding="SAME")
                upscale1 = tf.nn.bias_add(h_upscale1, upscale1_b)
                upscale1 = tf.depth_to_space(upscale1, 2)
                self._hidden = upscale1
        def define_decoder1():
            with tf.name_scope("upscale21"):
                upscale2_kernal = tf.get_variable(name="upscale2_kernal", initializer=tf.glorot_uniform_initializer,
                                                  shape=[3,3,512,1024], dtype=tf.float32)
                upscale2_b = tf.Variable(initial_value=tf.zeros([1024], dtype=tf.float32))
                h_upscale2 = tf.nn.conv2d(self._hidden, upscale2_kernal, strides=[1,1,1,1], padding="SAME")
                upscale2 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale2, upscale2_b), alpha=0.1)
                upscale2 = tf.depth_to_space(upscale2, 2)
            with tf.name_scope("upscale31"):
                upscale3_kernal = tf.get_variable(name="upscale3_kernal", initializer=tf.glorot_uniform_initializer,
                                                  shape=[3,3,256,512], dtype=tf.float32)
                upscale3_b = tf.Variable(initial_value=tf.zeros([512], dtype=tf.float32))
                h_upscale3 = tf.nn.conv2d(upscale2, upscale3_kernal, strides=[1,1,1,1], padding="SAME")
                upscale3 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale3, upscale3_b), alpha=0.1)
                upscale3 = tf.depth_to_space(upscale3, 2)
            with tf.name_scope("upscale41"):
                upscale4_kernal = tf.get_variable(name="upscale4_kernal", initializer=tf.glorot_uniform_initializer,
                                                  shape=[3,3,128,256], dtype=tf.float32)
                upscale4_b = tf.Variable(initial_value=tf.zeros([256], dtype=tf.float32))
                h_upscale4 = tf.nn.conv2d(upscale3, upscale4_kernal, strides=[1,1,1,1], padding="SAME")
                upscale4 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale4, upscale4_b), alpha=0.1)
                upscale4 = tf.depth_to_space(upscale4, 2)
            with tf.name_scope("conv51"):
                conv5_kernal = tf.get_variable(name="conv5_kernal", initializer=tf.glorot_uniform_initializer,
                                               shape=[5, 5, 64, self._channel], dtype=tf.float32)
                h_conv5 = tf.nn.conv2d(upscale4, conv5_kernal, strides=[1, 1, 1, 1], padding="SAME")
                conv5_b = tf.Variable(initial_value=tf.zeros([self._channel], dtype=tf.float32))
                conv5 = tf.nn.sigmoid(tf.nn.bias_add(h_conv5, conv5_b))
                self._reconstruct1 = conv5
        def define_decoder2():
            with tf.name_scope("upscale22"):
                upscale22_kernal = tf.get_variable(name="upscale22_kernal", initializer=tf.glorot_uniform_initializer,
                                                   shape=[3,3,512,1024], dtype=tf.float32)
                upscale22_b = tf.Variable(initial_value=tf.zeros([1024], dtype=tf.float32))
                h_upscale22 = tf.nn.conv2d(self._hidden, upscale22_kernal, strides=[1,1,1,1], padding="SAME")
                upscale22 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale22, upscale22_b), alpha=0.1)
                upscale22 = tf.depth_to_space(upscale22, 2)
            with tf.name_scope("upscale32"):
                upscale32_kernal = tf.get_variable(name="upscale32_kernal", initializer=tf.glorot_uniform_initializer,
                                                   shape=[3,3,256,512], dtype=tf.float32)
                upscale32_b = tf.Variable(initial_value=tf.zeros([512], dtype=tf.float32))
                h_upscale32 = tf.nn.conv2d(upscale22, upscale32_kernal, strides=[1,1,1,1], padding="SAME")
                upscale32 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale32, upscale32_b), alpha=0.1)
                upscale32 = tf.depth_to_space(upscale32, 2)
            with tf.name_scope("upscale42"):
                upscale42_kernal = tf.get_variable(name="upscale42_kernal", initializer=tf.glorot_uniform_initializer,
                                                   shape=[3,3,128,256], dtype=tf.float32)
                upscale42_b = tf.Variable(initial_value=tf.zeros([256], dtype=tf.float32))
                h_upscale42 = tf.nn.conv2d(upscale32, upscale42_kernal, strides=[1,1,1,1], padding="SAME")
                upscale42 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale42, upscale42_b), alpha=0.1)
                upscale42 = tf.depth_to_space(upscale42, 2)
            with tf.name_scope("conv52"):
                conv52_kernal = tf.get_variable(name="conv53_lernal", initializer=tf.glorot_uniform_initializer,
                                                shape=[5, 5, 64, self._channel], dtype=tf.float32)
                h_conv52 = tf.nn.conv2d(upscale42, conv52_kernal, strides=[1, 1, 1, 1], padding="SAME")
                conv52_b = tf.Variable(initial_value=tf.zeros([self._channel], dtype=tf.float32))
                conv52 = tf.nn.sigmoid(tf.nn.bias_add(h_conv52, conv52_b))
                self._reconstruct2 = conv52
        define_encoder()
        define_decoder1()
        define_decoder2()
    def reconstruct1(self, X):
        return self._sess.run(self._reconstruct1, feed_dict={self._x: X})
    def define_loss(self):
        self._loss1 = tf.reduce_mean(tf.pow(tf.subtract(self._x, self._reconstruct1), 2))
        self._optimizer1 = tf.train.AdamOptimizer(learning_rate=self._learning_rate,
                                                  beta1=0.5, beta2=0.999).minimize(self._loss1)
        self._loss2 = tf.reduce_mean(tf.pow(tf.subtract(self._x, self._reconstruct2), 2))
        self._optimizer2 = tf.train.AdamOptimizer(learning_rate=self._learning_rate,
                                                  beta1=0.5, beta2=0.999).minimize(self._loss2)
    def train(self, sourceTrainPath, destTrainPath):
        sourceImgObj = ImageTrainObject(sourceTrainPath, self._batch_size)
        sourceCount = sourceImgObj.DataCount // self._batch_size
        destImgObj = ImageTrainObject(destTrainPath, self._batch_size)
        destCount = destImgObj.DataCount // self._batch_size
        self._sess.run(tf.global_variables_initializer())
        # mpl.rcParams["xtick.labelsize"] = 6
        # mpl.rcParams["ytick.labelsize"] = 6
        # fig = plt.figure("cost")
        # ax = fig.add_subplot(211)
        # ax.grid(True)
        # bx = fig.add_subplot(212)
        # bx.grid(True)
        # sourceData = []
        # destData = []
        for step in range(self._max_step):
            source_avg_cost = 0
            dest_avg_cost = 0
            for time in range(sourceCount + destCount):
                warp, target = sourceImgObj.generateBatch()
                warp = warp / 255.0
                target = target / 255.0
                _, loss, tmp = self._sess.run([self._optimizer1, self._loss1, self._tmp], feed_dict={self._input: warp, self._x: target})
                source_avg_cost += (loss / (sourceCount + destCount))
                # sourceData.append(loss)
                # ax.plot(np.arange(len(sourceData)), np.sqrt(sourceData), linewidth=0.8, color="r")
                # plt.pause(0.01)
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- [source] cost: %3.5f" % loss,
                      "avg: %.5f" % tmp)
                warp, target = destImgObj.generateBatch()
                warp = warp / 255.0
                target = target / 255.0
                _, loss, tmp = self._sess.run([self._optimizer2, self._loss2, self._tmp], feed_dict={self._input: warp, self._x: target})
                dest_avg_cost += (loss / (sourceCount + destCount))
                # destData.append(loss)
                # bx.plot(np.arange(len(destData)), np.sqrt(destData), linewidth=0.8, color="b")
                # plt.pause(0.01)
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- [ dest ] cost: %3.5f" % loss,
                      "avg: %.5f" % tmp, "time:", time)
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- source_avg_cost: %3.5f" % source_avg_cost,
                  "- dest_avg_cost: %3.5f" % dest_avg_cost, "step:", step)
            tf.train.Saver().save(self._sess, save_path="{0}encoder".format(self._modelPath), global_step=step)
        # plt.show()
    def load_model(self):
        path = tf.train.latest_checkpoint(self._modelPath)
        print(path)
        tf.train.Saver().restore(self._sess, path)
    def showAll(self, batchSize=32):
        prePath = "F:/tensorflow/automodel/scrawler/video/trainImg/"
        resPath = r"F:/tensorflow/automodel/scrawler/video/res_2Img/"
        items = os.listdir(prePath)
        tmp_list = []
        tmp_list1 = []
        fileName_list = []
        totalFiles = len(items)
        for idx, file in enumerate(items):
            source = cv2.imread(r"{0}{1}".format(prePath, file))
            sour = cv2.resize(source, (64, 64))
            tmp_list1.append(sour)
            tmp_list.append(source)
            fileName_list.append(file)
            if len(tmp_list) == batchSize:
                sourceWarp, sourceTarget = get_training_data(np.array(tmp_list), batchSize)
                sourceWarp = sourceWarp / 255.0
                sourceTarget = sourceTarget / 255.0
                sour_target = np.array(tmp_list1, dtype=np.float32) / 255.0
                dest = self._sess.run(self._reconstruct2,
                                            feed_dict={self._x: sourceTarget, self._input: sour_target})
                for img, name in zip(dest, fileName_list):
                    dest = np.array(img * 255, dtype=np.uint8)
                    dest = cv2.resize(dest, (128, 128))
                    filePath = "{0}{1}".format(resPath, name)
                    cv2.imwrite(filePath, dest)
                    print(filePath)
                tmp_list = []
                tmp_list1 = []
                fileName_list = []
            elif idx == (totalFiles - 1):
                sourceWarp, sourceTarget = get_training_data(np.array(tmp_list), batchSize)
                sourceWarp = sourceWarp / 255.0
                sourceTarget = sourceTarget / 255.0
                sour_target = np.array(tmp_list1, dtype=np.float32) / 255.0
                dest = self._sess.run(self._reconstruct2,
                                      feed_dict={self._x: sourceTarget, self._input: sour_target})
                for img, name in zip(dest, fileName_list):
                    dest = np.array(img * 255, dtype=np.uint8)
                    dest = cv2.resize(dest, (128, 128))
                    filePath = "{0}{1}".format(resPath, name)
                    cv2.imwrite(filePath, dest)
                    print(filePath)
                tmp_list1 = []
                tmp_list = []
                fileName_list = []
    def generateImage(self):
        source = cv2.imread(r"F:/tensorflow/automodel/scrawler/video/trainImg/3524.jpg")
        sourceWarp, sourceTarget = get_training_data(np.array([source]), 1)
        print(sourceWarp.shape, sourceWarp.shape)
        sourceWarp = sourceWarp / 255.0
        sourceTarget = sourceTarget / 255.0
        source = cv2.resize(source, (64, 64))
        source = np.array([source], dtype=np.float32)
        source = source / 255.0
        dest, loss = self._sess.run([self._reconstruct2, self._loss1],
                                    feed_dict={self._x: sourceTarget, self._input: source})
        print(loss)
        sourceTarget = np.reshape(source, [64, 64, 3])
        dest = np.reshape(dest, [64, 64, 3])
        dest = np.array(dest * 255, dtype=np.uint8)
        fig = plt.figure("compare")
        ax = fig.add_subplot(121)
        b, g, r = cv2.split(sourceTarget)
        source = cv2.merge([r, g, b])
        ax.imshow(source)
        ax.axis("off")
        bx = fig.add_subplot(122)
        bx.axis("off")
        b, g, r = cv2.split(dest)
        dest = cv2.merge([r, g, b])
        bx.imshow(dest)
        plt.show()

def xavier_init(input_count, output_count, constant=1.0):
    low = -constant * np.sqrt(6.0 / (input_count + output_count))
    high = constant * np.sqrt(6.0 / (input_count + output_count))
    return tf.random_uniform((input_count, output_count),
                            minval=low, maxval=high,
                            dtype=tf.float32)

# class AutoEncoder:
#     def __init__(self, learning_rate, max_step,
#                  batch_size, modelPath, channel=3):
#         self._modelPath = modelPath
#         # if re.match(r"^/.+/[^.]+$", self._modelPath) is None:
#         #     raise Exception("filePath is invalid")
#         if self._modelPath[len(self._modelPath) - 1] != '/':
#             self._modelPath += '/'
#         self._channel = channel
#         self._learning_rate = learning_rate
#         self._max_step = max_step
#         self._batch_size = batch_size
#         self._sess = tf.Session()
#         self.define_network()
#         self.define_loss()
#     def define_network(self):
#         self._x = tf.placeholder(shape=[None, 64, 64, self._channel], dtype=tf.float32)
#         self._input = tf.placeholder(shape=[None, 64, 64, self._channel], dtype=tf.float32)
#         def define_encoder():
#             with tf.name_scope("conv1"):
#                 conv1_kernal = tf.get_variable(name="conv1_kernal", initializer=tf.glorot_uniform_initializer,
#                                                shape=[5,5,self._channel,64], dtype=tf.float32)
#                 h_conv1 = tf.nn.conv2d(self._input, conv1_kernal, strides=[1,2,2,1], padding="SAME")
#                 conv1_b = tf.Variable(initial_value=tf.zeros([64], dtype=tf.float32))
#                 conv1 = tf.nn.leaky_relu(tf.nn.bias_add(h_conv1, conv1_b), alpha=0.1)
#             with tf.name_scope("conv2"):
#                 conv2_kernal = tf.get_variable(name="conv2_kernal", initializer=tf.glorot_uniform_initializer,
#                                                shape=[5,5,64,128], dtype=tf.float32)
#                 h_conv2 = tf.nn.conv2d(conv1, conv2_kernal, strides=[1,2,2,1], padding="SAME")
#                 conv2_b = tf.Variable(initial_value=tf.zeros([128], dtype=tf.float32))
#                 conv2 = tf.nn.leaky_relu(tf.nn.bias_add(h_conv2, conv2_b), alpha=0.1)
#             with tf.name_scope("conv3"):
#                 conv3_kernal = tf.get_variable(name="conv3_kernal", initializer=tf.glorot_uniform_initializer,
#                                                shape=[5,5,128,256], dtype=tf.float32)
#                 h_conv3 = tf.nn.conv2d(conv2, conv3_kernal, strides=[1,2,2,1], padding="SAME")
#                 conv3_b = tf.Variable(initial_value=tf.zeros([256], dtype=tf.float32))
#                 conv3 = tf.nn.leaky_relu(tf.nn.bias_add(h_conv3, conv3_b), alpha=0.1)
#             with tf.name_scope("conv4"):
#                 conv4_kernal = tf.get_variable(name="conv4_kernal", initializer=tf.glorot_uniform_initializer,
#                                                shape=[5,5,256,512], dtype=tf.float32)
#                 h_conv4 = tf.nn.conv2d(conv3, conv4_kernal, strides=[1,2,2,1], padding="SAME")
#                 conv4_b = tf.Variable(initial_value=tf.zeros([512], dtype=tf.float32))
#                 conv4 = tf.nn.leaky_relu(tf.nn.bias_add(h_conv4, conv4_b), alpha=0.1)
#                 flatten1 = tf.layers.Flatten()(conv4)
#                 tmp_dim = int(flatten1.get_shape()[1])
#             with tf.name_scope("dense1"):
#                 w1 = tf.get_variable(name="w1", initializer=tf.glorot_uniform_initializer,
#                                      shape=[tmp_dim, 1024])
#                 b1 = tf.Variable(initial_value=tf.zeros([1024], dtype=tf.float32))
#                 link1 = tf.nn.bias_add(tf.matmul(flatten1, w1), b1)
#             with tf.name_scope("dense2"):
#                 w2 = tf.get_variable(name="w2", initializer=tf.glorot_uniform_initializer,
#                                      shape=[1024, tmp_dim])
#                 b2 = tf.Variable(initial_value=tf.zeros([tmp_dim], dtype=tf.float32))
#                 link2 = tf.nn.bias_add(tf.matmul(link1, w2), b2)
#                 link2 = tf.reshape(link2, [-1, 4, 4, 512])
#                 self._tmp = tf.reduce_mean(link1)
#             with tf.name_scope("upscale1"):
#                 upscale1_kernal = tf.Variable(initial_value=tf.truncated_normal(
#                     [3, 3, 512, 1024], stddev=0.1, dtype=tf.float32))
#                 upscale1_b = tf.Variable(initial_value=tf.zeros([1024], dtype=tf.float32))
#                 h_upscale1 = tf.nn.conv2d(link2, upscale1_kernal, strides=[1, 1, 1, 1], padding="SAME")
#                 upscale1 = tf.nn.bias_add(h_upscale1, upscale1_b)
#                 upscale1 = tf.depth_to_space(upscale1, 2)
#                 self._hidden = upscale1
#         def define_decoder1():
#             with tf.name_scope("upscale21"):
#                 upscale2_kernal = tf.get_variable(name="upscale2_kernal", initializer=tf.glorot_uniform_initializer,
#                                                   shape=[3,3,256,512], dtype=tf.float32)
#                 upscale2_b = tf.Variable(initial_value=tf.zeros([512], dtype=tf.float32))
#                 h_upscale2 = tf.nn.conv2d(self._hidden, upscale2_kernal, strides=[1,1,1,1], padding="SAME")
#                 upscale2 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale2, upscale2_b), alpha=0.1)
#                 upscale2 = tf.depth_to_space(upscale2, 2)
#             with tf.name_scope("upscale31"):
#                 upscale3_kernal = tf.get_variable(name="upscale3_kernal", initializer=tf.glorot_uniform_initializer,
#                                                   shape=[3,3,128,256], dtype=tf.float32)
#                 upscale3_b = tf.Variable(initial_value=tf.zeros([256], dtype=tf.float32))
#                 h_upscale3 = tf.nn.conv2d(upscale2, upscale3_kernal, strides=[1,1,1,1], padding="SAME")
#                 upscale3 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale3, upscale3_b), alpha=0.1)
#                 upscale3 = tf.depth_to_space(upscale3, 2)
#             with tf.name_scope("upscale41"):
#                 upscale4_kernal = tf.get_variable(name="upscale4_kernal", initializer=tf.glorot_uniform_initializer,
#                                                   shape=[3,3,64,128], dtype=tf.float32)
#                 upscale4_b = tf.Variable(initial_value=tf.zeros([128], dtype=tf.float32))
#                 h_upscale4 = tf.nn.conv2d(upscale3, upscale4_kernal, strides=[1,1,1,1], padding="SAME")
#                 upscale4 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale4, upscale4_b), alpha=0.1)
#                 upscale4 = tf.depth_to_space(upscale4, 2)
#             with tf.name_scope("conv51"):
#                 conv5_kernal = tf.get_variable(name="conv5_kernal", initializer=tf.glorot_uniform_initializer,
#                                                shape=[5, 5, 32, self._channel], dtype=tf.float32)
#                 h_conv5 = tf.nn.conv2d(upscale4, conv5_kernal, strides=[1, 1, 1, 1], padding="SAME")
#                 conv5_b = tf.Variable(initial_value=tf.zeros([self._channel], dtype=tf.float32))
#                 conv5 = tf.nn.sigmoid(tf.nn.bias_add(h_conv5, conv5_b))
#                 self._reconstruct1 = conv5
#         def define_decoder2():
#             with tf.name_scope("upscale22"):
#                 upscale22_kernal = tf.get_variable(name="upscale22_kernal", initializer=tf.glorot_uniform_initializer,
#                                                    shape=[3,3,256,512], dtype=tf.float32)
#                 upscale22_b = tf.Variable(initial_value=tf.zeros([512], dtype=tf.float32))
#                 h_upscale22 = tf.nn.conv2d(self._hidden, upscale22_kernal, strides=[1,1,1,1], padding="SAME")
#                 upscale22 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale22, upscale22_b), alpha=0.1)
#                 upscale22 = tf.depth_to_space(upscale22, 2)
#             with tf.name_scope("upscale32"):
#                 upscale32_kernal = tf.get_variable(name="upscale32_kernal", initializer=tf.glorot_uniform_initializer,
#                                                    shape=[3,3,128,256], dtype=tf.float32)
#                 upscale32_b = tf.Variable(initial_value=tf.zeros([256], dtype=tf.float32))
#                 h_upscale32 = tf.nn.conv2d(upscale22, upscale32_kernal, strides=[1,1,1,1], padding="SAME")
#                 upscale32 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale32, upscale32_b), alpha=0.1)
#                 upscale32 = tf.depth_to_space(upscale32, 2)
#             with tf.name_scope("upscale42"):
#                 upscale42_kernal = tf.get_variable(name="upscale42_kernal", initializer=tf.glorot_uniform_initializer,
#                                                    shape=[3,3,64,128], dtype=tf.float32)
#                 upscale42_b = tf.Variable(initial_value=tf.zeros([128], dtype=tf.float32))
#                 h_upscale42 = tf.nn.conv2d(upscale32, upscale42_kernal, strides=[1,1,1,1], padding="SAME")
#                 upscale42 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale42, upscale42_b), alpha=0.1)
#                 upscale42 = tf.depth_to_space(upscale42, 2)
#             with tf.name_scope("conv52"):
#                 conv52_kernal = tf.get_variable(name="conv53_lernal", initializer=tf.glorot_uniform_initializer,
#                                                 shape=[5, 5, 32, self._channel], dtype=tf.float32)
#                 h_conv52 = tf.nn.conv2d(upscale42, conv52_kernal, strides=[1, 1, 1, 1], padding="SAME")
#                 conv52_b = tf.Variable(initial_value=tf.zeros([self._channel], dtype=tf.float32))
#                 conv52 = tf.nn.sigmoid(tf.nn.bias_add(h_conv52, conv52_b))
#                 self._reconstruct2 = conv52
#         define_encoder()
#         define_decoder1()
#         define_decoder2()
#     def reconstruct1(self, X):
#         return self._sess.run(self._reconstruct1, feed_dict={self._x: X})
#     def define_loss(self):
#         self._loss1 = tf.reduce_mean(tf.pow(tf.subtract(self._x, self._reconstruct1), 2))
#         self._optimizer1 = tf.train.AdamOptimizer(learning_rate=self._learning_rate,
#                                                   beta1=0.5, beta2=0.999).minimize(self._loss1)
#         self._loss2 = tf.reduce_mean(tf.pow(tf.subtract(self._x, self._reconstruct2), 2))
#         self._optimizer2 = tf.train.AdamOptimizer(learning_rate=self._learning_rate,
#                                                   beta1=0.5, beta2=0.999).minimize(self._loss2)
#     def train(self, sourceTrainPath, destTrainPath):
#         sourceImgObj = ImageTrainObject(sourceTrainPath, self._batch_size)
#         sourceCount = sourceImgObj.DataCount // self._batch_size
#         destImgObj = ImageTrainObject(destTrainPath, self._batch_size)
#         destCount = destImgObj.DataCount // self._batch_size
#         self._sess.run(tf.global_variables_initializer())
#         # mpl.rcParams["xtick.labelsize"] = 6
#         # mpl.rcParams["ytick.labelsize"] = 6
#         # fig = plt.figure("cost")
#         # ax = fig.add_subplot(211)
#         # ax.grid(True)
#         # bx = fig.add_subplot(212)
#         # bx.grid(True)
#         # sourceData = []
#         # destData = []
#         for step in range(self._max_step):
#             source_avg_cost = 0
#             dest_avg_cost = 0
#             for time in range(sourceCount + destCount):
#                 warp, target = sourceImgObj.generateBatch()
#                 warp = warp / 255.0
#                 target = target / 255.0
#                 _, loss, tmp = self._sess.run([self._optimizer1, self._loss1, self._tmp], feed_dict={self._input: warp, self._x: target})
#                 source_avg_cost += (loss / (sourceCount + destCount))
#                 # sourceData.append(loss)
#                 # ax.plot(np.arange(len(sourceData)), np.sqrt(sourceData), linewidth=0.8, color="r")
#                 # plt.pause(0.01)
#                 print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- [source] cost: %3.5f" % loss,
#                       "avg: %3.5f" % tmp)
#                 warp, target = destImgObj.generateBatch()
#                 warp = warp / 255.0
#                 target = target / 255.0
#                 _, loss, tmp = self._sess.run([self._optimizer2, self._loss2, self._tmp], feed_dict={self._input: warp, self._x: target})
#                 dest_avg_cost += (loss / (sourceCount + destCount))
#                 # destData.append(loss)
#                 # bx.plot(np.arange(len(destData)), np.sqrt(destData), linewidth=0.8, color="b")
#                 # plt.pause(0.01)
#                 print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- [ dest ] cost: %3.5f" % loss,
#                       "avg: %3.5f" % tmp, "time:", time)
#             print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- source_avg_cost: %3.5f" % source_avg_cost,
#                   "- dest_avg_cost: %3.5f" % dest_avg_cost, "step:", step)
#             tf.train.Saver().save(self._sess, save_path="{0}encoder".format(self._modelPath), global_step=step)
#         # plt.show()
#     def load_model(self):
#         path = tf.train.latest_checkpoint(self._modelPath)
#         tf.train.Saver().restore(self._sess, path)
#     def showAll(self, batchSize=128):
#         prePath = "F:/tensorflow/automodel/scrawler/video/trainImg/"
#         resPath = r"F:/tensorflow/automodel/scrawler/video/resImg/"
#         items = os.listdir(prePath)
#         tmp_list = []
#         fileName_list = []
#         totalFiles = len(items)
#         for idx, file in enumerate(items):
#             source = cv2.imread(r"{0}{1}".format(prePath, file)) / 255.0
#             tmp_list.append(source)
#             fileName_list.append(file)
#             if len(tmp_list) == batchSize:
#                 matrix = np.array(tmp_list, dtype=np.float32)
#                 dest = self._sess.run(self._reconstruct2, feed_dict={self._x: matrix})
#                 for img, name in zip(dest, fileName_list):
#                     dest = np.array(img * 255, dtype=np.uint8)
#                     filePath = "{0}{1}".format(resPath, name)
#                     cv2.imwrite(filePath, dest)
#                     print(filePath)
#                 tmp_list = []
#                 fileName_list = []
#             elif idx == (totalFiles - 1):
#                 matrix = np.array(tmp_list, dtype=np.float32)
#                 dest = self._sess.run(self._reconstruct2, feed_dict={self._x: matrix})
#                 for img, name in zip(dest, fileName_list):
#                     dest = np.array(img * 255, dtype=np.uint8)
#                     filePath = "{0}{1}".format(resPath, name)
#                     cv2.imwrite(filePath, dest)
#                     print(filePath)
#                 tmp_list = []
#                 fileName_list = []
#     def generateImage(self):
#         source = cv2.imread(r"F:/tensorflow/automodel/scrawler/video-1/trainImg/18.jpg") / 255.0
#         source = np.reshape(source, [1, 128, 128, 3])
#         dest, loss = self._sess.run([self._reconstruct1, self._loss2], feed_dict={self._x: source})
#         print(loss)
#         source = np.reshape(source, [128, 128, 3])
#         dest = np.reshape(dest, [128, 128, 3])
#         dest = np.array(dest * 255, dtype=np.uint8)
#         fig = plt.figure("compare")
#         ax = fig.add_subplot(121)
#         b, g, r = cv2.split(source)
#         source = cv2.merge([r, g, b])
#         ax.imshow(source)
#         ax.axis("off")
#         bx = fig.add_subplot(122)
#         bx.axis("off")
#         b, g, r = cv2.split(dest)
#         dest = cv2.merge([r, g, b])
#         bx.imshow(dest)
#         plt.show()

class ConvolutionalAutoencoder:
    def __init__(self, learning_rate, max_step, batch_size):
        self._learning_rate = learning_rate
        self._max_step = max_step
        self._batch_size = batch_size
        self._sess = tf.Session()
        self.define_network()
        self.define_loss()
    def define_network(self):
        self._x = tf.placeholder(shape=[None, 128, 128, 3], dtype=tf.float32)
        def define_encoder():
            with tf.variable_scope("conv1") as scope:
                conv1_kernal = tf.get_variable(name="conv1_kernal", initializer=tf.truncated_normal(
                    shape=[5, 5, 3, 16], stddev=0.1, dtype=tf.float32))
                conv1_biases = tf.get_variable(name="conv1_biases", initializer=tf.zeros(shape=[16], dtype=tf.float32))
                h_conv1 = tf.nn.conv2d(self._x, conv1_kernal, strides=[1, 2, 2, 1], padding="SAME")
                conv1 = tf.nn.leaky_relu(tf.add(h_conv1, conv1_biases), alpha=0.1)
            with tf.variable_scope("conv2") as scope:
                conv2_kernal = tf.get_variable(name="conv2_kernal",
                                               initializer=tf.truncated_normal(
                                                   shape=[5, 5, 16, 32], stddev=0.1, dtype=tf.float32))
                conv2_biases = tf.get_variable(name="conv2_biases",
                                               initializer=tf.zeros(shape=[32], dtype=tf.float32))
                h_conv2 = tf.nn.conv2d(conv1, conv2_kernal, strides=[1, 2, 2, 1], padding="SAME")
                conv2 = tf.nn.leaky_relu(h_conv2 + conv2_biases, alpha=0.1)
                flatten1 = tf.layers.Flatten()(conv2)
                tmp_dim = int(flatten1.get_shape()[1])
            with tf.name_scope("dense1"):
                w1 = tf.Variable(initial_value=xavier_init(tmp_dim, 256))
                b1 = tf.Variable(initial_value=tf.zeros([256], dtype=tf.float32))
                link1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flatten1, w1), b1))
            with tf.name_scope("dense2"):
                w2 = tf.Variable(initial_value=xavier_init(256, tmp_dim))
                b2 = tf.Variable(initial_value=tf.zeros([tmp_dim], dtype=tf.float32))
                link2 = tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(link1, w2), b2), alpha=0.1)
                link2 = tf.reshape(link2, [-1, 32, 32, 32])
            with tf.name_scope("upscale1"):
                upscale1_kernal = tf.Variable(initial_value=tf.truncated_normal(
                        [3, 3, 32, 64], stddev=0.1, dtype=tf.float32))
                upscale1_b = tf.Variable(initial_value=tf.zeros([64], dtype=tf.float32))
                h_upscale1 = tf.nn.conv2d(link2, upscale1_kernal, strides=[1, 1, 1, 1], padding="SAME")
                upscale1 = tf.nn.leaky_relu(tf.nn.bias_add(h_upscale1, upscale1_b), alpha=0.1)
                upscale1 = tf.depth_to_space(upscale1, 2)
                self._hidden = upscale1
        def define_decoder():
            with tf.variable_scope("uconv1") as scope:
                uconv1_kernal = tf.get_variable(name="uconv1_kernal", initializer=tf.truncated_normal(
                    shape=[3, 3, 16, 32], dtype=tf.float32, stddev=0.1))
                uconv1_biases = tf.get_variable(name="uconv1_biases",
                                                initializer=tf.zeros(shape=[32], dtype=tf.float32))
                h_uconv1 = tf.nn.conv2d(self._hidden, uconv1_kernal, strides=[1, 1, 1, 1], padding="SAME")
                uconv1 = tf.depth_to_space(tf.nn.leaky_relu(tf.nn.bias_add(h_uconv1, uconv1_biases), alpha=0.1), 2)
            with tf.variable_scope("uconv2") as scope:
                uconv2_kernal = tf.get_variable(name="uconv2_kernal", initializer=tf.truncated_normal(
                    shape=[5, 5, 8, 3], stddev=0.1, dtype=tf.float32))
                uconv2_biases = tf.get_variable(name="uconv2_biases",
                                                initializer=tf.zeros(shape=[3], dtype=tf.float32))
                h_uconv2 = tf.nn.conv2d(uconv1, uconv2_kernal, strides=[1, 1, 1, 1], padding="SAME")
                uconv2 = tf.nn.leaky_relu(tf.nn.bias_add(h_uconv2, uconv2_biases), alpha=0.1)
                self._reconstruct = uconv2
        define_encoder()
        define_decoder()
    def define_loss(self):
        self._loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self._x, self._reconstruct), 2))
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)
    def get_hidden_output(self, X):
        return self._sess.run(self._hidden, feed_dict={self._x: X})
    def get_reconstruct(self, X):
        return self._sess.run(self._reconstruct, feed_dict={self._x: X})
    def generate(self, hidden=None):
        if hidden is None:
            raise Exception("There is no hidden input")
        return self._sess.run(self._reconstruct, feed_dict={self._hidden: hidden})
    def train(self):
        imgObj = ImageTrainObject("F:/tensorflow/automodel/scrawler/video/trainImg/", self._batch_size)
        count = imgObj.DataCount // self._batch_size
        self._sess.run(tf.global_variables_initializer())
        fig = plt.figure("loss_fig")
        mpl.rcParams["xtick.labelsize"] = 6
        mpl.rcParams["ytick.labelsize"] = 6
        ax = fig.add_subplot(111)
        ax.grid(True)
        data = []
        for step in range(self._max_step):
            avg_cost = 0
            for times in range(count):
                train = imgObj.generateBatch() / 255.0
                _, loss = self._sess.run([self._optimizer, self._loss], feed_dict={self._x: train})
                avg_cost += (loss / count)
                data.append(np.log(loss))
                ax.plot(np.arange(len(data)), np.array(data), linewidth=0.8, color="r")
                plt.draw()
                plt.pause(0.1)
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- cost: %.3f" % loss, " time: ", times)
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "- avg_cost: %.3f" % avg_cost, " step: ", step)
            tf.train.Saver().save(self._sess, save_path="{0}encoder".format("F:/tensorflow/automodel/model/"), global_step=step)
        plt.show()
    def showAll(self):
        tf.train.Saver().restore(self._sess, tf.train.latest_checkpoint("F:/tensorflow/automodel/model/"))
        prePath = "F:/tensorflow/automodel/scrawler/video/trainImg/"
        items = os.listdir(prePath)
        for file in items:
            source = cv2.imread(r"{0}{1}".format(prePath, file)) / 255.0
            source = np.reshape(source, [1, 128, 128, 3])
            dest = self.get_reconstruct(source)
            dest = np.reshape(dest, [128, 128, 3])
            dest = np.array(dest * 255, dtype=np.uint8)
            cv2.imwrite("{0}{1}".format(r"F:/tensorflow/automodel/scrawler/video/resultImg/", file), dest)
            print(file)
    def load_model(self):
        tf.train.Saver().restore(self._sess, tf.train.latest_checkpoint("F:/tensorflow/automodel/model/"))
        source = cv2.imread("F:/tensorflow/automodel/scrawler/video/trainImg/18.jpg")
        source = np.reshape(source, [1, 128, 128, 3]) / 255.0
        dest, loss = self._sess.run([self._reconstruct, self._loss], feed_dict={self._x: source})
        dest = np.reshape(dest, [128, 128, 3])
        dest = np.array(dest * 255, dtype=np.uint8)
        print(loss)
        cv2.imshow("aaa", dest)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    obj = AutoEncoder(5e-5, 100, 64, "F:/tensorflow/automodel/model_1/")
    obj.load_model()
    obj.generateImage()
    # obj.showAll()
    # obj.train(sourceTrainPath=r"F:/tensorflow/automodel/scrawler/video/trainImg/",
    #           destTrainPath=r"F:/tensorflow/automodel/scrawler/video-1/trainImg/")
    # obj = ConvolutionalAutoencoder(0.01, 5, 64)
    # obj.train()
    # obj.load_model()