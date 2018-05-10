'''
Created on 2018年5月6日

@author: IL MARE
'''
from tensorflow.python.framework import graph_util
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib import rnn

hidden_size = 64
class_num = 10
time_step = 28
layer_num = 3
batch_size = 16
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

def func():
    lstm_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    return lstm_cell

if __name__ == "__main__":
    path = r"G:/Machine-Learning-Study-Notes/python/RNN/modelFile/model.ckpt"
    mul_cell = rnn.MultiRNNCell([func() for _ in range(layer_num)], state_is_tuple=True)
    x = tf.placeholder(dtype=tf.float32, shape=[1, 784])
    input = tf.reshape(x, [-1, 28, 28])
    y = tf.placeholder(dtype=tf.float32, shape=[1, class_num])
    state = mul_cell.zero_state(1, dtype=tf.float32)
    outputs = []
    for i in range(28):
        output, state = mul_cell(input[:, i, :], state)
        outputs.append(output)
    h_state = outputs[-1]
    W = tf.Variable(tf.random_uniform([hidden_size, class_num]
                                      , -1, 1, dtype=tf.float32), 
                                      dtype=tf.float32)
    bais = tf.Variable(tf.zeros([1, class_num], dtype=tf.float32), dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bais)
    sess = tf.InteractiveSession()
    tf.train.Saver().restore(sess, path)
    mnist = input_data.read_data_sets("/MNIST_data", one_hot=True)
    count = mnist.test.images.shape[0]
    print(count)
    res = 0
    for i in range(count):
        batch_x, batch_y = mnist.test.images[i: i + 1], mnist.test.labels[i: i + 1]
        pre = sess.run(y_pre, feed_dict={x: batch_x, keep_prob: 1.0})
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pre, 1), tf.argmax(batch_y, 1)), tf.float32))
        res += accuracy.eval()
        if i % 500 == 0:
            print(res / (i + 1))
    print(res / count)
#     path = r"G:/Machine-Learning-Study-Notes/python/RNN/model.pb"
#     with tf.gfile.GFile(path, "rb") as fp:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(fp.read())
#     tf.import_graph_def(graph_def)
#     sess = tf.InteractiveSession()
#     y_pre = sess.graph.get_tensor_by_name("import/predict:0")
#     _x = sess.graph.get_tensor_by_name("import/x:0")
#     keep_prob = sess.graph.get_tensor_by_name("import/keep_prob:0")
#     mnist = input_data.read_data_sets("/MNIST_data", one_hot=True)
#     count = mnist.test.images.shape[0] // 16
#     res = 0
#     for i in range(count):
#         batch_x, batch_y = mnist.test.images[i * 16: i * 16 + 16], mnist.test.labels[i * 16: i * 16 + 16]
#         pre = sess.run(y_pre, feed_dict={_x: batch_x, keep_prob: 1.0})
#         accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pre, 1), tf.argmax(batch_y, 1)), tf.float32))
#         res += accuracy.eval()
#     print(res / count)