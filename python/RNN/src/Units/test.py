'''
Created on 2018年5月5日

@author: IL MARE
'''
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

hidden_size = 64
class_num = 10
time_step = 28
layer_num = 2
batch_size = 28
keep_prob = tf.placeholder(dtype=tf.float32)

def func():
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    return lstm_cell

if __name__ == "__main__":
    mnist = input_data.read_data_sets("/MNIST_data", one_hot=True)
    sess = tf.InteractiveSession()
    _x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
    x = tf.reshape(_x, [-1, 28, 28])
    inputs = tf.nn.dropout(x, keep_prob=keep_prob)
    y = tf.placeholder(dtype=np.float32, shape=[None, class_num])
    
    mul_cell = rnn.MultiRNNCell([func() for i in range(layer_num)], state_is_tuple=True)
    state = mul_cell.zero_state(batch_size, dtype=tf.float32)
    
    outputs, state = tf.nn.dynamic_rnn(mul_cell, inputs, initial_state=state, dtype=tf.float32)
    h_state = outputs[:, -1, :]
    W = tf.Variable(tf.truncated_normal([hidden_size, class_num],stddev=0.1, dtype=tf.float32))
    bais = tf.Variable(tf.zeros([class_num]))
    
    y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bais)
    
    loss = -tf.reduce_mean(y * tf.log(y_pre))
    train_op = tf.train.AdadeltaOptimizer(0.0001).minimize(loss)
    
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.global_variables_initializer().run()
    
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={_x: batch_x, y: batch_y, keep_prob:1.0})
        if i % 200 == 0:
            print(sess.run(accuracy, feed_dict={_x: batch_x, y: batch_y, keep_prob:1.0}))