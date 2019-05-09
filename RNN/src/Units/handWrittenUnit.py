'''
Created on 2018年5月5日

@author: IL MARE
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.framework import graph_util

hidden_size = 64
class_num = 10
time_step = 28
layer_num = 3
batch_size = 16
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

  
def func():
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    return lstm_cell

if __name__ == "__main__":
    mnist = input_data.read_data_sets("/MNIST_data", one_hot=True)
    mul_cell = rnn.MultiRNNCell([func() for _ in range(layer_num)], state_is_tuple=True)
    sess = tf.InteractiveSession()
    _x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784], name="x")
    x = tf.nn.dropout(tf.reshape(_x, [-1, 28, 28]), keep_prob=keep_prob)
    y = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10], name="y")
    state = init_state = mul_cell.zero_state(batch_size, dtype=tf.float32)
#     outputs, state = tf.nn.dynamic_rnn(mul_cell, x, initial_state=state, dtype=tf.float32)
    outputs = []
    for step in range(28):
        output, state = mul_cell(x[:, step, :], state)
        outputs.append(output)
    h_state = outputs[-1]
    W = tf.Variable(tf.random_uniform(shape=[hidden_size, class_num], dtype=tf.float32), dtype=tf.float32)
    b = tf.Variable(tf.zeros(shape=[1, 10]), dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(h_state, W) + b, name="predict")
    cross_entry = - y * tf.log(y_pre)
    loss = tf.reduce_mean(tf.reduce_sum(cross_entry, reduction_indices=[1]))
      
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    sess.run(tf.global_variables_initializer())
     
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1)), dtype=tf.float32))
    for i in range(500):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={_x: batch_x, y: batch_y, keep_prob: 1.0})
        if i % 200 == 0:
            print(accuracy.eval(feed_dict={_x: batch_x, y: batch_y, keep_prob: 1.0}))
    saver = tf.train.Saver()
    saver.save(sess, r"G:/Machine-Learning-Study-Notes/python/RNN/model.ckpt")
#     graph_def = tf.get_default_graph().as_graph_def()
#     output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ["predict", "x", "keep_prob"])
#     with tf.gfile.GFile(r"G:/Machine-Learning-Study-Notes/python/RNN/model.pb", "wb") as fp:
#         fp.write(output_graph_def.SerializeToString())