'''
Created on 2018年4月23日

@author: IL MARE
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    session = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10], tf.float32))
    b = tf.Variable(tf.zeros([1, 10], dtype=tf.float32))
    y = tf.nn.softmax(tf.matmul(x, w) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entry = -tf.reduce_mean(tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entry)
    session.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        session.run(train_step, {x: batch_x, y_: batch_y})
    accuracy = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), dtype=tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
    for i in range(20):
        tensor = mnist.test.images[i]
        tensor = tf.constant(value=tensor, shape=[784, 1])
        predict = tf.matmul(tf.transpose(w), tensor) + tf.transpose(b)
        label = mnist.test.labels[i]
        result = tf.equal(tf.argmax(label, 0), tf.argmax(predict, 0))
        print(result.eval())
