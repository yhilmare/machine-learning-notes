'''
Created on 2018年5月6日

@author: IL MARE
'''
from tensorflow.python.framework import graph_util
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

if __name__ == "__main__":
    path = r"G:/Machine-Learning-Study-Notes/python/RNN/model.pb"
    with tf.gfile.GFile(path, "rb") as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
    tf.import_graph_def(graph_def)
    sess = tf.InteractiveSession()
    y_pre = sess.graph.get_tensor_by_name("import/predict:0")
    _x = sess.graph.get_tensor_by_name("import/x:0")
    keep_prob = sess.graph.get_tensor_by_name("import/keep_prob:0")
    mnist = input_data.read_data_sets("/MNIST_data", one_hot=True)
    count = mnist.test.images.shape[0] // 16
    res = 0
    for i in range(count):
        batch_x, batch_y = mnist.test.images[i * 16: i * 16 + 16], mnist.test.labels[i * 16: i * 16 + 16]
        pre = sess.run(y_pre, feed_dict={_x: batch_x, keep_prob: 1.0})
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pre, 1), tf.argmax(batch_y, 1)), tf.float32))
        res += accuracy.eval()
    print(res / count)