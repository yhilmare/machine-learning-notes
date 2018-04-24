'''
Created on 2018年4月24日

@author: IL MARE
'''
import tensorflow as tf

if __name__ == "__main__":
    a = tf.constant([[0,0,0,0,1], [1,0,1,0,1],[0,0,0,0,1], [1,0,1,0,1]], dtype=tf.float32)
    b = tf.constant([[0,0,0,0,1], [0,0,1,0,1],[0,0,0,0,1], [1,0,1,0,1]], dtype=tf.float32)
    index = tf.equal(tf.argmax(a, 1), tf.argmax(b, 1))
    result = tf.reduce_mean(tf.cast(index, dtype=tf.float32))
    session = tf.InteractiveSession()
    print(result.eval())