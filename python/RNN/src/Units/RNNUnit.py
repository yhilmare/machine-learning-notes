'''
Created on 2018年4月23日

@author: IL MARE
'''
import tensorflow as tf

if __name__ == "__main__":
    hello = tf.constant("Hello,tensroflow")
    sess = tf.Session()
    print(sess.run(hello).decode("utf_8"))