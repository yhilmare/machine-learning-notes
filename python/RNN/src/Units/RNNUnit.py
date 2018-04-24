'''
Created on 2018年4月23日

@author: IL MARE
'''
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print(mnist.train.images.shape, mnist.validation.images.shape)
        