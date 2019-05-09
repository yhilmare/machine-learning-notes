'''
Created By ILMARE
@Date 2019-3-6
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sklearn
import cv2

def xavier_init(input_count, output_count, constant=1):
    low = -constant * np.sqrt(6.0 / (input_count + output_count))
    high = constant * np.sqrt(6.0 / (input_count + output_count))
    return tf.random_uniform((input_count, output_count),
                            minval=low, maxval=high,
                            dtype=tf.float32)

def standard_scale(X_train, X_test):
    pre = sklearn.preprocessing.StandardScaler().fit(X_train)
    return pre.transform(X_train), pre.transform(X_test)

def get_random_block_from_data(data, batch_size):
    start_idx = np.random.randint(0, len(data) - batch_size)
    return data[start_idx: start_idx + batch_size, :]

class AGNAutoEncoder:
    def __init__(self, input_size, hidden_size,
                 transfer_function=tf.nn.softplus,learning_rate=0.001,
                 scale=0.1, batch_size=128, max_step=20):
        self._learning_tate = learning_rate
        self._max_step = max_step
        self._batch_size = batch_size
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._transfer = transfer_function
        self._scale = tf.placeholder(tf.float32)
        self._training_scale = scale
        self._weight = self.init_weights()
        self._sess = tf.Session()
        self.defineNetwork()
    def defineNetwork(self):
        self._x = tf.placeholder(shape=[None, self._input_size], dtype=tf.float32)
        self._hidden = self._transfer(tf.add(tf.matmul(
            self._x + self._training_scale * tf.random_normal((self._input_size, )),
            self._weight['w1']), self._weight['b1']))
        self._reconstruction = tf.add(tf.matmul(self._hidden,
                                                self._weight['w2']), self._weight['b2'])
        self._loss = 0.5 * tf.reduce_sum(tf.pow(
            tf.subtract(self._reconstruction, self._x), 2.0))
        self._optimizer = tf.train.AdamOptimizer(self._learning_tate).minimize(self._loss)
    def init_weights(self):
        all_weight = dict()
        all_weight['w1'] = tf.Variable(initial_value=xavier_init(self._input_size,
                                                                 self._hidden_size), dtype=tf.float32)
        all_weight['b1'] = tf.Variable(initial_value=tf.zeros(shape=[self._hidden_size],
                                                              dtype=tf.float32))
        all_weight['w2'] = tf.Variable(initial_value=xavier_init(self._hidden_size,
                                                                 self._input_size), dtype=tf.float32)
        all_weight['b2'] = tf.Variable(initial_value=tf.zeros(shape=[self._input_size],
                                                              dtype=tf.float32))
        return all_weight
    def part_fit(self, X):
        _loss, _ = self._sess.run([self._loss, self._optimizer],
                                  feed_dict={self._x: X, self._scale: self._training_scale})
        return _loss
    def calculate_total_cost(self, X):
        _loss = self._sess.run([self._loss],
                               feed_dict={self._x: X, self._scale: self._training_scale})
        return _loss
    def transform(self, X):
        return self._sess.run(self._hidden, feed_dict={self._x: X,
                                                       self._scale: self._training_scale})
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self._weight['b1'])
        return self._sess.run(self._reconstruction, feed_dict={self._hidden: hidden})
    def reconstrct(self, X):
        return self._sess.run(self._reconstruction, feed_dict={self._x: X,
                                                               self._scale: self._training_scale})
    def getWeight(self):
        return self._sess.run(self._weight['w1'])
    def getBiases(self):
        return self._sess.run(self._weight['b1'])
    def train(self):
        self._sess.run(tf.global_variables_initializer())
        mnist = input_data.read_data_sets("/home/ilmare/dataSet/mnist", one_hot=True)
        train, _ = standard_scale(mnist.train.images, mnist.test.images)
        n_examples = int(mnist.train.num_examples)
        for idx in range(self._max_step):
            avg_cost = 0
            total_batch = n_examples // self._batch_size
            for batch in range(0, total_batch):
                train_tmp = get_random_block_from_data(train, self._batch_size)
                _cost = self.part_fit(train_tmp)
                avg_cost += _cost / n_examples * self._batch_size
            if (idx + 1) % 5 == 0:
                print("avg_cost: %.3f" % avg_cost, " step: ", idx)
        tf.train.Saver().save(self._sess, save_path="{0}autoencoder".format("/home/ilmare/Desktop/FaceReplace/model/"))
    def load_model(self):
        tf.train.Saver().restore(self._sess, tf.train.latest_checkpoint("/home/ilmare/Desktop/FaceReplace/model/"))
        mnist = input_data.read_data_sets("/home/ilmare/dataSet/mnist", one_hot=True)
        source = np.reshape(mnist.train.images[0], [1, 784])
        dest = self.reconstrct(source)
        source = np.reshape(source, [28, 28])
        dest = np.reshape(dest, [28, 28])
        print(source.shape, dest.shape)
        # fig = plt.figure("test")
        # ax = fig.add_subplot(121)
        # ax.imshow(source)
        # bx = fig.add_subplot(122)
        # bx.imshow(dest)
        # plt.show()
        cv2.imshow("lalal", dest)
        cv2.waitKey(0)