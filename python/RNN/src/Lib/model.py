'''
Created on 2018年5月8日

@author: IL MARE
'''
import collections
import numpy as np
from tensorflow.contrib import rnn
import tensorflow as tf
import time

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

class lstm_model:
    def __init__(self, hidden_size, num_layer, 
                 corpus, keep_prob, 
                 embedding_size, lr, max_step,
                 save_path, sampling=False):
        if not sampling:
            self._num_seq = corpus.num_seq
            self._num_step = corpus.num_step
        else:
            self._num_seq = 1
            self._num_step = 1
        self._save_path = save_path
        self._lr = lr
        self._max_step = max_step
        self._embedding_size = embedding_size
        self._hidden_size = hidden_size
        self._num_layer = num_layer
        self._corpus = corpus
        self._keep_prob = keep_prob
        tf.reset_default_graph()
        self.init_inputs()
        self.build_lstm()
        self.define_loss()
        self.define_gradients()
    def init_inputs(self):
        self._x = tf.placeholder(dtype=tf.int32, 
                                      shape=[self._num_seq, self._num_step])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", 
                                        shape=[self._corpus.word_num, 
                                               self._embedding_size], dtype=tf.float32)
            self._inputs = tf.nn.embedding_lookup(embedding, self._x)
    def build_lstm(self):
        def build_cell():
            cell = rnn.BasicLSTMCell(self._hidden_size, forget_bias=1.0, state_is_tuple=True)
            cell = rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
            return cell
        mul_cell = rnn.MultiRNNCell([build_cell() for _ in range(self._num_layer)], 
                                    state_is_tuple=True)
        self._init_state = mul_cell.zero_state(self._num_seq, dtype=tf.float32)
        outputs, self._final_state = tf.nn.dynamic_rnn(mul_cell, self._inputs, 
                                                       initial_state=self._init_state)
        outputs = tf.reshape(outputs, [-1, self._hidden_size])
        W = tf.Variable(tf.truncated_normal([self._hidden_size, self._corpus.word_num], 
                                            stddev=0.1, dtype=tf.float32))
        bais = tf.Variable(tf.zeros([1, self._corpus.word_num], 
                                    dtype=tf.float32), dtype=tf.float32)
        self._prediction = tf.nn.softmax(tf.matmul(outputs, W) + bais)
    def define_loss(self):
        self._y = tf.placeholder(dtype=tf.int32, shape=[self._num_seq, self._num_step])
        y_one_hot = tf.reshape(tf.one_hot(self._y, self._corpus.word_num), self._prediction.shape)
        self._loss = -tf.reduce_mean(tf.reduce_sum(y_one_hot * tf.log(self._prediction), 
                                                   reduction_indices=[1]))
    def define_gradients(self):
        vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, vars), 5)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._optimizer = optimizer.apply_gradients(zip(grads, vars))
    def train(self):
        with tf.Session() as sess:
            return_mat = []
            sess.run(tf.global_variables_initializer())
            state = sess.run(self._init_state)
            step = 0
            start = time.clock()
            for x, y in self._corpus.generate_batch():
                feed = {self._x: x, 
                        self._y: y,
                        self._init_state:state}
                loss, _, state = sess.run([self._loss, self._optimizer, 
                                           self._final_state], feed_dict = feed)
                return_mat.append(loss)
                step += 1
                if step % 10 == 0:
                    end = time.clock()
                    interval = end - start
                    yield return_mat
                    print("迭代次数：{0:d}/{2:d}，当前损失：{1:.3f}，迭代速度：{3:.3f} 秒/十次，约需要{4:.3f}秒"
                          .format(step, loss, self._max_step, interval, ((self._max_step - step) / 10) * interval))
                    start = time.clock()
                if step == self._max_step:
                    break
            tf.train.Saver().save(sess, "{0}model".format(self._save_path), global_step=step)
    def load_model(self):
        sess = tf.Session()
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(self._save_path))
        self._sess = sess
    def sampling(self, init_str, max_sample=30):
        sample = [c for c in init_str]
        pre = np.ones((self._corpus.word_num, ))
        state = self._sess.run(self._init_state)
        for c in sample:
            feed = {self._x: np.reshape(c, [1, 1]),
                    self._init_state: state}
            pre, state = self._sess.run([self._prediction, self._final_state], 
                                 feed_dict=feed)
        c = pick_top_n(pre, self._corpus.word_num)
        sample.append(c)
        for count in range(max_sample):
            x = np.zeros([1, 1])
            x[0][0] = c
            feed = {self._x: x, 
                    self._init_state: state}
            pre, state = self._sess.run([self._prediction, self._final_state], 
                                        feed_dict=feed)
            c = pick_top_n(pre, self._corpus.word_num)
            sample.append(c)
        return sample

class corpus:
    '''
            该对象用于构造语料库，参数解释：
    file_path:语料库所在位置
    num_seq:一个batch中所包含的句子数
    num_step:一个句子中包含的词的数目
    max_size:统计语料库中出现频度前max_size的字或词
    '''
    def __init__(self, file_path, num_seq=10, num_step=10, max_size=3500):
        self._file_path = file_path
        with open(self._file_path, "r", encoding="utf_8") as fp:
            self._buffer = fp.read()
        self._count = collections.Counter(self._buffer).most_common(max_size)
        self._word_to_int = dict()
        for word, _ in self._count:
            self._word_to_int[word] = len(self._word_to_int)
        self._int_to_word = dict(zip(self._word_to_int.values(), self._word_to_int.keys()))
        self._batch_size = num_seq * num_step
        num_batch = len(self._buffer) // self._batch_size
        self._buffer = self._buffer[: num_batch * self._batch_size]
        self._num_seq = num_seq
        self._num_step = num_step
    @property
    def num_seq(self):
        return self._num_seq
    @property
    def num_step(self):
        return self._num_step
    @property
    def file_path(self):
        return self._file_path
    @property
    def word_num(self):
        return len(self._int_to_word) + 1
    @property
    def batch_size(self):
        return self._batch_size
    @property
    def words(self):
        return self._buffer
    def sentence_to_int(self, sentence):
        return_mat = []
        for word in sentence:
            return_mat.append(self.word_to_int(word))
        return np.array(return_mat)
    def int_to_sentence(self, row):
        return_mat = []
        for index in row:
            return_mat.append(self.int_to_word(index))
        return "".join(return_mat)
    def word_to_int(self, word):
        return self._word_to_int.get(word, len(self._int_to_word))
    def int_to_word(self, index):
        return self._int_to_word.get(index, "<unk>")
    def text_to_attr(self):
        return_mat = []
        for _word in self._buffer:
            return_mat.append(self.word_to_int(_word))
        return np.array(return_mat)
    def attr_to_text(self, attr):
        return_mat = []
        for _attr in attr:
            return_mat.append(self.int_to_word(_attr))
        return return_mat
    def generate_batch(self):
        attrs = self.text_to_attr()
        attrs = np.reshape(attrs, [self.num_seq, -1])
        while True:
            np.random.shuffle(attrs)
            for index in range(0, attrs.shape[1], self.num_step):
                x = attrs[:, index: index + self.num_step]
                y = np.zeros_like(x)
                y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
                yield x, y