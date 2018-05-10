'''
Created on 2018年5月8日

@author: IL MARE
'''
import collections
import numpy as np
from tensorflow.contrib import rnn
import tensorflow as tf
import shelve
import os

path = r"G:/Machine-Learning-Study-Notes/python/RNN/modelFile/jay.txt"
save_path = r"G:/Machine-Learning-Study-Notes/python/RNN/modelFile/jay.ckpt"
obj_path = r"G:/Machine-Learning-Study-Notes/python/RNN/modelFile/corpus"

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

class lstm_model:
    def __init__(self, hidden_size, num_layer, 
                 corpus, keep_prob, 
                 embedding_size, lr, max_step, 
                 sampling=False):
        if not sampling:
            self._num_seq = corpus.num_seq
            self._num_step = corpus.num_step
        else:
            self._num_seq = 1
            self._num_step = 1
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
                                        shape=[self._corpus.word_num + 1, 
                                               self._embedding_size], dtype=tf.float32)
            self._inputs = tf.nn.embedding_lookup(embedding, self._x)
    def build_lstm(self):
        def build_cell():
            with tf.name_scope("lstm"):
                cell = rnn.BasicLSTMCell(self._hidden_size, 1.0, True)
                cell = rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
                return cell
        mul_cell = rnn.MultiRNNCell([build_cell() for _ in range(self._num_layer)], 
                                    state_is_tuple=True)
        self._init_state = state = mul_cell.zero_state(self._num_seq, dtype=tf.float32)
        outputs = []
        for step in range(self._num_step):
            output, state = mul_cell(self._inputs[:, step, :], state)
            outputs.append(output)
        W = tf.Variable(tf.truncated_normal([self._hidden_size, self._corpus.word_num], 
                                            stddev=0.1, dtype=tf.float32))
        bais = tf.Variable(tf.zeros([1, self._corpus.word_num], 
                                    dtype=tf.float32), dtype=tf.float32)
        self._prediction = []
        for h_state in outputs:
            self._prediction.append(tf.nn.softmax(tf.matmul(h_state, W) + bais))
    def define_loss(self):
        self._y = tf.placeholder(dtype=tf.int32, shape=[self._num_seq, self._num_step])
        y_one_hot = tf.one_hot(self._y, self._corpus.word_num)
        self._loss = 0
        count = len(self._prediction)
        for i in range(count):
            pre = self._prediction[i]
            self._loss += -tf.reduce_mean(tf.reduce_sum(y_one_hot[:, i, :] * 
                                                        tf.log(tf.clip_by_value(pre, 1e-8, 1)), 
                                                        reduction_indices=[1]))
        self._accuracy = 0
        for i in range(count):
            pre = self._prediction[i]
            self._accuracy += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_one_hot[:, i, :], 1), 
                                                         tf.argmax(pre, 1)), tf.float32))
        self._accuracy /= count
    def define_gradients(self):
        vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, vars), 5)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._optimizer = optimizer.apply_gradients(zip(grads, vars))
    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            for x, y in self._corpus.generate_batch():
                feed = {self._x: x, 
                        self._y: y}
                accuracy, loss, _, prediction = sess.run([self._accuracy, self._loss, 
                                                          self._optimizer, self._prediction], 
                                                         feed_dict = feed)
                step += 1
                if step % 200 == 0:
                    print("迭代次数：{0:d}，当前准曲率：{1:.3f}，当前损失：{2:.3f}".format(step, accuracy, loss))
                if step == self._max_step:
                    break
            tf.train.Saver().save(sess, save_path)
    def load_model(self):
        sess = tf.Session()
        tf.train.Saver().restore(sess, save_path)
        self._sess = sess
    def sampling(self, init_str, max_sample=30):
        sample = [c for c in init_str]
        self._sess.run(tf.global_variables_initializer())
        for c in sample:
            feed = {self._x: np.reshape(c, [1, 1])}
            pre = self._sess.run(self._prediction, feed_dict=feed)[0]
        c = pick_top_n(pre[0], 3500)
        sample.append(c)
        for count in range(max_sample):
            x = np.zeros([1, 1])
            x[0][0] = c
            feed = {self._x: x}
            pre = self._sess.run(self._prediction, feed_dict=feed)[0]
            c = pick_top_n(pre[0], 3500)
#             c = np.argsort(pre[0])[-1]
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
        return len(self._int_to_word)
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
        return self._int_to_word.get(index, "unknown")
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

def train():
    obj = corpus(path, 32, 26)
    with shelve.open(obj_path) as fp:
        fp["obj"] = obj
    model = lstm_model(hidden_size=128, num_layer=2, 
                     corpus=obj, keep_prob=1.0, 
                     embedding_size=128, max_step=10000,
                     lr=0.005)
    model.train()

def test(str, max_sample=200):
    obj = None
    assert os.path.exists("{0}.bak".format(obj_path)), "找不到文件"
    with shelve.open(obj_path) as fp:
        obj = fp["obj"]
    model = lstm_model(hidden_size=128, num_layer=2, 
                     corpus=obj, keep_prob=1.0, 
                     embedding_size=128, max_step=10000,
                     lr=0.001, sampling=True)
    model.load_model()
    sample = model.sampling(obj.sentence_to_int(str), max_sample)
    print(obj.int_to_sentence(sample))

if __name__ == "__main__":
#     train()
    test("何人无不见，此地自何如。", max_sample=300)