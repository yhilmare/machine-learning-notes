'''
Created on 2018年5月5日

@author: IL MARE
'''
import numpy as np
import copy
import time
import tensorflow as tf
import pickle

'''
n_seqs可以理解为一个batch中的句子数
n_step可以理解为一个句子中的单词数或汉字数
batch_size就是一个batch中一共包含的单词数或汉字数
n_batch即为全部训练文章中的所有词能够组成的batch总数
'''
def batch_generator(arr, n_seqs, n_steps):
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    '''
            截断训练集，让训练集的大小刚好为batch_size的整数倍
    '''
    arr = arr[:batch_size * n_batches]
    '''
            将整个训练集以句子长度划分为一个矩阵
    '''
    arr = arr.reshape((n_seqs, -1))
    '''
            以下代码产生训练集，每次函数返回一个x和y，其中，x是一个batch大小（n_seq,n_step），相应的标签y跟x也是相同的大小
            只是y将x矩阵的第一列与最后一列互换，产生了标签y。最后用yield返回。
    '''
    while True:
        '''
                        将arr中的元素随机排列
        '''
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y


class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)
            print(len(vocab))
            # max_vocab_process
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)