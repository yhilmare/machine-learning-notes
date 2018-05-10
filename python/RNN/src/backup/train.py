'''
Created on 2018年5月8日

@author: IL MARE
'''
import tensorflow as tf
from read_utils import TextConverter, batch_generator
from model import CharRNN
import os
import codecs

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('num_seqs', 100, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq')
tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.flags.DEFINE_string('input_file', '', 'utf8 encoded text file')
tf.flags.DEFINE_integer('max_steps', 100000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 10, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')


def main(_):
    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        text = f.read()
    converter = TextConverter(text, FLAGS.max_vocab)
    '''
            下面的代码是将文中出现频率前n的字或词按从高到低组成一个大小为n的列表存入到文件中
    '''
    converter.save_to_file(os.path.join(model_path, 'converter.pkl'))
    '''
           下面这行代码返回全文中每一个字或词在词频排列中的位置索引号所组成的列表，若字或词没有在高频词中则返回高频词列表的大小
    '''
    arr = converter.text_to_arr(text)
    '''
            产生一个g的训练集，g以迭代器的形式返回，每迭代一次返回一个batch，该batch的大小为(n_seq,n_step)
    '''
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)
    '''
            返回整个词汇表的大小
    '''
    print(converter.vocab_size)
    '''
            产生模型
    '''
    model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )
    '''
            训练模型
    '''
    model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()