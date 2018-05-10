'''
Created on 2018年4月25日

@author: IL MARE
'''

from Lib.model import lstm_model
from Lib.model import corpus
import shelve
import os

path = r"G:/Machine-Learning-Study-Notes/python/RNN/modelFile/train.txt"
save_path = r"G:/Machine-Learning-Study-Notes/python/RNN/modelFile/"

def train():
    obj = corpus(path, 32, 26, 3500)
    with shelve.open("{0}corpus.bak".format(save_path)) as fp:
        fp["obj"] = obj
    model = lstm_model(hidden_size=128, num_layer=2, 
                     corpus=obj, keep_prob=1.0, 
                     embedding_size=128, max_step=2000,
                     lr=0.005, save_path=save_path)
    model.train()

def test(str, max_sample=200):
    obj = None
    assert os.path.exists("{0}corpus.bak".format(save_path)), "找不到文件"
    with shelve.open("{0}corpus".format(save_path)) as fp:
        obj = fp["obj"]
    model = lstm_model(hidden_size=128, num_layer=2, 
                     corpus=obj, keep_prob=1.0, 
                     embedding_size=128, max_step=2000,
                     lr=0.001, save_path=save_path, sampling=True)
    model.load_model()
    sample = model.sampling(obj.sentence_to_int(str), max_sample)
    print(obj.int_to_sentence(sample))

if __name__ == "__main__":
#     train()
    test("", max_sample=300)