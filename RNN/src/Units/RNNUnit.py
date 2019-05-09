'''
Created on 2018年4月25日

@author: IL MARE
'''
import os
import sys
sys.path.append(os.getcwd())
from Lib.model import lstm_model
from Lib.model import corpus
import shelve
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import time

path = r"G:/Machine-Learning-Study-Notes/python/RNN/modelFile/train.txt"
save_path = r"G:/Machine-Learning-Study-Notes/python/RNN/modelFile/"

def train_model():
    obj = corpus(path, 50, 50, 3500)
    with shelve.open("{0}corpus".format(save_path)) as fp:
        fp["obj"] = obj
    model = lstm_model(hidden_size=128, num_layer=2, 
                     corpus=obj, keep_prob=1.0, 
                     embedding_size=128, max_step=10000,
                     lr=0.005, save_path=save_path)
    fig = plt.figure("cross-entropy")
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    ax = fig.add_subplot(111)
    ax.grid(True)
    for return_mat in model.train():
        x = np.arange((len(return_mat)))
        y = np.array(return_mat)
        ax.plot(x, y, linewidth=0.8, color="b")
        plt.pause(0.1)
    plt.show()

def test_model(init_str, max_sample=200):
    obj = None
    assert os.path.exists("{0}corpus.bak".format(save_path)), "找不到文件"
    with shelve.open("{0}corpus".format(save_path)) as fp:
        obj = fp["obj"]
    model = lstm_model(hidden_size=128, num_layer=2, 
                     corpus=obj, keep_prob=1.0, 
                     embedding_size=128, max_step=5000,
                     lr=0.005, save_path=save_path, sampling=True)
    model.load_model()
    sample = model.sampling(obj.sentence_to_int(init_str), max_sample)
    print(obj.int_to_sentence(sample))

if __name__ == "__main__":
#    train_model()
    test_model("", max_sample=1500)