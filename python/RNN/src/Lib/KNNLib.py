'''
Created on 2018年4月25日

@author: IL MARE
'''
import numpy as np
import shelve

def load_embed():
    with shelve.open("{0}parameter".format("G:/Machine-Learning-Study-Notes/python/RNN/src/Units/")) as fp:
        embedding_matrix = fp["word2vec"]
        reverse_dic = fp["reverse_dic"]
        dic = fp["dic"]
    return embedding_matrix, dic, reverse_dic

if __name__ == "__main__":
    a = [[1,2,3],[4,5,6],[7,8,9]]
    a = np.array(a)
    print(a[[0,1], [1,1]])