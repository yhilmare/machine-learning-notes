'''
Created on 2018年6月29日

@author: IL MARE
'''

import Libs.SimpleCNN as SimpleCNN
import Libs.AlexNet as AlexNet

def testSimpleCNN():
    CNN = SimpleCNN.SimpleCNN(0.001, 2000)
    CNN.loadModel()
    CNN.test()

def testAlexNet():
    alex = AlexNet.AlexNet(0.001, 1, 10, 2000)
    alex.loadModel()
    alex.test()

if __name__ == "__main__":
    testAlexNet()
#     testSimpleCNN()