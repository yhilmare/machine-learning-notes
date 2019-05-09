'''
Created on 2018年7月5日

@author: IL MARE
'''
from Libs.AverageFace import AverageFace 

data_path = r"G:\python\sources\nwpu\dectImage"

if __name__ == "__main__":
    obj = AverageFace(data_path, (700, 700))
    obj.showImage()