'''
Created on 2018年5月20日

@author: IL MARE
'''
n_queen = 8
result_mat = []

def backTrace(column, vaildSet, tagIndex, tag):
    global result_mat
    for index, col in enumerate(column):
        if col != -1:
            if index != tagIndex and abs(tag - col) == abs(tagIndex - index):
                return False
    if len(vaildSet) == 0:
        result_mat.append(column.copy())
        return True
    for vaild in vaildSet:
        column[tagIndex + 1] = vaild
        flag = backTrace(column, vaildSet - set([vaild]), tagIndex + 1, vaild)
        if not flag:
            column[tagIndex + 1] = -1
            
    
if __name__ == "__main__":
    column = [ -1 for i in range(n_queen)]
    vaildSet = set(range(n_queen))
    backTrace(column, vaildSet, -1, -1)
    for row in result_mat:
        print(row)
