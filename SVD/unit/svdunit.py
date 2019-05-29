'''
Created aBy ILMARE
@Date 2019-5-29
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2

mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8

def transform():
    A = np.asmatrix([[3, 1], [0.5, 1]], dtype=np.float32)
    eigvalue, eigvector = np.linalg.eigh(A)
    print(eigvalue)
    print(eigvector)
    print(A * eigvector)
    print(eigvector * np.diag(eigvalue) * eigvector.T)
    v = np.asmatrix([[0, 0], [0.975663, 0.21927527], [0, 0], [-0.4099776, 0.9120956]], dtype=np.float32)
    M = np.asmatrix([[3, 1], [0.5, 1]], dtype=np.float32)
    # v = np.asmatrix([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    # M = np.asmatrix([[0.5, 0.866], [-0.866, 0.5]], dtype=np.float32)
    v1 = (M * v.T).T
    print(v1)
    fig = plt.figure("test")
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.plot(v[0:2, 0], v[0:2, 1], color="black")
    ax.plot(v[2:4, 0], v[2:4, 1], color="black")
    ax.plot(v1[0:2, 0], v1[0:2, 1], color="red", linestyle="--")
    ax.plot(v1[2:4, 0], v1[2:4, 1], color="red", linestyle="--")
    ax.plot([0, 3], [0, 1], color="blue")
    ax.plot([0, 0.5], [0, 1], color="blue")
    # ax.plot(v1[:, 0], v1[:, 1])
    plt.show()

def svd():
    A = np.asmatrix([[0, 1], [1, 1], [1, 0]], dtype=np.float32)
    print(np.linalg.matrix_rank(A))
    eigvalue, V = np.linalg.eigh(A.T * A)
    _, M = np.linalg.eigh(A * A.T)
    print(V)
    print(M)
    sigma = np.diag(np.sqrt(eigvalue))
    sigma = np.vstack((np.asmatrix([0, 0]), sigma))
    print(sigma)
    print(M * sigma * V.T)

if __name__ == "__main__":
    img = cv2.imread(r"f:/96.jpg")
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    r = np.asmatrix(b, dtype=np.float32)
    print(r.shape)
    _, V = np.linalg.eigh(r.T * r)
    eigvalue, M = np.linalg.eigh(r * r.T)
    sigma = np.diag(np.sqrt(eigvalue))
    sigma = np.vstack((np.zeros(shape=(420, 540)), sigma))
    tmp = V * sigma * M.T
    # print(tmp.shape)
    # eigvalue, eigvector = np.linalg.eigh(r)
    # r = eigvector * np.diag(eigvalue) * eigvector.T
    # # U, sigma, VT = np.linalg.svd(r)
    # # r = U * np.diag(sigma) * VT
    # r = np.asmatrix(r, dtype=np.uint8)
    # print(r)
    tmp = np.asmatrix(tmp, dtype=np.uint8)
    cv2.imshow("asd", tmp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
