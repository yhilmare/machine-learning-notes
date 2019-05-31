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
    eigvalue, eigvector = np.linalg.eig(A)
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

def img():
    img = cv2.imread(r"f:/96.jpg")
    height = img.shape[0]
    width = img.shape[1]
    k = 2
    r = np.asmatrix(img[:, :, 0], dtype=np.float32)
    g = np.asmatrix(img[:, :, 1], dtype=np.float32)
    b = np.asmatrix(img[:, :, 2], dtype=np.float32)
    U1, sigma1, V1T = np.linalg.svd(r)
    sigma1 = sigma1[0: k]
    U2, sigma2, V2T = np.linalg.svd(g)
    sigma2 = sigma2[0: k]
    U3, sigma3, V3T = np.linalg.svd(b)
    sigma3 = sigma3[0: k]
    sigma1 = np.hstack(
        (np.vstack((np.diag(sigma1), np.zeros(shape=(height - k, k)))), np.zeros(shape=(height, width - k))))
    sigma2 = np.hstack(
        (np.vstack((np.diag(sigma2), np.zeros(shape=(height - k, k)))), np.zeros(shape=(height, width - k))))
    sigma3 = np.hstack(
        (np.vstack((np.diag(sigma3), np.zeros(shape=(height - k, k)))), np.zeros(shape=(height, width - k))))
    channel_1 = np.asmatrix(U1 * sigma1 * V1T, dtype=np.uint8)
    channel_2 = np.asmatrix(U2 * sigma2 * V2T, dtype=np.uint8)
    channel_3 = np.asmatrix(U3 * sigma3 * V3T, dtype=np.uint8)
    img = cv2.merge([channel_1, channel_2, channel_3])
    # print(img.shape)
    cv2.imshow("sdasd", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def svd(matrix):
    matrix = np.asmatrix(matrix, dtype=np.float32)
    height = matrix.shape[0]
    width = matrix.shape[1]
    if height < width:
        eig, U = np.linalg.eig(matrix * matrix.T)
        sigma = np.sqrt(eig.real)
        _, V = np.linalg.eig(matrix.T * matrix)
        return U.real, sigma, V.T.real
    else:
        _, U = np.linalg.eig(matrix * matrix.T)
        sigma, V = np.linalg.eig(matrix.T * matrix)
        sigma = np.sqrt(sigma.real)
        return U.real, sigma, V.real.T

if __name__ == "__main__":
    v = np.asmatrix([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    W = np.asmatrix([[-0.866, 0.5], [0.5, 0.866]], dtype=np.float32)
    v1 = (W * v.T).T
    print(v1.T)
    fig = plt.figure("test")
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.plot(v[:, 0], v[:, 1])
    ax.plot(v1[:, 0], v1[:, 1])

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.show()

