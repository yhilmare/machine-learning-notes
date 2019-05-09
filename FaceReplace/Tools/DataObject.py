'''
Created By ILMARE
@Date 2019-3-3
'''
import re
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}

def umeyama(src, dst, estimate_scale):
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = np.dot(dst_demean.T, src_demean) / num
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale
    return T

def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result

def random_warp(image):
    assert image.shape == (128, 128, 3)
    range_ = np.linspace(64 - 64, 64 + 64, 9)
    mapx = np.broadcast_to(range_, (9, 9))
    mapy = mapx.T
    mapx = mapx + np.random.normal(size=(9, 9), scale=2.5)
    mapy = mapy + np.random.normal(size=(9, 9), scale=2.5)
    interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
    interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:65:8, 0:65:8].T.reshape(-1, 2)
    mat = umeyama(src_points, dst_points, True)[0:2]
    target_image = cv2.warpAffine(image, mat, (64, 64))
    return warped_image, target_image


def get_training_data(images, batch_size):
    indices = np.random.randint(len(images), size=batch_size)
    for i, index in enumerate(indices):
        image = images[index]
        image = random_transform(image, **random_transform_args)
        warped_img, target_img = random_warp(image)
        if i == 0:
            warped_images = np.empty((batch_size,) + warped_img.shape, warped_img.dtype)
            target_images = np.empty((batch_size,) + target_img.shape, warped_img.dtype)
        warped_images[i] = warped_img
        target_images[i] = target_img
    return warped_images, target_images

class ImageTrainObject:
    def __init__(self, filePath, batchSize):
        self._filePath = filePath
        self._batchSize = batchSize
        # if re.match(r"^/.+/[^.]+$", self._filePath) is None:
        #     raise Exception("filePath is invalid")
        if self._filePath[len(self._filePath) - 1] != '/':
            self._filePath += '/'
        self._fileItems = os.listdir(self._filePath)
        if batchSize >= self.DataCount:
            raise Exception("Too big batchSize")
    @property
    def DataCount(self):
        return len(self._fileItems)
    def generateBatch(self):
        beginIdx = np.random.randint(0, self.DataCount - self._batchSize)
        destFile = self._fileItems[beginIdx: beginIdx + self._batchSize]
        return_mat = []
        for file in destFile:
            img = cv2.imread("{0}{1}".format(self._filePath, file))
            return_mat.append(img)
        return get_training_data(np.array(return_mat, dtype=np.uint8), self._batchSize)

if __name__ == "__main__":
    img = cv2.imread(r"F:\tensorflow\automodel\scrawler\video\trainImg\18.jpg")
    img = np.array([img], dtype=np.uint8)
    warp, target = get_training_data(img, 1)
    fig = plt.figure("compare")
    ax = fig.add_subplot(121)
    b, g, r = cv2.split(warp[0])
    source = cv2.merge([r, g, b])
    ax.imshow(source)
    ax.axis("off")
    bx = fig.add_subplot(122)
    bx.axis("off")
    b, g, r = cv2.split(target[0])
    dest = cv2.merge([r, g, b])
    bx.imshow(dest)
    plt.show()
    # filePath = r"F:/tensorflow/automodel/scrawler/video/trainImg/"
    # batchSize = 64
    # obj = ImageTrainObject(filePath, batchSize)
    # obj.generateBatch()
    # print(obj.DataCount)