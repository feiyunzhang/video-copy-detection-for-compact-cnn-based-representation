# -*- coding: UTF-8 -*-
import numpy as np
from sklearn import linear_model
import scipy.misc
from matplotlib import pyplot as plt


class KSVD(object):
    def __init__(self, n_components, max_iter=100, tol=1e-6,
                 n_nonzero_coefs=32):
        """
        稀疏模型Y = DX，Y为样本矩阵，使用KSVD动态更新字典矩阵D和稀疏矩阵X
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter: 最大迭代次数
        :param tol: 稀疏表示结果的容差
        :param n_nonzero_coefs: 稀疏度
        """
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        """
        初始化字典矩阵
        """
        u, s, v = np.linalg.svd(y)
        self.dictionary = u[:, :self.n_components]

    def _update_dict(self, y, d, x):
        """
        使用KSVD更新字典的过程
        """
        for i in range(self.n_components):
            print ("ksvd")
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue

            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d, x

    def fit(self, y):
        """
        KSVD迭代过程
        """
        self._initialize(y)
        for i in range(self.max_iter):
            x = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
            e = np.linalg.norm(y - np.dot(self.dictionary, x))
            print("ok")
            if e < self.tol:
                break
            self._update_dict(y, self.dictionary, x)

        self.sparsecode = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        return self.dictionary, self.sparsecode


if __name__ == '__main__':
    #im_ascent = scipy.misc.ascent().astype(np.float)
    im_ascent1 = np.load('/workspace/data/fyzhang/feature/test_data_pca.npy')
    im_ascent = im_ascent1.T
    ksvd = KSVD(2048)
    dictionary, sparsecode = ksvd.fit(im_ascent)
    print dictionary.shape
    print sparsecode.shape
    print dictionary
    print "code",sparsecode
    print (np.dot(sparsecode.T,dictionary)).shape
    print(np.dot(sparsecode.T,dictionary)-im_ascent1)
