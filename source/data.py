import pickle
import datetime, math, sys, time

from sklearn.datasets import fetch_mldata
import numpy as np
import math
import chainer
import chainer.functions as F
from chainer import FunctionSet, Variable, optimizers, cuda, serializers
import six


def augmentation(images, random_crop=True, random_flip=True):
    # random crop and random flip
    h, w = images.shape[2], images.shape[3]
    pad_size = 2
    aug_images = []
    padded_images = np.pad(images, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'reflect')
    for image in padded_images:
        if random_flip:
            image = image[:, :, ::-1] if np.random.uniform() > 0.5 else image
        if random_crop:
            offset_h = np.random.randint(0, 2 * pad_size)
            offset_w = np.random.randint(0, 2 * pad_size)
            image = image[:, offset_h:offset_h + h, offset_w:offset_w + w]
        else:
            image = image[:, pad_size:pad_size + h, pad_size:pad_size + w]
        aug_images.append(image)
    ret = np.stack(aug_images)
    assert ret.shape == images.shape
    return ret


class Data:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.index = np.arange(self.N)

    @property
    def N(self):
        return len(self.data)

    def _augmentation(self, images, trans=True, flip=True):
        # shape of `image' [N, K, W, H]
        assert images.ndim == 4
        return augmentation(images, trans, flip)

    def get(self, n=None, shuffle=True, aug_trans=True, aug_flip=True, gpu=-1):
        if shuffle:
            ind = np.random.permutation(self.data.shape[0])
        else:
            ind = np.arange(self.data.shape[0])
        if n is None:
            n = self.data.shape[0]
        index = ind[:n]
        batch_data = self.data[index]
        batch_label = self.label[index]
        if aug_trans or aug_flip:
            batch_data = self._augmentation(batch_data, aug_trans, aug_flip)
        if gpu > -1:
            return cuda.to_gpu(batch_data, device=gpu), \
                   cuda.to_gpu(batch_label, device=gpu)
        else:
            return batch_data, batch_label

    def put(self, data, label):
        if self.data is None:
            self.data = cuda.to_cpu(data)
            self.label = cuda.to_cpu(label)
        else:
            self.data = np.vstack([self.data, cuda.to_cpu(data)])
            self.label = np.hstack([self.label, cuda.to_cpu(label)]).reshape((self.data.shape[0]))
