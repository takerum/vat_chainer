import chainer
import chainer.functions as F
import chainer.links as L
import sys

sys.path.append('../../../')
from source.chainer_functions.misc import call_bn


class CNN(chainer.Chain):
    def __init__(self, n_outputs=10, dropout_rate=0.5, last_bn=False):
        self.dropout_rate = dropout_rate
        self.last_bn = last_bn
        super(CNN, self).__init__(
            c1=L.Convolution2D(3, 128 * self.dim_factor, ksize=3, stride=1, pad=1),
            c2=L.Convolution2D(128 * self.dim_factor, 128 * self.dim_factor, ksize=3, stride=1, pad=1),
            c3=L.Convolution2D(128 * self.dim_factor, 128 * self.dim_factor, ksize=3, stride=1, pad=1),
            c4=L.Convolution2D(128 * self.dim_factor, 256 * self.dim_factor, ksize=3, stride=1, pad=1),
            c5=L.Convolution2D(256 * self.dim_factor, 256 * self.dim_factor, ksize=3, stride=1, pad=1),
            c6=L.Convolution2D(256 * self.dim_factor, 256 * self.dim_factor, ksize=3, stride=1, pad=1),
            c7=L.Convolution2D(256 * self.dim_factor, 512 * self.dim_factor, ksize=3, stride=1, pad=0),
            c8=L.Convolution2D(512 * self.dim_factor, 256 * self.dim_factor, ksize=1, stride=1, pad=0),
            c9=L.Convolution2D(256 * self.dim_factor, 128 * self.dim_factor, ksize=1, stride=1, pad=0),
            l_cl=L.Linear(128 * self.dim_factor, n_outputs),
            bn1=L.BatchNormalization(128 * self.dim_factor),
            bn2=L.BatchNormalization(128 * self.dim_factor),
            bn3=L.BatchNormalization(128 * self.dim_factor),
            bn4=L.BatchNormalization(256 * self.dim_factor),
            bn5=L.BatchNormalization(256 * self.dim_factor),
            bn6=L.BatchNormalization(256 * self.dim_factor),
            bn7=L.BatchNormalization(512 * self.dim_factor),
            bn8=L.BatchNormalization(256 * self.dim_factor),
            bn9=L.BatchNormalization(128 * self.dim_factor),
        )
        if last_bn:
            self.add_link('bn_cl', L.BatchNormalization(n_outputs))

    def __call__(self, x, train=True, update_batch_stats=True):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(call_bn(self.bn1, h, test=not train, update_batch_stats=update_batch_stats), slope=0.1)
        h = self.c2(h)
        h = F.leaky_relu(call_bn(self.bn2, h, test=not train, update_batch_stats=update_batch_stats), slope=0.1)
        h = self.c3(h)
        h = F.leaky_relu(call_bn(self.bn3, h, test=not train, update_batch_stats=update_batch_stats), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, ratio=self.dropout_rate, train=train)

        h = self.c4(h)
        h = F.leaky_relu(call_bn(self.bn4, h, test=not train, update_batch_stats=update_batch_stats), slope=0.1)
        h = self.c5(h)
        h = F.leaky_relu(call_bn(self.bn5, h, test=not train, update_batch_stats=update_batch_stats), slope=0.1)
        h = self.c6(h)
        h = F.leaky_relu(call_bn(self.bn6, h, test=not train, update_batch_stats=update_batch_stats), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, ratio=self.dropout_rate, train=train)

        h = self.c7(h)
        h = F.leaky_relu(call_bn(self.bn7, h, test=not train, update_batch_stats=update_batch_stats), slope=0.1)
        h = self.c8(h)
        h = F.leaky_relu(call_bn(self.bn8, h, test=not train, update_batch_stats=update_batch_stats), slope=0.1)
        h = self.c9(h)
        h = F.leaky_relu(call_bn(self.bn9, h, test=not train, update_batch_stats=update_batch_stats), slope=0.1)
        h = F.average_pooling_2d(h, ksize=h.data.shape[2])
        logit = self.l_cl(h)
        if self.last_bn:
            logit = self.bn_cl(logit, test=not train, update_batch_stats=update_batch_stats)
        return logit
