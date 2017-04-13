import chainer
import chainer.functions as F
import chainer.links as L
import sys

sys.path.append('../../../')
from source.chainer_functions.misc import call_bn


class CNN(chainer.Chain):
    def __init__(self, n_outputs=10, dropout_rate=0.5, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__(
            c1=L.Convolution2D(3, 128, ksize=3, stride=1, pad=1),
            c2=L.Convolution2D(128, 128, ksize=3, stride=1, pad=1),
            c3=L.Convolution2D(128, 128, ksize=3, stride=1, pad=1),
            c4=L.Convolution2D(128, 256, ksize=3, stride=1, pad=1),
            c5=L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            c6=L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            c7=L.Convolution2D(256, 512, ksize=3, stride=1, pad=0),
            c8=L.Convolution2D(512, 256, ksize=1, stride=1, pad=0),
            c9=L.Convolution2D(256, 128, ksize=1, stride=1, pad=0),
            l_cl=L.Linear(128, n_outputs),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(128),
            bn4=L.BatchNormalization(256),
            bn5=L.BatchNormalization(256),
            bn6=L.BatchNormalization(256),
            bn7=L.BatchNormalization(512),
            bn8=L.BatchNormalization(256),
            bn9=L.BatchNormalization(128),
        )
        if top_bn:
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
        if self.top_bn:
            logit = call_bn(self.bn_cl, logit, test=not train, update_batch_stats=update_batch_stats)
        return logit
