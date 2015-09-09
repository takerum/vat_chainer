# copied from original batch noramlization function and modified for adversarial and virtual adversarial training.

from chainer.functions.normalization.batch_normalization import BatchNormalization
from chainer import Function
import cupy


class MyBatchNormalization(BatchNormalization):
    def __call__(self, x, test=False, finetune=False,update_batch_estimations=True):
        self.use_batch_mean = not test
        self.is_finetune = finetune
        self.update_batch_estimations = update_batch_estimations
        return Function.__call__(self, x)

    def forward(self, x_orig):
        xp = cupy.get_array_module(*x_orig)
        ldim, cdim, rdim = self._internal_shape(x_orig[0])
        x = x_orig[0].reshape(ldim, cdim, rdim)

        if self.use_batch_mean:
            mean = x.mean(axis=(0, 2), keepdims=True)
            var = x.var(axis=(0, 2), keepdims=True)
            var += self.eps
        else:
            mean = self.avg_mean
            var = self.avg_var

        self.std = xp.sqrt(var, dtype=var.dtype)
        x_mu = x - mean
        self.x_hat = x_mu / self.std
        y = self.gamma * self.x_hat
        y += self.beta

        # Compute exponential moving average
        if self.use_batch_mean and self.update_batch_estimations:
            if self.is_finetune:
                self.N[0] += 1
                decay = 1. / self.N[0]
            else:
                decay = self.decay

            m = ldim * rdim
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            self.avg_mean *= decay
            self.avg_mean += (1 - decay) * mean
            self.avg_var *= decay
            self.avg_var += (1 - decay) * adjust * var

        return y.reshape(x_orig[0].shape),
