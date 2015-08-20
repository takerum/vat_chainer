# copied from original batch noramlization function and modified for adversarial and virtual adversarial training.

from chainer.functions.batch_normalization import BatchNormalization
from chainer import Function
from chainer import cuda

def _kernel_with_I(args, expr, name):
    return cuda.elementwise(
        '{}, int cdim, int rdim'.format(args),
        'int I = i / rdim % cdim; {};'.format(expr),
        name)

_one = None


def _partial_reduce(x):
    global _one
    out_axis, sum_axis = x.shape
    one = _one
    if one is None or one.size < sum_axis:
        one = cuda.ones(sum_axis)
        _one = one
    one = one[:sum_axis]
    handle = cuda.get_cublas_handle()
    ret = cuda.empty(out_axis)
    cuda.cublas.cublasSgemv(handle, 't', sum_axis, out_axis,
                            numpy.float32(
                                1.0), x.gpudata, sum_axis, one.gpudata,
                            1, numpy.float32(0.0), ret.gpudata, 1)
    return ret

if cuda.available:
    @cuda.cutools.context_dependent_memoize
    def _create_reduction_kernel(shape0, expr1, expr2):
        return cuda.elementwise(
            '''
                float* ret1, float* ret2,
                const float* x, const float* y,
                float alpha, int shape12
            ''', '''
                float sum1 = 0, sum2 = 0;
                for (int j = 0; j < {0}; j++) {{
                    int I = j * shape12 + i;
                    sum1 += {1};
                    sum2 += {2};
                }}
                ret1[i] = sum1 * alpha;
                ret2[i] = sum2 * alpha;
            '''.format(shape0, expr1, expr2), 'bn_asix02')


def _cusum_axis02(x, y=None, expr1='x[I]', expr2='x[I] * x[I]', mean=False):
    with cuda.using_cumisc():
        shape = x.shape
        ret1 = cuda.empty_like(x[0])
        ret2 = cuda.empty_like(x[0])
        if y is None:
            y = x
        alpha = 1.0
        if mean:
            alpha = 1.0 / (shape[0] * shape[2])

        # In most cases shape[0] is constant.
        # Therefore, the kernel is compiled only once.
        # If shape[0] is small, Compiler will perform loop unrolling.
        _create_reduction_kernel(shape[0], expr1, expr2)(
            ret1, ret2, x, y, alpha, shape[1] * shape[2])

        if shape[2] != 1:
            ret1 = _partial_reduce(ret1)
            ret2 = _partial_reduce(ret2)
        ret_shape = (1, shape[1], 1)
        return (ret1.reshape(ret_shape), ret2.reshape(ret_shape))

class MyBatchNormalization(BatchNormalization):
    def __call__(self, x, test=False, finetune=False,update_batch_estimations=True):
        self.use_batch_mean = not test
        self.is_finetune = finetune
        self.update_batch_estimations = update_batch_estimations
        return Function.__call__(self, x)

    def forward_cpu(self, x_orig):
        ldim, cdim, rdim = self._internal_shape(x_orig[0])
        x = x_orig[0].reshape(ldim, cdim, rdim)

        if self.use_batch_mean:
            mean = x.mean(axis=(0, 2), keepdims=True)
            var = x.var(axis=(0, 2), keepdims=True) + self.eps
        else:
            mean = self.avg_mean
            var = self.avg_var

        self.std = numpy.sqrt(var)
        x_mu = x - mean
        self.x_hat = x_mu / self.std
        y = self.gamma * self.x_hat + self.beta

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

    def forward_gpu(self, x_orig):
        ldim, cdim, rdim = self._internal_shape(x_orig[0])
        x = x_orig[0].reshape(ldim, cdim, rdim)

        if self.use_batch_mean:
            mean, sqmean = _cusum_axis02(x, mean=True)
            var = sqmean  # reuse buffer
            cuda.elementwise(
                'float* var, const float* mean, float eps',
                'var[i] = var[i] - mean[i] * mean[i] + eps',
                'bn_var')(var, mean, self.eps)
        else:
            mean = self.avg_mean
            var = self.avg_var

        y = cuda.empty_like(x_orig[0])
        _kernel_with_I(
            '''
                float* y, const float* x,
                const float* mean, const float* var,
                const float* gamma, const float* beta
            ''',
            'y[i] = (x[i] - mean[I]) * rsqrtf(var[I]) * gamma[I] + beta[I];',
            'bn_fwd')(y, x, mean, var, self.gamma, self.beta, cdim, rdim)

        # Compute exponential moving average
        if self.use_batch_mean and self.update_batch_estimations:
            if self.is_finetune:
                self.N[0] += 1
                decay = 1. / self.N[0]
            else:
                decay = self.decay

            m = ldim * rdim
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            cuda.elementwise(
                '''
                   float* avg_mean, const float* mean,
                   float* avg_var, const float* var,
                   float decay, float adjust
                ''', '''
                   avg_mean[i] = decay * avg_mean[i]
                                 + (1 - decay) * mean[i];
                   avg_var[i]  = decay * avg_var[i]
                                 + (1 - decay) * adjust * var[i];
                ''',
                'bn_moving_avg')(
                    self.avg_mean, mean, self.avg_var, var, decay, adjust)

        return y,
