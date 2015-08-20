import numpy
from chainer import cuda
from chainer import function
from chainer.utils import type_check


class KLDivergenceForVAT(function.Function):

    def __init__(self, unchain_py=True, use_cudnn=True):
        self.unchain_py = unchain_py
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        py_type, py_tilde_type = in_types

        type_check.expect(
            py_type.dtype == numpy.float32,
            py_type.ndim == 2,
            py_tilde_type.dtype == numpy.float32,
            py_tilde_type.ndim == 2,

            py_type.shape[0] == py_tilde_type.shape[0],
        )
    def check_type_backward(self, in_types, out_types):
        type_check.expect(
            in_types.size() == 2,
            out_types.size() == 1,
        )
        y_type, = out_types
        type_check.expect(y_type.ndim == 0)  # means scalar

    def forward_cpu(self, inputs):
        """
        return (1/N) * \sum_i^N \sum_j^L [py_ij * log(py_ij) - py_ij * log(py_tilde_ij)]
        """
        py,py_tilde = inputs
        kl = py * ( numpy.log(py) - numpy.log(py_tilde) )
        ret = numpy.mean(numpy.sum(kl,axis=1,keepdims=True),axis=0,keepdims=True)
        return ret.reshape(()),

    def forward_gpu(self, inputs):
        py,py_tilde = inputs
        c = py.shape[1]
        kl = cuda.empty_like(py)
        cuda.elementwise(
                'float *kl, const float* py, const float* py_tilde, int c',
                'kl[i] = py[i] * ( log(py[i]) -  log(py_tilde[i]) )'
                ,'kldivergence')(kl, py, py_tilde, c)
        ret = cuda.gpuarray.sum(kl)/py.shape[0]
        return ret,


    def backward_cpu(self, inputs, grad_outputs):
        """
        (gradient w.r.t py) = log(py) + 1 - log(py_tilde)
        (gradient w.r.t py_tilde) = - py/py_tilde
        """
        py,py_tilde = inputs
        coeff = grad_outputs[0]/py.shape[0]
        if(self.unchain_py):
            ret_py = None
        else:
            ret_py = coeff * ( numpy.log(py) - numpy.log(py_tilde) + 1)
        ret_py_tilde = -coeff * py/py_tilde
        #print ret_py_tilde
        return ret_py,ret_py_tilde

    def backward_gpu(self, inputs, grad_outputs):
        py,py_tilde = inputs
        coeff = grad_outputs[0]/py.shape[0]
        if(self.unchain_py):
            ret_py = None
        else:
            ret_py = cuda.empty_like(py)
            cuda.elementwise(
                'float* ret_py, const float* py, const float* py_tilde, const float* coeff',
                'ret_py[i] = *coeff * ( log(py[i]) - log(py_tilde[i]) + 1)',
                'grad_wrt_py')(ret_py, py, py_tilde, coeff)

        ret_py_tilde = cuda.empty_like(py_tilde)
        cuda.elementwise(
            'float* ret_py_tilde, const float* py, const float* py_tilde, const float* coeff',
            'ret_py_tilde[i] = - *coeff * py[i] / py_tilde[i]',
            'grad_wrt_py_tilde')(ret_py_tilde, py, py_tilde, coeff)

        return ret_py, ret_py_tilde


def kldivergence_for_vat(py, py_tilde, unchain_py, use_cudnn=True):
    """Computes KL divergence between y and _y:KL[p(y|x)||p(_y|x)] (softmax activation only)

    Args:
        py (Variable): Variable holding a matrix whose (i, j)-th element
            indicates normalized probability of the class j at the i-th
            example.
        py_tilde (Variable): Variable holding a matrix whose (i, j)-th element
            indicates normalized probability of the class j at the i-th
            example (assumed to be probability y given "perturbed x").

    Returns:
        Variable: A variable holding a scalar array of the KL divergence loss.



    """
    return KLDivergenceForVAT(unchain_py, use_cudnn)(py, py_tilde)
