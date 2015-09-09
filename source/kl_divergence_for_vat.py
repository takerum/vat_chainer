import numpy
from chainer import cuda
from chainer import function
from chainer.utils import type_check


class CategoricalKLDivergence(function.Function):

    def __init__(self, unchain_py=True):
        self.unchain_py = unchain_py

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs[0])
        """
        return (1/N) * \sum_i^N \sum_j^L [py_ij * log(py_ij) - py_ij * log(py_tilde_ij)]
        """
        py,py_tilde = inputs
        kl = py * ( xp.log(py) - xp.log(py_tilde) )
        ret = xp.mean(xp.sum(kl,axis=1,keepdims=True),axis=0,keepdims=True)
        return ret.reshape(()),


    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs[0])
        """
        (gradient w.r.t py) = log(py) + 1 - log(py_tilde)
        (gradient w.r.t py_tilde) = - py/py_tilde
        """
        py,py_tilde = inputs
        coeff = grad_outputs[0]/py.shape[0]
        if(self.unchain_py):
            ret_py = None
        else:
            ret_py = coeff * ( xp.log(py) - xp.log(py_tilde) + 1)
        ret_py_tilde = -coeff * py/py_tilde
        return ret_py,ret_py_tilde



def categorical_kl_divergence(py, py_tilde, unchain_py):
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
    return CategoricalKLDivergence(unchain_py)(py, py_tilde)
