from chainer import Function, FunctionSet, gradient_check, Variable, optimizers
import cupy

def as_mat(x):
    return x.reshape(x.shape[0], x.size // x.shape[0])

class InputGradientKeeper(Function):

    def __call__(self, inputs):
        self.init_gx(inputs)
        return super(InputGradientKeeper, self).__call__(inputs)

    def init_gx(self, inputs):
        xp = cupy.get_array_module(*inputs.data)
        self.gx = as_mat(xp.zeros_like(inputs.data))

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, grad_outputs):
        self.gx.fill(0)
        self.gx += as_mat(grad_outputs[0])
        return grad_outputs

