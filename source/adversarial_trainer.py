from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
import numpy as numpy
from  input_gradient_keeper import InputGradientKeeper
from nn import NN

def as_mat(x):
    return x.reshape(x.shape[0], x.size // x.shape[0])

def normalize_axis1_cpu(x):
    x_scaled = x / (1e-6 + numpy.max(numpy.abs(x), axis=1, keepdims=True))
    return x_scaled / numpy.sqrt(1e-6 + numpy.sum(x_scaled ** 2, axis=1, keepdims=True))

def normalize_axis1_gpu(x):
    nx = cuda.empty_like(x)
    maxes = cuda.empty((x.shape[0],), dtype=numpy.float32)
    c = x.shape[1]
    cuda.elementwise(
        'float* maxes, const float* x, int c',
        '''
                const float* row = x + i * c;
                float maxval = row[0];
                for (int j = 1; j < c; ++j) {
                    if (maxval < row[j]) {
                        maxval = row[j];
                    }
                }
            maxes[i] = maxval;
            ''', 'gx_rowmax')(maxes, x, c)
    cuda.elementwise(
        'float* y, const float* x, const float* maxes, int c, float k',
        'y[i] = x[i] / (maxes[i / c] + k)',
        'gx_scaled_with_max')(nx, x, maxes, c, 1e-6)
    coeff = maxes
    cuda.elementwise(
        'float* coeff, const float* y, int c,float k',
        '''
               const float* row = y + i * c;
               float sum = 0;
               for (int j = 0; j < c; ++j) {
                 sum += row[j]*row[j];
               }
               coeff[i] = 1 / sqrt(sum+k);
        ''', 'gx_norm')(coeff, nx, c, 1e-6)
    cuda.elementwise(
        'float* nx, const float* coeff, int c',
        'nx[i] = nx[i] * coeff[i / c]',
        'normalize_gx')(nx, coeff, c)
    return nx

def normalize_axis1(x):
    if(isinstance(x,cuda.GPUArray)):
        return normalize_axis1_gpu(x)
    else:
        return normalize_axis1_cpu(x)

def perturbation_with_L2_norm_constraint(x,norm):
    return norm * normalize_axis1(x)

def perturbation_with_max_norm_constraint(x,norm):
    if(isinstance(x, cuda.GPUArray)):
        ptb = cuda.empty_like(x)
        gx_sign = cuda.empty_like(x)
        ar_pl_one = cuda.empty_like(x).fill(1.0)
        ar_mn_one = cuda.empty_like(x).fill(-1.0)
        cuda.gpuarray.if_positive(x, ar_pl_one, ar_mn_one, out=gx_sign)
        cuda.elementwise(
            'float* y, const float* gx_sign, float eps',
            'y[i] = eps * gx_sign[i]',
            'normalize_gx')(ptb, gx_sign, norm)
        return ptb
    else:
        return norm * numpy.sign(x)


class AdversarialTrainer(object):

    def __init__(self,nn,out_act_type='Softmax',epsilon=1.0,norm_constraint_type='L2',lamb=1.0):
        self.nn = nn
        self.out_act_type = out_act_type
        self.epsilon = epsilon
        self.norm_constraint_type = norm_constraint_type
        self.lamb = lamb

    def accuracy(self,x,t):
        return F.accuracy(self.nn.y_given_x(x,test=True),t)

    def accuracy_for_adversarial_examples(self,x,t):
        xadv,loss = self.get_adversarial_examples(x,t,test=True)
        return F.accuracy(self.nn.y_given_x(xadv,test=True),t)

    def cost_fitness(self,x,t,test=False,update_batchstat_estimation=True):
        """
        :param x: input (Variable instance)
        :param t: target (Variable instance)
        :return: standard fitness loss ( cross entropy or mean squared error )
        """
        y = self.nn.y_given_x(x,test,update_batchstat_estimation)
        if(self.out_act_type == 'Softmax' ):
            return F.softmax_cross_entropy(y,t)
        elif(self.out_act_type == 'Linear'):
            return F.mean_squared_error(y,t)
        else:
            raise NotImplementedError()

    def cost_adversarial_training(self,x,t,test=False):
        xadv, ptb, cost_fitness = self.get_adversarial_examples(x,t,test=test)
        cost_fitness_adv = self.cost_fitness(xadv,t,test=test, update_batchstat_estimation=False)
        return cost_fitness, self.lamb*cost_fitness_adv

    def get_adversarial_examples(self,x,t,test=False):
        input_gradient_keeper = InputGradientKeeper()
        x_ = input_gradient_keeper(x)
        cost_fitness = self.cost_fitness(x_,t,test=test, update_batchstat_estimation=True)
        cost_fitness.backward()
        gx = input_gradient_keeper.gx
        if (self.norm_constraint_type == 'L2'):
            ptb = perturbation_with_L2_norm_constraint(gx,self.epsilon)
        else:
            ptb = perturbation_with_max_norm_constraint(gx,self.epsilon)
        xadv = x + ptb.reshape(x.data.shape)
        return xadv, ptb, cost_fitness




