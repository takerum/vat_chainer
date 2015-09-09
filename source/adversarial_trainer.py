import cupy
import chainer.functions as F
from  input_gradient_keeper import InputGradientKeeper
from nn import NN

def as_mat(x):
    return x.reshape(x.shape[0], x.size // x.shape[0])

def normalize_axis1(x):
    xp = cupy.get_array_module(*x)
    abs_x = abs(x)
    x = x / (1e-6 + abs_x.max(axis=1,keepdims=True))
    x_norm_2 = x**2
    return x / xp.sqrt(1e-6 + x_norm_2.sum(axis=1,keepdims=True))


def perturbation_with_L2_norm_constraint(x,norm):
    return norm * normalize_axis1(x)

def perturbation_with_max_norm_constraint(x,norm):
    xp = cupy.get_array_module(*x)
    return norm * xp.sign(x)


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




