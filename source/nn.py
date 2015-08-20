from chainer import FunctionSet
from abc import ABCMeta, abstractmethod

class NN(FunctionSet):
    __metaclass__ = ABCMeta
    @abstractmethod
    def y_given_x(self,x,test=False,update_batchstat_estimation=True):
        """
        :param x: input ( Variable instance )
        :param test: ( in evaluation mode if test == True)
        :param update_batchstat_estimation: boolean for batch normalization. if True update batch stat with current mini-batch for test mode.
        :return: neural network output ( Variable instance )
        """
        raise NotImplementedError()

    @abstractmethod
    def py_given_y(self,y):
        raise NotImplementedError()
    
    def py_given_x(self,x,test=False,update_batchstat_estimation=True):
        return self.py_given_y(self.y_given_x(x,test,update_batchstat_estimation))
