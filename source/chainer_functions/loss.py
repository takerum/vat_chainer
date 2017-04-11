import chainer
import chainer.functions as F
from chainer import FunctionSet, Variable, optimizers, cuda, serializers


def kl_binary(p_logit, q_logit):
    if isinstance(p_logit, chainer.Variable):
        xp = cuda.get_array_module(p_logit.data)
    else:
        xp = cuda.get_array_module(p_logit)
    p_logit = F.concat([p_logit, xp.zeros(p_logit.shape, xp.float32)], 1)
    q_logit = F.concat([q_logit, xp.zeros(q_logit.shape, xp.float32)], 1)
    return kl_categorical(p_logit, q_logit)


def kl_categorical(p_logit, q_logit):
    if isinstance(p_logit, chainer.Variable):
        xp = cuda.get_array_module(p_logit.data)
    else:
        xp = cuda.get_array_module(p_logit)
    p = F.softmax(p_logit)
    _kl = F.sum(p * (F.log_softmax(p_logit) - F.log_softmax(q_logit)), 1)
    return F.sum(_kl) / xp.prod(xp.array(_kl.shape))


def cross_entropy(logit, y):
    # y should be one-hot encoded probability
    return - F.sum(y * F.log_softmax(logit)) / logit.shape[0]


def kl(p_logit, q_logit):
    if p_logit.shape[1] == 1:
        return kl_binary(p_logit, q_logit)
    else:
        return kl_categorical(p_logit, q_logit)


def distance(p_logit, q_logit, dist_type="KL"):
    if dist_type == "KL":
        return kl(p_logit, q_logit)
    else:
        raise NotImplementedError


def entropy_y_x(p_logit):
    p = F.softmax(p_logit)
    return - F.sum(p * F.log_softmax(p_logit)) / p_logit.shape[0]


def at_loss(forward, x, y, train=True, epsilon=8.0):
    ce = cross_entropy(forward(x, train=train, update_batch_stats=False), y)
    ce.backward()
    d = x.grad
    xp = cuda.get_array_module(x.data)
    d = d / xp.sqrt(xp.sum(d ** 2, axis=range(1, len(d.shape)), keepdims=True))
    d_var = Variable(d.astype(xp.float32))
    return cross_entropy(forward(x + epsilon * d_var, train=train, update_batch_stats=False), y)


def vat_loss(forward, distance, x, train=True, epsilon=8.0, xi=1e-6, Ip=1, p_logit=None):
    if p_logit is None:
        p_logit = forward(x, train=train, update_batch_stats=False).data  # unchain
    else:
        assert not isinstance(p_logit, Variable)

    xp = cuda.get_array_module(x.data)
    d = xp.random.normal(size=x.shape)
    d = d / xp.sqrt(xp.sum(d ** 2, axis=range(1, len(d.shape)), keepdims=True))
    for ip in range(Ip):
        d_var = Variable(d.astype(xp.float32))
        p_d_logit = forward(x + xi * d_var, train=train, update_batch_stats=False)
        kl_loss = distance(p_logit, p_d_logit)
        kl_loss.backward()
        d = d_var.grad
        d = d / xp.sqrt(xp.sum(d ** 2, axis=range(1, len(d.shape)), keepdims=True))
    d_var = Variable(d.astype(xp.float32))
    p_adv_logit = forward(x + epsilon * d_var, train=train, update_batch_stats=False)
    return distance(p_logit, p_adv_logit)
