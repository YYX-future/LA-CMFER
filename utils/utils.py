import random
import torch

from collections import Iterable
import numpy as np

def setup_seed(seed=3047):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot

class OptimWithSheduler:
    """
    usage::

        op = optim.SGD(lr=1e-3, params=net.parameters()) # create an optimizer
        scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=100, power=1, max_iter=100) # create a function
        that receives two keyword arguments:step, initial_lr
        opw = OptimWithSheduler(op, scheduler) # create a wrapped optimizer
        with OptimizerManager(opw): # use it as an ordinary optimizer
            loss.backward()
    """
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1


def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    '''
    change as initial_lr * (1 + gamma * min(1.0, iter / max_iter) ) ** (- power)
    as known as inv learning rate sheduler in caffe,
    see https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto

    the default gamma and power come from <Domain-Adversarial Training of Neural Networks>

    code to see how it changes(decays to %20 at %10 * max_iter under default arg)::

        from matplotlib import pyplot as plt

        ys = [inverseDecaySheduler(x, 1e-3) for x in range(10000)]
        xs = [x for x in range(10000)]

        plt.plot(xs, ys)
        plt.show()

    '''
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))


class OptimizerManager:
    """
    automatic call op.zero_grad() when enter, call op.step() when exit
    usage::

        with OptimizerManager(op): # or with OptimizerManager([op1, op2])
            b = net.forward(a)
            b.backward(torch.ones_like(b))

    """
    def __init__(self, optims):
        self.optims = optims if isinstance(optims, Iterable) else [optims]

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None # release reference, to avoid imexplicit reference
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True