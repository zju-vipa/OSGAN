import random

import numpy as np

import torch
import torch.nn.functional as F


def negate_grads_hook(module, grad_input, grad_output):
    if isinstance(grad_input, tuple):
        result = []
        for item in grad_input:
            if isinstance(item, torch.Tensor):
                result.append(torch.neg(item))
            else:
                result.append(item)
        return tuple(result)
    if isinstance(grad_input, torch.Tensor):
        return torch.neg(grad_input)


class GradientHook(object):
    def __init__(self, module, is_negate=True):
        self.module = module
        self.is_negate = is_negate
        self.handle = None

    def set_negate(self, is_negate):
        self.is_negate = is_negate

    def negate_grads_func(self, module, grad_input, grad_output):
        # if is_negate is false, not negate grads
        if not self.is_negate:
            return

        if isinstance(grad_input, tuple):
            result = []
            for item in grad_input:
                if isinstance(item, torch.Tensor):
                    result.append(torch.neg(item))
                else:
                    result.append(item)
            return tuple(result)
        if isinstance(grad_input, torch.Tensor):
            return torch.neg(grad_input)

    def set_negate_grads_hook(self):
        self.handle = self.module. \
            register_backward_hook(self.negate_grads_func)

    def __del__(self):
        if self.handle:
            self.handle.remove()


class FeatureHook(object):
    def __init__(self, net, names, in_or_out='output'):
        self.net = net
        self.names = names
        self.in_or_out = in_or_out
        self.features = list()
        self.handles = list()
        self.add_hooks()

    def hook_func_input(self, module, input, ouput):
        self.features.append(input[0])

    def hook_func_output(self, module, input, ouput):
        self.features.append(ouput)

    def add_hooks(self):
        for name in self.names:
            module_cur = self.net
            if '.' in name:
                names_split = name.split('.')
                for nm in names_split:
                    module_cur = module_cur._modules[nm]
            else:
                module_cur = module_cur._modules[name]

            if self.in_or_out == 'input':
                self.handles.append(
                    module_cur.register_forward_hook(self.hook_func_input))
            elif self.in_or_out == 'output':
                self.handles.append(
                    module_cur.register_forward_hook(self.hook_func_output))
            else:
                ValueError('Invalid argument: {}'.format(self.in_or_out))

    def get_feat(self):
        return self.features

    def clear_feat(self):
        self.features = list()

    def __del__(self):
        for handle in self.handles:
            handle.remove()
        del self.features


def get_gradient_ratios(lossA, lossB, x_f, eps=1e-6):
    grad_lossA_xf = torch.autograd.grad(torch.sum(lossA), x_f, retain_graph=True)[0]
    grad_lossB_xf = torch.autograd.grad(torch.sum(lossB), x_f, retain_graph=True)[0]
    gamma = grad_lossA_xf / grad_lossB_xf

    return gamma

