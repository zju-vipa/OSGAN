import torch
# import torch.nn as nn

class GradientScaler(torch.autograd.Function):
    factor = 1.0
    
    @staticmethod
    def forward(ctx, input):
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        factor = GradientScaler.factor
        return factor.view(-1, 1, 1, 1)*grad_output
        # return torch.neg(grad_output)
