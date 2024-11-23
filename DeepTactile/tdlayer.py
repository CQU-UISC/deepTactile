import torch
import torch.nn as nn
from to_graph import *
from to_STMnistgraph import *
from torch_geometric.nn import TAGConv

# Hyperparameters
thresh = 0.5  # Neuronal threshold
lens = 0.5    # Approximation function parameter
decay = 0.2   # Leakage rate

class ActFun(torch.autograd.Function):
    """
    Custom activation function for spiking neurons.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()


act_fun = ActFun.apply


def mem_update(mem, spike, x):
    """
    Membrane potential update for spiking neurons.

    Args:
        mem: Current membrane potential
        spike: Current spike status
        x: Input stimulus

    Returns:
        Updated membrane potential and spike
    """
    mem = mem * decay * (1 - spike) + x
    spike = act_fun(mem)
    return mem, spike


def mem_update_conv(ops, x, edge_idxs, mem, spike):
    """
    Membrane potential update for graph convolutional layers.

    Args:
        ops: Graph convolutional layer
        x: Node features
        edge_idxs: Edge index for graph
        mem: Current membrane potential
        spike: Current spike status

    Returns:
        Updated membrane potential and spike
    """
    mem = mem * decay * (1 - spike) + ops(x, edge_idxs)
    spike = act_fun(mem)
    return mem, spike


def mem_update_FC(ops, x, mem, spike):
    """
    Membrane potential update for fully connected layers.

    Args:
        ops: Fully connected layer
        x: Input features
        mem: Current membrane potential
        spike: Current spike status

    Returns:
        Updated membrane potential and spike
    """
    mem = mem * decay * (1 - spike) + ops(x)
    spike = act_fun(mem)
    return mem, spike


class TF_Layer(nn.Module):
    """
    Data transformation layer for spiking graph convolution.

    Args:
        layer: Graph convolutional layer
        device: Device to process data
    """
    def __init__(self, layer, device='cuda:0'):
        super(TF_Layer, self).__init__()
        self.layer = layer
        self.device = device

    def forward(self, x, edge_index):
        times_window = x.size(-1)
        x = x.split(1, -1)
        x_ = x[0].squeeze(-1).to(self.device)
        output = torch.zeros(self.layer(x_, edge_index).shape + (times_window,), device=self.device)
        for step in range(times_window):
            x_in = x[step].squeeze(-1).to(self.device)
            output[..., step] = self.layer(x_in, edge_index)
        return output


class TF_Layer_FC(nn.Module):
    """
    Data transformation layer for spiking fully connected layers.

    Args:
        layer: Fully connected layer
        device: Device to process data
    """
    def __init__(self, layer, device='cuda:0'):
        super(TF_Layer_FC, self).__init__()
        self.layer = layer
        self.device = device

    def forward(self, x):
        times_window = x.size(-1)
        x = x.split(1, -1)
        x_ = x[0].squeeze(-1)
        x_ = torch.flatten(x_, 1)
        outputs = torch.zeros(self.layer(x_).shape + (times_window,), device=self.device)
        for step in range(times_window):
            x_in = x[step].squeeze(-1).to(self.device)
            x_in = torch.flatten(x_in, 1)
            outputs[..., step] = self.layer(x_in)
        return outputs


class LIFSpike(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) spiking neuron model.

    This module simulates a spiking neuron with temporal dynamics.
    """
    def __init__(self):
        super(LIFSpike, self).__init__()

    def forward(self, x):
        times_window = x.size(-1)
        u = torch.zeros(x.shape[:-1], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(times_window):
            u, out[..., step] = mem_update(u, out[..., max(step - 1, 0)], x[..., step])
        return out


class EBNorm(nn.BatchNorm2d):
    """
    Event-Based Batch Normalization (tdBN).

    Normalizes input across spatial and temporal dimensions.
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(EBNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            mean = input.mean([0, 1, 3])
            var = input.var([0, 1, 3], unbiased=False)
            n = input.numel() / input.size(2)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = thresh * (input - mean[None, None, :, None]) / (torch.sqrt(var[None, None, :, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, None, :, None] + self.bias[None, None, :, None]

        return input
