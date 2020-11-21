import torch
from torch.nn import Conv2d
from torch.nn import Module

from networks.complexFunctions import complex_relu, complex_max_pool2d


class ComplexMaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, input):
        input_r, input_i = input[:, 0, :, :, :], input[:, 1, :, :, :]

        output_r, output_i = complex_max_pool2d(input_r, input_i, kernel_size=self.kernel_size,
                                                stride=self.stride, padding=self.padding,
                                                dilation=self.dilation, ceil_mode=self.ceil_mode,
                                                return_indices=self.return_indices)
        output_r = output_r.unsqueeze(1)
        output_i = output_i.unsqueeze(1)
        return torch.cat((output_r, output_i), 1)


class ComplexReLU(Module):
    def forward(self, input_r, input_i):
        return complex_relu(input_r, input_i)


class ComplexConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        input_r, input_i = input[:, 0, :, :, :], input[:, 1, :, :, :]

        output_r = self.conv_r(input_r) - self.conv_i(input_i)
        output_r = output_r.unsqueeze(1)

        output_i = self.conv_r(input_i) + self.conv_i(input_r)
        output_i = output_i.unsqueeze(1)

        return torch.cat((output_r, output_i), 1)
