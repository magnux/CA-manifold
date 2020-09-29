#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
based on: https://github.com/wavefrontshaping/complexPyTorch
"""
import torch
from functools import partial
from torch.nn.functional import relu, max_pool2d, dropout, dropout2d, conv1d, conv2d, conv3d


def complex_relu(input_r, input_i):
    return relu(input_r), relu(input_i)


def complex_max_pool2d(input_r, input_i, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):
    return max_pool2d(input_r, kernel_size, stride, padding, dilation, ceil_mode, return_indices), \
           max_pool2d(input_i, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


def complex_dropout(input_r, input_i, p=0.5, training=True, inplace=False):
    return dropout(input_r, p, training, inplace), \
           dropout(input_i, p, training, inplace)


def complex_dropout2d(input_r, input_i, p=0.5, training=True, inplace=False):
    return dropout2d(input_r, p, training, inplace), \
           dropout2d(input_i, p, training, inplace)


def _complex_conv(input_r, input_i, weight_r, weight_i, bias_r=None, bias_i=None,
                  stride=1, padding=0, dilation=1, groups=1, conv_func=None):
    return (conv_func(input_r, weight_r, bias_r, stride, padding, dilation, groups) -
            conv_func(input_i, weight_i, bias_i, stride, padding, dilation, groups),
            conv_func(input_i, weight_r, bias_r, stride, padding, dilation, groups) +
            conv_func(input_r, weight_i, bias_i, stride, padding, dilation, groups))


complex_conv1d = partial(_complex_conv, conv_func=conv1d)
complex_conv2d = partial(_complex_conv, conv_func=conv2d)
complex_conv3d = partial(_complex_conv, conv_func=conv3d)


def _complex_conv_sc(input, weight, bias, stride=1, padding=0, dilation=1, groups=1, conv_func=None):
    input_r, input_i = torch.split(input, input.size(1) // 2, dim=1)
    weight_r, weight_i = torch.split(weight, weight.size(1) // 2, dim=1)
    bias_r, bias_i = torch.split(bias, bias.size(0) // 2, dim=1)
    return torch.cat(_complex_conv(input_r, input_i, weight_r, weight_i, bias_r, bias_i,
                                   stride, padding, dilation, groups, conv_func), dim=1)


complex_conv1d_sc = partial(_complex_conv_sc, conv_func=conv1d)
complex_conv2d_sc = partial(_complex_conv_sc, conv_func=conv2d)
complex_conv3d_sc = partial(_complex_conv_sc, conv_func=conv3d)
