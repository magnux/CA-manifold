import torch
import torch.nn as nn
import math


class ResidualBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, kernel_size=3, stride=1, padding=1, groups=1, conv_fn=nn.Conv2d, bias=True, act_layer=nn.ReLU):
        super(ResidualBlock, self).__init__()

        self.fin = fin
        self.fout = fout
        self.fhidden = max((fin + fout), 1) if fhidden is None else fhidden

        self.block = nn.Sequential(
            conv_fn(self.fin, self.fhidden, kernel_size, stride=1, padding=padding, groups=groups, bias=bias),
            act_layer(True),
            conv_fn(self.fhidden, self.fhidden, kernel_size, stride=1, padding=padding, groups=groups, bias=bias),
            act_layer(True),
            conv_fn(self.fhidden, self.fout, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        )
        self.shortcut = conv_fn(self.fin, self.fout, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)
