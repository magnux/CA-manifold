import torch.nn as nn


class SequentialCond(nn.Sequential):

    def forward(self, input, lat):
        for module in self:
            input = module(input, lat)
        return input
