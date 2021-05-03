import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.linearresidualblock import LinearResidualBlock


class DynaResidualBlock(nn.Module):
    def __init__(self, lat_size, fin, fout, fhidden=None, dim=2, kernel_size=1, stride=1, padding=0, norm_weights=False, weights_noise_scale=0.):
        super(DynaResidualBlock, self).__init__()

        self.lat_size = lat_size if lat_size > 3 else 512
        self.fin = fin
        self.fout = fout
        self.fhidden = max((fin + fout), 1) if fhidden is None else fhidden
        self.dim = dim

        if dim == 1:
            self.f_conv = F.conv1d
        elif dim == 2:
            self.f_conv = F.conv2d
        elif dim == 3:
            self.f_conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.k_in_size = self.fhidden * self.fin * (kernel_size ** dim)
        self.k_mid_size = self.fhidden * self.fhidden * (kernel_size ** dim)
        self.k_out_size = self.fout * self.fhidden * (kernel_size ** dim)
        self.k_short_size = self.fout * self.fin * (kernel_size ** dim)
        
        self.b_in_size = self.fhidden if not norm_weights else 0
        self.b_mid_size = self.fhidden if not norm_weights else 0
        self.b_out_size = self.fout if not norm_weights else 0
        self.b_short_size = self.fout if not norm_weights else 0

        k_total_size = (self.k_in_size + self.k_mid_size + self.k_out_size + self.k_short_size +
                        self.b_in_size + self.b_mid_size + self.b_out_size + self.b_short_size)

        self.dyna_k = nn.Sequential(
            nn.Linear(lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, k_total_size, self.lat_size * 2),
        )

        self.prev_lat = None
        self.k_in, self.k_mid, self.k_out, self.k_short = None, None, None, None
        self.b_in, self.b_mid, self.b_out, self.b_short = 0, 0, 0, 0
        self.kernel_size = [kernel_size for _ in range(self.dim)]
        self.stride = stride
        self.padding = padding
        self.norm_weights = norm_weights
        self.weights_noise_scale = weights_noise_scale
        self.linear_fa = LinearFA(k_total_size, k_total_size)

    def forward(self, x, lat):
        batch_size = x.size(0)
        
        if self.prev_lat is None or self.prev_lat.data_ptr() != lat.data_ptr():
            ks = self.dyna_k(lat)
            self.linear_fa.weights_noise_scale = self.weights_noise_scale
            ks = self.linear_fa(ks)
            k_in, k_mid, k_out, k_short, b_in, b_mid, b_out, b_short = torch.split(ks, [self.k_in_size, self.k_mid_size,
                                                                                        self.k_out_size, self.k_short_size,
                                                                                        self.b_in_size, self.b_mid_size,
                                                                                        self.b_out_size, self.b_short_size], dim=1)
            self.k_in = k_in.view([batch_size, self.fhidden, self.fin] + self.kernel_size)
            self.k_mid = k_mid.view([batch_size, self.fhidden, self.fhidden] + self.kernel_size)
            self.k_out = k_out.view([batch_size, self.fout, self.fhidden] + self.kernel_size)
            self.k_short = k_short.view([batch_size, self.fout, self.fin] + self.kernel_size)

            if self.norm_weights:
                self.k_in = self.k_in * torch.rsqrt((self.k_in ** 2).sum(dim=[i for i in range(2, self.dim + 3)], keepdim=True) + 1e-8)
                self.k_mid = self.k_mid * torch.rsqrt((self.k_mid ** 2).sum(dim=[i for i in range(2, self.dim + 3)], keepdim=True) + 1e-8)
                self.k_out = self.k_out * torch.rsqrt((self.k_out ** 2).sum(dim=[i for i in range(2, self.dim + 3)], keepdim=True) + 1e-8)
                self.k_short = self.k_short * torch.rsqrt((self.k_short ** 2).sum(dim=[i for i in range(2, self.dim + 3)], keepdim=True) + 1e-8)

            self.k_in = self.k_in.reshape([batch_size * self.fhidden, self.fin] + self.kernel_size)
            self.k_mid = self.k_mid.reshape([batch_size * self.fhidden, self.fhidden] + self.kernel_size)
            self.k_out = self.k_out.reshape([batch_size * self.fout, self.fhidden] + self.kernel_size)
            self.k_short = self.k_short.reshape([batch_size * self.fout, self.fin] + self.kernel_size)

            if not self.norm_weights:
                self.b_in = b_in.view([batch_size, self.fhidden]).reshape([1, batch_size * self.fhidden] + [1 for _ in range(self.dim)])
                self.b_mid = b_mid.view([batch_size, self.fhidden]).reshape([1, batch_size * self.fhidden] + [1 for _ in range(self.dim)])
                self.b_out = b_out.view([batch_size, self.fout]).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])
                self.b_short = b_short.view([batch_size, self.fout]).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])
            
            self.prev_lat = lat

        x_new = x.reshape([1, batch_size * self.fin] + [x.size(d + 2) for d in range(self.dim)])
        x_new_s = self.f_conv(x_new, self.k_short, stride=self.stride, padding=self.padding, groups=batch_size) + self.b_short
        x_new = self.f_conv(x_new, self.k_in, stride=1, padding=self.padding, groups=batch_size) + self.b_in
        x_new = F.relu(x_new, True)
        x_new = self.f_conv(x_new, self.k_mid, stride=1, padding=self.padding, groups=batch_size) + self.b_mid
        x_new = F.relu(x_new, True)
        x_new = self.f_conv(x_new, self.k_out, stride=self.stride, padding=self.padding, groups=batch_size) + self.b_out
        x_new = x_new + x_new_s
        x_new = x_new.reshape([batch_size, self.fout] + [x.size(d + 2) for d in range(self.dim)])
        
        return x_new


def linear_fa_backward_hook(module, grad_input, grad_output):
    if grad_input[1] is not None:
        grad_input_fa = ((1. - module.weights_noise_scale) * grad_output[0].mm(module.weight) +
                         module.weights_noise_scale * grad_output[0].mm(module.weight_fa))
    else:
        # No layer below, thus no gradient w.r.t. input
        grad_input_fa = None

    if len(grad_input) == 3:
        return (grad_input[0], grad_input_fa) + grad_input[2:]
    else:
        # No gradient w.r.t. bias
        return (grad_input_fa,) + grad_input[1:]


class LinearFA(nn.Module):
    """
    Implementation of a linear module which uses random feedback weights
    in its backward pass, as described in Lillicrap et al., 2016:
    https://www.nature.com/articles/ncomms13276
    """

    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, weights_noise_scale=0.):
        super(LinearFA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_fa', torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.weights_noise_scale = weights_noise_scale

        self.register_backward_hook(linear_fa_backward_hook)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.weight_fa, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
