import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn.init import _calculate_fan_in_and_fan_out, uniform_, normal_
import math
from torch.nn import functional as F
import numpy as np
import functools


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.bs, self.bs, w // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.bs ** 2), h // self.bs, w // self.bs)
        return x


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.bs, self.bs, c // (self.bs ** 2), h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(n, c // (self.bs ** 2), h * self.bs, w * self.bs)
        return x


class IDAct(nn.Module):
    def forward(self, input):
        return input


class L2NormConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        init=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.Tensor(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.bias = None

        self.beta = nn.Parameter(
            torch.zeros([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.gamma = nn.Parameter(
            torch.ones([1, out_channels, 1, 1], dtype=torch.float32)
        )
        # init
        if callable(init):
            self.init_fn = init
        else:
            self.init_fn = lambda: False
        normal_(self.weight, mean=0.0, std=0.05)
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            uniform_(self.bias, -bound, bound)

    def forward(self, x, it=None):
        W_norm = F.normalize(self.weight, dim=[1, 2, 3], p=2)
        x = F.conv2d(
            x, W_norm, self.bias, stride=self.stride, padding=self.padding
        )
        # training attribute is inherited from nn.Module
        if self.init_fn() and self.training:
            mean = torch.mean(x, dim=[0, 2, 3], keepdim=True)
            var = torch.var(x, dim=[0, 2, 3], keepdim=True)
            self.gamma.data = 1.0 / torch.sqrt(var + 1e-10)
            self.beta.data = -mean * self.gamma

        return self.gamma * x + self.beta


class LayerNormConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):

        x = self.conv(x)
        return self.norm(x)


class NormConv2d(nn.Module):
    """
    Convolutional layer with l2 weight normalization and learned scaling parameters
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        super().__init__()
        self.beta = nn.Parameter(
            torch.zeros([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.gamma = nn.Parameter(
            torch.ones([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.conv = weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            name="weight",
        )

    def forward(self, x):
        # weight normalization
        # self.conv.weight = normalize(self.conv.weight., dim=[0, 2, 3])
        out = self.conv(x)
        out = self.gamma * out + self.beta
        return out


class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None, conv_layer=NormConv2d):
        super().__init__()
        if out_channels == None:
            self.down = conv_layer(
                channels, channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.down = conv_layer(
                channels, out_channels, kernel_size=3, stride=2, padding=1
            )

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(
        self, in_channels, out_channels, subpixel=True, conv_layer=NormConv2d
    ):
        super().__init__()
        if subpixel:
            self.up = conv_layer(in_channels, 4 * out_channels, 3, padding=1)
            self.op2 = DepthToSpace(block_size=2)
        else:
            # channels have to be bisected because of formely concatenated skips connections
            self.up = conv_layer(in_channels, out_channels, 3, padding=1)
            self.op2 = nn.Upsample(scale_factor=2, mode="bilinear")
            # self.up = nn.Upsample(scale_factor=2, mode="bilinear")
            # self.op2 = conv_layer(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        out = self.up(x)
        out = self.op2(out)
        return out


class VunetRNB(nn.Module):
    def __init__(
        self,
        channels,
        a_channels=None,
        residual=False,
        kernel_size=3,
        activate=True,
        conv_layer=NormConv2d,
        act_fn=None,
        dropout_prob=0.
    ):
        super().__init__()
        self.residual = residual
        self.dout = nn.Dropout(p=dropout_prob)

        if self.residual:
            assert a_channels is not None
            self.nin = conv_layer(a_channels, channels, kernel_size=1)

        if activate:
            if act_fn is None:
                self.act_fn = nn.ELU()
            else:
                self.act_fn = act_fn
        else:
            self.act_fn = IDAct()

        in_c = 2 * channels if self.residual else channels
        self.conv = conv_layer(
            in_channels=in_c,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x, a=None):
        residual = x
        if a is not None:
            assert self.residual
            a = self.act_fn(a)
            a = self.nin(a)
            residual = torch.cat([residual, a], dim=1)

        residual = self.act_fn(residual)
        residual = self.dout(residual)
        residual = self.conv(residual)

        return x + residual


class BasicFullyConnectedNet(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False, use_bn=False,
                 out_dim=None):
        super(BasicFullyConnectedNet, self).__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(
            nn.Linear(hidden_dim, dim if out_dim is None else out_dim))
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(
                input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


_norm_options = {
        "in": nn.InstanceNorm2d,
        "bn": nn.BatchNorm2d,
        "an": ActNorm}

class GINActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(
                input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            alpha = torch.prod(std + 1e-6)
            self.scale.data.copy_(alpha / (std + 1e-6))

    def get_scale(self):
        scale = self.scale[:, :-1, :, :]
        lastscale = 1.0 / (torch.prod(scale) + 1e-6)
        lastscale = lastscale * torch.ones(1, 1, 1, 1).to(lastscale)
        scale = torch.cat((scale, lastscale), dim=1)
        return scale

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.get_scale() * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            logdet = torch.zeros(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.get_scale() - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class GIN2ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(
                input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            alpha = torch.prod(std + 1e-6)
            self.scale.data.copy_(alpha / (std + 1e-6))

    def get_scale(self):
        scale = self.scale
        totalscale = torch.prod(scale, dim=1, keepdim=True)
        scale = scale / (
                    totalscale + 1e-6)  # TODO this might be an issue scale -> 0
        return scale

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.get_scale() * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            # TODO better return real logdet?
            logdet = torch.zeros(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.get_scale() - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


# karpathy's made + conditioning


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return nn.functional.linear(input, self.mask * self.weight, self.bias)


class ARFullyConnectedNet(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1,
                 natural_ordering=False, ncond=0):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
        self.ncond = ncond

        # define a simple MLP neural net
        self.net = nn.ModuleList()
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.append(MaskedLinear(h0, h1))

        if self.ncond > 0:
            self.condnet = nn.ModuleList()
            hs = [ncond] + hidden_sizes + [nout]
            for h0, h1 in zip(hs, hs[1:]):
                self.condnet.append(MaskedLinear(h0, h1))

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1: return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(
            self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1,
                                    size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        for l, m in zip(self.net, masks):
            l.set_mask(m)

    def forward(self, x, y=None):
        assert len(x.shape) == 2
        assert x.shape[1] == self.nin
        if self.ncond > 0:
            assert y is not None
            assert len(y.shape) == 2
            assert y.shape[1] == self.ncond
            assert y.shape[0] == x.shape[0]
            for i in range(len(self.net)):
                if i > 0:
                    x = nn.functional.relu(x)
                    y = nn.functional.relu(y)
                y = self.condnet[i](y)
                x = self.net[i](x) + y
            return x
        else:
            assert y is None
            for i in range(len(self.net)):
                if i > 0:
                    x = nn.functional.relu(x)
                x = self.net[i](x)
            return x


class BasicUnConnectedNet(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False,
                 out_dim=None):
        super().__init__()
        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        assert self.out_dim % self.dim == 0
        self.factor = self.out_dim // self.dim

        layers = []
        layers.append(nn.Conv1d(in_channels=1, out_channels=hidden_dim,
                                kernel_size=1))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Conv1d(in_channels=hidden_dim,
                                    out_channels=hidden_dim, kernel_size=1))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Conv1d(in_channels=hidden_dim,
                                out_channels=self.factor, kernel_size=1))
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        assert len(x.shape) == 2
        xs = x.shape
        x = x[:, None, :]  # (bs,1,dim)
        x = self.main(x)  # (bs, out_dim, dim)
        x = x.reshape(-1, self.out_dim)
        return x


class FeatureLayer(nn.Module):
    def __init__(self, scale, in_channels=None, norm='AN', width_multiplier=1):
        super().__init__()
        self.scale = scale
        self.norm = _norm_options[norm.lower()]
        self.wm = width_multiplier
        if in_channels is None:
            self.in_channels = int(self.wm*64*min(2**(self.scale-1), 16))
        else:
            self.in_channels = in_channels
        self.out_channels = int(self.wm*64*min(2**self.scale, 16))
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        Norm = functools.partial(self.norm, affine=True)
        Activate = lambda: nn.LeakyReLU(0.2)
        self.sub_layers = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False),
                Norm(num_features=self.out_channels),
                Activate()])


class DenseEncoderLayer(nn.Module):
    def __init__(self, scale, spatial_size, out_size, in_channels=None,
                 width_multiplier=1):
        super().__init__()
        self.scale = scale
        self.wm = width_multiplier
        self.in_channels = int(self.wm*64*min(2**(self.scale-1), 16))
        if in_channels is not None:
            print('Warning: Ignoring `scale` parameter in DenseEncoderLayer due to given number of input channels.')
            self.in_channels = in_channels
        self.out_channels = out_size
        self.kernel_size = spatial_size
        self.build()

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        self.sub_layers = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=0,
                    bias=True)])