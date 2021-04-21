import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.modules import BasicFullyConnectedNet, ActNorm, GINActNorm, GIN2ActNorm


class ConditionalFlow(nn.Module):
    """Flat version. Feeds an embedding into the flow in every block"""
    def __init__(self, in_channels, embedding_dim, hidden_dim, hidden_depth,
                 n_flows, conditioning_option="none", activation='lrelu'):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = embedding_dim
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.conditioning_option = conditioning_option

        self.sub_layers = nn.ModuleList()
        if self.conditioning_option.lower() != "none":
            self.conditioning_layers = nn.ModuleList()
        for flow in range(self.n_flows):
            self.sub_layers.append(ConditionalFlatDoubleCouplingFlowBlock(
                                   self.in_channels, self.cond_channels, self.mid_channels,
                                   self.num_blocks, activation=activation)
                                   )
            if self.conditioning_option.lower() != "none":
                self.conditioning_layers.append(nn.Conv2d(self.cond_channels, self.cond_channels, 1))

    def forward(self, x, embedding, reverse=False):
        hconds = list()
        hcond = embedding[:,:,None,None]
        self.last_outs = []
        self.last_logdets = []
        for i in range(self.n_flows):
            if self.conditioning_option.lower() == "parallel":
                hcond = self.conditioning_layers[i](embedding)
            elif self.conditioning_option.lower() == "sequential":
                hcond = self.conditioning_layers[i](hcond)
            hconds.append(hcond)
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x, hconds[i])
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, hconds[i], reverse=True)
            return x

    def reverse(self, out, xcond):
        return self(out, xcond, reverse=True)


class UnconditionalFlow(nn.Module):
    """Flat"""
    def __init__(self, in_channels, hidden_dim, hidden_depth, n_flows, activation='lrelu'):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):
            self.sub_layers.append(UnconditionalFlatDoubleCouplingFlowBlock(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks, activation=activation)
                                   )

    def forward(self, x, reverse=False):
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
            return x

    def reverse(self, out):
        return self(out, reverse=True)


class UnconditionalFlow2(nn.Module):
    """Flat"""
    def __init__(self, in_channels, hidden_dim, hidden_depth, n_flows):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):
            self.sub_layers.append(UnconditionalFlatDoubleCouplingFlowBlock2(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks)
                                   )

    def forward(self, x, reverse=False):
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
            return x

    def reverse(self, out):
        return self(out, reverse=True)


class UnconditionalGINFlow(nn.Module):
    """Flat"""
    def __init__(self, in_channels, hidden_dim, hidden_depth, n_flows):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):
            self.sub_layers.append(UnconditionalFlatDoubleCouplingGINFlowBlock(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks)
                                   )

    def forward(self, x, reverse=False):
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
            return x

    def reverse(self, out):
        return self(out, reverse=True)


class UnconditionalGIN2Flow(nn.Module):
    """Flat"""
    def __init__(self, in_channels, hidden_dim, hidden_depth, n_flows):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):
            self.sub_layers.append(UnconditionalFlatDoubleCouplingGIN2FlowBlock(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks)
                                   )

    def forward(self, x, reverse=False):
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
            return x

    def reverse(self, out):
        return self(out, reverse=True)


class UnconditionalNICEFlow(nn.Module):
    """Flat"""
    def __init__(self, in_channels, hidden_dim, hidden_depth, n_flows):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):
            self.sub_layers.append(UnconditionalFlatDoubleCouplingNICEFlowBlock(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks)
                                   )

    def forward(self, x, reverse=False):
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
            return x

    def reverse(self, out):
        return self(out, reverse=True)


class DoubleVectorCouplingBlock(nn.Module):
    """In contrast to VectorCouplingBlock, this module assures alternating chunking in upper and lower half."""
    def __init__(self, in_channels, hidden_dim, hidden_depth=2):
        super(DoubleVectorCouplingBlock, self).__init__()
        assert in_channels % 2 == 0
        self.s = nn.ModuleList([BasicFullyConnectedNet(dim=in_channels // 2, depth=hidden_depth, hidden_dim=hidden_dim,
                                                       use_tanh=True) for _ in range(2)])
        self.t = nn.ModuleList([BasicFullyConnectedNet(dim=in_channels // 2, depth=hidden_depth, hidden_dim=hidden_dim,
                                                       use_tanh=False) for _ in range(2)])

    def forward(self, x, reverse=False):
        assert len(x.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                x_ = x[idx_keep] * (scale.exp()) + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale.view(x.size(0), -1), dim=1)
                logdet = logdet + logdet_
            return x[:,:,None,None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply])) * (self.s[i](x[idx_apply]).neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]


class DoubleVectorCouplingBlock2(nn.Module):
    """Support uneven inputs"""
    def __init__(self, in_channels, hidden_dim, hidden_depth=2):
        super().__init__()
        dim1 = (in_channels // 2) + (in_channels % 2)
        dim2 = in_channels // 2
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=True),
        ])
        self.t = nn.ModuleList([
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False),
            BasicFullyConnectedNet(dim=dim1, out_dim=dim2, depth=hidden_depth,
                                   hidden_dim=hidden_dim, use_tanh=False),
        ])

    def forward(self, x, reverse=False):
        assert len(x.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                x_ = x[idx_keep] * (scale.exp()) + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale.view(x.size(0), -1), dim=1)
                logdet = logdet + logdet_
            return x[:,:,None,None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply])) * (self.s[i](x[idx_apply]).neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]


class GINDoubleVectorCouplingBlock(nn.Module):
    """Volume preserving version. In contrast to VectorCouplingBlock, this module assures alternating chunking in upper and lower half."""
    def __init__(self, in_channels, hidden_dim, hidden_depth=2):
        super().__init__()
        assert in_channels % 2 == 0
        self.s = nn.ModuleList([BasicFullyConnectedNet(
            dim=in_channels // 2, depth=hidden_depth, hidden_dim=hidden_dim,
            out_dim=in_channels // 2 - 1,
            use_tanh=True) for _ in range(2)])
        self.t = nn.ModuleList([BasicFullyConnectedNet(
            dim=in_channels // 2, depth=hidden_depth, hidden_dim=hidden_dim,
            use_tanh=False) for _ in range(2)])

    def get_scale(self, scale):
        assert len(scale.shape) == 2
        if scale.shape[1] == 0:
            return torch.zeros(scale.shape[0],1)
        lastscale = -torch.sum(scale, dim=1, keepdim=True)
        scale = torch.cat((scale, lastscale), dim=1)
        return scale

    def forward(self, x, reverse=False):
        assert len(x.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        if not reverse:
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                scale = self.get_scale(scale)
                x_ = x[idx_keep] * (scale.exp()) + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
            logdet = torch.zeros(x.shape[0]).to(x)
            return x[:,:,None,None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                scale = self.get_scale(scale)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply])) * (scale.neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]


class GIN2DoubleVectorCouplingBlock(nn.Module):
    """Volume preserving version but uniform scaling. In contrast to VectorCouplingBlock, this module assures alternating chunking in upper and lower half."""
    def __init__(self, in_channels, hidden_dim, hidden_depth=2):
        super().__init__()
        assert in_channels % 2 == 0
        self.s = nn.ModuleList([BasicFullyConnectedNet(
            dim=in_channels // 2, depth=hidden_depth, hidden_dim=hidden_dim,
            out_dim=in_channels // 2,
            use_tanh=True) for _ in range(2)])
        self.t = nn.ModuleList([BasicFullyConnectedNet(
            dim=in_channels // 2, depth=hidden_depth, hidden_dim=hidden_dim,
            use_tanh=False) for _ in range(2)])

    def get_scale(self, scale):
        assert len(scale.shape) == 2
        totalscale = torch.sum(scale, dim=1, keepdim=True)
        scale = scale - totalscale
        return scale

    def forward(self, x, reverse=False):
        assert len(x.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        if not reverse:
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                scale = self.get_scale(scale)
                x_ = x[idx_keep] * (scale.exp()) + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
            logdet = torch.zeros(x.shape[0]).to(x)
            # TODO better return real logdet?
            return x[:,:,None,None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                scale = self.get_scale(scale)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply])) * (scale.neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]


class NICEDoubleVectorCouplingBlock(nn.Module):
    """Volume preserving version by simply not scaling. In contrast to VectorCouplingBlock, this module assures alternating chunking in upper and lower half."""
    def __init__(self, in_channels, hidden_dim, hidden_depth=2):
        super().__init__()
        assert in_channels % 2 == 0
        self.t = nn.ModuleList([BasicFullyConnectedNet(
            dim=in_channels // 2, depth=hidden_depth, hidden_dim=hidden_dim,
            use_tanh=False) for _ in range(2)])

    def forward(self, x, reverse=False):
        assert len(x.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        if not reverse:
            for i in range(len(self.t)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                x_ = x[idx_keep] + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
            logdet = torch.zeros(x.shape[0]).to(x)
            return x[:,:,None,None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.t))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply]))
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]


class ConditionalDoubleVectorCouplingBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, hidden_dim, depth=2):
        super(ConditionalDoubleVectorCouplingBlock, self).__init__()
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels//2+cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=True,
                                   out_dim=in_channels//2) for _ in range(2)])
        self.t = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels//2+cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=False,
                                   out_dim=in_channels//2) for _ in range(2)])

    def forward(self, x, xc, reverse=False):
        assert len(x.shape) == 4
        assert len(xc.shape) == 4
        x = x.squeeze(-1).squeeze(-1)
        xc = xc.squeeze(-1).squeeze(-1)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1)
                scale = self.s[i](conditioner_input)
                x_ = x[idx_keep] * scale.exp() + self.t[i](conditioner_input)
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale, dim=1)
                logdet = logdet + logdet_
            return x[:,:,None,None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1)
                x_ = (x[idx_keep] - self.t[i](conditioner_input)) * self.s[i](conditioner_input).neg().exp()
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:,:,None,None]


class UnconditionalFlatDoubleCouplingFlowBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, hidden_depth, activation="lrelu"):
        super().__init__()
        __possible_activations = {"lrelu": lambda: InvLeakyRelu(alpha=0.95), "none":IgnoreLeakyRelu}
        self.norm_layer = ActNorm(in_channels, logdet=True)
        self.coupling = DoubleVectorCouplingBlock(in_channels,
                                                  hidden_dim,
                                                  hidden_depth)
        self.activation = __possible_activations[activation]()
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.activation(h)
            logdet += ld
            h, ld = self.coupling(h)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, reverse=True)
            h = self.activation(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out):
        return self.forward(out, reverse=True)


class UnconditionalFlatDoubleCouplingFlowBlock2(nn.Module):
    def __init__(self, in_channels, hidden_dim, hidden_depth):
        super().__init__()
        self.norm_layer = ActNorm(in_channels, logdet=True)
        self.coupling = DoubleVectorCouplingBlock2(in_channels,
                                                   hidden_dim,
                                                   hidden_depth)
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.coupling(h)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out):
        return self.forward(out, reverse=True)


class UnconditionalFlatDoubleCouplingGINFlowBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, hidden_depth):
        super().__init__()
        self.norm_layer = GINActNorm(in_channels, logdet=True)
        self.coupling = GINDoubleVectorCouplingBlock(in_channels,
                                                     hidden_dim,
                                                     hidden_depth)
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.coupling(h)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out):
        return self.forward(out, reverse=True)


class UnconditionalFlatDoubleCouplingGIN2FlowBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, hidden_depth):
        super().__init__()
        self.norm_layer = GIN2ActNorm(in_channels, logdet=True)
        self.coupling = GIN2DoubleVectorCouplingBlock(in_channels,
                                                     hidden_dim,
                                                     hidden_depth)
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.coupling(h)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out):
        return self.forward(out, reverse=True)


class UnconditionalFlatDoubleCouplingNICEFlowBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, hidden_depth):
        super().__init__()
        #self.norm_layer = GINActNorm(in_channels, logdet=True)
        self.coupling = NICEDoubleVectorCouplingBlock(in_channels,
                                                      hidden_dim,
                                                      hidden_depth)
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            #h, ld = self.norm_layer(h)
            #logdet += ld
            h, ld = self.coupling(h)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, reverse=True)
            #h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out):
        return self.forward(out, reverse=True)


class ConditionalFlatDoubleCouplingFlowBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, hidden_dim, hidden_depth, activation="lrelu"):
        super().__init__()
        __possible_activations = {"lrelu": InvLeakyRelu, "none":IgnoreLeakyRelu}
        self.norm_layer = ActNorm(in_channels, logdet=True)
        self.coupling = ConditionalDoubleVectorCouplingBlock(in_channels,
                                                             cond_channels,
                                                             hidden_dim,
                                                             hidden_depth)
        self.activation = __possible_activations[activation]()
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, xcond, reverse=False):
        if not reverse:
            h = x
            logdet = 0.0
            h, ld = self.norm_layer(h)
            logdet += ld
            h, ld = self.activation(h)
            logdet += ld
            h, ld = self.coupling(h, xcond)
            logdet += ld
            h, ld = self.shuffle(h)
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, xcond, reverse=True)
            h = self.activation(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out, xcond):
        return self.forward(out, xcond, reverse=True)


class Shuffle(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Shuffle, self).__init__()
        self.in_channels = in_channels
        idx = torch.randperm(in_channels)
        self.register_buffer('forward_shuffle_idx', nn.Parameter(idx, requires_grad=False))
        self.register_buffer('backward_shuffle_idx', nn.Parameter(torch.argsort(idx), requires_grad=False))

    def forward(self, x, reverse=False, conditioning=None):
        if not reverse:
            return x[:, self.forward_shuffle_idx, ...], 0
        else:
            return x[:, self.backward_shuffle_idx, ...]


class OrthogonalPermute(nn.Module):
    """For some orthogonal matrix O: O^(-1) = O^T"""
    def __init__(self, in_channels, **kwargs):
        super(OrthogonalPermute, self).__init__()
        print('WARNING: OrthogonalPermute induces invertibility issues!?')
        self.in_channels = in_channels
        omatrix = torch.empty(in_channels, in_channels)
        nn.init.orthogonal_(omatrix)
        self.register_buffer('forward_orthogonal', omatrix)
        self.register_buffer('backward_orthogonal', omatrix.t())

    def forward(self, x, reverse=False):
        twodim = False
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)
            twodim = True
        if not reverse:
            if not twodim:
                return F.conv2d(x, self.forward_orthogonal.unsqueeze(2).unsqueeze(3)), 0
            return F.conv2d(x, self.forward_orthogonal.unsqueeze(2).unsqueeze(3)).squeeze(), 0
        else:
            if not twodim:
                return F.conv2d(x, self.backward_orthogonal.unsqueeze(2).unsqueeze(3))
            return F.conv2d(x, self.backward_orthogonal.unsqueeze(2).unsqueeze(3)).squeeze()


class IgnoreLeakyRelu(nn.Module):
    """performs identity op."""
    def __init__(self, alpha=0.9):
        super().__init__()

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        h = input
        return h, 0.0

    def reverse(self, input):
        h = input
        return h


class InvLeakyRelu(nn.Module):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        scaling = (input >= 0).to(input) + (input < 0).to(input)*self.alpha
        h = input*scaling
        return h, 0.0

    def reverse(self, input):
        scaling = (input >= 0).to(input) + (input < 0).to(input)*self.alpha
        h = input/scaling
        return h


class InvParametricRelu(InvLeakyRelu):
    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
