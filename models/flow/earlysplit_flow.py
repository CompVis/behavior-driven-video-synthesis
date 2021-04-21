"""Early Splitting (a.k.a Gaussinization) of chunks of input"""

import torch
import torch.nn as nn
import numpy as np

from edflow.util import retrieve

from models.flow.blocks import UnconditionalFlow, ConditionalFlow


class ConditionalFlatSplitFlow(nn.Module):
    """Early Gaussinization; uses ConditionalDoubleVectorCouplingBlock as building blocks.
    Note: If dim is not a power of 2, the first split will be executed such that dim_1 = 2**int(log2(dim))"""
    def __init__(self, n_scale, dim, n_flow_sub, submodule_depth, hidden_dim_mulitplier, embedding_dim=128,
                 conditioning_option='none'):
        super().__init__()
        self.n_scale = n_scale
        self.z_dim = dim
        dims = self.make_dims(n_scale, dim)
        # need to duplicate last entry to allow consistent forward/backward passing
        self.all_dims = dims + [dims[-1]]
        print(' Splitting code at:', self.all_dims)
        diffs = np.abs(np.diff(np.array(dims)))
        assert diffs[-1] > 1
        self.dim_diffs = list(diffs) + [diffs[-1]//2, diffs[-1]//2]
        print(' Backsplitting dims: ', self.dim_diffs)
        self.subflows = nn.ModuleList()
        for dim in dims:
            print('append flow {}->{}'.format(dim, dim))
            self.subflows.append(ConditionalFlow(in_channels=dim, embedding_dim=embedding_dim,
                                                 hidden_dim=dim*hidden_dim_mulitplier,
                                                 hidden_depth=submodule_depth, n_flows=n_flow_sub,
                                                 conditioning_option=conditioning_option))

    def forward(self, input_, condition, reverse=False):
        if not reverse:
            return self.f(input_, condition)
        else:
            return self.g(input_, condition)

    def f(self, x, condition):
        # forward pass
        z_early_out = []
        logdet = 0
        current_dim = self.z_dim
        for i in range(len(self.subflows)):
            x, sublogdet = self.subflows[i](x, condition)
            # self.all_dims holds input dimensions to subflows
            dim_out, dim_in = current_dim - self.all_dims[i+1], self.all_dims[i+1]
            z_early, x = torch.split(x, [dim_out, dim_in], dim=1)
            z_early_out.append(z_early)
            logdet = logdet + sublogdet

            current_dim = dim_in

        z_early_out.append(x)
        return torch.cat(z_early_out, dim=1), logdet

    def g(self, z, condition):
        # inverse pass
        z_split = torch.split(z, self.dim_diffs, dim=1)  # z_split is a tuple
        z_tmp = torch.cat((z_split[-2], z_split[-1]), dim=1)
        z_tmp = self.subflows[-1](z_tmp, condition, reverse=True)
        idx_tmp_z = 3
        for i in reversed(range(len(self.subflows) - 1)):
            z_tmp = torch.cat((z_split[-idx_tmp_z], z_tmp), dim=1)
            z_tmp = self.subflows[i](z_tmp, condition, reverse=True)
            idx_tmp_z += 1
        x = z_tmp
        return x, None

    def make_dims(self, n_factor, z_dim):
        dims = [z_dim, 2**int(np.log2(z_dim))]
        for n in range(1, n_factor):
            dims.append(dims[-1]//2)
        assert dims[-1] > 0
        # self.all_dims holds input dimensions to subflows
        return dims


class UnconditionalFlatSplitFlow(nn.Module):
    """Early Gaussinization; uses DoubleVectorCouplingBlock as building blocks.
    Note: If dim is not a power of 2, the first split will be executed such that dim_1 = 2**int(log2(dim))"""
    def __init__(self, n_scale, dim, n_flow_sub, submodule_depth, hidden_dim_mulitplier):
        super().__init__()
        self.n_scale = n_scale
        self.z_dim = dim
        dims = self.make_dims(n_scale, dim)
        # need to duplicate last entry to allow consistent forward/backward passing
        self.all_dims = dims + [dims[-1]]
        print(' Splitting code at:', self.all_dims)
        diffs = np.abs(np.diff(np.array(dims)))
        assert diffs[-1] > 1
        self.dim_diffs = list(diffs) + [diffs[-1]//2, diffs[-1]//2]
        print(' Backsplitting dims: ', self.dim_diffs)
        self.subflows = nn.ModuleList()
        for dim in dims:
            print('append flow {}->{}'.format(dim, dim))
            self.subflows.append(UnconditionalFlow(in_channels=dim,
                                                   hidden_dim=dim*hidden_dim_mulitplier,
                                                   hidden_depth=submodule_depth, n_flows=n_flow_sub,
                                                   )
                                 )

    def forward(self, input_, reverse=False):
        if not reverse:
            return self.f(input_)
        else:
            return self.g(input_)

    def f(self, x):
        # forward pass
        z_early_out = []
        logdet = 0
        current_dim = self.z_dim
        for i in range(len(self.subflows)):
            x, sublogdet = self.subflows[i](x)
            # self.all_dims holds input dimensions to subflows
            dim_out, dim_in = current_dim - self.all_dims[i+1], self.all_dims[i+1]
            z_early, x = torch.split(x, [dim_out, dim_in], dim=1)
            z_early_out.append(z_early)
            logdet = logdet + sublogdet
            current_dim = dim_in

        z_early_out.append(x)
        return torch.cat(z_early_out, dim=1), logdet

    def g(self, z):
        # inverse pass
        z_split = torch.split(z, self.dim_diffs, dim=1)  # z_split is a tuple
        z_tmp = torch.cat((z_split[-2], z_split[-1]), dim=1)
        z_tmp = self.subflows[-1](z_tmp, reverse=True)
        idx_tmp_z = 3
        for i in reversed(range(len(self.subflows) - 1)):
            z_tmp = torch.cat((z_split[-idx_tmp_z], z_tmp), dim=1)
            z_tmp = self.subflows[i](z_tmp, reverse=True)
            idx_tmp_z += 1
        x = z_tmp
        return x, None

    def make_dims(self, n_factor, z_dim):
        dims = [z_dim, 2**int(np.log2(z_dim))]
        for n in range(1, n_factor):
            dims.append(dims[-1]//2)
        assert dims[-1] > 0
        # self.all_dims holds input dimensions to subflows
        return dims


class SupervisedTransformer(nn.Module):
    """Early Splitting (mimics multiscale architecture in glow for flat inputs)"""
    def __init__(self, config):
        super().__init__()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        self.config = config
        self.n_split = retrieve(config, "Transformer/n_split")
        in_channel = retrieve(config, "Transformer/in_channels")
        n_flow = retrieve(config, "Transformer/n_flow_sub")
        depth_submodules = retrieve(config, "Transformer/hidden_depth")
        hidden_dim_mulitplier = retrieve(config, "Transformer/hidden_dim_multiplier")
        embedding_dim = retrieve(config, "Transformer/embedding_dim")
        self.n_classes = retrieve(config, "Transformer/num_classes")
        conditioning_option = retrieve(config, "Transformer/conditioning_option", default='none')

        self.flow = ConditionalFlatSplitFlow(n_scale=self.n_split, dim=in_channel, n_flow_sub=n_flow,
                                             submodule_depth=depth_submodules,
                                             hidden_dim_mulitplier=hidden_dim_mulitplier,
                                             embedding_dim=embedding_dim,
                                             conditioning_option=conditioning_option)
        self.embedder = nn.Linear(self.n_classes, embedding_dim, bias=False)

    def embed(self, labels):
        # make one-hot from label
        one_hot = torch.nn.functional.one_hot(labels, num_classes=self.n_classes).float()
        # embed it via embedding layer
        embedding = self.embedder(one_hot)
        return embedding

    def forward(self, input, label, **dummykwargs):
        embedding = self.embed(label)
        out, logdet = self.flow(input, embedding)
        return out, logdet  # logdet must be of shape bs x 1 to be usable in loss_01.py

    def reverse(self, out, label):
        embedding = self.embed(label)
        return self.flow(out, embedding, reverse=True)[0]

    def get_last_layer(self):
        return getattr(self.flow.subflows[-1].sub_layers[-1].coupling.t[-1].main[-1], 'weight')


class UnsupervisedTransformer(nn.Module):
    """Early Splitting (mimics multiscale architecture in glow for flat inputs)"""
    def __init__(self, config):
        super().__init__()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        self.config = config
        self.n_split = retrieve(config, "Transformer/n_split")
        in_channel = retrieve(config, "Transformer/in_channels")
        n_flow = retrieve(config, "Transformer/n_flow_sub")
        depth_submodules = retrieve(config, "Transformer/hidden_depth")
        hidden_dim_mulitplier = retrieve(config, "Transformer/hidden_dim_multiplier")

        self.flow = ConditionalFlatSplitFlow(n_scale=self.n_split, dim=in_channel, n_flow_sub=n_flow,
                                             submodule_depth=depth_submodules,
                                             hidden_dim_mulitplier=hidden_dim_mulitplier,
                                             )

    def forward(self, input):
        out, logdet = self.flow(input)
        return out, logdet  # logdet must be of shape bs x 1 to be usable in loss_01.py

    def reverse(self, out):
        return self.flow(out, reverse=True)[0]

    def get_last_layer(self):
        return getattr(self.flow.subflows[-1].sub_layers[-1].coupling.t[-1].main[-1], 'weight')
