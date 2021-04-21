"""Vanilla Flow consisting of ActNorm, DoubleVectorCouplingBlock and Shuffle"""
import torch
import torch.nn as nn
from models.flow.blocks import (
    ConditionalFlow,
    UnconditionalFlow,
    UnconditionalFlow2,
    UnconditionalGINFlow,
    UnconditionalGIN2Flow,
    UnconditionalNICEFlow,
)



class SupervisedTransformer(nn.Module):
    """Vanilla version. No multiscale support."""

    def __init__(self, **kwargs):
        super().__init__()
        # self.config = config
        in_channels = kwargs["flow_in_channels"]
        mid_channels = kwargs["flow_mid_channels"]
        hidden_depth = kwargs["flow_hidden_depth"]
        n_flows = kwargs["n_flows"]
        conditioning_option = kwargs["flow_conditioning_option"]
        embedding_channels = (
            kwargs["flow_embedding_channels"]
            if "flow_embedding_channels" in kwargs
            else kwargs["flow_in_channels"]
        )
        self.num_class_channels = (
            kwargs["flow_num_classes"]
            if "flow_num_classes"
            else kwargs["flow_in_channels"]
        )

        self.emb_channels = embedding_channels
        self.in_channels = in_channels

        self.flow = ConditionalFlow(
            in_channels=in_channels,
            embedding_dim=self.emb_channels,
            hidden_dim=mid_channels,
            hidden_depth=hidden_depth,
            n_flows=n_flows,
            conditioning_option=conditioning_option,
        )

        self.embedder = nn.Linear(
            self.num_class_channels, self.emb_channels, bias=False
        )

    def embed(self, labels, labels_are_one_hot=False):
        # make one-hot from label
        if not labels_are_one_hot:
            one_hot = torch.nn.functional.one_hot(
                labels, num_classes=self.num_class_channels
            ).float()
        else:
            one_hot = labels
        # embed it via embedding layer
        embedding = self.embedder(one_hot)
        return embedding

    def sample(self, shape, label):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde, label)
        return sample

    def forward(
        self, input, label, reverse=False, train=False, labels_are_one_hot=False
    ):
        if reverse:
            return self.reverse(input, label)
        embedding = self.embed(label, labels_are_one_hot=labels_are_one_hot)
        out, logdet = self.flow(input, embedding)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out, label, labels_are_one_hot=False):
        embedding = self.embed(label, labels_are_one_hot=labels_are_one_hot)
        return self.flow(out, embedding, reverse=True)

    def get_last_layer(self):
        return getattr(
            self.flow.sub_layers[-1].coupling.t[-1].main[-1], "weight"
        )


class UnsupervisedTransformer(nn.Module):
    """Vanilla version. No multiscale support."""

    def __init__(self, **kwargs):
        super().__init__()
        # self.config = config

        in_channels = kwargs["flow_in_channels"]
        mid_channels = kwargs["flow_mid_channels"]
        hidden_depth = kwargs["flow_hidden_depth"]
        n_flows = kwargs["n_flows"]

        self.in_channels = in_channels

        self.flow = UnconditionalFlow(
            in_channels=in_channels,
            hidden_dim=mid_channels,
            hidden_depth=hidden_depth,
            n_flows=n_flows,
        )

    def sample(self, shape):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde)
        return sample

    def forward(self, input, reverse=False, train=False):
        if reverse:
            return self.reverse(input)
        out, logdet = self.flow(input)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out):
        return self.flow(out, reverse=True)

    def get_last_layer(self):
        return getattr(
            self.flow.sub_layers[-1].coupling.t[-1].main[-1], "weight"
        )


class UnsupervisedTransformer2(nn.Module):
    """To support uneven dims and get rid of leaky relu thing"""

    def __init__(self, **kwargs):
        super().__init__()
        
        in_channels = kwargs["flow_in_channels"]
        mid_channels = kwargs["flow_mid_channels"]
        hidden_depth = kwargs["flow_hidden_depth"]
        n_flows = kwargs["n_flows"]

        self.in_channels = in_channels

        self.flow = UnconditionalFlow2(
            in_channels=in_channels,
            hidden_dim=mid_channels,
            hidden_depth=hidden_depth,
            n_flows=n_flows,
        )

    def sample(self, shape,device="cpu"):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde)
        return sample.squeeze(dim=-1).squeeze(dim=-1)

    def forward(self, input, reverse=False, train=False):
        if len(input.shape) == 2:
            input = input[:,:,None,None]
        if reverse:
            return self.reverse(input)
        out, logdet = self.flow(input)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out):
        if len(out.shape) == 2:
            out = out[:,:,None,None]
        return self.flow(out, reverse=True)

    def get_last_layer(self):
        return getattr(
            self.flow.sub_layers[-1].coupling.t[-1].main[-1], "weight"
        )


# class TransformerFade(nn.Module):
#     """Two latents: Fa(ctor) and De(nsity) variables."""
#
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         in_channels = retrieve(config, "Transformer/in_channels")
#         mid_channels = retrieve(config, "Transformer/mid_channels")
#         hidden_depth = retrieve(config, "Transformer/hidden_depth")
#         n_flows = retrieve(config, "Transformer/n_flows")
#         bevp = retrieve(config, "Transformer/bevp", default=True)
#         benice = retrieve(config, "Transformer/benice", default=False)
#         be2 = retrieve(config, "Transformer/be2", default=False)
#
#         self.in_channels = in_channels
#
#         if bevp:
#             if benice:
#                 self.flowg = UnconditionalNICEFlow(
#                     in_channels=in_channels,
#                     hidden_dim=mid_channels,
#                     hidden_depth=hidden_depth,
#                     n_flows=n_flows,
#                 )
#             else:
#                 if not be2:
#                     self.flowg = UnconditionalGINFlow(
#                         in_channels=in_channels,
#                         hidden_dim=mid_channels,
#                         hidden_depth=hidden_depth,
#                         n_flows=n_flows,
#                     )
#                 else:
#                     self.flowg = UnconditionalGIN2Flow(
#                         in_channels=in_channels,
#                         hidden_dim=mid_channels,
#                         hidden_depth=hidden_depth,
#                         n_flows=n_flows,
#                     )
#         else:
#             self.flowg = UnconditionalFlow(
#                 in_channels=in_channels,
#                 hidden_dim=mid_channels,
#                 hidden_depth=hidden_depth,
#                 n_flows=n_flows,
#             )
#         self.flowf = UnconditionalFlow(
#             in_channels=in_channels,
#             hidden_dim=mid_channels,
#             hidden_depth=hidden_depth,
#             n_flows=n_flows,
#         )
#
#     def sample(self, shape):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         z_tilde = torch.randn(shape).to(device)
#         sample = self.reverse(z_tilde)
#         return sample
#
#     def forward(self, input, reverse=False, train=False):
#         if reverse:
#             return self.reverse(input)
#         gz, gzlogdet = self.flowg(input)
#         fgz, fzlogdet = self.flowf(gz)
#         fgzlogdet = gzlogdet + fzlogdet
#         return gz, gzlogdet, fgz, fgzlogdet
#
#     def reverse(self, out):
#         return self.flowg(self.flowf(out, reverse=True), reverse=True)
#
#     def get_last_layer(self):
#         return getattr(
#             self.flow.sub_layers[-1].coupling.t[-1].main[-1], "weight"
#         )
#
#
# class FlatFactorTransformer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#
#         self.n_factors = retrieve(config, "Transformer/n_factors", default=2)
#         self.factor_config = retrieve(
#             config, "Transformer/factor_config", default=list()
#         )
#
#         in_channels = retrieve(config, "Transformer/in_channels")
#         mid_channels = retrieve(config, "Transformer/mid_channels")
#         hidden_depth = retrieve(config, "Transformer/hidden_depth")
#         n_flows = retrieve(config, "Transformer/n_flows")
#         activation = retrieve(config, "Transformer/activation", default="lrelu")
#
#         self.in_channels = in_channels
#
#         self.flow = UnconditionalFlow(
#             in_channels=in_channels,
#             hidden_dim=mid_channels,
#             hidden_depth=hidden_depth,
#             n_flows=n_flows,
#             activation=activation,
#         )
#
#     def forward(self, input):
#         out, logdet = self.flow(input)
#         if self.factor_config:
#             out = torch.split(out, self.factor_config, dim=1)
#         else:
#             out = torch.chunk(out, self.n_factors, dim=1)
#         return out, logdet
#
#     def reverse(self, out):
#         out = torch.cat(out, dim=1)
#         return self.flow.reverse(out)
