"""Flow consisting of ActNorm, DoubleVectorCouplingBlock and Shuffle. Additionally, powerful conditioning encodings are
learned."""

import torch
import torch.nn as nn
import numpy as np


from lib.modules import ActNorm,FeatureLayer,DenseEncoderLayer
from models.flow.blocks import ConditionalFlow
#from trex.model.ae.model import FeatureLayer, DenseEncoderLayer


class DenseEmbedder(nn.Module):
    """Supposed to map small-scale features (e.g. labels) to some given latent dim"""
    def __init__(self, in_dim, up_dim, depth=4, given_dims=None):
        super().__init__()
        self.net = nn.ModuleList()
        if given_dims is not None:
            assert given_dims[0] == in_dim
            assert given_dims[-1] == up_dim
            dims = given_dims
        else:
            dims = np.linspace(in_dim, up_dim, depth).astype(int)
        for l in range(len(dims)-2):
            self.net.append(nn.Conv2d(dims[l], dims[l + 1], 1))
            self.net.append(ActNorm(dims[l + 1]))
            self.net.append(nn.LeakyReLU(0.2))

        self.net.append(nn.Conv2d(dims[-2], dims[-1], 1))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x.squeeze(-1).squeeze(-1)


class Embedder(nn.Module):
    """Embeds a 4-dim tensor onto dense latent code, much like the classic encoder."""
    def __init__(self, in_spatial_size, in_channels, emb_dim, n_down=4):
        super().__init__()
        self.feature_layers = nn.ModuleList()
        norm = 'an'  # hard coded
        bottleneck_size = in_spatial_size // 2**n_down
        self.feature_layers.append(FeatureLayer(0, in_channels=in_channels, norm=norm))
        for scale in range(1, n_down):
            self.feature_layers.append(FeatureLayer(scale, norm=norm))
        self.dense_encode = DenseEncoderLayer(n_down, bottleneck_size, emb_dim)
        if n_down == 1:
            # add some extra parameters to make model a little more powerful ? # TODO
            print(" Warning: Embedder for ConditionalTransformer has only one down-sampling step. You might want to "
                  "increase its capacity.")

    def forward(self, input):
        h = input
        for layer in self.feature_layers:
            h = layer(h)
        h = self.dense_encode(h)
        return h.squeeze(-1).squeeze(-1)


class ConditionalTransformer(nn.Module):
    """Conditional Transformer supposed to map ."""
    def __init__(self, **kwargs):
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        super().__init__()

        in_channels = kwargs[ "flow_in_channels"]
        mid_channels = kwargs[ "flow_mid_channels"]
        hidden_depth = kwargs["flow_hidden_depth"]
        n_flows = kwargs["n_flows"]
        conditioning_option =  kwargs["conditioning_option"]
        flowactivation = "lrelu"
        embedding_channels =  kwargs["embedding_channels"] if "embedding_channels" in kwargs else in_channels
        n_down = kwargs["embedder_down"]
        self.emb_channels = embedding_channels
        self.in_channels = in_channels

        self.flow = ConditionalFlow(in_channels=in_channels, embedding_dim=self.emb_channels, hidden_dim=mid_channels,
                                    hidden_depth=hidden_depth, n_flows=n_flows, conditioning_option=conditioning_option,
                                    activation=flowactivation)
        conditioning_spatial_size = kwargs["conditioning_spatial_size"]
        conditioning_in_channels = kwargs["conditioning_in_channels"]
        if conditioning_spatial_size == 1:
            self.embedder = DenseEmbedder(conditioning_in_channels, in_channels)
        else:
            self.embedder = Embedder(conditioning_spatial_size, conditioning_in_channels, in_channels, n_down=n_down)

    def embed(self, conditioning):
        # embed it via embedding layer
        embedding = self.embedder(conditioning)
        return embedding

    def sample(self, shape, conditioning):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde, self.embed(conditioning))
        return sample

    def forward(self, input, conditioning, reverse=False, train=False):
        if len(input.shape) == 2:
            conditioning = conditioning[:,:,None,None]
            input = input[:,:,None,None]
        if reverse:
            assert False    
            # TODO this looks dangerously wrong as self.reverse _also_ applies
            # self.embed
            return self.reverse(input, self.embed(conditioning))
        embedding = self.embed(conditioning)
        out, logdet = self.flow(input, embedding)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out, conditioning):
        if len(out.shape) == 2:
            out = out[:,:,None,None]
            conditioning = conditioning[:,:,None,None]
        embedding = self.embed(conditioning)
        return self.flow(out, embedding, reverse=True)

    def get_last_layer(self):
        return getattr(self.flow.sub_layers[-1].coupling.t[-1].main[-1], 'weight')
