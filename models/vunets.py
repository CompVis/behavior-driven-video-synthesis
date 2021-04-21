import torch
from torch import nn
from torch.nn import ModuleList
from lib.modules import (
    Upsample,
    Downsample,
    NormConv2d,
    SpaceToDepth,
    DepthToSpace,
    VunetRNB,
    L2NormConv2d,
    LayerNormConv2d,
)
import numpy as np
from functools import partial


class VunetOrg(nn.Module):
    def __init__(self, init_fn=None, n_channels_x=3,**kwargs):
        super().__init__()
        self.spatial_size = kwargs["spatial_size"]
        self.n_scales = (
            (
                1
                + int(np.round(np.log2(kwargs["spatial_size"])))
                - kwargs["bottleneck_factor"]  # bottleneck factor = 2
            )
            if kwargs["n_scales"] < 6
            else kwargs["n_scales"]
        )
        self.n_scales_x = (
            self.n_scales - kwargs["box_factor"] if n_channels_x > 3 else self.n_scales
        )
        dropout_prob = kwargs["dropout_prob"] if "dropout_prob" in kwargs else 0.
        self.n_channels_x = n_channels_x
        n_latent_scales = kwargs["n_latent_scales"]
        if kwargs["conv_layer_type"] == "l1":
            conv_layer = NormConv2d
            conv_t = "L1NormConv2d"
        elif kwargs["conv_layer_type"] == "l2":
            conv_layer = partial(L2NormConv2d, init=init_fn, bias=False)
            conv_t = "L2NormConv2d"
        else:
            raise NotImplementedError("No conv layers others than l1 and l2 normalized ones are available.")

        print("Vunet using " + conv_t + " as conv layers.")
        self.eu = EncUp(
            self.n_scales_x,
            n_filters=kwargs["nf_start"],  # 64
            max_filters=kwargs["nf_max"],  # 128
            conv_layer=conv_layer,
            nf_in=self.n_channels_x,
            dropout_prob=dropout_prob
        )
        self.ed = EncDown(
            n_filters=kwargs["nf_max"],
            nf_in=kwargs["nf_max"],
            conv_layer=conv_layer,
            n_scales=n_latent_scales,
            subpixel_upsampling=kwargs["subpixel_upsampling"],
            dropout_prob=dropout_prob
        )  # nf_max = 128
        self.du = DecUp(
            self.n_scales,
            n_filters=kwargs["nf_start"],
            max_filters=kwargs["nf_max"],
            conv_layer=conv_layer,
            dropout_prob=dropout_prob
        )
        self.dd = DecDown(
            self.n_scales,
            kwargs["nf_max"],
            kwargs["nf_start"],
            nf_out=3,
            conv_layer=conv_layer,
            n_latent_scales=n_latent_scales,
            subpixel_upsampling=kwargs["subpixel_upsampling"],
            dropout_prob=dropout_prob
        )

    def forward(self, x, c):
        # x: shape image
        # c: stickman
        hs = self.eu(x)
        es, qs, zs_posterior = self.ed(hs)

        gs = self.du(c)
        imgs, ds, ps, zs_prior = self.dd(gs, zs_posterior, training=True)

        activations = hs, qs, gs, ds
        return imgs, qs, ps, activations

    def test_forward(self, c):
        # sample appearance
        gs = self.du(c)
        imgs, ds, ps, zs_prior = self.dd(gs, [], training=False)
        return imgs

    def transfer(self, x, c):
        hs = self.eu(x)
        es, qs, zs_posterior = self.ed(hs)
        zs_mean = list(qs)

        gs = self.du(c)
        imgs, _, _, _ = self.dd(gs, zs_mean, training=True)
        return imgs


class EncUp(nn.Module):
    def     __init__(
        self, n_scales, n_filters, max_filters, nf_in=3, conv_layer=NormConv2d, dropout_prob=0.
    ):
        super().__init__()
        # number of residual block per scale
        self.n_rnb = 2
        self.n_scales = n_scales
        self.nin = conv_layer(
            in_channels=nf_in, out_channels=n_filters, kernel_size=1
        )

        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        nf = n_filters
        for i in range(self.n_scales):
            for n in range(self.n_rnb):
                self.blocks.append(VunetRNB(channels=nf, conv_layer=conv_layer,dropout_prob=dropout_prob))

            if i + 1 < self.n_scales:
                out_c = min(2 * nf, max_filters)
                self.downs.append(Downsample(nf, out_c))
                nf = out_c

    def forward(self, x, **kwargs):
        # x is an image, which defines the appearance of a person
        hs = []

        h = self.nin(x)

        for i in range(self.n_scales):

            for n in range(self.n_rnb):
                h = self.blocks[2 * i + n](h)
                hs.append(h)

            if i + 1 < self.n_scales:
                h = self.downs[i](h)

        return hs


def latent_sample(p):
    mean = p
    stddev = 1.0
    eps = torch.randn_like(mean)

    return mean + stddev * eps


class EncDown(nn.Module):
    def __init__(self, n_filters, nf_in, subpixel_upsampling, n_scales=2, conv_layer=NormConv2d, dropout_prob=0.):
        super().__init__()
        self.nin = conv_layer(nf_in, n_filters, kernel_size=1)
        self.n_scales = n_scales
        self.n_rnb = 2
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.make_latent_params = nn.ModuleList()
        nf = n_filters
        for i in range(self.n_scales):

            for n in range(self.n_rnb // 2):
                self.blocks.append(
                    VunetRNB(channels=nf, a_channels=nf, residual=True,dropout_prob=dropout_prob)
                )

            self.make_latent_params.append(
                conv_layer(nf, nf, kernel_size=3, padding=1)
            )

            for n in range(self.n_rnb // 2):
                self.blocks.append(
                    VunetRNB(channels=nf, a_channels=2 * nf, residual=True,dropout_prob=dropout_prob)
                )

            self.ups.append(
                Upsample(nf, nf, subpixel=True)
            )

        self.fin_block = VunetRNB(channels=nf, a_channels=nf, residual=True,dropout_prob=dropout_prob)

    def forward(self, gs):
        hs = []  # hidden units
        qs = []  # posteriors
        zs = []  # samples from posterior

        h = self.nin(gs[-1])
        for i in range(self.n_scales):

            h = self.blocks[2 * i](h, gs.pop())
            hs.append(h)

            # post params
            q = self.make_latent_params[i](h)
            qs.append(q)

            # post sample
            z = latent_sample(q)
            zs.append(z)

            gz = torch.cat([gs.pop(), z], dim=1)
            h = self.blocks[2 * i + 1](h, gz)
            hs.append(h)

            h = self.ups[i](h)

        # last resnet_block
        h = self.fin_block(h, gs.pop())
        hs.append(h)
        return hs, qs, zs


class DecUp(nn.Module):
    def __init__(
        self, n_scales, n_filters, max_filters, nf_in=3, conv_layer=NormConv2d, dropout_prob=0.
    ):
        super().__init__()
        # number of residual block per scale
        self.n_rnb = 2
        self.n_scales = n_scales
        self.nin = conv_layer(
            in_channels=nf_in, out_channels=n_filters, kernel_size=1
        )

        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        nf = n_filters
        for i in range(self.n_scales):
            for n in range(self.n_rnb):
                self.blocks.append(VunetRNB(channels=nf, conv_layer=conv_layer,dropout_prob=dropout_prob))

            if i + 1 < self.n_scales:
                out_c = min(2 * nf, max_filters)
                self.downs.append(Downsample(nf, out_c))
                nf = out_c

    def forward(self, c):
        # x is an image, which defines the shape and body pose of the person
        hs = []

        h = self.nin(c)

        for i in range(self.n_scales):

            for n in range(self.n_rnb):
                h = self.blocks[2 * i + n](h)
                hs.append(h)

            if i + 1 < self.n_scales:
                h = self.downs[i](h)

        return hs


class DecDownAlter(nn.Module):
    def __init__(
        self,
        n_scales,
        nf_in,
        nf_last,
        nf_out,
        subpixel_upsampling,
        conv_layer=NormConv2d,
        n_latent_scales=2,
        dropout_prob=0.
    ):
        super().__init__()
        self.n_rnb = 2
        self.n_scales = n_scales
        self.n_latent_scales = n_latent_scales
        self.nin = conv_layer(nf_in, nf_in, kernel_size=1)
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        # autoregressive stuff
        # self.latent_nins = nn.ModuleDict()
        # self.auto_lp = nn.ModuleDict()
        self.auto_blocks = nn.ModuleList()
        # last conv
        self.out_conv = conv_layer(nf_last, nf_out, kernel_size=3, padding=1)
        # for reordering
        self.depth_to_space = DepthToSpace(block_size=2)
        self.space_to_depth = SpaceToDepth(block_size=2)

        nf_h_latent_scales = nf_in

        nf = nf_in
        for i in range(self.n_scales):

            for n in range(self.n_rnb // 2):
                self.blocks.append(
                    VunetRNB(
                        channels=nf,
                        a_channels=nf,
                        residual=True,
                        conv_layer=conv_layer,
                        dropout_prob=dropout_prob
                    )
                )

            if i < self.n_latent_scales:
                self.auto_blocks.append(VunetRNB(channels=nf,a_channels=nf,residual=True,conv_layer=conv_layer,dropout_prob=dropout_prob))

            for n in range(self.n_rnb // 2):
                self.blocks.append(
                    VunetRNB(
                        channels=nf,
                        a_channels=nf,
                        residual=True,
                        conv_layer=conv_layer,
                        dropout_prob=dropout_prob
                    )
                )

            if i + 1 < self.n_scales:
                out_c = min(nf_in, nf_last * 2 ** (n_scales - (i + 2)))
                if subpixel_upsampling:
                    subpixel = True
                else:
                    subpixel = True if i < self.n_latent_scales else False
                self.ups.append(Upsample(nf, out_c, subpixel=subpixel))
                nf = out_c

    def forward(self, gs, zs_posterior, training):
        gs = list(gs)
        zs_posterior = list(zs_posterior)

        hs = []
        ps = []
        zs = []

        h = self.nin(gs[-1])
        lat_count = 0
        for i in range(self.n_scales):

            h = self.blocks[2 * i](h, gs.pop())
            hs.append(h)

            if i < self.n_latent_scales:

                if training:
                    from_dist = zs_posterior.pop(0)
                else:
                    from_dist = torch.randn_like(h)

                h = self.auto_blocks[lat_count](h,from_dist)
                # hs.append(h)
                lat_count+=1
                # scale = f"l_{i}"
                # if training:
                #     zs_posterior_groups = self.__split_groups(zs_posterior[0])
                # p_groups = []
                # z_groups = []
                # pre = self.auto_blocks[scale][0](h)
                # p_features = self.space_to_depth(pre)
                #
                # for l in range(4):
                #     p_group = self.auto_lp[scale][l](p_features)
                #     p_groups.append(p_group)
                #     prior_s = torch.randn_like(p_group)
                #     z_groups.append(prior_s)
                #
                #
                #
                #     if training:
                #         feedback = zs_posterior_groups.pop(0)
                #     else:
                #         feedback = prior_s
                #
                #     if l + 1 < 4:
                #         p_features = self.auto_blocks[scale][l + 1](
                #             p_features, feedback
                #         )
                # if training:
                #     assert not zs_posterior_groups
                #
                # p = self.__merge_groups(p_groups)
                # ps.append(p)
                #
                # z_prior = self.__merge_groups(z_groups)
                # zs.append(z_prior)
                #
                # if training:
                #     z = zs_posterior.pop(0)
                # else:
                #     z = z_prior
                #
                # h = torch.cat([h, z], dim=1)
                # h = self.latent_nins[scale](h)
                h = self.blocks[2 * i + 1](h, gs.pop())
                hs.append(h)
            else:
                h = self.blocks[2 * i + 1](h, gs.pop())
                hs.append(h)

            if i + 1 < self.n_scales:
                h = self.ups[i](h)

        assert not gs
        if training:
            assert not zs_posterior

        params = self.out_conv(hs[-1])

        # returns imgs, activations, prior params and samples
        return params


    def __split_groups(self, x):
        # split along channel axis
        sec_size = x.shape[1]
        return list(torch.split(self.space_to_depth(x), sec_size, dim=1))

    def __merge_groups(self, x):
        # merge groups along channel axis
        return self.depth_to_space(torch.cat(x, dim=1))

class VunetAlter(nn.Module):
    def __init__(self, init_fn=None, n_channels_x=3,**kwargs):
        super().__init__()
        self.spatial_size = kwargs["spatial_size"]
        self.n_scales = (
            (
                1
                + int(np.round(np.log2(kwargs["spatial_size"])))
                - kwargs["bottleneck_factor"]  # bottleneck factor = 2
            )
            if kwargs["n_scales"] < 6
            else kwargs["n_scales"]
        )
        self.n_scales_x = (
            self.n_scales - kwargs["box_factor"] if n_channels_x > 3 else self.n_scales
        )
        self.n_channels_x = n_channels_x
        n_latent_scales = kwargs["n_latent_scales"]
        dropout_prob = kwargs["dropout_prob"] if "dropout_prob" in kwargs else 0.
        if kwargs["conv_layer_type"] == "l1":
            conv_layer = NormConv2d
            conv_t = "L1NormConv2d"
        elif kwargs["conv_layer_type"] == "l2":
            conv_layer = partial(L2NormConv2d, init=init_fn, bias=False)
            conv_t = "L2NormConv2d"
        else:
            conv_layer = LayerNormConv2d
            conv_t = "LayerNormConv2d"

        print("Vunet using " + conv_t + " as conv layers.")
        self.eu = EncUp(
            self.n_scales_x,
            n_filters=kwargs["nf_start"],  # 64
            max_filters=kwargs["nf_max"],  # 128
            conv_layer=conv_layer,
            nf_in=n_channels_x,
            dropout_prob=dropout_prob
        )
        self.ed = EncDownAlter(
            n_filters=kwargs["nf_max"],
            nf_in=kwargs["nf_max"],
            conv_layer=conv_layer,
            n_scales=n_latent_scales,
            subpixel_upsampling=kwargs["subpixel_upsampling"],
            dropout_prob=dropout_prob
        )  # nf_max = 128
        self.du = DecUp(
            self.n_scales,
            n_filters=kwargs["nf_start"],
            max_filters=kwargs["nf_max"],
            conv_layer=conv_layer,
            dropout_prob=dropout_prob
        )
        self.dd = DecDownAlter(
            self.n_scales,
            kwargs["nf_max"],
            kwargs["nf_start"],
            nf_out=3,
            conv_layer=conv_layer,
            n_latent_scales=n_latent_scales,
            subpixel_upsampling=kwargs["subpixel_upsampling"],
            dropout_prob=dropout_prob
        )

    def forward(self, x, c):
        # x: shape image
        # c: stickman
        hs = self.eu(x)
        es, means, logstds, zs_posterior = self.ed(hs)

        gs = self.du(c)
        imgs = self.dd(gs, zs_posterior, training=True)

        activations = hs, means, logstds
        return imgs, means, logstds, activations

    def test_forward(self, c):
        # sample appearance
        gs = self.du(c)
        imgs = self.dd(gs, [], training=False)
        return imgs

    def transfer(self, x, c):
        hs = self.eu(x)
        es, means, logstds, zs_posterior = self.ed(hs)
        zs_mean = list(means)

        gs = self.du(c)
        imgs = self.dd(gs, zs_mean, training=True)
        return imgs




class EncDownAlter(nn.Module):
    def __init__(self, n_filters, nf_in, subpixel_upsampling, n_scales=2, conv_layer=NormConv2d, dropout_prob=0.):
        super().__init__()
        self.nin = conv_layer(nf_in, n_filters, kernel_size=1)
        self.n_scales = n_scales
        self.n_rnb = 2
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.make_latent_params = nn.ModuleList()
        self.make_logstds = nn.ModuleList()
        nf = n_filters
        for i in range(self.n_scales):

            for n in range(self.n_rnb // 2):
                self.blocks.append(
                    VunetRNB(channels=nf, a_channels=nf, residual=True,dropout_prob=dropout_prob)
                )

            self.make_latent_params.append(
                conv_layer(nf, nf, kernel_size=3, padding=1)
            )

            self.make_logstds.append(
                conv_layer(nf, nf, kernel_size=3, padding=1)
            )

            for n in range(self.n_rnb // 2):
                self.blocks.append(
                    VunetRNB(channels=nf, a_channels=2 * nf, residual=True)
                )

            self.ups.append(
                Upsample(nf, nf, subpixel=True)
            )

        self.fin_block = VunetRNB(channels=nf, a_channels=nf, residual=True,dropout_prob=dropout_prob)
        self.squash = nn.Sigmoid()

    def forward(self, gs):
        hs = []  # hidden units
        means = []  # posteriors
        zs = []  # samples from posterior
        log_stds = []


        h = self.nin(gs[-1])
        for i in range(self.n_scales):

            h = self.blocks[2 * i](h, gs.pop())
            hs.append(h)

            # post params
            mu = self.make_latent_params[i](h)
            means.append(mu)
            logstd = self.make_logstds[i](h)
            logstd = self.squash(logstd)
            log_stds.append(logstd)


            # post sample
            z = self.reparametrize(mu,logstd)
            zs.append(z)

            gz = torch.cat([gs.pop(), z], dim=1)
            h = self.blocks[2 * i + 1](h, gz)
            hs.append(h)

            h = self.ups[i](h)

        # last resnet_block
        h = self.fin_block(h, gs.pop())
        hs.append(h)
        return hs, means,log_stds, zs

    def reparametrize(self,mu,logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu


class DecDown(nn.Module):
    def __init__(
        self,
        n_scales,
        nf_in,
        nf_last,
        nf_out,
        subpixel_upsampling,
        conv_layer=NormConv2d,
        n_latent_scales=2, dropout_prob=0.
    ):
        super().__init__()
        self.n_rnb = 2
        self.n_scales = n_scales
        self.n_latent_scales = n_latent_scales
        self.nin = conv_layer(nf_in, nf_in, kernel_size=1)
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        # autoregressive stuff
        self.latent_nins = nn.ModuleDict()
        self.auto_lp = nn.ModuleDict()
        self.auto_blocks = nn.ModuleDict()
        # last conv
        self.out_conv = conv_layer(nf_last, nf_out, kernel_size=3, padding=1)
        # for reordering
        self.depth_to_space = DepthToSpace(block_size=2)
        self.space_to_depth = SpaceToDepth(block_size=2)

        nf_h_latent_scales = nf_in

        nf = nf_in
        for i in range(self.n_scales):

            for n in range(self.n_rnb // 2):
                self.blocks.append(
                    VunetRNB(
                        channels=nf,
                        a_channels=nf,
                        residual=True,
                        conv_layer=conv_layer,
                        dropout_prob=dropout_prob
                    )
                )

            if i < self.n_latent_scales:
                scale = f"l_{i}"
                self.latent_nins.update(
                    {
                        scale: conv_layer(
                            nf_h_latent_scales * 2,
                            nf_h_latent_scales,
                            kernel_size=1,
                        )
                    }
                )

                # autoregressive_stuff
                clp = ModuleList()
                cb = ModuleList()
                for l in range(4):

                    clp.append(
                        conv_layer(
                            4 * nf_h_latent_scales,
                            nf_h_latent_scales,
                            kernel_size=3,
                            padding=1,
                        )
                    )
                    if l == 0:
                        cb.append(VunetRNB(channels=nf_h_latent_scales,dropout_prob=dropout_prob))
                    else:
                        cb.append(
                            VunetRNB(
                                channels=4 * nf_h_latent_scales,
                                a_channels=nf_h_latent_scales,
                                residual=True,
                                dropout_prob=dropout_prob
                            )
                        )

                self.auto_lp.update({scale: clp})
                self.auto_blocks.update({scale: cb})

            for n in range(self.n_rnb // 2):
                self.blocks.append(
                    VunetRNB(
                        channels=nf,
                        a_channels=nf,
                        residual=True,
                        conv_layer=conv_layer,
                        dropout_prob=dropout_prob
                    )
                )

            if i + 1 < self.n_scales:
                out_c = min(nf_in, nf_last * 2 ** (n_scales - (i + 2)))
                if subpixel_upsampling:
                    subpixel = True
                else:
                    subpixel = True if i < self.n_latent_scales else False
                self.ups.append(Upsample(nf, out_c, subpixel=subpixel))
                nf = out_c

    def forward(self, gs, zs_posterior, training):
        gs = list(gs)
        zs_posterior = list(zs_posterior)

        hs = []
        ps = []
        zs = []

        h = self.nin(gs[-1])
        for i in range(self.n_scales):

            h = self.blocks[2 * i](h, gs.pop())
            hs.append(h)

            if i < self.n_latent_scales:
                scale = f"l_{i}"
                if training:
                    zs_posterior_groups = self.__split_groups(zs_posterior[0])
                p_groups = []
                z_groups = []
                pre = self.auto_blocks[scale][0](h)
                p_features = self.space_to_depth(pre)

                for l in range(4):
                    p_group = self.auto_lp[scale][l](p_features)
                    p_groups.append(p_group)
                    z_group = latent_sample(p_group)
                    z_groups.append(z_group)

                    if training:
                        feedback = zs_posterior_groups.pop(0)
                    else:
                        feedback = z_group

                    if l + 1 < 4:
                        p_features = self.auto_blocks[scale][l + 1](
                            p_features, feedback
                        )
                if training:
                    assert not zs_posterior_groups

                p = self.__merge_groups(p_groups)
                ps.append(p)

                z_prior = self.__merge_groups(z_groups)
                zs.append(z_prior)

                if training:
                    z = zs_posterior.pop(0)
                else:
                    z = z_prior

                h = torch.cat([h, z], dim=1)
                h = self.latent_nins[scale](h)
                h = self.blocks[2 * i + 1](h, gs.pop())
                hs.append(h)
            else:
                h = self.blocks[2 * i + 1](h, gs.pop())
                hs.append(h)

            if i + 1 < self.n_scales:
                h = self.ups[i](h)

        assert not gs
        if training:
            assert not zs_posterior

        params = self.out_conv(hs[-1])

        # returns imgs, activations, prior params and samples
        return params, hs, ps, zs

    def __split_groups(self, x):
        # split along channel axis
        sec_size = x.shape[1]
        return list(torch.split(self.space_to_depth(x), sec_size, dim=1))

    def __merge_groups(self, x):
        # merge groups along channel axis
        return self.depth_to_space(torch.cat(x, dim=1))


class Regressor(nn.Module):

    def __init__(self,n_out,n_latent_scales,nf_max,latent_widths, linear_width_factor, n_linear=2,**kwargs):
        super().__init__()

        self.n_stages = n_latent_scales
        self.n_linear = n_linear
        self.linear_width = self.n_stages * nf_max * linear_width_factor
        self.embedders = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.act_fn = nn.ReLU()

        for i in range(self.n_stages):
            self.embedders.append(nn.Conv2d(in_channels=nf_max,out_channels=linear_width_factor*nf_max,kernel_size=latent_widths[i]))

        for i in range(self.n_linear):
            arg_in = 2 if self.linear_width // 2**(self.n_linear-i) > n_out else 1
            arg_out =2 if self.linear_width // 2**(self.n_linear-i-1) > n_out else 1
            if i == n_linear-1:
                self.linears.append(nn.Linear(in_features=self.linear_width// arg_in**i,out_features=n_out ))
            else:
                self.linears.append(nn.Linear(in_features=self.linear_width// arg_in**i,out_features=self.linear_width// arg_out**(i+1)))



    def forward(self, embeddings:list):
        out = []
        for e, embedder in zip(reversed(embeddings),self.embedders):
           out.append(self.act_fn(embedder(e)).squeeze(dim=-1).squeeze(dim=-1))

        out = torch.cat(out,dim=-1)

        for i in range(self.n_linear):
            if i < self.n_linear-1:
                out = self.act_fn(self.linears[i](out))
            else:
                out = self.linears[i](out)

        return out

