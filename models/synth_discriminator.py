import torch
from torch import nn
import functools
from lib.modules import VunetRNB, NormConv2d, Downsample
from lib.utils import toggle_grad
from torch import autograd
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events

class PatchGANDiscriminator(nn.Module):
    def __init__(
        self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d
    ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the first conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PartDiscriminator(nn.Module):
    def __init__(
        self, n_scales, part_size, nf_in=3, conv_layer=NormConv2d, max_filters=256, dropout_prob=0.
    ):
        super().__init__()
        # number of residual block per scale
        self.n_rnb = 2
        self.n_scales = n_scales
        self.nin = conv_layer(in_channels=nf_in, out_channels=16, kernel_size=3)

        blocks = []
        nf = 16
        spatial_size = part_size
        for i in range(self.n_scales):

            blocks.append(VunetRNB(channels=nf, conv_layer=conv_layer,dropout_prob=dropout_prob))

            out_c = min(2 * nf, max_filters)
            blocks.append(Downsample(nf, out_c))
            nf = out_c
            spatial_size = spatial_size // 2

        self.feature_extractor = nn.Sequential(*blocks)
        self.n_linear_units = nf * spatial_size ** 2
        self.classifier = nn.Linear(
            in_features=self.n_linear_units, out_features=1
        )

    def forward(self, x):
        h = self.nin(x)
        h = self.feature_extractor(h)

        h = h.view(-1, self.n_linear_units)
        out = self.classifier(h)

        return out


class DiscTrainer(object):
    def __init__(
        self,
        generator,
        config, # pass config["training"]
        discriminator= PartDiscriminator,
        grad_pen=False,
        lambda_gp=10,
        grad_weighting=False,
        **kwargs
    ):

        self.disc = discriminator(n_scales=config["pd_scales"],part_size=kwargs["spatial_size"] // 4)
        self.opt = None
        self.loss = nn.BCEWithLogitsLoss()
        self.use_gp = grad_pen
        self.lambda_gp = lambda_gp
        self.gw = grad_weighting
        # adam betas has to be a tupple
        self.adam_betas = config["adam_beta"]
        self.save_intervall = config["save_intervall"]

        self.generator = generator
        self.parallel = isinstance(generator,nn.DataParallel)

    def train_disc(self, real_x, fake_x,retain_graph=False):
        toggle_grad(self.disc, True)
        toggle_grad(self.generator, False)
        self.disc.train()
        self.opt.zero_grad()

        # train on real
        real_x.requires_grad_(True)
        disc_on_real = self.disc(real_x)
        real_loss = self.loss(disc_on_real, torch.ones_like(disc_on_real))

        if self.use_gp:
            real_loss.backward(create_graph=True)
            reg = self.lambda_gp * compute_grad2(disc_on_real, real_x).mean()
            reg.backward()
        else:
            real_loss.backward(retain_graph=retain_graph)

        # train on fake
        fake_x.requires_grad_()
        disc_on_fake = self.disc(fake_x)
        fake_loss = self.loss(disc_on_fake, torch.zeros_like(disc_on_real))
        if self.gw:
            fake_loss.backward(retain_graph=True)
        else:
            fake_loss.backward(retain_graph=retain_graph)

        self.opt.step()

        toggle_grad(self.disc, False)
        toggle_grad(self.generator, True)

        dloss = real_loss + fake_loss

        out = {
            "dloss": dloss.item(),
            "dloss_r": real_loss.item(),
            "dloss_f": fake_loss.item(),
        }
        if self.use_gp:
            out.update({"gp": reg.item()})

        return out

    def get_genloss(self, x_fake, pre_loss, last_layer_weight):
        toggle_grad(self.generator, True)
        toggle_grad(self.disc, False)
        disc_on_fake = self.disc(x_fake)
        gen_loss = self.loss(disc_on_fake, torch.ones_like(disc_on_fake))

        if self.gw:
            self.generator.zero_grad()
            # these are all other loss parts except for gan part of synthesis model
            grad_mean_normal_loss = torch.mean(
                autograd.grad(pre_loss, last_layer_weight, retain_graph=True)[0]
            )
            self.generator.zero_grad()
            # this is the gan part of the loss of the synthesis model
            grad_mean_gen_loss = torch.mean(
                autograd.grad(gen_loss, last_layer_weight, retain_graph=True)[0]
            )

            loss_weight = torch.abs(grad_mean_normal_loss / grad_mean_gen_loss)
            loss_weight.requires_grad_(False)
        else:
            loss_weight = 1.0

        return gen_loss, loss_weight

    def init_training(self,devices,lr,d_ckpt=None,o_ckpt=None):

        if d_ckpt is not None:
            self.disc.load_state_dict(d_ckpt)

        if self.parallel:
            self.disc = nn.DataParallel(self.disc,device_ids=devices)
        self.disc.to(devices[0])

        self.opt = torch.optim.Adam(params=[{"params":self.disc.parameters(),"name" : "disc"}],lr=lr,betas=self.adam_betas)
        if o_ckpt is not None:
            self.opt.load_state_dict(o_ckpt)


    def update_lr(self,lr):
        for p in self.opt.param_groups:
            p["lr"] = lr

    def checkpoint(self,save_dir, trainer, name):
        ckpt_handler = ModelCheckpoint(
            save_dir,
            name,
            save_interval=self.save_intervall,
            n_saved=1,
            require_empty=False,
        )

        save_dict =  {
            "disc": self.disc.module if self.parallel else self.disc,
            "opt": self.opt,
        }
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, ckpt_handler, save_dict
        )

def compute_grad2(d_out, x_in,allow_unused=False):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        only_inputs=True,
    )
    grad_dout = grad_dout[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg
