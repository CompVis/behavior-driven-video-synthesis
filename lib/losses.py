import torch
from torch import nn
from torch.nn import functional as F
from lib.utils import bounding_box_batch, get_member
from models.pose_discriminator import MIDisc, MIDiscConv1
from lib.utils import toggle_grad
from torch.optim import Adam
from collections import namedtuple


VGGOutput = namedtuple(
    "VGGOutput",
    ["input", "relu1_2", "relu2_2", "relu3_2", "relu4_2", "relu5_2"],
)


def weight_decay(weights: list):
    # reshaped = [weight.reshape(-1) for weight in weights]
    tests = [
        torch.dot(weight.reshape(-1), weight.reshape(-1)) for weight in weights
    ]
    weight_norms = torch.stack(tests, dim=-1)
    return torch.sum(weight_norms)


def latent_kl(prior_mean, posterior_mean):
    """

    :param prior_mean:
    :param posterior_mean:
    :return:
    """
    kl = 0.5 * torch.pow(prior_mean - posterior_mean, 2)
    kl = torch.sum(kl, dim=[1, 2, 3])
    kl = torch.mean(kl)

    return kl


def aggregate_kl_loss(prior_means, posterior_means):
    kl_loss = torch.sum(
        torch.cat(
            [
                latent_kl(p, q).unsqueeze(dim=-1)
                for p, q in zip(
                    list(prior_means.values()), list(posterior_means.values())
                )
            ],
            dim=-1,
        )
    )
    return kl_loss


def compute_kl_loss(prior_means, posterior_means):
    kl_loss = torch.sum(
        torch.cat(
            [
                latent_kl(p, q).unsqueeze(dim=-1)
                for p, q in zip(prior_means, posterior_means)
            ],
            dim=-1,
        )
    )
    return kl_loss


def compute_kl_with_prior(means, logstds):
    kl_out = torch.mean(
        torch.cat(
            [
                kl_loss(m.reshape(m.size(0),-1), l.reshape(l.size(0),-1)).unsqueeze(dim=-1)
                for m, l in zip(means, logstds)
            ],
            dim=-1,
        )
    )
    return kl_out


def vgg_loss(custom_vgg, target, pred, weights=None):
    """

    :param custom_vgg:
    :param target:
    :param pred:
    :return:
    """
    target_feats = custom_vgg(target)
    pred_feats = custom_vgg(pred)
    target_feats = VGGOutput(**target_feats)
    pred_feats = VGGOutput(**pred_feats)

    names = list(pred_feats._asdict().keys())
    if weights is None:
        losses = {}

        for i, (tf, pf) in enumerate(zip(target_feats, pred_feats)):
            loss = get_member(custom_vgg, "loss_weights")[i] * torch.mean(
                torch.abs(tf - pf)
            ).unsqueeze(dim=-1)
            losses.update({names[i]: loss})
    else:

        losses = {
            names[0]: get_member(custom_vgg, "loss_weights")[0]
            * torch.mean(weights * torch.abs(target_feats[0] - pred_feats[0]))
            .unsqueeze(dim=-1)
            .to(torch.float)
        }

        for i, (tf, pf) in enumerate(zip(target_feats[1:], pred_feats[1:])):
            loss = get_member(custom_vgg, "loss_weights")[i + 1] * torch.mean(
                torch.abs(tf - pf)
            ).unsqueeze(dim=-1)

            losses.update({names[i + 1]: loss})

    return losses


def zoom_loss(target, pred, kps, img_sizes, custom_vgg, spatial_size):

    resized_pred = bounding_box_batch(kps, pred, img_sizes, spatial_size)

    return vgg_loss(custom_vgg, target, resized_pred)


class GANLoss(nn.Module):
    """
    The GAN loss; 'loss_type'-parameter defines the loss function which is actually computed
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        if loss_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif loss_type == "mse":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(
                f'The loss type for GANLoss must be either "vanilla" or "mse", but is actually {loss_type}.'
            )

        self.loss_type = loss_type

    def forward(self, pred, target):

        return self.loss(pred, target)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class SequentialDiscLoss(nn.Module):
    def __init__(self, loss_type: str = "bce"):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        elif loss_type == "mse":
            loss_layers = [nn.Sigmoid(), nn.MSELoss()]
            self.loss = nn.Sequential(*loss_layers)
        else:
            self.loss = None

        assert self.loss_type in ["bce", "mse", "hinge"]

    def forward(self, pred, target, mode="real"):
        if self.loss_type in ["bce", "mse"]:
            return self.loss(pred, target)
        elif self.loss_type == "hinge":
            assert mode in ["real", "fake", "gen"]
            if mode == "real":
                # discriminator training for real
                return torch.mean(torch.nn.ReLU()(1.0 - pred))
            elif mode == "fake":
                # discriminator training for fake
                return torch.mean(torch.nn.ReLU()(1.0 + pred))
            else:
                # generator training
                return -torch.mean(pred)
        else:
            raise ValueError("Invalid loss type.")


class MILoss:
    def __init__(self, input_dim, device,**kwargs):

        n_layer = (
            kwargs["n_layer_c"]
            if not "n_layer_midisc" in kwargs
            else kwargs["n_layer_midisc"]
        )
        nf_hidden = (
            kwargs["dim_hidden_c"]
            if not "nf_hidden_midisc" in kwargs
            else kwargs["nf_hidden_midisc"]
        )
        if hasattr(kwargs, "conv_midisc") and kwargs.conv_midisc:
            self.disc = MIDiscConv1(n_layer, input_dim, nf_hidden)
            print("Using convolutional mi disc.")
        else:
            self.disc = MIDisc(n_layer, input_dim, nf_hidden)
            print("Using linear mi disc.")

        self.disc.to(device)
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.disc_opt = Adam(
            params=[{"params": self.disc.parameters(), "name": "mi_disc"}],
            lr=kwargs.lr_init,
            weight_decay=kwargs["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.disc_opt, milestones=kwargs["tau"], gamma=kwargs["gamma"]
        )
        self.sigm = nn.Sigmoid()

    def train_disc(self, zb_joint, zb_marg, seq_len=0):

        # enable gradient
        toggle_grad(self.disc, True)
        self.disc.train()
        self.disc_opt.zero_grad()

        disc_joint = self.disc(zb_joint).squeeze()
        joint_p = torch.mean(self.sigm(disc_joint))
        out_dict = {"mi_true_p": joint_p.item()}


        disc_marg = self.disc(zb_marg).squeeze()
        marg_p = torch.mean(self.sigm(disc_marg))
        out_dict.update({"mi_fake_p": marg_p.item()})

        loss_joint = (
            self.loss(disc_joint, torch.ones_like(disc_joint)) / seq_len
        )
        loss_marg = self.loss(disc_marg, torch.zeros_like(disc_marg))

        out_dict.update({"mi_disc_loss_joint": loss_joint.item()})
        out_dict.update({"mi_disc_loss_marg": loss_marg.item()})

        loss = loss_joint + loss_marg
        out_dict.update({"mi_disc_loss": loss.item()})

        loss.backward(retain_graph=True)
        self.disc_opt.step()

        return out_dict

    def train_gen(self, zb_joint, zb_marg):
        # disable gradient
        toggle_grad(self.disc, False)
        self.disc.eval()
        zb_joint.requires_grad_(True)
        disc_joint = self.disc(zb_joint).squeeze()
        zb_marg.requires_grad_(True)
        disc_marg = self.disc(zb_marg).squeeze()

        loss_joint = self.loss(disc_joint, torch.ones_like(disc_joint))
        loss_marg = self.loss(disc_marg, torch.zeros_like(disc_marg))

        return -(loss_joint + loss_marg)

    def load(self, ckpt):
        if ckpt is not None:
            self.disc.load_state_dict(ckpt["mi_disc"])
            self.disc_opt.load_state_dict(ckpt["mi_optimizer"])

    def get_save_dict(self):
        return {"mi_disc": self.disc, "mi_optimizer": self.disc_opt}


def kl_loss(mu, logstd):
    # mu and logstd are b x k x d x d
    # make them into b*d*d x k

    dim = mu.shape[1]
    std = torch.exp(logstd)
    kl = torch.sum(-logstd + 0.5 * (std ** 2 + mu ** 2), dim=-1) - (0.5 * dim)

    return kl.mean()


class FlowLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        # self.config = config

    def forward(self, sample, logdet):
        nll_loss = torch.mean(nll(sample))
        assert len(logdet.shape) == 1
        nlogdet_loss = -torch.mean(logdet)
        loss = nll_loss + nlogdet_loss
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample)))
        log = {
            "flow_loss": loss.item(),
            "reference_nll_loss": reference_nll_loss.item(),
            "nlogdet_loss": nlogdet_loss.item(),
            "nll_loss": nll_loss.item(),
        }
        return loss, log

class FlowLossUncond(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample, logdet):
        nll_loss = torch.mean(nll(sample))
        assert len(logdet.shape) == 1
        nlogdet_loss = -torch.mean(logdet)
        loss = nll_loss + nlogdet_loss
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample)))
        log = {
                    "flow_loss": loss, "reference_nll_loss": reference_nll_loss,
                    "nlogdet_loss": nlogdet_loss, "nll_loss": nll_loss,
               }
        return loss, log


def nll(sample):
    return 0.5 * torch.sum(torch.pow(sample, 2), dim=[1, 2, 3])
