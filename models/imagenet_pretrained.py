from torch import nn
import torch
from lib.utils import scale_img
from torchvision.models import inception_v3
import torch.nn.functional as F


class PerceptualVGG(nn.Module):
    def __init__(self, vgg, weights):
        super().__init__()
        # self.vgg = vgg19(pretrained=True)
        if isinstance(vgg, torch.nn.DataParallel):
            self.vgg_layers = vgg.module.features
        else:
            self.vgg_layers = vgg.features

        self.loss_weights = weights

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
            .unsqueeze(dim=0)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1),
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)
            .unsqueeze(dim=0)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1),
        )
        self.target_layers = {
            "3": "relu1_2",
            "8": "relu2_2",
            "13": "relu3_2",
            "22": "relu4_2",
            "31": "relu5_2",
        }

    def forward(self, x):
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std

        out = {"input": x}
        # out = {}
        # normalize in between 0 and 1
        # x = scale_img(x)
        # normalize appropriate for vgg
        # x = torch.stack([self.input_transform(el) for el in torch.unbind(x)],dim=0)

        for name, submodule in self.vgg_layers._modules.items():
            # x = submodule(x)
            if name in self.target_layers:
                x = submodule(x)
                out[self.target_layers[name]] = x
            else:
                x = submodule(x)

        return out


class FIDInceptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.v3 = inception_v3(pretrained=True,aux_logits=False)
        self.v3.aux_logits = False
        # self.v3 = inception_v3(pretrained=True)


        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
                .unsqueeze(dim=0)
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1),
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)
                .unsqueeze(dim=0)
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1),
        )

        self.resize = nn.Upsample(size=(299,299),mode="bilinear")


    def forward(self, x):
        x = self.resize(x)
        # normalize in between 0 and 1
        x = scale_img(x)
        # normalize to demanded values
        x = (x - self.mean) / self.std


        for name, submodule in self.v3._modules.items():
            # if name == 'AuxLogits':
            #     continue
            x = submodule(x)
            if name == "Mixed_7c":
                break
            elif name == "Conv2d_4a_3x3" or name == "Conv2d_2b_3x3":
                x = F.avg_pool2d(x, kernel_size=3, stride=2)

        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = torch.flatten(out, 1)

        return out
