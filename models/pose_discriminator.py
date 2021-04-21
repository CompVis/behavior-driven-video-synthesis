import torch
from torch import nn
import numpy as np

from lib.modules import L2NormConv2d, VunetRNB


class MIDisc(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, drop_prob=0.0):
        super().__init__()

        # build encoder
        feat = nn.ModuleList()
        n_in = input_dim  # x and y coords
        for k in range(n_layers):
            enc_tmp = nn.Linear(n_in, hidden_dim)
            feat.append(enc_tmp)
            feat.append(nn.Dropout(drop_prob))
            feat.append(nn.LeakyReLU(0.2))

            n_in = hidden_dim

        self.feat = nn.Sequential(*feat)

        self.classifier = nn.Linear(n_in, 1)

    def forward(self, x):
        h = self.feat(x)
        return self.classifier(h)


class MIDiscConv1(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, drop_prob=0.0):
        super().__init__()

        feat = []
        feat.append(L2NormConv2d(input_dim, hidden_dim, kernel_size=1))
        for k in range(n_layers):
            enc_tmp = VunetRNB(
                hidden_dim,
                kernel_size=1,
                conv_layer=L2NormConv2d,
                act_fn=nn.LeakyReLU(),
                dropout_prob=drop_prob
            )
            feat.append(enc_tmp)

        self.extractor = nn.Sequential(*feat)
        self.classifier = nn.Sequential(
            nn.LeakyReLU(), L2NormConv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

    def forward(self, x):
        if len(x.shape) != 4:
            x = x.reshape((-1, x.shape[1], 1, 1))
        h = self.extractor(x)
        h = self.classifier(h)

        return torch.sum(h, dim=[1, 2, 3]).unsqueeze(dim=-1)


class Sequence_disc(nn.Module):
    def __init__(
        self,
        n_in,
        n_layers_rnn=1,
        dim_hidden_rnn=256,
        n_layers_class=2,
        dim_hidden_class=128,
        input_type="poses",
        architecture="lstm",
        root_translation_changes=False
    ):
        super(Sequence_disc, self).__init__()

        # build feature rnn
        self.n_layers_rnn = n_layers_rnn
        self.dim_hidden_rnn = dim_hidden_rnn
        self.input_type = input_type
        self.root_translation_changes = root_translation_changes
        if self.root_translation_changes:
            assert self.input_type == "poses"
        if architecture == "gru":
            self.rnn = nn.GRU(
                input_size=n_in,
                hidden_size=dim_hidden_rnn,
                num_layers=n_layers_rnn,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=n_in,
                hidden_size=dim_hidden_rnn,
                num_layers=n_layers_rnn,
                batch_first=True,
            )
        self.hidden = self.init_hidden(bs=1, device="cpu")

        # build classifier
        classifier = []
        n_in = self.dim_hidden_rnn
        for k in range(n_layers_class):
            enc_tmp = nn.Linear(n_in, dim_hidden_class)
            torch.nn.init.xavier_uniform_(enc_tmp.weight)
            classifier.append(enc_tmp)
            classifier.append(nn.ReLU())

            n_in = dim_hidden_class

        classifier.append(nn.Linear(n_in, 1))

        self.classifier = nn.Sequential(*classifier)

        self.BCE = nn.BCEWithLogitsLoss()
        # self.sig = nn.Sigmoid()

    def init_hidden(self, bs, device):
        # num_layers x bs x dim_hidden
        if isinstance(self.rnn, nn.LSTM):
            self.hidden = (
                torch.zeros((self.n_layers_rnn, bs, self.dim_hidden_rnn)).to(
                    device
                ),
                torch.zeros((self.n_layers_rnn, bs, self.dim_hidden_rnn)).to(
                    device
                ),
            )
        else:
            self.hidden = torch.zeros(
                (self.n_layers_rnn, bs, self.dim_hidden_rnn)
            ).to(device)

    def forward(self, x):
        if self.input_type == "changes":
            x = x[:, 1:] - x[:, :-1]
        elif self.root_translation_changes:
                x[:, :, :3] = torch.cat(
                    [
                        x[:, 1:, :3] - x[:, :-1, :3],
                        torch.zeros(
                            (x.shape[0], 1, 3),
                            dtype=x.dtype,
                            device=x.get_device()
                            if x.get_device() >= 0
                            else "cpu",
                        ),
                    ], dim=1
                )
        elif self.input_type=="angles":
            x = x[...,3:]
        elif self.input_type == "translation":
            x = x[...,:3:]

        self.init_hidden(x.shape[0], device=x.get_device())
        fmap_out = []
        _, self.hidden = self.rnn(x, self.hidden)
        fmap_out.append(self.hidden[0])
        out = self.classifier(self.hidden[0])

        return out, fmap_out

    def loss(self, pred_gen, pred_orig):
        # L_disc1 = torch.mean(torch.nn.ReLU()(1.0 - pred_orig))
        # L_disc2 = torch.mean(torch.nn.ReLU()(1.0 + pred_gen))
        # L_gen = -torch.mean(pred_gen)

        L_disc1 = self.BCE(pred_gen, torch.zeros_like(pred_gen))
        L_disc2 = self.BCE(pred_orig, torch.ones_like(pred_gen))
        L_gen = self.BCE(pred_gen, torch.ones_like(pred_gen))

        return ((L_disc1 + L_disc2) / 2), L_gen

    def fmap_loss(self, fmap1, fmap2, loss="l1"):
        recp_loss = 0
        for idx in range(len(fmap1)):
            if loss == "l1":
                recp_loss += torch.mean(torch.abs((fmap1[idx] - fmap2[idx])))
            if loss == "l2":
                recp_loss += torch.mean((fmap1[idx] - fmap2[idx]) ** 2)
        return recp_loss / len(fmap1)


class Sequence_disc_conv(nn.Module):
    def __init__(
        self,
        n_kps,
        seq_len,
        temp_window=10,
        temp_stride=5,
        n_filter=16,
        n_layers_class=2,
        dim_hidden_class=128,
        use_sgm=True,
    ):
        super(Sequence_disc_conv, self).__init__()

        # build temporal Convolution
        temp_conv1 = nn.Conv2d(1, n_filter, (n_kps, temp_window), stride=(1, temp_stride))
        torch.nn.init.xavier_uniform_(temp_conv1.weight)
        self.temp_conv1 = temp_conv1
        n_out = int(np.floor((seq_len - temp_window) / temp_stride)) + 1

        temp_conv2 = nn.Conv2d(1, n_filter, (n_out, 3), stride=1)
        torch.nn.init.xavier_uniform_(temp_conv2.weight)
        self.temp_conv2 = temp_conv2
        n_out = int(n_filter - 3 + 1)

        # build classifier
        classifier = nn.ModuleList()
        n_in = n_out * n_filter
        for k in range(n_layers_class):
            enc_tmp = nn.Linear(n_in, dim_hidden_class)
            torch.nn.init.xavier_uniform_(enc_tmp.weight)
            classifier.append(enc_tmp)
            classifier.append(nn.ReLU())

            n_in = dim_hidden_class

        classifier.append(nn.Linear(n_in, 1))
        torch.nn.init.xavier_uniform_(enc_tmp.weight)

        if use_sgm:
            classifier.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):

        out = self.temp_conv1(x.unsqueeze(1))
        out = torch.nn.functional.relu(out)

        out = out.permute(0, 2, 3, 1)
        out = self.temp_conv2(out)
        out = out.view(-1, out.size(1) * out.size(2) * out.size(3))

        out = self.classifier(out.squeeze())

        return out


def conv1D(in_planes, out_planes, stride=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1D(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=planes)
        self.relu = nn.ReLU()
        self.conv2 = conv1D(planes, planes)
        self.bn2 = nn.GroupNorm(num_groups=4, num_channels=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Sequence_disc_michael(nn.Module):
    def __init__(
        self,
        layers,
        n_kps,
        out_dim=1,
        conditioning="none",
        compare_sequences=False,
    ):

        self.inplanes = 64
        super(Sequence_disc_michael, self).__init__()
        self.conditioning = conditioning
        self.compare_sequences = compare_sequences
        linear_dim = 13*32


        self.conv1 = nn.Conv1d(
            n_kps * 2 if self.compare_sequences else n_kps,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(BasicBlock, 32, layers[1], stride=2)
        # self.layer3 = self._make_layer(BasicBlock, 32, layers[2], stride=2)
        # self.layer4 = self._make_layer(BasicBlock, 16, layers[3], stride=2)
        self.fc = nn.Linear(linear_dim, out_dim, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_normal_(m.weight)

    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(
                    num_channels=planes * block.expansion, num_groups=16
                ),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, label=None):

        # breakpoint()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)

        out = x.reshape(x.size(0), -1)

        return self.fc(out), x

    def loss(self, pred_gen, pred_orig):
        L_disc1 = torch.mean(torch.nn.ReLU()(1.0 - pred_orig))
        L_disc2 = torch.mean(torch.nn.ReLU()(1.0 + pred_gen))
        L_gen = -torch.mean(pred_gen)
        # BCE = nn.BCELoss()
        # sig = nn.Sigmoid()
        # pred_gen, pred_orig = sig(pred_gen), sig(pred_orig)
        # L_disc1 = BCE(pred_gen, torch.zeros_like(pred_gen))
        # L_disc2 = BCE(pred_orig, torch.ones_like(pred_gen))
        # L_gen = BCE(pred_gen, torch.ones_like(pred_gen))

        return (L_disc1 + L_disc2) / 2, L_gen

    def gp(self, x_gen, x_orig, c):
        if self.input_type == "changes":
            x_gen = x_gen[:, :, 1:] - x_gen[:, :, :-1]
        elif self.input_type == "combined":
            x_gen = torch.cat(
                [x_gen[:, :, 1:] - x_gen[:, :, :-1], x_gen], dim=2
            )
        disc_interpolates, *_ = self.forward(x_gen, c)
        dev = x_gen.get_device() if x_gen.get_device() >= 0 else "cpu"
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=x_gen,
            grad_outputs=torch.ones(disc_interpolates.size()).to(dev),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(x_gen.size(0), -1)  # flat the data
        gradient_penalty = (
            ((gradients + 1e-16).norm(2, dim=1) - 1.0) ** 2
        ).mean()
        return gradient_penalty

    def fmap_loss(self, fmap1, fmap2, loss="l1"):
        recp_loss = 0
        for idx in range(len(fmap1)):
            if loss == "l1":
                recp_loss += torch.mean(torch.abs((fmap1[idx] - fmap2[idx])))
            if loss == "l2":
                recp_loss += torch.mean((fmap1[idx] - fmap2[idx]) ** 2)
        return recp_loss / len(fmap1)

    def class_loss(self, l_p, label):
        # l_gt    = torch.nn.functional.one_hot(label, l_p.shape[1])
        _, pred = torch.max(l_p, dim=1)
        acc = (pred == label).sum().item() / l_p.shape[0]
        loss = torch.nn.CrossEntropyLoss()(l_p, label.cuda())
        return loss, acc


class ResnetBlock(nn.Module):
    """
    Pre activated resnet block
    """

    def __init__(
        self,
        nin,
        n_out,
        n_hidden=None,
        kernel_size=3,
        stride=(1, 1),
        padding=1,
        bias=True,
        actvfn=nn.ReLU,
    ):
        super().__init__()
        if n_hidden is None:
            n_hidden = n_out
        self.actvfn = actvfn()
        self.conv1 = nn.Conv2d(
            nin,
            n_hidden,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )
        self.bn1 = nn.GroupNorm(num_channels=nin, num_groups=nin // 8)
        self.conv2 = nn.Conv2d(
            n_hidden, n_out, kernel_size=kernel_size, padding=padding, bias=bias
        )
        self.bn2 = nn.GroupNorm(num_channels=n_hidden, num_groups=n_hidden // 8)
        if nin != n_out or max(stride) > 1:
            self.shortcut = nn.Conv2d(
                nin,
                n_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        else:
            self.shortcut = None

    def forward(self, x):
        if self.shortcut is None:
            res = x
        else:
            res = self.shortcut(x)

        x = self.bn1(x)
        x = self.conv1(self.actvfn(x))
        x = self.bn2(x)
        x = self.conv2(self.actvfn(x))

        return x + res


class SelfAttention(nn.Module):
    def __init__(self, n_channels, down_factor):
        super().__init__()
        intermediate_channels = n_channels // down_factor
        self.Wf = nn.Conv2d(
            out_channels=intermediate_channels,
            in_channels=n_channels,
            kernel_size=1,
            bias=False,
        )
        self.Wg = nn.Conv2d(
            out_channels=intermediate_channels,
            in_channels=n_channels,
            kernel_size=1,
            bias=False,
        )
        self.Wh = nn.Conv2d(
            out_channels=n_channels // 2,
            in_channels=n_channels,
            kernel_size=1,
            bias=False,
        )
        self.Wv = nn.Conv2d(
            in_channels=n_channels // 2,
            out_channels=n_channels,
            kernel_size=1,
            bias=False,
        )

        self.gamma = self.beta = nn.Parameter(
            torch.zeros([1, 1, 1, 1], dtype=torch.float32), requires_grad=True
        )
        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        bs = x.shape[0]
        width = x.shape[-1]
        height = x.shape[-2]

        f = self.Wf(x)
        f = torch.reshape(f, [bs, f.shape[1], -1]).permute([0, 2, 1])

        g = self.Wg(x)
        g = self.pool(g)
        g = torch.reshape(g, [bs, g.shape[1], -1]).permute([0, 2, 1])
        g = torch.transpose(g, dim0=1, dim1=2)

        attn = torch.matmul(f, g)
        attn = self.softmax(attn)

        h = self.Wh(x)
        h = self.pool(h)
        h = torch.reshape(h, [bs, h.shape[1], -1]).permute([0, 2, 1])

        v = torch.matmul(attn, h)
        v = torch.reshape(v, [bs, height, width, -1]).permute([0, 3, 1, 2])

        v = self.Wv(v)

        return x + self.beta * v


if __name__ == "__main__":
    ## Test 3dconvnet with dummy input
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # model = Sequence_disc_conv(n_kps=16, seq_len=50).cuda()
    # print("Number of parameters in generator", sum(p.numel() for p in model.parameters()))
    # dummy = torch.rand((10, 50, 16)).cuda()
    layer = [3, 4, 6, 3]
    model = Sequence_disc_michael(layer, n_kps=22, n_labels=1).cuda()
    print(
        "Number of parameters in generator",
        sum(p.numel() for p in model.parameters()),
    )
    dummy = torch.rand((10, 22, 10)).cuda()
    print(model(dummy, torch.rand((10)).cuda())[0].shape)
