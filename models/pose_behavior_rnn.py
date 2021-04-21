import torch
from torch import nn
from lib.modules import NormConv2d


class Classifier(nn.Module):
    def __init__(self, n_in, n_c):
        super(Classifier, self).__init__()
        self.dim = 256
        self.GRU = nn.GRU(n_in, self.dim, 1, batch_first=True)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(self.dim, n_c)

    def forward(self, x):
        _, h_0 = self.GRU(x)
        x = self.fc1(h_0.reshape(x.shape[0], -1))
        return x

class Classifier_action(nn.Module):
    def __init__(self, n_in, n_c, dropout=0, dim=256):
        super().__init__()

        self.RNN = nn.LSTM(n_in, dim, 1, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(dim, 128)
        self.act = nn.ReLU()
        self.fc3 = nn.Linear(128, n_c)

    def forward(self, x):
        h_0, _ = self.RNN(x)
        x = self.act(self.fc1(h_0[:, -1]))
        return self.fc3(x), x


class Regressor(nn.Module):
    def __init__(self, n_dim, n_key):
        super(Regressor, self).__init__()

        self.act = nn.ReLU()
        self.fc1 = nn.Linear(n_dim, n_dim // 2)
        self.fc2 = nn.Linear(n_dim // 2, n_dim // 4)
        self.fc3 = nn.Linear(n_dim // 4, n_key)

    def forward(self, x):
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))


class Regressor_fly(nn.Module):
    def __init__(self, n_dim, n_key):
        super(Regressor_fly, self).__init__()

        self.act = nn.ReLU()
        self.fc1 = nn.Linear(n_dim, n_dim)
        self.fc2 = nn.Linear(n_dim, n_dim // 2)
        self.fc3 = nn.Linear(n_dim // 2, n_dim // 4)
        self.fc4 = nn.Linear(50, 128)
        self.fc5 = nn.Linear(128 + n_dim // 4, n_key)

    def forward(self, x, c):
        x = self.act(self.fc3(self.act(self.fc2(self.act(self.fc1(x))))))
        c = self.act(self.fc4(c))
        x = torch.cat((x, c), dim=1)
        return self.fc5(x)


class Classifier_action_beta(nn.Module):
    def __init__(self, n_in, n_c):
        super(Classifier_action_beta, self).__init__()

        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(n_in, n_c)

    def forward(self, x):
        return self.fc1(x)


class Decoder(nn.Module):
    def __init__(self, n_in, n_out, n_layer, dim_hidden):
        super(Decoder, self).__init__()

        # build encoder
        dec = nn.ModuleList()
        for k in range(n_layer):
            dec_tmp = nn.Linear(n_in, dim_hidden)
            torch.nn.init.xavier_uniform_(dec_tmp.weight)
            dec.append(dec_tmp)
            dec.append(nn.ReLU())

            n_in = dim_hidden

        # add target mapping
        dec_tmp = nn.Linear(n_in, n_out)
        torch.nn.init.xavier_uniform_(dec_tmp.weight)
        dec.append(dec_tmp)

        self.dec = nn.Sequential(*dec)

    def forward(self, x):
        return self.dec(x)


class CEncoder(nn.Module):
    def __init__(self, n_in, n_layers, dim_hidden, dim_bn):
        super(CEncoder, self).__init__()

        # build encoder
        enc = nn.ModuleList()
        for k in range(n_layers):
            enc_tmp = nn.Linear(n_in, dim_hidden)
            torch.nn.init.xavier_uniform_(enc_tmp.weight)
            enc.append(enc_tmp)
            enc.append(nn.ReLU())

            n_in = dim_hidden

        enc_tmp = nn.Linear(n_in, dim_bn)
        torch.nn.init.xavier_uniform_(enc_tmp.weight)
        enc.append(enc_tmp)

        self.cond_enc = nn.Sequential(*enc)

    def forward(self, x):
        return self.cond_enc(x)


class BEncoder(nn.Module):
    def __init__(
        self,
        n_in,
        n_layers,
        dim_hidden,
        use_linear,
        dim_linear,
        ib=False,
    ):
        super(BEncoder, self).__init__()


        self.rnn = nn.LSTM(
            input_size=n_in,
            hidden_size=dim_hidden,
            num_layers=n_layers,
            batch_first=True,
        )

        self.n_layer = n_layers
        self.dim_hidden = dim_hidden
        self.ib = ib
        self.hidden = self.init_hidden(bs=1, device="cpu")

        self.use_linear = use_linear
        self.linear = None
        if self.use_linear:
            self.linear = nn.Linear(self.dim_hidden, dim_linear)

        if self.ib:
            # functions to map tp mu and sigma
            self.mu_fn = NormConv2d(self.dim_hidden, self.dim_hidden, 1)
            self.std_fn = NormConv2d(self.dim_hidden, self.dim_hidden, 1)

    def init_hidden(self, bs, device):
        # num_layers x bs x dim_hidden
        self.hidden = (
            torch.zeros((self.n_layer, bs, self.dim_hidden)).to(device),
            torch.zeros((self.n_layer, bs, self.dim_hidden)).to(device),
        )

    def set_hidden(self, bs, device, hidden=None, cell=None):
        if hidden is None and cell is None:
            self.init_hidden(bs, device)
        elif hidden is None:
            self.hidden = (torch.zeros_like(cell), cell)
        elif cell is None:
            self.hidden = (hidden, torch.zeros_like(hidden))
        else:
            self.hidden = (hidden, cell)

    def forward(self, x, sample=False):
        out, self.hidden = self.rnn(x, self.hidden)

        # if self.use_linear:
        #     out = self.linear(out[:, -1].squeeze(1))
        # else:
        #     out = out.squeeze(dim=1)
        pre = self.hidden[0][-1]
        if self.ib:
            mu = (
                self.mu_fn(pre.unsqueeze(dim=-1).unsqueeze(dim=-1))
                .squeeze(dim=-1)
                .squeeze(dim=-1)
            )
            logstd = (
                self.std_fn(pre.unsqueeze(dim=-1).unsqueeze(dim=-1))
                .squeeze(dim=-1)
                .squeeze(dim=-1)
            )
            if sample:
                out = self._sample(mu)
            else:
                out = self.reparametrize(mu, logstd)
            return (out, mu, logstd, pre)

        return pre

    def reparametrize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def _sample(self, mu):
        return torch.randn_like(mu)

class FCResnet(nn.Module):

    def __init__(self,in_cn, out_cn):
        super().__init__()

        self.fc1 = nn.Linear(in_cn,int(out_cn/2))
        self.fc2 = nn.Linear(int(out_cn/2),int(out_cn/2))
        self.fc3 = nn.Linear(int(out_cn / 2), out_cn)
        self.shortcut = nn.Linear(in_cn, out_cn)

        self.act_fn = nn.ReLU()
        #self.norm = nn.LayerNorm(out_cn)
        self.norm = nn.LayerNorm(out_cn, elementwise_affine=False)

    def forward(self, x):
        sc = self.shortcut(x)

        out = self.act_fn(self.fc1(x))
        out = self.act_fn(self.fc2(out))
        out = self.act_fn(self.fc3(out))

        # out = self.fc1(x)
        # out = self.fc2(out)
        # out = self.fc3(out)

        out = out+sc

        return self.norm(out)

class MTVAE(nn.Module):

    def __init__(self,config, n_dim_im, device):
        super().__init__()
        self.config = config
        self.n_in = n_dim_im
        self.device = device

        self.lstm_enc = nn.LSTM(
                input_size=self.n_in,
                hidden_size=1024,
                num_layers=1,
                batch_first=True,
            )
        self.lstm_dec = nn.LSTM(
                input_size=1024,
                hidden_size=1024,
                num_layers=1,
                batch_first=True,
            )


        self.latent_enc = FCResnet(1024,1024)
        self.latent_dec = FCResnet(1536,1024)

        self.make_keypoints = nn.Linear(1024,self.n_in)
        self.inv_z = nn.Linear(512,512)
        self.make_h_dec=nn.Linear(2048,1024)
        self.make_c_dec= nn.Linear(2048,1024)

        self.div = self.config["n_cond"]



        self.make_mu = nn.Linear(1024,512)
        self.cov = nn.Linear(1024, 512)


    def forward(self,input_source,input_tgt,transfer=False,sample_prior=False):
        bs = input_source.size(0)
        seq_a = input_source[:,:self.div]
        seq_b = input_source[:, self.div:]

        seq_c = input_tgt

        h0 = torch.randn((1,bs,1024)).to(self.device)
        c0 = torch.randn((1,bs,1024)).to(self.device)

        # get motion encodings
        out_a, (hn_a,cn_a) = self.lstm_enc(seq_a,(h0,c0))
        #out_a = nn.LayerNorm(out_a.shape[1:],elementwise_affine=False)(out_a)
        out_b, (hn_b, cn_b) = self.lstm_enc(seq_b, (h0, c0))
        #out_b = nn.LayerNorm(out_b.shape[1:], elementwise_affine=False)(out_b)
        out_c, (hn_c, cn_c) = self.lstm_enc(seq_c, (h0, c0))
        #out_c = nn.LayerNorm(out_c.shape[1:], elementwise_affine=False)(out_c)

        e_a = out_a[:,-1]
        e_b = out_b[:,-1]
        e_c = out_c[:,-1]

        # latent encoder
        e_diff = e_b - e_a

        params = self.latent_enc(e_diff)
        mu = params[:,:int(params.size(1)/2)]
        logstd = params[:, int(params.size(1) / 2):]

        if sample_prior:
            z = torch.randn_like(mu)
        else:
            z = self.reparametrize(mu,logstd)

        #latent decoder
        inv_z = self.inv_z(z)
        if transfer:
            z_in_dec = torch.cat([inv_z, e_c], dim=-1)
            out_latent_dec = self.latent_dec(z_in_dec)
            dec_in = out_latent_dec + e_c
        else:
            z_in_dec = torch.cat([inv_z,e_a],dim=-1)
            out_latent_dec = self.latent_dec(z_in_dec)
            dec_in = out_latent_dec + e_a

        dec_in = nn.LayerNorm(dec_in.size(-1),elementwise_affine=False)(dec_in)

        out_cycle = self.make_cycle(e_a,dec_in)

        init_hidden_dec_past = hn_c.squeeze(0) if transfer else hn_a.squeeze(0)
        pre_dec = torch.cat([init_hidden_dec_past,dec_in],dim=1)

        h0_dec = nn.Tanh()(self.make_h_dec(pre_dec))[None]
        c0_dec = self.make_c_dec(pre_dec)[None]

        # lstm decoder
        dec_in = torch.stack([dec_in]*seq_b.size(1),dim=1)
        out_dec, *_ = self.lstm_dec(dec_in,(h0_dec,c0_dec))

        out_kp =[]
        for i in range(out_dec.size(1)):
            kp = self.make_keypoints(out_dec[:,i])
            out_kp.append(kp)

        return torch.stack(out_kp,dim=1), mu, logstd, out_cycle

    def make_cycle(self,e_a,out_latent_dec):

        enc_cycle_in = out_latent_dec - e_a
        params = self.latent_enc(enc_cycle_in)
        mu = params[:, :int(params.size(-1) / 2)]
        logstd = params[:, int(params.size(-1) / 2):]
        return self.reparametrize(mu,logstd)



    def reparametrize(self,mu,logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu


class RNNDecoder(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_layers_lstm,
        n_layers_lin,
        dim_hidden_lstm,
        use_linear,
        dim_hidden_lin,
        rnn_type="lstm",
    ):
        super(RNNDecoder, self).__init__()
        self.rnn_type = rnn_type

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=n_in,
                hidden_size=dim_hidden_lstm,
                num_layers=n_layers_lstm,
                batch_first=True,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=n_in,
                hidden_size=dim_hidden_lstm,
                num_layers=n_layers_lstm,
                batch_first=True,
            )
        else:
            raise ValueError("Unknown recurrent architecture for rnn decoder.")

        self.n_layers_lstm = n_layers_lstm
        self.n_layers_lin = n_layers_lin
        self.dim_hidden_lstm = dim_hidden_lstm
        self.hidden = self.init_hidden(bs=1, device="cpu")

        self.use_linear = use_linear
        self.linear = None
        if self.use_linear:
            # self.linear = nn.Linear(dim_hidden_lstm, n_out)
            n_in = self.dim_hidden_lstm

            # build encoder
            dec = nn.ModuleList()
            for k in range(n_layers_lin):
                enc_tmp = nn.Linear(n_in, dim_hidden_lin)
                torch.nn.init.xavier_uniform_(enc_tmp.weight)
                dec.append(enc_tmp)
                dec.append(nn.ReLU())

                n_in = dim_hidden_lin

            enc_tmp = nn.Linear(n_in, n_out)
            torch.nn.init.xavier_uniform_(enc_tmp.weight)
            dec.append(enc_tmp)

            self.dec = nn.Sequential(*dec)

    def init_hidden(self, bs, device):
        # num_layers x bs x dim_hidden
        if self.rnn_type == "lstm":
            self.hidden = (
                torch.zeros((self.n_layers_lstm, bs, self.dim_hidden_lstm)).to(
                    device
                ),
                torch.zeros((self.n_layers_lstm, bs, self.dim_hidden_lstm)).to(
                    device
                ),
            )
        elif self.rnn_type == "gru":
            self.hidden = torch.zeros(
                (self.n_layers_lstm, bs, self.dim_hidden_lstm)
            ).to(device)

    def set_hidden(self, bs, device, hidden=None, cell=None):
        if self.rnn_type == "lstm":
            if hidden is None and cell is None:
                self.init_hidden(bs, device)
            elif hidden is None:
                self.hidden = (torch.zeros_like(cell), cell)
            elif cell is None:
                self.hidden = (hidden, torch.zeros_like(hidden))
            else:
                self.hidden = (hidden, cell)
        elif self.rnn_type == "gru":
            if hidden is None:
                self.init_hidden(bs, device)
            else:
                self.hidden = hidden

    def forward(self, x):

        out, self.hidden = self.rnn(x, self.hidden)
        if self.use_linear:
            out = self.dec(out.squeeze(dim=1))
        else:
            out = out.squeeze(dim=1)

        return out


class ResidualRNNDecoder(nn.Module):
    def __init__(
        self, n_in_out, n_hidden, rnn_type="lstm", use_nin=False
    ):
        super().__init__()
        self.n_in_out = n_in_out
        self.n_hidden = n_hidden
        self.rnn_type = rnn_type
        self.use_nin = use_nin

        if self.rnn_type == "gru":
            self.rnn = nn.GRUCell(self.n_in_out, self.n_hidden)
        else:
            self.rnn = nn.LSTMCell(self.n_in_out, self.n_hidden)

            self.n_out = nn.Linear(self.n_hidden, self.n_in_out)

        if self.use_nin:
            self.n_in = nn.Linear(self.n_in_out, self.n_in_out)



        self.init_hidden(bs=1, device="cpu")

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(dim=1)
        elif len(x.shape) != 2:
            raise TypeError("invalid shape of tensor.")

        res = x
        if self.use_nin:
            x = self.n_in(x)

        if self.rnn_type == "lstm":
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden[0]
        else:
            self.hidden = self.rnn(x, self.hidden)
            out_rnn = self.hidden

        out = self.n_out(out_rnn)

        return out + res, res

    def init_hidden(self, bs, device):
        # num_layers x bs x dim_hidden

        if self.rnn_type == "lstm":
            self.hidden = (
                torch.zeros((bs, self.n_hidden)).to(device),
                torch.zeros((bs, self.n_hidden)).to(device),
            )
        elif self.rnn_type == "gru":
            self.hidden = torch.zeros((bs, self.n_hidden)).to(device)


    def set_hidden(self, bs, device, hidden=None, cell=None):
        if self.rnn_type == "lstm":
            if hidden is None and cell is None:
                self.init_hidden(bs, device)
            elif hidden is None:
                self.hidden = (torch.zeros_like(cell), cell)
            elif cell is None:
                self.hidden = (hidden, torch.zeros_like(hidden))
            else:
                self.hidden = (hidden, cell)
        elif self.rnn_type == "gru":
            if hidden is None:
                self.init_hidden(bs, device)
            else:
                self.hidden = hidden



class ResidualBehaviorNet(nn.Module):
    def __init__(self, n_kps, **kwargs):
        super().__init__()

        self.dec_type = (
            kwargs["decoder_arch"] if "decoder_arch" in kwargs else "lstm"
        )
        self.use_nin_dec = (
            kwargs["linear_in_decoder"]
            if "linear_in_decoder" in kwargs
            else False
        )
        self.ib = (
            kwargs["information_bottleneck"]
            if "information_bottleneck" in kwargs
            else False
        )
        self.dim_hidden_b = kwargs["dim_hidden_b"]
        # (n_in, n_layers, dim_hidden, use_linear, dim_linear)

        self.b_enc = BEncoder(
            n_kps,
            1,
            self.dim_hidden_b,
            use_linear=False,
            dim_linear=1,
            ib=self.ib,
        )

        self.decoder = ResidualRNNDecoder(
            n_in_out= n_kps,
            n_hidden=self.dim_hidden_b,
            rnn_type=self.dec_type,
            use_nin=self.use_nin_dec,
        )

    def forward(self, x1, x2, len, start_frame=0, sample=False):

        b = self.infer_b(x1, sample)

        xs, cs, zs_gen, _ = self.generate_seq(
            b[0] if self.ib else b, x2, len, start_frame=start_frame
        )
        if self.ib:
            b, mu, logstd, pre = b
            return xs, cs, zs_gen, b, mu, logstd, pre

        return xs, cs, zs_gen, b

    def infer_b(self, s, sample):
        """

        :param s: The input sequence from which b is inferred
        :return:
        """


        self.b_enc.init_hidden(s.shape[0], device=s.get_device())
        outs = self.b_enc(s, sample)
        # if isinstance(outs, tuple) and not self.ib:
        #     outs = outs[0]

        # return only last element as this enc defines a many to one mapping
        return outs

    def generate_seq(self, b, x_pose, len, start_frame):
        xs = []
        cs = []
        zs = []
        # start pose
        x_start = x_pose[:, start_frame]
        x = x_start

        self.decoder.set_hidden(
            b.shape[0], device=b.get_device(), hidden=b, cell=b
        )

        for i in range(len):

            x, v = self.decoder(x)

            xs.append(x)
            # changes are here velocities
            cs.append(v)

        xs = torch.stack(xs, dim=1)
        cs = torch.stack(cs, dim=1)

        return xs, cs, zs, b
