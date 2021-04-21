import torch
import numpy as np
import yaml
import wandb

WANDB_DISABLE_CODE = True
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Average
from glob import glob
from os import path
from tqdm.autonotebook import tqdm
import os
from functools import partial

from torch.utils.data.sampler import (
    RandomSampler,
)
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import torch.nn as nn
from torchvision import transforms as tt
from scipy.spatial.distance import pdist

from data import get_dataset
from data.samplers import SequenceSampler
from experiments.experiment import Experiment, GREEN, RED, BLUE, YELLOW, ENDC
from models.pose_behavior_rnn import ResidualBehaviorNet, Classifier, Classifier_action, Classifier_action_beta, Regressor, Regressor_fly
from models.pose_discriminator import Sequence_disc_michael
from models.flow.simple_flow import UnsupervisedTransformer2
from lib.utils import linear_var,prepare_input
from lib.logging import (
    visualize_transfer3d,
    make_hist,
    latent_interpolate,
    make_eval_grid,
    eval_nets,
)
from lib.losses import kl_loss, FlowLoss

class BehaviorNet(Experiment):
    def __init__(self, config, dirs, device):
        super().__init__(config, dirs, device)
        bsize = self.config["training"]["batch_size"]
        print(
            f"Device of experiment is {device}; batch_size training is {bsize}."
        )
        assert config["data"]["dataset"] in ["Human3.6m", "HumanEva"]
        config["training"]["device"]=self.device

        ######### set data params ###############
        self.data_keys = [
            "keypoints",
            "matched_keypoints",
            "paired_keypoints",
            "paired_sample_ids",
            "action",
        ]
        self.imax = 0

        ########## seed setting ##########
        torch.manual_seed(self.config["general"]["seed"])
        torch.cuda.manual_seed(self.config["general"]["seed"])
        np.random.seed(self.config["general"]["seed"])
        # random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.config["general"]["seed"])
        rng = np.random.RandomState(self.config["general"]["seed"])

        self.config["training"]["tau"] = [int(float(v) * self.config["training"]["n_epochs"]) for v in self.config["training"]["tau"]]

        self.gamma = self.config["training"]["gamma_init"]


        if self.config["data"]["dataset"] == "HumanEva":
            assert self.config["data"]["keypoint_type"] == "keypoints"
        else:
            assert self.config["data"]["keypoint_type"] in [
                "keypoints_3d_world",
                "angle_world_expmap",
            ]

        synth_model_params = self.config['logging']['synth_params']
        synth_model_ckpt = glob(path.join('/'.join(synth_model_params.split('/')[:-1]),'*.pth'))[0]

        if path.isfile(synth_model_params) and path.isfile(synth_model_ckpt):

            with open(synth_model_params, "r") as f:
                self.synth_params = yaml.load(f, Loader=yaml.FullLoader)

            self.synth_ckpt = torch.load(
                synth_model_ckpt,
                map_location="cpu",
            )
        else:
            print(RED + f'Warning: No ckpt and hyoerparameters for a pretrained synthesis model found under "{synth_model_ckpt}" and "{synth_model_params}". No videos will be rendered...')
            self.synth_ckpt = None
            self.synth_params = None


        self.kl_in_eps = False
        self.last_kl = np.finfo(float).max

        self.only_flow = self.config["training"]["only_flow"]

        # determine n_epochs for flow training: if training flow immediately after first stage model, then use 5 epochs as in the evaluation protocols.
        self.n_flow_epochs = self.config['training']['n_epochs'] if self.only_flow else 5

        
    def __update_gamma(self, avg_kl):
        with torch.no_grad():
            new_gamma = self.gamma - self.config["training"]["gamma_step"] * (
                self.imax - avg_kl
            )
            return max(new_gamma, 0)

    def _fallback_ckpt(self):
        fallback_name = self.config["architecture"]["dim_hidden_b"]

        print(
            BLUE
            + f"Intending to train only flow but no checkpoint is available. Use default vae checkpoint for given b_hidden_dim which is {fallback_name}."
            + ENDC
        )

        mod_ckpt, op_ckpt = self._load_ckpt(
            f"@b{fallback_name}",
            dir="/" + "/".join(self.dirs["ckpt"].split("/")[:-1]),
        )
        return mod_ckpt, op_ckpt

    def run_training(self):
        def get_loss(seq_pred, seq_target):

            # component-wise recon_loss_weight

            loss_seq_pred = seq_pred
            loss_seq_target = seq_target

            recon_loss = rec_loss(
                loss_seq_pred, loss_seq_target
            )  # skip init pose which is not predicted by model

            # aggregate
            recon_loss_tot = torch.mean(recon_loss)
            recon_loss_per_seq = torch.mean(recon_loss, dim=[0, 2])

            return recon_loss_tot, recon_loss_per_seq

        ########## init wandb ###########
        print(GREEN + "*************** START TRAINING *******************")

        # fixme check if this works
        for (key, val) in self.config.items():
            print(GREEN + f"{key}: {val}")  # print to console
            wandb.config.update({key: val})  # update wandb config

        print(
            GREEN + "**************************************************" + ENDC
        )

        ########## checkpoints ##########
        if self.config["general"]["restart"] or self.only_flow:
            mod_ckpt, op_ckpt = self._load_ckpt("reg_ckpt")
            # flow_ckpt, flow_op_ckpt = self._load_ckpt("flow_ckpt")


        else:
            mod_ckpt = op_ckpt = None

        flow_ckpt = flow_op_ckpt = None

        if self.only_flow and mod_ckpt is None and op_ckpt is None:
            mod_ckpt, op_ckpt = self._fallback_ckpt()
            if mod_ckpt is None and op_ckpt is None:
                print(
                    YELLOW
                    + f"Warning: Intending to train only flow but no checkpoint is available. Train VAE from scratch."
                    + ENDC
                )
                self.only_flow = False

        dataset, image_transforms = get_dataset(self.config["data"])
        transforms = tt.Compose([tt.ToTensor()])
        train_dataset = dataset(
            transforms,
            data_keys=self.data_keys,
            mode="train",
            label_transfer=True,
            debug=self.config["general"]["debug"],
            crop_app=True,
            **self.config["data"]

        )

        # if seq_length is pruned, use min seq_length, such that the seq_length of test_dataset lower or equal than that of the train dataset
        collect_len = train_dataset.seq_length
        self.collect_recon_loss_seq = {
            k: np.zeros(shape=[k])
            for k in range(collect_len[0], collect_len[-1])
        }
        self.collect_count_seq_lens = np.zeros(shape=[collect_len[-1]])
        # adapt sequence_length
        self.config["data"]["seq_length"] = (
            min(self.config["data"]["seq_length"][0], train_dataset.seq_length[0]),
            min(self.config["data"]["seq_length"][1], train_dataset.seq_length[1]),
        )

        train_sampler = RandomSampler(data_source=train_dataset)

        seq_sampler_train = SequenceSampler(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.config["training"]["batch_size"],
            drop_last=True,
        )

        train_loader = DataLoader(
            train_dataset,
            num_workers=0 if self.config['general']['debug'] else self.config["data"]["n_data_workers"],
            batch_sampler=seq_sampler_train,
        )

        # test data
        t_datakeys = [key for key in self.data_keys] + ["action", "sample_ids","intrinsics",
                                                        "intrinsics_paired",
                                                        "extrinsics",
                                                        "extrinsics_paired",]
        test_dataset = dataset(
            image_transforms,
            data_keys=t_datakeys,
            mode="test",
            debug=self.config["general"]["debug"],
            label_transfer=True,
            **self.config["data"]
        )
        assert (
                test_dataset.action_id_to_action is not None
        )
        rand_sampler_test = RandomSampler(data_source=test_dataset)
        seq_sampler_test = SequenceSampler(
            test_dataset,
            rand_sampler_test,
            batch_size=self.config["training"]["batch_size"],
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            num_workers=0 if self.config['general']['debug'] else self.config["data"]["n_data_workers"],
            batch_sampler=seq_sampler_test,
        )

        rand_sampler_transfer = RandomSampler(data_source=test_dataset)
        seq_sampler_transfer = SequenceSampler(
            test_dataset,
            rand_sampler_transfer,
            batch_size=1,
            drop_last=True,
        )
        transfer_loader = DataLoader(
            test_dataset,
            batch_sampler=seq_sampler_transfer,
            num_workers=0 if self.config['general']['debug'] else self.config["data"]["n_data_workers"],
        )

        # compare_dataset = dataset(
        #     transforms,
        #     data_keys=t_datakeys,
        #     mode="train",
        #     label_transfer=True,
        #     debug=self.config["general"]["debug"],
        #     crop_app=True,
        #     **self.config["data"]
        # )
        # if self.only_flow:
        #     rand_sampler_compare = RandomSampler(data_source=compare_dataset)
        #     seq_sampler_compare = SequenceSampler(
        #         compare_dataset,
        #         rand_sampler_compare,
        #         batch_size=self.config["training"]["batch_size"],
        #         drop_last=True,
        #     )
        #     compare_loader = DataLoader(
        #         compare_dataset, batch_sampler=seq_sampler_compare
        #     )

        ## Classifier action
        n_actions = len(train_dataset.action_id_to_action)
        classifier_action = Classifier_action(len(train_dataset.dim_to_use), n_actions, dropout=0, dim=512).to(self.device)
        optimizer_classifier = Adam(classifier_action.parameters(), lr=0.0001, weight_decay=1e-4)
        print("Number of parameters in classifier action", sum(p.numel() for p in classifier_action.parameters()))

        n_actions = len(train_dataset.action_id_to_action)
        # classifier_action2 = Classifier_action(len(train_dataset.dim_to_use), n_actions, dropout=0, dim=512).to(self.device)
        classifier_action2 = Sequence_disc_michael([2, 1, 1, 1], len(train_dataset.dim_to_use), out_dim=n_actions).to(self.device)
        optimizer_classifier2 = Adam(classifier_action2.parameters(), lr=0.0001, weight_decay=1e-5)
        print("Number of parameters in classifier action", sum(p.numel() for p in classifier_action2.parameters()))

        # Classifier beta
        classifier_beta = Classifier_action_beta(self.config["architecture"]["dim_hidden_b"], n_actions).to(self.device)
        optimizer_classifier_beta = Adam(classifier_beta.parameters(), lr=0.001)
        print("Number of parameters in classifier on beta", sum(p.numel() for p in classifier_beta.parameters()))
        # Regressor  
        regressor = Regressor_fly(self.config["architecture"]["dim_hidden_b"], len(train_dataset.dim_to_use)).to(self.device)
        optimizer_regressor = Adam(regressor.parameters(), lr=0.0001)
        print("Number of parameters in regressor", sum(p.numel() for p in regressor.parameters()))

        ########## load network and optimizer ##########
        net = ResidualBehaviorNet(n_kps=len(train_dataset.dim_to_use),
                                  information_bottleneck=True,
                                  **self.config["architecture"]
                                  )

        print(
            "Number of parameters in VAE model",
            sum(p.numel() for p in net.parameters()),
        )
        if self.config["general"]["restart"] or self.only_flow:
            if mod_ckpt is not None:
                print(
                    BLUE
                    + f"***** Initializing VAE from checkpoint! *****"
                    + ENDC
                )
                net.load_state_dict(mod_ckpt)
        net.to(self.device)

        to_optim = [
            {"params": net.b_enc.parameters(), "name": "z_enc"},
            {"params": net.decoder.parameters(), "name": "dec"},
        ]

        optimizer = Adam(
            to_optim, lr=self.config["training"]["lr_init"]  # , weight_decay=self.config["training"]["weight_decay"]
        )
        wandb.watch(net, log="all", log_freq=len(train_loader))
        if self.config["general"]["restart"] or self.only_flow:
            if op_ckpt is not None:
                optimizer.load_state_dict(op_ckpt)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.config["training"]["tau"], gamma=self.config["training"]["gamma"]
        )
        rec_loss = nn.MSELoss(reduction="none")

        if op_ckpt is not None and not self.config["general"]["debug"] and not self.only_flow:
            optimizer.load_state_dict(op_ckpt)
            # note this may not work for different optimizers
            start_it = list(optimizer.state_dict()["state"].values())[-1][
                "step"
            ]
        else:
            start_it = 0




        flow_mid_channels = self.config["architecture"]["dim_hidden_b"] * self.config["architecture"]["flow_mid_channels_factor"]


        latent_flow = UnsupervisedTransformer2(
            flow_in_channels=self.config["architecture"]["dim_hidden_b"],
            n_flows=self.config["architecture"]["n_flows"],
            flow_hidden_depth=self.config["architecture"]["flow_hidden_depth"],
            flow_mid_channels=flow_mid_channels,
        )

        # if flow_ckpt is not None:
        #     print(
        #         BLUE
        #         + f"***** Initializing Flow model from checkpoint! *****"
        #         + ENDC
        #     )
        #     net.load_state_dict(mod_ckpt)

        print(
            "Number of parameters in flow model",
            sum(p.numel() for p in latent_flow.parameters()),
        )

        latent_flow.to(self.device)
        self.flow_lr = self.config["training"]["flow_lr"] * self.config["training"]["batch_size"]

        flow_optimizer = Adam(
            params=[
                {"params": latent_flow.parameters(), "name": "latent_flow"}
            ],
            lr=self.flow_lr,
            betas=(0.5, 0.9),
            weight_decay=self.config["training"]["weight_decay"],
        )
        if flow_op_ckpt is not None:
            flow_optimizer.load_state_dict(flow_op_ckpt)

        flow_loss = FlowLoss()

        ############## DISCRIMINATOR ##############################
        n_kps = len(train_dataset.dim_to_use)

        # make gan loss weights
        print(
            f"len of train_dataset: {len(train_dataset)}, len of train_loader: {len(train_loader)}"
        )


        if self.config["training"]["imax_scaling"] == "ascend":
            # this is as in beta vae
            start_val_imax = 0
            end_val_imax = self.config["training"]["information_max"]
        elif self.config["training"]["imax_scaling"] == "descend":
            start_val_imax = self.config["training"]["information_max"]
            end_val_imax = 0
        else:
            start_val_imax = end_val_imax = self.config["training"]["information_max"]



        # 10 epochs of fine tuning
        total_steps = (self.config["training"]["n_epochs"] - 10) * len(train_loader)

        if not self.only_flow:

            if self.config['general']['restart']:
                if total_steps <= start_it:
                    raise ValueError(f'Start iteration: {start_it} > end iteration: {total_steps} --> Invalid! Aborting!')

                total_steps -= start_it
                self.config['training']['n_epochs'] = self.config['training']['n_epochs'] - int(np.ceil((start_it / len(train_loader))))


            print(f'Train VAE model for {self.config["training"]["n_epochs"]} epochs')




        adjust_imax = partial(
            linear_var,
            start_it=0,
            end_it=total_steps,
            start_val=start_val_imax,
            end_val=end_val_imax,
            clip_min=min(start_val_imax, end_val_imax),
            clip_max=max(start_val_imax, end_val_imax),
        )



        def train_fn(engine, batch):
            # if self.only_flow:
            #     net.eval()
            # else:
            #     net.train()
            # # reference keypoints with label #1
            # kps = batch["keypoints"].to(torch.float)
            #
            # # keypoints for cross label transfer, label #2
            # kps_cross = batch["paired_keypoints"].to(torch.float)
            #
            # p_id = batch["paired_sample_ids"].to(torch.int)
            #
            # seq_b, target_self = prepare_input(kps,self.device)
            # # seq_recon, target_rec = prepare_input(kps_rec)
            # seq_2, target_t = prepare_input(kps_cross,self.device)
            #
            # seq_len = seq_b.shape[1]
            # # reconstruct second sequence with inferred b
            #
            # labels = batch['action'][:, 0] - 2
            #
            # with torch.no_grad() if self.only_flow else torch.enable_grad():
            #     seq_start = seq_b
            #
            #     seq_inp_net = seq_b
            #
            #
            #
            #     xs, cs, _, bs, mu_s, logstd_s, pre_s = net(
            #         seq_inp_net, seq_start, seq_len
            #     )
            #
            #     # compute losses
            #     recon_loss, recon_loss_per_seq = get_loss(xs, target_self)
            #
            #     kl_loss_avg = kl_loss(mu_s, logstd_s)
            #
            #     # this time use the sequence, which was reconstructed as target for the starting pose
            #     seq_start_t = seq_b
            #     xt, ct, _, bt, mu_t, logstd_t, pre_t = net(
            #         seq_2, seq_start_t, seq_len
            #     )
            #
            #     # first loss part
            #     tuning = 1. if self.config["architecture"]["cvae"] else self.gamma
            #     loss = (
            #             self.config["training"]["recon_loss_weight"] * recon_loss
            #             + tuning * kl_loss_avg
            #     )
            #
            #
            #     if self.config["training"]["use_regressor"]:
            #         for i in range(5):
            #             rand_index  = torch.randint(0, seq_len, (1,))
            #             rand_one_hot = torch.nn.functional.one_hot(rand_index.repeat(mu_s.size(0)), num_classes=seq_len).to(self.device)
            #             loss_regressor = torch.mean((regressor(mu_s.detach(), rand_one_hot.float()) - seq_b[:, rand_index].squeeze())**2)
            #             optimizer_regressor.zero_grad()
            #             loss_regressor.backward(retain_graph=i<4)
            #             optimizer_regressor.step()
            #
            #         rand_one_hot = torch.nn.functional.one_hot(rand_index.repeat(mu_s.size(0)), num_classes=seq_len).to(self.device)
            #         loss_regressor = torch.mean((regressor(mu_s, rand_one_hot.float()) - seq_b[:, rand_index].squeeze()) ** 2)
            #
            #         loss -= torch.clamp(loss_regressor, max=0.45) * self.config["training"]["weight_regressor"]
            #         loss -= torch.clamp(loss_regressor, max=0.7) * self.config["training"]["weight_regressor"]
            #
            #     if not self.only_flow and engine.state.epoch < self.config["training"]["n_epochs"] - 10:
            #
            #         optimizer.zero_grad()
            #         loss.backward()
            #         optimizer.step()
            #         # update gamma after step of optimizer
            #         self.gamma = self.__update_gamma(kl_loss_avg)
            #
            # ## Train classifier on action
            # predict = classifier_action(seq_b)[0]
            # loss_classifier_action = nn.CrossEntropyLoss()(predict, labels.to(self.device))
            # optimizer_classifier.zero_grad()
            # loss_classifier_action.backward()
            # optimizer_classifier.step()
            # _, labels_pred = torch.max(nn.Sigmoid()(predict), dim=1)
            # acc_action = torch.sum(labels_pred.cpu() == labels).float()/labels_pred.shape[0]
            #
            # predict = classifier_action2((seq_b[:, 1:]-seq_b[:, :-1]).transpose(1,2))[0]
            # loss_classifier_action2 = nn.CrossEntropyLoss()(predict, labels.to(self.device))
            # optimizer_classifier2.zero_grad()
            # loss_classifier_action2.backward()
            # optimizer_classifier2.step()
            # _, labels_pred = torch.max(nn.Sigmoid()(predict), dim=1)
            # acc_action2 = torch.sum(labels_pred.cpu() == labels).float()/labels_pred.shape[0]
            #
            # ## Train classifier on beta
            # _, _, _, _, mu_s, *_ = net(seq_b, seq_b, seq_len)
            # predict = classifier_beta(mu_s)
            # loss_classifier_action_beta = nn.CrossEntropyLoss()(predict, labels.to(self.device))
            # optimizer_classifier_beta.zero_grad()
            # loss_classifier_action_beta.backward()
            # optimizer_classifier_beta.step()
            # _, labels_pred = torch.max(nn.Sigmoid()(predict), dim=1)
            # acc_action_beta = torch.sum(labels_pred.cpu() == labels).float()/labels_pred.shape[0]
            #
            # out_dict = {}
            # # this is only run if flow training is enabled
            # if self.only_flow:
            #     # train flow to sample from entire support of posterior
            #
            #     gauss, logdet = latent_flow(bs.detach())
            #     f_loss, flow_dict = flow_loss(gauss, logdet)
            #
            #     flow_optimizer.zero_grad()
            #     f_loss.backward()
            #     flow_optimizer.step()
            #
            #     out_dict.update(flow_dict)
            #
            # # add info to out_dict
            # out_dict['loss_classifier_action'] = loss_classifier_action.detach().item()
            # out_dict['acc_classifier_action'] = acc_action.item()
            # out_dict['loss_classifier_action2'] = loss_classifier_action2.detach().item()
            # out_dict['acc_classifier_action2'] = acc_action2.item()
            #
            #
            # # if engine.state.epoch >= self.config["training"]["n_epochs"] - 10:
            # out_dict['loss_classifier_action_beta'] = loss_classifier_action_beta.detach().item()
            # out_dict['acc_action_beta'] = acc_action_beta.item()
            # out_dict["loss"] = loss.detach().item()
            # out_dict["gamma"] = self.gamma
            # out_dict["kl_loss"] = kl_loss_avg.detach().item()
            #
            # out_dict["mu_s"] = torch.mean(mu_s).item()
            # out_dict["logstd_s"] = torch.mean(logstd_s).item()
            # # if self.config["training"]["use_regressor"]:
            # #     out_dict["loss_regressor"] = torch.mean(loss_regressor).item()
            # out_dict["imax"] = self.imax
            # out_dict["loss_recon"] = recon_loss.detach().item()
            # out_dict["loss_per_seq_recon"] = (
            #     recon_loss_per_seq.detach().cpu().numpy()
            # )
            # out_dict["seq_len"] = seq_len
            #
            # return out_dict
            if self.only_flow:
                net.eval()
            else:
                net.train()
                # reference keypoints with label #1
            kps = batch["keypoints"].to(torch.float)

            # keypoints for cross label transfer, label #2
            kps_cross = batch["paired_keypoints"].to(torch.float)

            p_id = batch["paired_sample_ids"].to(torch.int)

            seq_b, target_self = prepare_input(kps, self.device)
            # seq_recon, target_rec = prepare_input(kps_rec)
            seq_2, target_t = prepare_input(kps_cross, self.device)

            seq_len = seq_b.shape[1]
            # reconstruct second sequence with inferred b

            labels = batch['action'][:, 0] - 2

            with torch.no_grad() if self.only_flow else torch.enable_grad():
                seq_start = seq_b

                seq_inp_net = seq_b

                xs, cs, _, bs, mu_s, logstd_s, pre_s = net(
                    seq_inp_net, seq_start, seq_len
                )

                # compute losses
                recon_loss, recon_loss_per_seq = get_loss(xs, target_self)

                kl_loss_avg = kl_loss(mu_s, logstd_s)

                # this time use the sequence, which was reconstructed as target for the starting pose
                seq_start_t = seq_b
                xt, ct, _, bt, mu_t, logstd_t, pre_t = net(
                    seq_2, seq_start_t, seq_len
                )

                # first loss part
                tuning = 1. if self.config["architecture"]["cvae"] else self.gamma
                loss = (
                        self.config["training"]["recon_loss_weight"] * recon_loss
                        + tuning * kl_loss_avg
                )

                out_dict = {}

                if not self.only_flow:

                    if self.config["training"]["use_regressor"] and not self.only_flow:
                        for _ in range(5):
                            rand_index = torch.randint(0, seq_len, (1,))
                            rand_one_hot = torch.nn.functional.one_hot(rand_index.repeat(mu_s.size(0)),
                                                                       num_classes=seq_len).to(self.device)
                            loss_regressor = torch.mean(
                                (regressor(mu_s, rand_one_hot.float()) - seq_b[:, rand_index].squeeze()) ** 2)
                            optimizer_regressor.zero_grad()
                            loss_regressor.backward(retain_graph=True)
                            optimizer_regressor.step()

                        loss -= torch.clamp(loss_regressor, max=0.45) * self.config["training"]["weight_regressor"]
                        loss -= torch.clamp(loss_regressor, max=0.7) * self.config["training"]["weight_regressor"]

                    if engine.state.epoch < self.config["training"]["n_epochs"] - 10:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # update gamma after step of optimizer
                        self.gamma = self.__update_gamma(kl_loss_avg)

                    ## Train classifier on action
                    predict = classifier_action(seq_b)[0]
                    loss_classifier_action = nn.CrossEntropyLoss()(predict, labels.to(self.device))
                    optimizer_classifier.zero_grad()
                    loss_classifier_action.backward()
                    optimizer_classifier.step()
                    _, labels_pred = torch.max(nn.Sigmoid()(predict), dim=1)
                    acc_action = torch.sum(labels_pred.cpu() == labels).float() / labels_pred.shape[0]

                    predict = classifier_action2((seq_b[:, 1:] - seq_b[:, :-1]).transpose(1, 2))[0]
                    loss_classifier_action2 = nn.CrossEntropyLoss()(predict, labels.to(self.device))
                    optimizer_classifier2.zero_grad()
                    loss_classifier_action2.backward()
                    optimizer_classifier2.step()
                    _, labels_pred = torch.max(nn.Sigmoid()(predict), dim=1)
                    acc_action2 = torch.sum(labels_pred.cpu() == labels).float() / labels_pred.shape[0]

                    ## Train classifier on beta
                    _, _, _, _, mu_s, *_ = net(seq_b, seq_b, seq_len)
                    predict = classifier_beta(mu_s)
                    loss_classifier_action_beta = nn.CrossEntropyLoss()(predict, labels.to(self.device))
                    optimizer_classifier_beta.zero_grad()
                    loss_classifier_action_beta.backward()
                    optimizer_classifier_beta.step()
                    _, labels_pred = torch.max(nn.Sigmoid()(predict), dim=1)
                    acc_action_beta = torch.sum(labels_pred.cpu() == labels).float() / labels_pred.shape[0]

                    # add info to out_dict
                    out_dict['loss_classifier_action'] = loss_classifier_action.detach().item()
                    out_dict['acc_classifier_action'] = acc_action.item()
                    out_dict['loss_classifier_action2'] = loss_classifier_action2.detach().item()
                    out_dict['acc_classifier_action2'] = acc_action2.item()

                    # if engine.state.epoch >= self.config["training"]["n_epochs"] - 10:
                    out_dict['loss_classifier_action_beta'] = loss_classifier_action_beta.detach().item()
                    out_dict['acc_action_beta'] = acc_action_beta.item()
                    out_dict["loss"] = loss.detach().item()



            # this is only run if flow training is enabled
            if self.only_flow:
                # train flow to sample from entire support of posterior

                gauss, logdet = latent_flow(bs.detach())
                f_loss, flow_dict = flow_loss(gauss, logdet)

                flow_optimizer.zero_grad()
                f_loss.backward()
                flow_optimizer.step()

                out_dict.update(flow_dict)



            out_dict["mu_s"] = torch.mean(mu_s).item()
            out_dict["logstd_s"] = torch.mean(logstd_s).item()
            # if self.config["training"]["use_regressor"]:
            #     out_dict["loss_regressor"] = torch.mean(loss_regressor).item()
            out_dict["imax"] = self.imax
            out_dict["loss_recon"] = recon_loss.detach().item()
            out_dict["loss_per_seq_recon"] = (
                recon_loss_per_seq.detach().cpu().numpy()
            )
            out_dict["seq_len"] = seq_len
            out_dict["gamma"] = self.gamma
            out_dict["kl_loss"] = kl_loss_avg.detach().item()


            return out_dict

        ##### CREATE TRAINING RUN #####
        trainer = Engine(train_fn)
        pbar = ProgressBar()
        pbar.attach(
            trainer,
            output_transform=lambda x: {
                key: x[key] for key in x if "per_seq" not in key
            },
        )

        # compute averages for all outputs of train function which are specified in the list
        loss_avg = Average(output_transform=lambda x: x["loss"])
        loss_avg.attach(trainer, "loss")
        recon_loss_avg = Average(output_transform=lambda x: x["loss_recon"])
        recon_loss_avg.attach(trainer, "loss_recon")
        kl_loss_avg = Average(output_transform=lambda x: x["kl_loss"])
        kl_loss_avg.attach(trainer, "kl_loss")
        gamma_avg = Average(output_transform=lambda x: x["gamma"])
        gamma_avg.attach(trainer, "gamma")
        mu_s_avg = Average(output_transform=lambda x: x["mu_s"])
        mu_s_avg.attach(trainer, "mu_s")
        logstd_s_avg = Average(output_transform=lambda x: x["logstd_s"])
        logstd_s_avg.attach(trainer, "logstd_s")

        loss_classifier = Average(output_transform=lambda x: x["loss_classifier_action"] if "loss_classifier_action" in x else 0)
        loss_classifier.attach(trainer, "loss_classifier_action")
        acc_classifier = Average(output_transform=lambda x: x["acc_classifier_action"]if "acc_classifier_action" in x else 0)
        acc_classifier.attach(trainer, "acc_classifier_action")

        loss_classifier_action2 = Average(output_transform=lambda x: x["loss_classifier_action2"] if "loss_classifier_action2" in x else 0)
        loss_classifier_action2.attach(trainer, "loss_classifier_action2")
        acc_classifier_action2 = Average(output_transform=lambda x: x["acc_classifier_action2"]if "acc_classifier_action2" in x else 0)
        acc_classifier_action2.attach(trainer, "acc_classifier_action2")

        loss_classifier_action_beta = Average(output_transform=lambda x: x["loss_classifier_action_beta"] if "loss_classifier_action_beta" in x else 0)
        loss_classifier_action_beta.attach(trainer, "loss_classifier_action_beta")
        acc_action_beta = Average(output_transform=lambda x: x["acc_action_beta"] if "acc_action_beta" in x else 0)
        acc_action_beta.attach(trainer, "acc_action_beta")


        loss_regressor = Average(output_transform=lambda x: x["loss_regressor"] if "loss_regressor" in x else 0)
        loss_regressor.attach(trainer, "loss_regressor")

        # loss_avg = Average(output_transform=lambda x: x["loss"])
        # loss_avg.attach(trainer, "loss")

        ##### TRAINING HOOKS ######
        @trainer.on(Events.ITERATION_COMPLETED)
        def collect_training_info(engine):
            it = engine.state.iteration
            self.imax = adjust_imax(it)

            seq_len = engine.state.output["seq_len"]
            self.collect_recon_loss_seq[seq_len] += engine.state.output[
                "loss_per_seq_recon"
            ]
            self.collect_count_seq_lens[seq_len] += self.config["training"]["batch_size"]

        @trainer.on(Events.EPOCH_COMPLETED)
        def update_optimizer_params(engine):
            scheduler.step()

        def log_wandb(engine):

            wandb.log(
                {
                    "epoch": engine.state.epoch,
                    "iteration": engine.state.iteration,
                }
            )

            print(
                f"Logging metrics: Currently, the following metrics are tracked: {list(engine.state.metrics.keys())}"
            )
            for key in engine.state.metrics:
                val = engine.state.metrics[key]
                wandb.log({key + "-epoch-avg": val})
                print(ENDC + f" [metrics] {key}:{val}")


            if self.only_flow:
                loss_avg = engine.state.metrics["flow_loss"]
            else:
                for k in range(collect_len[0], collect_len[1], 5):
                    recon_seq_av = (
                            self.collect_recon_loss_seq[k]
                            / self.collect_count_seq_lens[k]
                    )
                    make_hist(recon_seq_av, k, engine.state.epoch)

                # reset
                self.collect_recon_loss_seq = {
                    k: np.zeros(shape=[k])
                    for k in range(collect_len[0], collect_len[-1])
                }
                self.collect_count_seq_lens = np.zeros(shape=[collect_len[-1]])

                loss_avg = engine.state.metrics["loss"]

            print(GREEN + f"Epoch {engine.state.epoch} summary:")
            print(ENDC + f" [losses] loss overall:{loss_avg}")

        def eval_model(engine):
            eval_nets(
                net,
                test_loader,
                self.device,
                engine.state.epoch,
                20000,
                flow=latent_flow if self.only_flow else None,
                cf_action=classifier_action,
                cf_action_beta=classifier_beta,
                cf_action2=classifier_action2,
                debug=self.config["general"]["debug"]
            )


        def transfer_behavior_test(engine):
            visualize_transfer3d(
                net,
                transfer_loader,
                self.device,
                name="Test-Set: ",
                dirs=self.dirs,
                revert_coord_space=False,
                epoch=engine.state.epoch,
                synth_params=self.synth_params,
                synth_ckpt=self.synth_ckpt,
                flow=latent_flow if self.only_flow else None,
                n_vid_to_generate=self.config["logging"]["n_vid_to_generate"]
            )

        # compare predictions on train and test set
        def eval_grid(engine):
            if self.config["data"]["dataset"] != "HumanEva":
                make_eval_grid(
                    net,
                    transfer_loader,
                    self.device,
                    dirs=self.dirs,
                    revert_coord_space=False,
                    epoch=engine.state.epoch,
                    synth_ckpt=self.synth_ckpt,
                    synth_params=self.synth_params,
                )

        def latent_interpolations(engine):
            latent_interpolate(
                net,
                transfer_loader,
                self.device,
                dirs=self.dirs,
                epoch=engine.state.epoch,
                synth_params=self.synth_params,
                synth_ckpt=self.synth_ckpt,
                n_vid_to_generate=self.config["logging"]["n_vid_to_generate"]
            )

        ckpt_handler_reg = ModelCheckpoint(
            self.dirs["ckpt"], "reg_ckpt", n_saved=100, require_empty=False
        )
        save_dict = {"model": net, "optimizer": optimizer}
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=1), ckpt_handler_reg, save_dict
        )

        flow_ckpt_handler = ModelCheckpoint(
            self.dirs["ckpt"], "flow_ckpt", n_saved=100, require_empty=False
        )

        flow_sd = {"model": latent_flow, "optimizer": flow_optimizer}

        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_wandb)

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, eval_model
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(
                every=5 #10
            ),
            transfer_behavior_test,
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(
                every=5 #10
            ),
            latent_interpolations,
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(
                every=3 #3
            ),
            eval_grid,
        )

        ####### RUN TRAINING ##############
        if not self.only_flow:
            print(
                BLUE
                + "*************** Train VAE *******************"
                + ENDC
            )
            trainer.run(
                train_loader,
                max_epochs=self.config["training"]["n_epochs"],
                epoch_length=10 if self.config["general"]["debug"] else len(train_loader),
            )
            print(
                BLUE
                + "*************** VAE training ends *******************"
                + ENDC
            )

        del trainer

        # enable flow training
        self.only_flow = True

        # prepare flow trainer
        flow_trainer = Engine(train_fn)
        pbar = ProgressBar()
        pbar.attach(
            flow_trainer,
            output_transform=lambda x: {
                key: x[key] for key in x if "per_seq" not in key
            },
        )

        loss_flow_avg = Average(output_transform=lambda x: x["flow_loss"])
        loss_flow_avg.attach(flow_trainer, "flow_loss")
        loss_r_nll_avg = Average(
            output_transform=lambda x: x["reference_nll_loss"]
        )
        loss_r_nll_avg.attach(flow_trainer, "reference_nll_loss")
        nlogdet_loss_avg = Average(
            output_transform=lambda x: x["nlogdet_loss"]
        )
        nlogdet_loss_avg.attach(flow_trainer, "nlogdet_loss")
        loss_nll_avg = Average(output_transform=lambda x: x["nll_loss"])
        loss_nll_avg.attach(flow_trainer, "nll_loss")

        flow_trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.config["logging"]["n_epoch_eval"]), eval_model
        )
        flow_trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.config["logging"]["n_epoch_eval"]), latent_interpolations
        )
        flow_trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=2 * self.config["logging"]["n_epoch_eval"]),
            transfer_behavior_test,
        )
        flow_trainer.add_event_handler(Events.EPOCH_COMPLETED, log_wandb)
        flow_trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.config["logging"]["n_epoch_eval"]),
            flow_ckpt_handler,
            flow_sd,
        )

        @flow_trainer.on(Events.ITERATION_COMPLETED(every=500))
        def log_flow_losses(engine):

            wandb.log(
                {
                    "flow_loss": engine.state.output["flow_loss"],
                    "nlogdet_loss": engine.state.output["nlogdet_loss"],
                    "nll_loss": engine.state.output["nll_loss"],
                    "reference_nll_loss": engine.state.output[
                        "reference_nll_loss"
                    ],
                }
            )

        print(
            YELLOW + "*************** Train Flow *******************" + ENDC
        )
        flow_trainer.run(
            train_loader,
            max_epochs=self.n_flow_epochs,
            epoch_length=10 if self.config["general"]["debug"] else len(train_loader),
        )
        print(
            YELLOW
            + "*************** Flow training ends *******************"
            + ENDC
        )

    def run_inference(self):
        from data.data_conversions_3d import (
            revert_output_format,
        )
        from models.vunets import VunetAlter


        # rng = np.random.seed(int(time.time()))

        save_dir = path.join(self.dirs["generated"], "inference")
        os.makedirs(save_dir, exist_ok=True)
        print(RED, f"+++++++++++++++++++++ save_dir: {save_dir} +++++++++++++++++++++++" ,ENDC)
        # self.config["logging"]["visualization"] = False

        # get checkpoints
        mod_ckpt, _ = self._load_ckpt("reg_ckpt")
        flow_ckpt, _ = self._load_ckpt("flow_ckpt")
            # flow_ckpt = None

        if mod_ckpt is None:
            mod_ckpt, _ = self._fallback_ckpt()

        dataset, image_transforms = get_dataset(self.config["data"])



        t_datakeys = [key for key in self.data_keys] + ["action"] + ["sample_ids"]
        test_dataset = dataset(
            image_transforms,
            data_keys=t_datakeys,
            mode="test",
            crop_app=False,
            label_transfer=True,
            **self.config["data"]
        )


        vunet = None
        if self.synth_ckpt is not None and self.synth_params is not None:
            print("Loading synth model for generation of real images")
            n_channels_x = (
                3 * len(test_dataset.joint_model.norm_T)
                if self.synth_params['data']["inplane_normalize"]
                else 3
            )
            vunet = VunetAlter(
                n_channels_x=n_channels_x, **self.synth_params["architecture"],**self.synth_params["data"]
            )
            vunet.load_state_dict(self.synth_ckpt)
            vunet = vunet.to(self.device)
            vunet.eval()

            assert test_dataset.complete_datadict is not None

        eval_metric = True

        rand_sampler_test = RandomSampler(data_source=test_dataset)
        seq_sampler_test = SequenceSampler(
            test_dataset, rand_sampler_test, batch_size=64,
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, num_workers=1, batch_sampler=seq_sampler_test
        )

        net = ResidualBehaviorNet(n_kps=len(test_dataset.dim_to_use),
        information_bottleneck = True,
        ** self.config["architecture"]
        )

        net.load_state_dict(mod_ckpt)
        net.to(self.device)

        flow_mid_channels = self.config["architecture"]["dim_hidden_b"] * self.config["architecture"]["flow_mid_channels_factor"]
        flow = None

        if flow_ckpt is not None:
            flow = UnsupervisedTransformer2(
                flow_in_channels=self.config["architecture"]["dim_hidden_b"],
                n_flows=self.config["architecture"]["n_flows"],
                flow_hidden_depth=self.config["architecture"]["flow_hidden_depth"],
                flow_mid_channels=flow_mid_channels,
            )

            flow.load_state_dict(flow_ckpt)
            flow.to(self.device)


        betafile = path.join(self.dirs["log"], "betas_trainset.npy")

        ## Metrics arrays
        ADE = []
        ADE_c = []
        FDE_c = []
        FDE = []
        ASD = []
        FSD = []
        distance_mu = []
        recon_std= []
        distance_std = []
        X_prior = []
        X_cross = []
        X_flow = []
        X_orig  = []
        X_embed = []
        recon_mu = []
        X_labels = []
        X_self = []
        num_samples = 0


        data_iter = tqdm(test_loader, desc="Evaluation metrics")
        for i, batch in enumerate(data_iter):
            with torch.no_grad():

                ### Evaluate metrics, ASD ...
                kps1 = batch["keypoints"].to(dtype=torch.float32)
                kps2 = batch["paired_keypoints"].to(dtype=torch.float32)

                x_s, target_s = prepare_input(kps1,self.device)
                x_t, target_t = prepare_input(kps2,self.device)
                kps3, sample_ids_3 = batch["matched_keypoints"]
                x_related, _ = prepare_input(kps3,self.device)

                data_b_s = x_s
                dev = data_b_s.get_device() if data_b_s.get_device() >= 0 else "cpu"

                # eval - reconstr.
                seq_len = data_b_s.size(1)
                l = seq_len

                seq_pred_s, c_s, _, b, mu, logstd, pre = net(
                    data_b_s[:, :l], x_s, seq_len
                )

                # sample new behavior
                _, _, _, sampled_prior, *_ = net(
                    data_b_s, x_s, seq_len, sample=True
                )

                ## Draw n samples from prior and evaluate below

                if eval_metric:
                    skip = 4
                    fsids = [
                        test_loader.dataset._sample_valid_seq_ids(
                            [ids[-1], batch["sample_ids"].shape[1] - 1]
                        )
                        for ids in batch["sample_ids"][::skip].cpu().numpy()
                    ]
                    future_seqs = torch.stack(
                        [
                            torch.tensor(
                                test_loader.dataset._get_keypoints(sids),
                                device=dev,
                            )
                            for sids in fsids
                        ],
                        dim=0,
                    )

                    n_samples = 50
                    seq_samples = []
                    for _ in range(n_samples):
                        if flow:
                            gsamples1 = torch.randn_like(b[::skip])

                            b_fs1 = flow.reverse(gsamples1)

                            b_fs1 = b_fs1.squeeze(dim=-1).squeeze(dim=-1)
                            seq_s, *_ = net.generate_seq(
                                b_fs1, target_s[::skip], len=seq_len, start_frame=target_s.shape[1] - 1
                            )
                        else:
                            seq_s, *_ = net(x_s[::skip, :l], target_s[::skip], seq_len,
                                        sample=True,
                                        start_frame=target_s.shape[1] - 1)
                        dev = (
                            seq_s.get_device() if seq_s.get_device() >= 0 else "cpu"
                        )
                        seq_s = torch.stack(
                            [
                                torch.tensor(
                                    revert_output_format(
                                        s.cpu().numpy(),
                                        test_loader.dataset.data_mean,
                                        test_loader.dataset.data_std,
                                        test_loader.dataset.dim_to_ignore,
                                    ),
                                    device=dev,
                                )
                                for s in seq_s
                            ],
                            dim=0,
                        )
                        seq_samples.append(seq_s)
                    seq_samples = torch.stack(seq_samples, dim=1)
                    seq_gt = torch.stack(
                        [
                            torch.tensor(
                                revert_output_format(
                                    s.cpu().numpy(),
                                    test_loader.dataset.data_mean,
                                    test_loader.dataset.data_std,
                                    test_loader.dataset.dim_to_ignore,
                                ),
                                device=dev,
                            )
                            for s in future_seqs
                        ],
                        dim=0,
                    ).unsqueeze(1)

                    seq_samples = seq_samples.reshape(*seq_samples.shape[:3],len(test_loader.dataset.joint_model.kps_to_use), 3)
                    seq_gt = seq_gt.reshape(*seq_gt.shape[:3],len(test_loader.dataset.joint_model.kps_to_use), 3)[:, :, 1:]

                    # average pairwise distance; average self distance; average final distance
                    for samples in seq_samples:
                        dist_ASD = 0
                        dist_FSD = 0

                        for seq_q in samples:
                            dist = torch.mean(torch.norm((seq_q - samples).reshape(samples.shape[0], seq_len, -1), dim=2), dim=1)
                            dist_ASD += np.sort(dist.cpu().numpy())[1] ## take 2nd value since 1st value is 0 distance with itself
                            dist_f = torch.norm((seq_q[-1] - samples[:, -1]).reshape(samples.shape[0], -1), dim=1)
                            dist_FSD += np.sort(dist_f.cpu().numpy())[1] ## take 2nd value since 1st value is 0 distance with itself

                        ASD.append(dist_ASD.item() / n_samples)
                        FSD.append(dist_FSD.item() / n_samples)

                    # average displacement error
                    ADE.append(torch.mean((torch.min(torch.mean(torch.norm((seq_samples - seq_gt).reshape(seq_gt.shape[0], n_samples, seq_len, -1), dim=3), dim=2), dim=1)[0])).item())
                    # final displacement error
                    FDE.append((torch.mean(torch.min(torch.norm((seq_samples[:, :, -1] - seq_gt[:, :, -1]).reshape(seq_gt.shape[0], n_samples, -1), dim=2), dim=1)[0])).item())


                    if i%10 == 0:
                        update = f"ASD:{np.mean(ASD):.2f}, FSD:{np.mean(FSD):.2f}, ADE:{np.mean(ADE):.2f}, FDE:{np.mean(FDE):.2f}"
                        data_iter.set_description(update)


                if num_samples < 25000:
                    labels = batch["action"][:, 0] - 2
                    seq_cross, _, _, _, mu, *_ = net(data_b_s[:, :l], x_t, seq_len, sample=False)
                    seq_pred_mu_cross, *_ = net.generate_seq(mu, x_t, seq_len, start_frame=0)
                    seq_pred_mu_s, *_ = net.generate_seq(mu,x_s,seq_len,start_frame=0)
                    _, _, _, _, mu2, *_ = net(seq_cross[:, :l], x_t, seq_len, sample=False)
                    _, _, _, _, mu3, *_ = net(x_related[:, :l], x_t, seq_len, sample=False)

                    recon_mu.append(torch.mean(torch.norm(mu-mu2, dim=1)).item())
                    recon_std.append(torch.std(torch.norm(mu-mu2, dim=1)).item())
                    distance_mu.append(torch.mean(torch.norm(mu-mu3, dim=1)).item())
                    distance_std.append(torch.std(torch.norm(mu-mu3, dim=1)).item())
                    samples_prior, *_ = net(x_s, target_s, seq_len, sample=True, start_frame=target_s.shape[1]-1)

                    ## Log metric
                    ADE_c.append(torch.mean(torch.norm((seq_cross-x_s), dim=2)).item())
                    FDE_c.append(torch.mean(torch.norm((seq_cross[:, -1]-x_s[:, -1]), dim=1)).item())
                    ## Accumulate sequences for evalution with classifiers
                    X_prior.append(samples_prior.cpu())
                    X_cross.append(seq_pred_mu_cross.cpu())
                    X_orig.append(x_s.cpu())
                    X_embed.append(mu.cpu())
                    X_self.append(seq_pred_mu_s.cpu())
                    X_labels.append(labels)
                    num_samples += x_s.shape[0]

                    ## Sample also from flow
                    if flow:
                        gsamples = torch.randn_like(b)
                        flow_samples = flow.reverse(gsamples).squeeze(2).squeeze(2)
                        seq_flow, *_ = net.generate_seq(flow_samples, target_s, seq_len, start_frame=target_s.shape[1]-1)
                        X_flow.append(seq_flow.cpu())
                else:
                    break


        ### PRINT RESULTS FROM 3 Characters METRICS ######
        print("ADE cross task {0:.2f} and FDE cross task {1:.2f}".format(np.mean(ADE_c), np.mean(FDE_c)))
        print('MU RECON', np.mean(recon_mu), 'STD RECON', np.mean(recon_std), 'divide:', np.mean(recon_mu)/np.mean(recon_std))
        print('X RECON', np.mean(distance_mu), 'STD X', np.mean(distance_std), 'divide:', np.mean(distance_mu)/np.mean(distance_std))
        # exit()
        ### Train Classifiers on real vs fake task
        # Concatenate data
        X_orig = torch.stack(X_orig, dim=0).reshape(-1, x_s.shape[1], 51)
        X_prior = torch.stack(X_prior, dim=0).reshape(-1, x_s.shape[1], 51)
        X_cross = torch.stack(X_cross, dim=0).reshape(-1, x_s.shape[1], 51)
        X_self = torch.stack(X_self, dim=0).reshape(-1, x_s.shape[1], 51)
        X_embed = torch.stack(X_embed, dim=0).reshape(-1, self.config["architecture"]["dim_hidden_b"])
        X_labels = torch.stack(X_labels, dim=0).reshape(-1)

        if flow:
            X_flow = torch.stack(X_flow, dim=0).reshape(-1, x_s.shape[1], 51)

        bs = 256
        iterations = 2000
        epochs = iterations//(num_samples//bs)
        times = [0, 10, 20, 30, 40, 49]
        for start in times:
            loss1 = []
            loss2 = []
            loss3 = []
            loss_regressor = []
            acc1 = []
            acc2 = []
            acc_flow = []
            acc_self = []
            loss_self = []
            loss_flow = []
            acc_action= []
            DE = []

            # Define classifier on prior samples
            class_real1 = Classifier(51, 1).to(self.device)
            optimizer_classifier_real1 = SGD(class_real1.parameters(), lr=0.001, momentum=0.9)

            # Define classifier on cross samples
            class_real2 = Classifier(51, 1).to(self.device)
            optimizer_classifier_real2 = SGD(class_real2.parameters(), lr=0.001, momentum=0.9)
            print("Number of parameters in classifier", sum(p.numel() for p in class_real2.parameters()))

            if flow:
                # Define classifier on cross samples
                class_flow =  Classifier(51, 1).to(self.device)
                optimizer_classifier_flow = SGD(class_flow.parameters(), lr=0.001, momentum=0.9)

            # Define classifier on prior samples
            class_real_self = Classifier(51, 1).to(self.device)
            optimizer_classifier_real_self = SGD(class_real2.parameters(), lr=0.001, momentum=0.9)

            # Define regressor to reconstruct
            regressor = Regressor(self.config["architecture"]["dim_hidden_b"], 51).to(self.device)
            optimizer_regressor = Adam(regressor.parameters(), lr=0.001)


            # # Define classifier on prior samples
            class_real_self = Classifier(51, 1).to(self.device)
            optimizer_classifier_real_self = SGD(class_real2.parameters(), lr=0.001, momentum=0.9)

            ## Binary Cross entropy loss
            cls_loss = nn.BCEWithLogitsLoss(reduction="mean")

            data_iterator = tqdm(range(epochs), desc="Train classifier", total=epochs)
            for idx in data_iterator:

                for i in range(num_samples//bs):

                    # Select data/batch
                    x_true = X_orig[i*bs:(i+1)*bs].to(self.device)
                    x_s    = X_prior[i*bs:(i+1)*bs].to(self.device)
                    x_c    = X_cross[i*bs:(i+1)*bs].to(self.device)
                    if flow:
                        x_f    = X_flow[i*bs:(i+1)*bs].to(self.device)
                    x_mu   = X_embed[i*bs:(i+1)*bs].to(self.device)
                    x_start = X_orig[i*bs:(i+1)*bs, start].to(self.device)
                    x_self = X_self[i*bs:(i+1)*bs].to(self.device)
                    x_l = X_labels[i*bs:(i+1)*bs].to(self.device)

                    # Train classifier on prior samples
                    predict = class_real1(x_s)
                    target = torch.zeros_like(predict)
                    loss_classifier_gen = cls_loss(predict, target)
                    acc1.append(torch.mean(nn.Sigmoid()(predict)).item())

                    predict = class_real1(x_true)
                    target = torch.ones_like(predict)
                    loss_classifier_gt = cls_loss(predict, target)

                    loss_class_real1 = loss_classifier_gen + loss_classifier_gt
                    loss1.append(loss_class_real1.item())
                    optimizer_classifier_real1.zero_grad()
                    loss_class_real1.backward()
                    optimizer_classifier_real1.step()

                    # Train classifier on cross samples
                    predict = class_real2(x_c)
                    target = torch.zeros_like(predict)
                    loss_classifier_gen = cls_loss(predict, target)
                    acc2.append(torch.mean(nn.Sigmoid()(predict)).item())

                    predict = class_real2(x_true)#
                    target = torch.ones_like(predict)
                    loss_classifier_gt = cls_loss(predict, target)

                    loss_class_real2 = loss_classifier_gen + loss_classifier_gt
                    loss2.append(loss_class_real2.item())
                    optimizer_classifier_real2.zero_grad()
                    loss_class_real2.backward()
                    optimizer_classifier_real2.step()

                    # Train classifier on self reconstructions
                    predict = class_real_self(x_self)
                    target = torch.zeros_like(predict)
                    loss_classifier_gen = cls_loss(predict, target)
                    acc_self.append(torch.mean(nn.Sigmoid()(predict)).item())

                    predict = class_real_self(x_true)
                    target = torch.ones_like(predict)
                    loss_classifier_gt = cls_loss(predict,target)

                    loss_class_real_self = loss_classifier_gen + loss_classifier_gt
                    loss_self.append(loss_class_real_self.item())
                    optimizer_classifier_real_self.zero_grad()
                    loss_class_real_self.backward()
                    optimizer_classifier_real_self.step()

                    if flow:
                        # Train classifier on flow samples
                        predict = class_flow(x_f)
                        target = torch.zeros_like(predict)
                        loss_classifier_gen = cls_loss(predict, target)
                        acc_flow.append(torch.mean(nn.Sigmoid()(predict)).item())

                        predict = class_flow(x_true)
                        target = torch.ones_like(predict)
                        loss_classifier_gt = cls_loss(predict,target)

                        loss_class_real_self = loss_classifier_gen + loss_classifier_gt
                        loss_flow.append(loss_class_real_self.item())
                        optimizer_classifier_flow.zero_grad()
                        loss_class_real_self.backward()
                        optimizer_classifier_flow.step()

                    ## Train regressor
                    predict = regressor(x_mu)
                    loss_regressor_ = torch.mean(torch.norm(predict - x_start, dim=1))
                    optimizer_regressor.zero_grad()
                    loss_regressor_.backward()
                    optimizer_regressor.step()
                    loss_regressor.append(loss_regressor_.item())



                update = "Acc Prior:{0:.2f}, Acc Cross:{1:.2f}, Acc Self: {2:.2f} Loss_regressor:{3:.2f}".format(np.mean(acc1[-20:]), np.mean(acc2[-20:]),np.mean(acc_self[-20:]),np.mean(loss_regressor[-20:]))
                if flow:
                    update = "Acc Prior:{0:.2f}, Acc Cross:{1:.2f}, Acc Self: {2:.2f} Acc Flow: {3:.2f} Loss_regressor:{4:.2f}".format(np.mean(acc1[-20:]), np.mean(acc2[-20:]),np.mean(acc_self[-20:]),np.mean(acc_flow[-20:]),np.mean(loss_regressor[-20:]))
                data_iterator.set_description(update)

            ## FINAL EVALUATION AFTER TRAINING ###
            loss_regressor = []
            acc1 = []
            acc2 = []
            acc_flow = []
            acc_action = []
            acc_self = []
            DE = []

            for i in range(num_samples//bs):

                # Select data/batch
                x_true = X_orig[i*bs:(i+1)*bs].to(self.device)
                x_s    = X_prior[i*bs:(i+1)*bs].to(self.device)
                x_c    = X_cross[i*bs:(i+1)*bs].to(self.device)
                if flow:
                    x_f    = X_flow[i*bs:(i+1)*bs].to(self.device)
                x_mu   = X_embed[i*bs:(i+1)*bs].to(self.device)
                x_start = X_orig[i*bs:(i+1)*bs, start].to(self.device)
                x_self = X_self[i*bs:(i+1)*bs].to(self.device)
                x_l = X_labels[i*bs:(i+1)*bs].to(self.device)

                DE.append(torch.mean(torch.norm(x_c[:, start] - x_start, dim=1)).item())

                # Train classifier on prior samples
                predict = class_real1(x_s)
                target = torch.zeros_like(predict)
                loss_classifier_gen = cls_loss(predict, target)
                acc1.append(torch.mean(nn.Sigmoid()(predict)).item())

                # Train classifier on cross samples
                predict = class_real2(x_c)#
                target = torch.zeros_like(predict)
                loss_classifier_gen = cls_loss(predict, target)
                acc2.append(torch.mean(nn.Sigmoid()(predict)).item())

                # Train classifier on self reconstructions
                predict = class_real_self(x_self)
                target = torch.zeros_like(predict)
                loss_classifier_gen = cls_loss(predict, target)
                acc_self.append(torch.mean(nn.Sigmoid()(predict)).item())

                if flow:
                    # Train classifier on flow samples
                    predict = class_flow(x_f)
                    target = torch.zeros_like(predict)
                    loss_classifier_gen = cls_loss(predict, target)
                    acc_flow.append(torch.mean(nn.Sigmoid()(predict)).item())

                ## Train regressor
                predict = regressor(x_mu)
                loss_regressor_ = torch.mean(torch.norm(predict - x_start, dim=1))
                optimizer_regressor.zero_grad()
                loss_regressor_.backward()
                optimizer_regressor.step()
                loss_regressor.append(loss_regressor_.item())




            update = "Acc Prior:{0:.2f}, Acc Cross:{1:.2f}, Acc Self: {2:.2f} Loss_regressor:{3:.2f} DE:{4:.2f}".format(np.mean(acc1), np.mean(acc2), np.mean(acc_self), np.mean(loss_regressor), np.mean(DE))
            if flow:
                update = "Acc Prior:{0:.2f}, Acc Cross:{1:.2f}, Acc Self: {2:.2f} Acc Flow: {3:.2f} Loss_regressor:{4:.2f}, DE:{5:.2f}".format(np.mean(acc1), np.mean(acc2), np.mean(acc_self), np.mean(acc_flow), np.mean(loss_regressor), np.mean(DE))
            print("FINAL:", update)