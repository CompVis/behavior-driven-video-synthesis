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

from data import get_dataset
from data.samplers import SequenceSampler
from experiments.experiment import Experiment, GREEN, RED, BLUE, YELLOW, ENDC
from models.pose_behavior_rnn import MTVAE, Classifier, Classifier_action, Classifier_action_beta, Regressor, Regressor_fly
from models.pose_discriminator import Sequence_disc_michael

from lib.utils import linear_var, prepare_input
from lib.logging_mtvae import (
    eval_nets,
    visualize_transfer3d
)



def kl_loss(mu, logstd):
    dim = mu.size(-1)
    bs = mu.size(0)
    kl = 0.5 * (-1.0 - logstd + torch.exp(logstd) + mu ** 2)
    kl_sum = torch.sum(kl)
    return kl_sum / (bs * dim)

class MTVAEModel(Experiment):
    def __init__(self, config, dirs, device):
        super().__init__(config, dirs, device)
        bsize = self.config["training"]["batch_size"]
        print(
            f"Device of experiment is {device}; batch_size training is {bsize}."
        )
        assert config["data"]["dataset"] in ["Human3.6m"]
        config["training"]["device"] = self.device

        ######### set data params ###############
        self.data_keys = [
            "keypoints",
            "matched_keypoints",
            "paired_keypoints",
            "paired_sample_ids",
            "action",
        ]

        ########## seed setting ##########
        torch.manual_seed(self.config["general"]["seed"])
        torch.cuda.manual_seed(self.config["general"]["seed"])
        np.random.seed(self.config["general"]["seed"])
        # random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.config["general"]["seed"])
        rng = np.random.RandomState(self.config["general"]["seed"])

        # self.config["training"]["tau"] = [int(float(v) * self.config["training"]["n_epochs"]) for v in self.config["training"]["tau"]]



        assert self.config["data"]["keypoint_type"] in [
            "keypoints_3d_world",
            "angle_world_expmap",
        ]

        if "angle" not in self.config["data"]["keypoint_type"]:
            self.config["data"]["small_joint_model"] = False





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
        if self.config["general"]["restart"]:
            mod_ckpt, op_ckpt = self._load_ckpt("reg_ckpt")
            # flow_ckpt, flow_op_ckpt = self._load_ckpt("flow_ckpt")


        else:
            mod_ckpt = op_ckpt = None



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
        # collect_len = train_dataset.seq_length
        # self.collect_recon_loss_seq = {
        #     k: np.zeros(shape=[k])
        #     for k in range(collect_len[0], collect_len[-1])
        # }
        # self.collect_count_seq_lens = np.zeros(shape=[collect_len[-1]])
        # # adapt sequence_length
        # self.config["data"]["seq_length"] = (
        #     min(self.config["data"]["seq_length"][0], train_dataset.seq_length[0]),
        #     min(self.config["data"]["seq_length"][1], train_dataset.seq_length[1]),
        # )

        train_sampler = RandomSampler(data_source=train_dataset)

        seq_sampler_train = SequenceSampler(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.config["training"]["batch_size"],
            drop_last=True,
        )

        train_loader = DataLoader(
            train_dataset,
            num_workers=0 if self.config["general"]["debug"] else self.config["data"]["n_data_workers"],
            batch_sampler=seq_sampler_train,
        )

        # test data
        t_datakeys = [key for key in self.data_keys] + ["action", "sample_ids", "intrinsics",
                                                        "intrinsics_paired",
                                                        "extrinsics",
                                                        "extrinsics_paired", ]
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
            num_workers=0 if self.config["general"]["debug"] else self.config["data"]["n_data_workers"] ,
            batch_sampler=seq_sampler_test,
        )
        #
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
            num_workers=0 if self.config["general"]["debug"] else self.config["data"]["n_data_workers"],
        )
        #
        # compare_dataset = dataset(
        #     transforms,
        #     data_keys=t_datakeys,
        #     mode="train",
        #     label_transfer=True,
        #     debug=self.config["general"]["debug"],
        #     crop_app=True,
        #     **self.config["data"]
        # )

        ## Classifier action
        # n_actions = len(train_dataset.action_id_to_action)
        # classifier_action = Classifier_action(len(train_dataset.dim_to_use), n_actions, dropout=0, dim=512).to(self.device)
        # optimizer_classifier = Adam(classifier_action.parameters(), lr=0.0001, weight_decay=1e-4)
        # print("Number of parameters in classifier action", sum(p.numel() for p in classifier_action.parameters()))
        #
        n_actions = len(train_dataset.action_id_to_action)
        # # classifier_action2 = Classifier_action(len(train_dataset.dim_to_use), n_actions, dropout=0, dim=512).to(self.device)
        # classifier_action2 = Sequence_disc_michael([2, 1, 1, 1], len(train_dataset.dim_to_use), out_dim=n_actions).to(self.device)
        # optimizer_classifier2 = Adam(classifier_action2.parameters(), lr=0.0001, weight_decay=1e-5)
        # print("Number of parameters in classifier action", sum(p.numel() for p in classifier_action2.parameters()))

        # Classifier beta
        classifier_beta = Classifier_action_beta(512, n_actions).to(self.device)
        optimizer_classifier_beta = Adam(classifier_beta.parameters(), lr=0.001)
        print("Number of parameters in classifier on beta", sum(p.numel() for p in classifier_beta.parameters()))
        # # Regressor
        # regressor = Regressor_fly(self.config["architecture"]["dim_hidden_b"], len(train_dataset.dim_to_use)).to(self.device)
        # optimizer_regressor = Adam(regressor.parameters(), lr=0.0001)
        # print("Number of parameters in regressor", sum(p.numel() for p in regressor.parameters()))

        ########## load network and optimizer ##########
        net = MTVAE(self.config["architecture"],len(train_dataset.dim_to_use),self.device)

        print(
            "Number of parameters in VAE model",
            sum(p.numel() for p in net.parameters()),
        )
        if self.config["general"]["restart"]:
            if mod_ckpt is not None:
                print(
                    BLUE
                    + f"***** Initializing VAE from checkpoint! *****"
                    + ENDC
                )
                net.load_state_dict(mod_ckpt)
        net.to(self.device)

        optimizer = Adam(
            net.parameters(), lr=self.config["training"]["lr_init"], weight_decay=self.config["training"]["weight_decay"]
        )
        wandb.watch(net, log="all", log_freq=len(train_loader))
        if self.config["general"]["restart"]:
            if op_ckpt is not None:
                optimizer.load_state_dict(op_ckpt)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=self.config["training"]["tau"], gamma=self.config["training"]["gamma"]
        # )
        # rec_loss = nn.MSELoss(reduction="none")

        ############## DISCRIMINATOR ##############################
        n_kps = len(train_dataset.dim_to_use)

        # make gan loss weights
        print(
            f"len of train_dataset: {len(train_dataset)}, len of train_loader: {len(train_loader)}"
        )


        # 10 epochs of fine tuning
        total_steps = (self.config["training"]["n_epochs"] - 10) * len(train_loader)

        get_kl_weight = partial(
            linear_var,
            start_it=0,
            end_it=total_steps,
            start_val=1e-5,
            end_val=1,
            clip_min=0,
            clip_max=1,
        )



        def train_fn(engine, batch):
            net.train()
            # reference keypoints with label #1
            kps = batch["keypoints"].to(torch.float).to(self.device)

            # keypoints for cross label transfer, label #2
            kps_cross = batch["paired_keypoints"].to(torch.float).to(self.device)

            p_id = batch["paired_sample_ids"].to(torch.int)
            # reconstruct second sequence with inferred b

            labels = batch['action'][:, 0] - 2

            out_seq, mu, logstd, out_cycle = net(kps,kps_cross)

            ps = torch.randn_like(out_cycle,requires_grad=False)


            cycle_loss = torch.mean(torch.abs(out_cycle-ps))
            kps_loss = torch.mean(torch.abs(out_seq-kps[:,net.div:]))
            l_kl = kl_loss(mu,logstd)

            k_vel = self.config["training"]["k_vel"]
            vel_tgt = kps[:,net.div:net.div+k_vel] - kps[:,net.div-1:net.div+k_vel-1]

            vel_pred = out_seq[:,:k_vel] - torch.cat([kps[:,net.div-1].unsqueeze(1),out_seq[:,:k_vel-1]],dim=1)
            motion_loss = torch.mean(torch.abs(vel_tgt-vel_pred))

            kl_weight = get_kl_weight(engine.state.iteration)

            loss = kps_loss + kl_weight * l_kl + self.config["training"]["weight_motion"] * motion_loss \
                   + self.config["training"]["weight_cycle"] * cycle_loss

            #
            #
            if engine.state.epoch < self.config["training"]["n_epochs"] - 10:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            out_dict = {"loss": loss.detach().item(), "motion_loss": motion_loss.detach().item(),
                        "rec_loss": kps_loss.detach().item(), "cycle_loss": cycle_loss.detach().item(),
                        "kl_loss":l_kl.detach().item(),"kl_weight": kl_weight}
            #
            #
            # ## Train classifier on action
            # predict = classifier_action(seq_b)[0]
            # loss_classifier_action = nn.CrossEntropyLoss()(predict, labels.to(self.device))
            # optimizer_classifier.zero_grad()
            # loss_classifier_action.backward()
            # optimizer_classifier.step()
            # _, labels_pred = torch.max(nn.Sigmoid()(predict), dim=1)
            # acc_action = torch.sum(labels_pred.cpu() == labels).float() / labels_pred.shape[0]
            #
            # predict = classifier_action2((seq_b[:, 1:] - seq_b[:, :-1]).transpose(1, 2))[0]
            # loss_classifier_action2 = nn.CrossEntropyLoss()(predict, labels.to(self.device))
            # optimizer_classifier2.zero_grad()
            # loss_classifier_action2.backward()
            # optimizer_classifier2.step()
            # _, labels_pred = torch.max(nn.Sigmoid()(predict), dim=1)
            # acc_action2 = torch.sum(labels_pred.cpu() == labels).float() / labels_pred.shape[0]
            #
            # ## Train classifier on beta
            # if engine.state.epoch >= self.config["training"]["n_epochs"] - 10:
            net.eval()
            with torch.no_grad():
                _, mu, *_ = net(kps,kps_cross)
            predict = classifier_beta(mu)
            loss_classifier_action_beta = nn.CrossEntropyLoss()(predict, labels.to(self.device))
            optimizer_classifier_beta.zero_grad()
            loss_classifier_action_beta.backward()
            optimizer_classifier_beta.step()
            _, labels_pred = torch.max(nn.Sigmoid()(predict), dim=1)
            acc_action_beta = torch.sum(labels_pred.cpu() == labels).float() / labels_pred.shape[0]
            #
            # out_dict = {}
            # # this is only run if flow training is enable
            #
            # # add info to out_dict
            # out_dict['loss_classifier_action'] = loss_classifier_action.detach().item()
            # out_dict['acc_classifier_action'] = acc_action.item()
            # out_dict['loss_classifier_action2'] = loss_classifier_action2.detach().item()
            # out_dict['acc_classifier_action2'] = acc_action2.item()
            #
            # # if engine.state.epoch >= self.config["training"]["n_epochs"] - 10:
            out_dict['loss_classifier_action_beta'] = loss_classifier_action_beta.detach().item()
            out_dict['acc_action_beta'] = acc_action_beta.item()
            # out_dict["loss"] = loss.detach().item()
            # out_dict["kl_loss"] = kl_loss_avg.detach().item()
            #
            # out_dict["mu_s"] = torch.mean(mu_s).item()
            # out_dict["logstd_s"] = torch.mean(logstd_s).item()
            # # if self.config["training"]["use_regressor"]:
            # #     out_dict["loss_regressor"] = torch.mean(loss_regressor).item()
            # out_dict["loss_recon"] = recon_loss.detach().item()
            # out_dict["loss_per_seq_recon"] = (
            #     recon_loss_per_seq.detach().cpu().numpy()
            # )
            # out_dict["seq_len"] = seq_len
            #
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
        # fixme this can be used to log as soon as losses for mtvae are defined and named
        loss_avg = Average(output_transform=lambda x: x["loss"])
        loss_avg.attach(trainer, "loss")
        recon_loss_avg = Average(output_transform=lambda x: x["rec_loss"])
        recon_loss_avg.attach(trainer, "rec_loss")
        kl_loss_avg = Average(output_transform=lambda x: x["kl_loss"])
        kl_loss_avg.attach(trainer, "kl_loss")
        kl_loss_avg = Average(output_transform=lambda x: x["motion_loss"])
        kl_loss_avg.attach(trainer, "motion_loss")
        kl_loss_avg = Average(output_transform=lambda x: x["cycle_loss"])
        kl_loss_avg.attach(trainer, "cycle_loss")
        # mu_s_avg = Average(output_transform=lambda x: x["mu_s"])
        # mu_s_avg.attach(trainer, "mu_s")
        # logstd_s_avg = Average(output_transform=lambda x: x["logstd_s"])
        # logstd_s_avg.attach(trainer, "logstd_s")
        #
        # loss_classifier = Average(output_transform=lambda x: x["loss_classifier_action"] if "loss_classifier_action" in x else 0)
        # loss_classifier.attach(trainer, "loss_classifier_action")
        # acc_classifier = Average(output_transform=lambda x: x["acc_classifier_action"] if "acc_classifier_action" in x else 0)
        # acc_classifier.attach(trainer, "acc_classifier_action")
        #
        # loss_classifier_action2 = Average(output_transform=lambda x: x["loss_classifier_action2"] if "loss_classifier_action2" in x else 0)
        # loss_classifier_action2.attach(trainer, "loss_classifier_action2")
        # acc_classifier_action2 = Average(output_transform=lambda x: x["acc_classifier_action2"] if "acc_classifier_action2" in x else 0)
        # acc_classifier_action2.attach(trainer, "acc_classifier_action2")
        #
        loss_classifier_action_beta = Average(output_transform=lambda x: x["loss_classifier_action_beta"] if "loss_classifier_action_beta" in x else 0)
        loss_classifier_action_beta.attach(trainer, "loss_classifier_action_beta")
        acc_action_beta = Average(output_transform=lambda x: x["acc_action_beta"] if "acc_action_beta" in x else 0)
        acc_action_beta.attach(trainer, "acc_action_beta")

        # loss_avg = Average(output_transform=lambda x: x["loss"])
        # loss_avg.attach(trainer, "loss")

        ##### TRAINING HOOKS ######
        # @trainer.on(Events.ITERATION_COMPLETED)
        # def collect_training_info(engine):
        #     it = engine.state.iteration
        #
        #     self.collect_recon_loss_seq[seq_len] += engine.state.output[
        #         "loss_per_seq_recon"
        #     ]
        #     self.collect_count_seq_lens[seq_len] += self.config["training"]["batch_size"]

        # @trainer.on(Events.EPOCH_COMPLETED)
        # def update_optimizer_params(engine):
        #     scheduler.step()

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



            # reset
            # self.collect_recon_loss_seq = {
            #     k: np.zeros(shape=[k])
            #     for k in range(collect_len[0], collect_len[-1])
            # }
            # self.collect_count_seq_lens = np.zeros(shape=[collect_len[-1]])

            loss_avg = engine.state.metrics["loss"]

            print(GREEN + f"Epoch {engine.state.epoch} summary:")
            print(ENDC + f" [losses] loss overall:{loss_avg}")

        def eval_model(engine):
            eval_nets(
                net,
                test_loader,
                self.device,
                engine.state.epoch,
                cf_action_beta=classifier_beta,
                debug=self.config["general"]["debug"]
            )
        #
        #
        def transfer_behavior_test(engine):
            visualize_transfer3d(
                net,
                transfer_loader,
                self.device,
                name="Test-Set: ",
                dirs=self.dirs,
                revert_coord_space=False,
                epoch=engine.state.epoch,
                n_vid_to_generate=self.config["logging"]["n_vid_to_generate"]
            )

        # # compare predictions on train and test set
        # def eval_grid(engine):
        #     if self.config["data"]["dataset"] != "HumanEva":
        #         make_eval_grid(
        #             net,
        #             transfer_loader,
        #             self.device,
        #             dirs=self.dirs,
        #             revert_coord_space=False,
        #             epoch=engine.state.epoch,
        #             synth_ckpt=self.synth_ckpt,
        #             synth_params=self.synth_params,
        #         )

        # def latent_interpolations(engine):
        #     latent_interpolate(
        #         net,
        #         transfer_loader,
        #         self.device,
        #         dirs=self.dirs,
        #         epoch=engine.state.epoch,
        #         synth_params=self.synth_params,
        #         synth_ckpt=self.synth_ckpt,
        #         n_vid_to_generate=self.config["logging"]["n_vid_to_generate"]
        #     )

        ckpt_handler_reg = ModelCheckpoint(
            self.dirs["ckpt"], "reg_ckpt", n_saved=100, require_empty=False
        )
        save_dict = {"model": net, "optimizer": optimizer}
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=1), ckpt_handler_reg, save_dict
        )

        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_wandb)

        def log_outputs(engine):
            for key in engine.state.output:
                val = engine.state.output[key]
                wandb.log({key + "-epoch-step": val})

        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=10 if self.config["general"]["debug"] else 1000),log_outputs)
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, eval_model
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(
                every=1 if self.config["general"]["debug"] else 3
            ),
            transfer_behavior_test,
        )
        # trainer.add_event_handler(
        #     Events.EPOCH_COMPLETED(
        #         every=10
        #     ),
        #     latent_interpolations,
        # )
        # trainer.add_event_handler(
        #     Events.EPOCH_COMPLETED(
        #         every=3
        #     ),
        #     eval_grid,
        # )

        ####### RUN TRAINING ##############
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

    def run_inference(self):
        from data.data_conversions_3d import (
            revert_output_format,
        )
        from models.vunets import VunetAlter
        from lib.logging import make_enrollment
        from os import makedirs
        import time
        from lib.figures import make_enrollment_figure
        from lib.logging import prepare_videos, get_synth_input_fix
        from data.data_conversions_3d import project_onto_image_plane
        import cv2
        from lib.logging import visualize_transfer3d
        from lib.figures import nearest_neighbours, latent_interpolate_eval, make_eval_grid, write_video, sample_examples, sample_examples_single


        save_dir = path.join(self.dirs["generated"], "inference")
        os.makedirs(save_dir, exist_ok=True)
        print(RED, f"+++++++++++++++++++++ save_dir: {save_dir} +++++++++++++++++++++++", ENDC)
        # self.config["logging"]["visualization"] = False

        # get checkpoints
        mod_ckpt, _ = self._load_ckpt("reg_ckpt")
        # flow_ckpt = None

        if mod_ckpt is None:
            raise FileNotFoundError("No model ckpt found.")

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

        rand_sampler_test = RandomSampler(data_source=test_dataset)
        seq_sampler_test = SequenceSampler(
            test_dataset, rand_sampler_test, batch_size=256,
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, num_workers=1, batch_sampler=seq_sampler_test
        )

        # load model
        net = MTVAE(self.config["architecture"],n_dim_im=len(test_dataset.dim_to_use),device=self.device)

        net.load_state_dict(mod_ckpt)
        net.to(self.device)


        betafile = path.join(self.dirs["log"], "betas_trainset.npy")

        ## Metrics arrays
        APD = []
        ADE = []
        FDE = []
        ASD = []
        FSD = []
        distance_mu = []
        recon_std = []
        distance_std = []
        X_prior = []
        X_cross = []
        X_orig = []
        X_embed = []
        recon_mu = []
        X_labels = []
        X_self = []
        num_samples = 0

        data_iter = tqdm(test_loader, desc="Evaluation metrics")
        for i, batch in enumerate(data_iter):
            with torch.no_grad():

                kps1 = batch["keypoints"].to(dtype=torch.float32)
                # kps1_change = batch["kp_change"].to(dtype=torch.float32)
                ids1 = batch["sample_ids"][0].cpu().numpy()
                id1 = ids1[0]
                label_id1 = test_loader.dataset.datadict["action"][id1]

                kps2 = batch["paired_keypoints"].to(dtype=torch.float32)
                # kps2_change = batch["paired_change"].to(dtype=torch.float32)
                ids2 = batch["paired_sample_ids"][0].cpu().numpy()
                id2 = ids2[0]
                label_id2 = test_loader.dataset.datadict["action"][id2]

                kps3 = batch["matched_keypoints"][0].to(torch.float32).to(self.device)

                data1 = kps1.to(self.device)
                data2 = kps2.to(self.device)

                x_in = data1
                target_s = x_in[:,net.div:]

                x_tr = data2
                target_t = x_tr[:,net.div]

                x_rel = kps3


                # recon
                seq_pred_s, mu, logstd, _ = net(x_in,x_tr)
                seq_len = seq_pred_s.shape[1]
                assert seq_len == 50

                # sample new behavior
                sampled_prior = net(x_in, x_tr, sample_prior=True)

                ## Draw n samples from prior and evaluate below
                eval_metric = False
                if eval_metric:
                    skip = 4
                    # fsids = [
                    #     test_loader.dataset._sample_valid_seq_ids(
                    #         [ids[-1], batch["sample_ids"].shape[1] - 1]
                    #     )
                    #     for ids in batch["sample_ids"][::skip].cpu().numpy()
                    # ]
                    # future_seqs = torch.stack(
                    #     [
                    #         torch.tensor(
                    #             test_loader.dataset._get_keypoints(sids),
                    #             device=self.device,
                    #         )
                    #         for sids in fsids
                    #     ],
                    #     dim=0,
                    # )[:,net.div:]
                    # in this setting the future sequence is already included
                    future_seqs = target_s[::skip]

                    n_samples = 50
                    seq_samples = []
                    for _ in range(n_samples):

                        seq_s, *_ = net(x_in[::skip], x_tr[::skip],
                                        sample_prior=True)
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
                                    device=self.device,
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

                    seq_samples = seq_samples.reshape(*seq_samples.shape[:3], len(test_loader.dataset.joint_model.kps_to_use), 3)
                    seq_gt = seq_gt.reshape(*seq_gt.shape[:3], len(test_loader.dataset.joint_model.kps_to_use), 3)

                    # average pairwise distance; average self distance; average final distance
                    for samples in seq_samples:
                        dist_APD = 0
                        dist_ASD = 0
                        dist_FSD = 0
                        for seq_q in samples:
                            dist = torch.norm((seq_q - samples).reshape(samples.shape[0], -1), dim=1)
                            dist_APD += torch.sum(dist) / (n_samples - 1)
                            dist = torch.mean(torch.norm((seq_q - samples).reshape(samples.shape[0], seq_len, -1), dim=2), dim=1)
                            dist_ASD += np.sort(dist.cpu().numpy())[1]  ## take 2nd value since 1st value is 0 distance with itself
                            dist_f = torch.norm((seq_q[-1] - samples[:, -1]).reshape(samples.shape[0], -1), dim=1)
                            dist_FSD += np.sort(dist_f.cpu().numpy())[1]  ## take 2nd value since 1st value is 0 distance with itself

                        APD.append(dist_APD.item() / n_samples)
                        ASD.append(dist_ASD.item() / n_samples)
                        FSD.append(dist_FSD.item() / n_samples)

                    # average displacement error
                    ADE.append(torch.mean((torch.min(torch.mean(torch.norm((seq_samples - seq_gt).reshape(seq_gt.shape[0], n_samples, seq_len, -1), dim=3), dim=2), dim=1)[0])).item())
                    # final displacement error
                    FDE.append((torch.mean(torch.min(torch.norm((seq_samples[:, :, -1] - seq_gt[:, :, -1]).reshape(seq_gt.shape[0], n_samples, -1), dim=2), dim=1)[0])).item())

                    if i % 10 == 0:
                        update = "APD:{0:.2f}, ASD:{1:.2f}, FSD:{2:.2f}, ADE:{3:.2f}, FDE:{4:.2f}".format(np.mean(APD), np.mean(ASD), np.mean(FSD), np.mean(ADE), np.mean(FDE))
                        data_iter.set_description(update)



                if num_samples < 25000:
                    labels = batch["action"][:, 0] - 2



                    seq_cross, mu, *_ = net(x_in,x_tr,transfer=True)
                    seq_cross_with_cond = torch.cat([x_tr[:,:net.div],seq_cross],dim=1)
                    # self recon
                    seq_pred_mu_s, *_ = net(x_in,x_tr)


                    _, mu2, *_ = net(seq_cross_with_cond, x_tr)
                    _, mu3, *_ = net(x_rel, x_tr)

                    recon_mu.append(torch.mean(torch.norm(mu - mu2, dim=1)).item())
                    recon_std.append(torch.std(torch.norm(mu - mu2, dim=1)).item())
                    distance_mu.append(torch.mean(torch.norm(mu - mu3, dim=1)).item())
                    distance_std.append(torch.std(torch.norm(mu - mu3, dim=1)).item())
                    samples_prior, *_ = net(x_in, x_tr, sample_prior=True)

                    ## Log metric
                    ## Accumulate sequences for evalution with classifiers
                    X_prior.append(samples_prior.cpu())
                    X_cross.append(seq_cross.cpu())
                    X_orig.append(x_in[:,net.div:].cpu())
                    X_embed.append(mu.cpu())
                    X_self.append(seq_pred_mu_s.cpu())
                    X_labels.append(labels)
                    num_samples += x_in.shape[0]
                else:
                    break

        ### PRINT RESULTS FROM 3 Characters METRICS ######
        print('MU RECON', np.mean(recon_mu), 'STD RECON', np.mean(recon_std), 'divide:', np.mean(recon_mu) / np.mean(recon_std))
        print('X RECON', np.mean(distance_mu), 'STD X', np.mean(distance_std), 'divide:', np.mean(distance_mu) / np.mean(distance_std))
        # breakpoint()
        # exit()
        ### Train Classifiers on real vs fake task
        # Concatenate data
        X_orig = torch.stack(X_orig, dim=0).reshape(-1, seq_len, 51)
        X_prior = torch.stack(X_prior, dim=0).reshape(-1, seq_len, 51)
        X_cross = torch.stack(X_cross, dim=0).reshape(-1, seq_len, 51)
        X_self = torch.stack(X_self, dim=0).reshape(-1, seq_len, 51)
        X_embed = torch.stack(X_embed, dim=0).reshape(-1, 512)
        X_labels = torch.stack(X_labels, dim=0).reshape(-1)


        bs = 256
        iterations = 2000
        epochs = iterations // (num_samples // bs)
        times = [0, 10, 20, 30, 40, 49]
        for start in times:
            loss1 = []
            loss2 = []
            loss_regressor = []
            acc1 = []
            acc2 = []
            acc_self = []
            loss_self = []

            # Define classifier on prior samples
            class_real1 = Classifier(51, 1).to(self.device)
            optimizer_classifier_real1 = SGD(class_real1.parameters(), lr=0.001, momentum=0.9)

            # Define classifier on cross samples
            class_real2 = Classifier(51, 1).to(self.device)
            optimizer_classifier_real2 = SGD(class_real2.parameters(), lr=0.001, momentum=0.9)
            print("Number of parameters in classifier", sum(p.numel() for p in class_real2.parameters()))

            # Define classifier on prior samples
            class_real_self = Classifier(51, 1).to(self.device)
            optimizer_classifier_real_self = SGD(class_real2.parameters(), lr=0.001, momentum=0.9)

            # Define regressor to reconstruct
            regressor = Regressor(512, 51).to(self.device)
            optimizer_regressor = Adam(regressor.parameters(), lr=0.001)

            # # Define classifier on prior samples
            class_real_self = Classifier(51, 1).to(self.device)
            optimizer_classifier_real_self = SGD(class_real2.parameters(), lr=0.001, momentum=0.9)

            ## Binary Cross entropy loss
            cls_loss = nn.BCEWithLogitsLoss(reduction="mean")

            data_iterator = tqdm(range(epochs), desc="Train classifier", total=epochs)
            for idx in data_iterator:

                for i in range(num_samples // bs):

                    # Select data/batch
                    x_true = X_orig[i * bs:(i + 1) * bs].to(self.device)
                    x_s = X_prior[i * bs:(i + 1) * bs].to(self.device)
                    x_c = X_cross[i * bs:(i + 1) * bs].to(self.device)
                    x_mu = X_embed[i * bs:(i + 1) * bs].to(self.device)
                    x_start = X_orig[i * bs:(i + 1) * bs, start].to(self.device)
                    x_self = X_self[i * bs:(i + 1) * bs].to(self.device)
                    x_l = X_labels[i * bs:(i + 1) * bs].to(self.device)

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

                    predict = class_real2(x_true)  #
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
                    loss_classifier_gt = cls_loss(predict, target)

                    loss_class_real_self = loss_classifier_gen + loss_classifier_gt
                    loss_self.append(loss_class_real_self.item())
                    optimizer_classifier_real_self.zero_grad()
                    loss_class_real_self.backward()
                    optimizer_classifier_real_self.step()


                    ## Train regressor
                    predict = regressor(x_mu)
                    loss_regressor_ = torch.mean(torch.norm(predict - x_start, dim=1))
                    optimizer_regressor.zero_grad()
                    loss_regressor_.backward()
                    optimizer_regressor.step()
                    loss_regressor.append(loss_regressor_.item())

                    # # Train action recognition
                    # predict = classifier_action(x_c)
                    # loss_classifier_action = nn.CrossEntropyLoss()(predict, x_l)
                    # optimizer_classifier_action.zero_grad()
                    # loss_classifier_action.backward()
                    # optimizer_classifier_action.step()
                    # labels_pred = torch.topk(nn.Sigmoid()(predict), 2, dim=1)[1]
                    # correct1 = (torch.sum(labels_pred[:, 0] == x_l).float()).item()
                    # correct2 = (torch.sum(labels_pred[:, 1] == x_l).float()).item()
                    # acc_action.append((correct1+correct2)/labels_pred.shape[0])

                    # _, labels_pred = torch.max(nn.Sigmoid()(predict), dim=1)
                    # acc_action.append((torch.sum(labels_pred == x_l).float()/labels_pred.shape[0]).item())

                update = "Acc Prior:{0:.2f}, Acc Cross:{1:.2f}, Acc Self: {2:.2f} Loss_regressor:{3:.2f}".format(np.mean(acc1[-20:]), np.mean(acc2[-20:]), np.mean(acc_self[-20:]),
                                                                                                                 np.mean(loss_regressor[-20:]))
                data_iterator.set_description(update)

            ## FINAL EVALUATION AFTER TRAINING ###
            loss_regressor = []
            acc1 = []
            acc2 = []
            acc_flow = []
            acc_action = []
            acc_self = []
            DE = []

            for i in range(num_samples // bs):

                # Select data/batch
                x_true = X_orig[i * bs:(i + 1) * bs].to(self.device)
                x_s = X_prior[i * bs:(i + 1) * bs].to(self.device)
                x_c = X_cross[i * bs:(i + 1) * bs].to(self.device)
                x_mu = X_embed[i * bs:(i + 1) * bs].to(self.device)
                x_start = X_orig[i * bs:(i + 1) * bs, start].to(self.device)
                x_self = X_self[i * bs:(i + 1) * bs].to(self.device)
                x_l = X_labels[i * bs:(i + 1) * bs].to(self.device)

                DE.append(torch.mean(torch.norm(x_c[:, start] - x_start, dim=1)).item())

                # Train classifier on prior samples
                predict = class_real1(x_s)
                target = torch.zeros_like(predict)
                loss_classifier_gen = cls_loss(predict, target)
                acc1.append(torch.mean(nn.Sigmoid()(predict)).item())

                # Train classifier on cross samples
                predict = class_real2(x_c)  #
                target = torch.zeros_like(predict)
                loss_classifier_gen = cls_loss(predict, target)
                acc2.append(torch.mean(nn.Sigmoid()(predict)).item())

                # Train classifier on self reconstructions
                predict = class_real_self(x_self)
                target = torch.zeros_like(predict)
                loss_classifier_gen = cls_loss(predict, target)
                acc_self.append(torch.mean(nn.Sigmoid()(predict)).item())

                ## Train regressor
                predict = regressor(x_mu)
                loss_regressor_ = torch.mean(torch.norm(predict - x_start, dim=1))
                optimizer_regressor.zero_grad()
                loss_regressor_.backward()
                optimizer_regressor.step()
                loss_regressor.append(loss_regressor_.item())

                # # Train action recognition
                # predict = classifier_action(x_c)
                # labels_pred = torch.topk(nn.Sigmoid()(predict), 2, dim=1)[1]
                # correct1 = (torch.sum(labels_pred[:, 0] == x_l).float()).item()
                # correct2 = (torch.sum(labels_pred[:, 1] == x_l).float()).item()
                # acc_action.append((correct1+correct2)/labels_pred.shape[0])

            update = "Acc Prior:{0:.2f}, Acc Cross:{1:.2f}, Acc Self: {2:.2f} Loss_regressor:{3:.2f} DE:{4:.2f}".format(np.mean(acc1), np.mean(acc2), np.mean(acc_self), np.mean(loss_regressor),
                                                                                                                        np.mean(DE))
            print("FINAL:", update)