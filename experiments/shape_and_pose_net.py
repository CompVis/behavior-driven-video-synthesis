import torch
from torch import nn
from tqdm.autonotebook import tqdm
from experiments.experiment import Experiment
from data.samplers import PerPersonSampler
from data import get_dataset
from data.human36m import Human36mDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.vunets import VunetAlter, Regressor
from models.imagenet_pretrained import PerceptualVGG
from lib.losses import compute_kl_with_prior, vgg_loss
from lib.utils import (
    linear_var,
    add_summary_writer,
    scale_img,
    n_parameters,
    make_img_grid,
    get_member,
    get_area_sampling_dist,
    parallel_data_prefetch,
)
from lib.metrics import compute_ssim, inception_score
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from torchvision.models import vgg19
from os import path
from math import ceil
import os
from functools import partial
import numpy as np
import wandb
from copy import deepcopy

WANDB_DISABLE_CODE = True

class ShapePoseNet(Experiment):
    def __init__(self, config, dirs, device):
        super().__init__(config, dirs, device)
        batch_size = self.config["training"]["batch_size"]
        print(
            f"Device of experiment is {device}; batch_size training is {batch_size}."
        )
        self.lr = self.config["training"]["lr"]
        self.imax = 0
        self.data_keys = [
            "pose_img",
            "stickman",
            "sample_ids",
            "app_img",
            "keypoints",
        ]

        self.seq_length = 0
        self.global_step = 0

        ########## seed setting ##########
        torch.manual_seed(self.config["general"]["seed"])
        torch.cuda.manual_seed(self.config["general"]["seed"])
        np.random.seed(self.config["general"]["seed"])
        # random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.config["general"]["seed"])
        rng = np.random.RandomState(self.config["general"]["seed"])
        self.flags_to_discard = ["mode"]

        self.gamma = 0

        if self.config["data"]["dataset"] in ["HumanGait", "iPER"]:
            self.config["data"]["crop_app"] = False

        if self.config["data"]["dataset"] not in ["Human3.6m", "HumanEva"]:
            self.config["data"]["keypoint_type"] = "norm_keypoints"

        if self.config["data"]["keypoint_type"] == "keypoints_3d_world":
            self.config["data"]["small_joint_model"] = False




    def __update_gamma(self, avg_kl):
        with torch.no_grad():
            new_gamma = self.gamma - self.config["training"]["gamma_step"] * (self.imax - avg_kl)
            return max(new_gamma, 0)

    def run_training(self):
        if self.config["general"]["restart"]:
            mod_ckpt, op_ckpt = self._load_ckpt("reg_ckpt")
            reg_ckpt, op_reg = self._load_ckpt("regressor")

            # remove gamma from optimizer param groups and set self.gamma
            for g in op_ckpt["param_groups"]:
                self.gamma = g["gamma"]
                del g["gamma"]

        else:
            mod_ckpt = op_ckpt = None
            reg_ckpt = op_reg = None

        dataset, transforms = get_dataset(self.config["data"])
        train_dataset = dataset(
            transforms,
            data_keys=self.data_keys,
            mode="train",
            debug=self.config["general"]["debug"],
            **self.config["data"],
            **self.config["training"]
        )
        # update datakeys, if new ones have been added
        self.data_keys = train_dataset.datakeys
        print(f"Length of train dataset is {len(train_dataset)}")

        # compute sampling distribution
        if self.config["data"]["sampling"] == "full":
            area_distribution = parallel_data_prefetch(
                partial(get_area_sampling_dist, kp_subset=None),
                train_dataset.datadict["keypoints"],
                self.config["data"]["n_data_workers"],
            )
            sampling_distribution = area_distribution / np.sum(
                area_distribution
            )
        elif self.config["data"]["sampling"] == "body":
            area_distribution = parallel_data_prefetch(
                partial(
                    get_area_sampling_dist,
                    kp_subset=train_dataset.joint_model.body,
                ),
                train_dataset.datadict["keypoints"],
                self.config["data"]["n_data_workers"],
            )
            sampling_distribution = area_distribution / np.sum(
                area_distribution
            )

        elif self.config["data"]["sampling"] == "pid":
            upids, counts = np.unique(
                train_dataset.datadict["p_ids"], return_counts=True
            )
            sampling_distribution = np.zeros_like(
                train_dataset.datadict["p_ids"], dtype=np.float
            )
            for pid, n in zip(upids, counts):
                sampling_distribution[
                    train_dataset.datadict["p_ids"] == pid
                ] = (1.0 / n)

            assert np.all(sampling_distribution > 0.0)
            sampling_distribution = sampling_distribution / np.sum(
                sampling_distribution
            )

        else:
            sampling_distribution = None
        sampler = PerPersonSampler(
            train_dataset, sampling_dist=sampling_distribution
        )
        train_loader = DataLoader(
            train_dataset,
            self.config["training"]["batch_size"],
            sampler=sampler,
            drop_last=True,
            num_workers=0 if self.config['general']['debug'] else self.config["data"]["n_data_workers"],
        )
        test_dataset = dataset(
            transforms,
            data_keys=self.data_keys,
            mode="test",
            debug=self.config["general"]["debug"],
            **self.config["data"],
            **self.config["training"]
        )
        print(f"Length of test dataset is {len(test_dataset)}")
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["logging"]["n_test_samples"],
            shuffle=True,
            drop_last=True,
            num_workers=0 if self.config['general']['debug'] else self.config["data"]["n_data_workers"]
        )
        self.test_iterator = iter(test_loader)



        met_loader = DataLoader(
            test_dataset,
            batch_size=self.config["metrics"]["test_batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0 if self.config['general']['debug'] else self.config["data"]["n_data_workers"]
        )

        n_channels_x = (
            3 * len(train_dataset.joint_model.norm_T)
            if self.config["data"]["inplane_normalize"]
            else 3
        )
        if self.config["architecture"]["conv_layer_type"] == "l2":

            def init_fn():
                return self.global_step <= self.config["training"]["n_init_batches"]

            vunet = VunetAlter(
                init_fn=init_fn, n_channels_x=n_channels_x, **self.config["architecture"],**self.config["data"]
            )
        else:
            vunet = VunetAlter(n_channels_x=n_channels_x, **self.config["architecture"],**self.config["data"])

        if mod_ckpt is not None and not self.config["general"]["debug"]:
            vunet.load_state_dict(mod_ckpt)

        if self.parallel:
            vunet = nn.DataParallel(vunet, device_ids=self.all_devices)

        vunet.to(self.all_devices[0])
        n_trainable_params = n_parameters(vunet)
        print(f"Number of trainable params is {n_trainable_params}")
        ssize=self.config["data"]["spatial_size"]
        print(f"Spatial size is {ssize}")

        vgg = vgg19(pretrained=True)
        if self.parallel:
            vgg = nn.DataParallel(vgg, device_ids=self.all_devices)

        vgg.to(self.all_devices[0])
        vgg.eval()

        custom_vgg = PerceptualVGG(vgg,self.config["training"]["vgg_weights"])
        if self.parallel:
            custom_vgg = nn.DataParallel(
                custom_vgg, device_ids=self.all_devices
            )

        custom_vgg.to(self.all_devices[0])

        optimizer = Adam(
            [
                {"params": get_member(vunet, "eu").parameters(), "name": "eu"},
                {"params": get_member(vunet, "ed").parameters(), "name": "ed"},
                {"params": get_member(vunet, "du").parameters(), "name": "du"},
                {"params": get_member(vunet, "dd").parameters(), "name": "dd"},
            ],
            lr=self.config["training"]["lr"],
            betas=self.config["training"]["adam_betas"],
        )

        if op_ckpt is not None and not self.config["general"]["debug"]:
            optimizer.load_state_dict(op_ckpt)
            # note this may not work for different optimizers
            start_it = list(optimizer.state_dict()["state"].values())[-1][
                "step"
            ]
        else:
            start_it = 0

        latent_widths = [
            self.config["data"]["spatial_size"] // (2 ** (vunet.n_scales - i))
            for i in range(self.config["architecture"]["n_latent_scales"], 0, -1)
        ]
        regressor = Regressor(
            len(test_dataset.joint_model.kps_to_use) * 2,
            latent_widths=latent_widths,
            **self.config["architecture"]
        )

        if mod_ckpt is not None and not self.config["general"]["debug"]:
            regressor.load_state_dict(reg_ckpt)
        regressor = regressor.to(self.all_devices[0])
        optimizer_regressor = Adam(regressor.parameters(), lr=0.001)
        if op_reg is not None and not self.config["general"]["debug"]:
            optimizer_regressor.load_state_dict(op_reg)

        print(
            "Number of parameters in regressor",
            sum(p.numel() for p in regressor.parameters()),
        )

        self.global_step = start_it

        if self.config["training"]["end_iteration"] <= start_it:
            raise ValueError(
                "The start iteration is higher or equal than the end iteration. If you want to resume training, adapt end iteration"
            )

        if self.config["general"]["debug"]:
            n_epoch = 1
            n_overall_epoch = 1
        else:
            n_epoch = int(
                ceil(
                    float(self.config["training"]["end_iteration"] - start_it)
                    * self.config["training"]["batch_size"]
                    / len(train_dataset)
                )
            )
            n_overall_epoch = int(
                ceil(
                    float(self.config["training"]["end_iteration"])
                    * self.config["training"]["batch_size"]
                    / len(train_dataset)
                )
            )

        print(f"Starting training for {n_epoch} Epochs!")

        total_steps = n_overall_epoch * len(train_dataset) // self.config["training"]["batch_size"]

        print(f"Overall {total_steps} train steps to take...")

        adjust_lr = partial(
            linear_var,
            start_it=0,
            end_it=total_steps,
            start_val=self.config["training"]["lr"],
            end_val=0,
            clip_min=0,
            clip_max=self.config["training"]["lr"],
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


        adjust_imax = partial(
            linear_var,
            start_it=0,
            end_it=total_steps,
            start_val=start_val_imax,
            end_val=end_val_imax,
            clip_min=min(start_val_imax, end_val_imax),
            clip_max=max(start_val_imax, end_val_imax),
        )

        self.lr = adjust_lr(start_it)
        self.imax = adjust_imax(start_it)
        print(
            f"Learning rate after adjusting it for the first time is {self.lr}"
        )
        print(
            f"Informmation max after adjusting it for the first time is {self.imax}"
        )

        ish36m = isinstance(train_dataset, Human36mDataset)

        self.kl_avg = []
        self.kl_mean = 0
        self.kl_it = 0


        for pg in optimizer.param_groups:
            pg["lr"] = self.lr

        def train_fn(engine, batch):
            vunet.train()

            self.global_step = engine.state.iteration

            if self.parallel:
                imgs = {name: batch[name] for name in train_dataset.datakeys}
            else:
                imgs = {
                    name: batch[name].to(self.device) for name in train_dataset.datakeys
                }
                # app_img = imgs["app_img"]
            target_img = imgs["pose_img"]
            shape_img = imgs["stickman"]
            # keypoints correspond to pose_img
            if not ish36m:
                pose_img = imgs["pose_img_inplane"]
            else:
                pose_img = target_img

            # apply vunet
            with torch.enable_grad():
                out_img, means, logstds, activations = vunet(
                    pose_img, shape_img
                )

            # if self.parallel and weights is not None:
            #     weights = weights.to(self.all_devices[-1])
            likelihood_loss_dict = vgg_loss(
                custom_vgg, target_img, out_img,
            )
            likelihoods = torch.stack(
                [likelihood_loss_dict[key] for key in likelihood_loss_dict],
                dim=0,
            )
            likelihood_loss = self.config["training"]["ll_weight"] * torch.sum(likelihoods)
            kl_loss = compute_kl_with_prior(means, logstds)

            loss = likelihood_loss
            tuning = (
                torch.tensor(1.0, device=self.device)
                if self.config["architecture"]["cvae"]
                else torch.tensor(self.gamma, device=self.device)
            )
            if engine.state.iteration > self.config["training"]["n_init_batches"]:
                loss = (loss + tuning * kl_loss).to(self.device)

            if self.config["training"]["train_regressor"]:
                reg_imgs = imgs["reg_imgs"]
                reg_targets = imgs["reg_targets"]

                for i in range(reg_imgs.shape[1]):
                    with torch.no_grad():
                        _, means, _, _ = vunet.ed(vunet.eu(reg_imgs[:, i]))

                    preds = regressor(means)
                    tgts = reg_targets[:, i].reshape(reg_targets[:, i].shape[0], -1)
                    loss_regressor = torch.norm(
                        (preds - tgts), dim=1
                    ).mean()
                    optimizer_regressor.zero_grad()
                    loss_regressor.backward(retain_graph=True)
                    optimizer_regressor.step()
                # loss_regressor = torch.exp(-4*torch.clamp(loss_regressor, max=0.08))
                loss_reg_model = torch.clamp(loss_regressor, max=1.2)
                loss -= loss_reg_model * self.config["training"]["weight_regressor"]

            log_lr = torch.tensor(self.lr)

            optimizer.zero_grad()
            loss.backward()
            # optimize
            optimizer.step()

            # fixme this won't work with loaded checkpoint
            # keep moving average
            if (engine.state.iteration - start_it) < 100:
                self.kl_avg.append(kl_loss.detach().cpu().numpy())
            else:
                self.kl_avg.pop(0)
                self.kl_avg.append(kl_loss.detach().cpu().numpy())

            self.gamma = self.__update_gamma(kl_loss)

            output_dict = {
                "loss": loss.item(),
                "likelihood_loss": likelihood_loss.item(),
                "kl_loss": kl_loss.item(),
                "learning_rate": log_lr,
                "gamma": self.gamma,
                "imax": self.imax,
                "kl_mean": self.kl_mean,
            }
            if self.config["training"]["train_regressor"]:
                output_dict.update(
                    {
                        "loss_reg": loss_regressor.item(),
                        "weight_regressor": self.config["training"]["weight_regressor"],
                    }
                )

            likelihood_loss_dict = {
                key: likelihood_loss_dict[key].item()
                for key in likelihood_loss_dict
            }
            output_dict.update(likelihood_loss_dict)
            return output_dict

        trainer = Engine(train_fn)

        # checkpointing
        ckpt_handler = ModelCheckpoint(
            self.dirs["ckpt"], "reg_ckpt", n_saved=10, require_empty=False
        )
        save_dict = {
            "model": vunet.module if self.parallel else vunet,
            "optimizer": optimizer,
        }
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=self.config["logging"]["ckpt_steps"]),
            ckpt_handler,
            save_dict,
        )
        ckpt_reg = ModelCheckpoint(
            self.dirs["ckpt"], "regressor", n_saved=10, require_empty=False
        )
        reg_sdict = save_dict = {
            "model": regressor,
            "optimizer": optimizer_regressor,
        }
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=self.config["logging"]["ckpt_steps"]),
            ckpt_reg,
            reg_sdict,
        )

        # tensorboard logs
        tb_dir = self.dirs["log"]
        writer = add_summary_writer(vunet, tb_dir)

        @trainer.on(Events.ITERATION_COMPLETED)
        def adjust_params(engine):
            # update learning rate and kl_loss weight
            it = engine.state.iteration

            self.lr = adjust_lr(it)
            self.imax = adjust_imax(it)
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr
                if "gamma" in pg:
                    pg["gamma"] = self.gamma
                else:
                    pg.update({"gamma": self.gamma})



        pbar = ProgressBar(ascii=True)
        pbar.attach(trainer, output_transform=lambda x: x)

        @trainer.on(Events.ITERATION_COMPLETED(every=self.config["logging"]["log_steps"]))
        def log(engine):

            wandb.log({"iteration": engine.state.iteration})

            it = engine.state.iteration
            data = engine.state.batch
            if self.parallel:
                imgs = {
                    name: data[name] for name in data if name != "sample_ids"
                }
            else:
                imgs = {
                    name: data[name].to(self.device)
                    for name in data
                    if name not in ["sample_ids"]
                }
            app_img = imgs["app_img"]
            shape_img = imgs["stickman"]
            target_img = imgs["pose_img"]
            if not isinstance(train_dataset, Human36mDataset):
                pose_img = imgs["pose_img_inplane"]

            else:
                pose_img = target_img

            vunet.eval()
            # visualize current performance on train set
            with torch.no_grad():
                # here's the difference to vunet: use NO app img but target image --> no person id necessary
                out_img, _, _, _ = vunet(pose_img, shape_img)

                out_img = scale_img(out_img)
                shape_img = scale_img(shape_img)
                target_img = scale_img(target_img)
                pose_img = scale_img(pose_img)

                writer.add_images(
                    "appearance_images",
                    target_img
                    if self.config["data"]["inplane_normalize"]
                    else pose_img[: self.config["logging"]["n_logged_img"]],
                    it,
                )
                writer.add_images(
                    "shape_images", shape_img[: self.config["logging"]["n_logged_img"]], it
                )
                writer.add_images(
                    "target_images", target_img[: self.config["logging"]["n_logged_img"]], it
                )
                writer.add_images(
                    "transferred_images", out_img[: self.config["logging"]["n_logged_img"]], it
                )

                [
                    writer.add_scalar(key, val, it)
                    for key, val in engine.state.output.items()
                ]

            # test
            try:
                batch = next(self.test_iterator)
            except StopIteration:
                self.test_iterator = iter(test_loader)
                batch = next(self.test_iterator)

            if self.parallel:
                imgs = {name: batch[name] for name in self.data_keys}
            else:
                imgs = {
                    name: batch[name].to(self.device) for name in self.data_keys
                }

            app_img = imgs["app_img"]
            timg = imgs["pose_img"]
            shape_img = imgs["stickman"]
            if self.config["data"]["inplane_normalize"]:
                pose_img = imgs["pose_img_inplane"]
                pose_ids = np.squeeze(imgs["sample_ids"].cpu().numpy()).tolist()
                app_img_disp = test_dataset._get_app_img(
                    ids=pose_ids, inplane_norm=False
                ).squeeze()
            else:
                pose_img = timg
                app_img_disp = app_img

            with torch.no_grad():
                # test reconstruction
                rec_img, _, _, _ = vunet(pose_img, shape_img)
                rec_img = scale_img(rec_img)

                # test appearance transfer
                if self.parallel:
                    tr_img = vunet.module.transfer(
                        app_img.to(self.device), shape_img.to(self.device)
                    )
                else:
                    tr_img = vunet.transfer(app_img, shape_img)
                tr_img = scale_img(tr_img)

                # test sampling mode
                if self.parallel:
                    sampled = vunet.module.test_forward(
                        shape_img.to(self.device)
                    )
                else:
                    sampled = vunet.test_forward(shape_img)
                sampled = scale_img(sampled)

            # scale also imputs
            app_img = scale_img(app_img_disp)
            timg = scale_img(timg)
            shape_img = scale_img(shape_img)

            writer.add_images(
                "test-reconstruct",
                make_img_grid(
                    [
                        timg.to(self.device),
                        shape_img.to(self.device),
                        rec_img.to(self.device),
                    ]
                ),
                it,
            )
            writer.add_images(
                "test-transfer",
                make_img_grid(
                    [
                        app_img.to(self.device),
                        shape_img.to(self.device),
                        tr_img.to(self.device),
                    ]
                ),
                it,
            )
            writer.add_images(
                "test-sample",
                make_img_grid(
                    [shape_img.to(self.device), sampled.to(self.device)]
                ),
            )

        infer_dir = path.join(self.dirs["generated"], "test_inference")
        if not path.isdir(infer_dir):
            os.makedirs(infer_dir)

        @trainer.on(Events.ITERATION_COMPLETED)
        def compute_eval_metrics(engine):
            # computes evaluation metrics and saves checkpoints
            if (engine.state.iteration + 1) % self.config["metrics"]["n_it_metrics"] == 0:
                # compute metrics
                vunet.eval()
                tr_imgs = []
                rec_imgs = []
                n_samples = 40 if self.config['general']['debug'] else self.config["metrics"]["max_n_samples"]
                for i, batch in enumerate(
                    tqdm(
                        met_loader,
                        total=n_samples
                        // met_loader.batch_size,
                        desc=f"Synthesizing {n_samples} images for IS computation."
                    )
                ):
                    if i * met_loader.batch_size >= n_samples:
                        break
                    if self.parallel:
                        imgs = {name: batch[name] for name in self.data_keys}
                    else:
                        imgs = {
                            name: batch[name].to(self.device)
                            for name in self.data_keys
                        }

                    app_img = imgs["app_img"]
                    timg = (
                        imgs["pose_img_inplane"]
                        if self.config["data"]["inplane_normalize"]
                        else imgs["pose_img"]
                    )
                    shape_img = imgs["stickman"]

                    with torch.no_grad():
                        rec_img, _, _, _ = vunet(timg, shape_img)
                        tr_img = vunet.transfer(app_img, shape_img)

                    tr_img_cp = deepcopy(tr_img)
                    rec_img_cp = deepcopy(rec_img)
                    rec_imgs.append(rec_img_cp.detach().cpu())
                    tr_imgs.append(tr_img_cp.detach().cpu())

                    del rec_img
                    del timg
                    del shape_img
                    del app_img
                    del tr_img

                tr_imgs = torch.cat(tr_imgs, dim=0)
                rec_imgs = torch.cat(rec_imgs, dim=0)

                tr_dataset = torch.utils.data.TensorDataset(tr_imgs)
                rec_dataset = torch.utils.data.TensorDataset(rec_imgs)

                is_rec, std_rec = inception_score(
                    rec_dataset,
                    self.device,
                    resize=True,
                    batch_size=self.config["metrics"]["test_batch_size"],
                    debug=self.config['general']['debug']
                )
                is_tr, std_tr = inception_score(
                    tr_dataset,
                    self.device,
                    batch_size=self.config["metrics"]["test_batch_size"],
                    resize=True,
                    debug=self.config['general']['debug']
                )

                ssim = compute_ssim(
                    vunet,
                    self.all_devices,
                    data_keys=self.data_keys,
                    debug=self.config["general"]["debug"],
                    **self.config["data"],
                    **self.config["training"],
                    **self.config["metrics"]
                )

                # add to tensorboard
                it = engine.state.iteration
                # writer.add_scalar("fid", fid, it)
                writer.add_scalar("ssim", ssim, it)
                writer.add_scalar("is_recon", is_rec, it)
                writer.add_scalar("is_transfer", is_tr, it)
                writer.add_scalar("std_is_recon", std_rec, it)
                writer.add_scalar("std_is_transfer", std_tr, it)

                # save checkpoint to separate dir which contains the checkpoints based on metrics
                save_dir = path.join(self.dirs["ckpt"], "epoch_ckpts")
                os.makedirs(save_dir, exist_ok=True)

                torch.save(
                    vunet.state_dict(),
                    path.join(
                        save_dir,
                        f"model@e{engine.state.epoch}@ssim={ssim}-is={is_rec}.pth",
                    ),
                )
                torch.save(
                    optimizer.state_dict(),
                    path.join(
                        save_dir,
                        f"opt@e{engine.state.epoch}@ssim={ssim}-is={is_rec}.pth",
                    ),
                )

        @trainer.on(Events.STARTED)
        def set_start_it(engine):
            engine.state.iteration = start_it + 1
            print(f"Engine starting from iteration #{engine.state.iteration}.")

        @trainer.on(Events.ITERATION_STARTED)
        def stop(engine):
            it = engine.state.iteration
            if it >= self.config["training"]["end_iteration"]:
                print(
                    f"Current iteration is {it}: Training terminating after this iteration."
                )
                engine.terminate()

        trainer.run(train_loader, max_epochs=n_epoch)

    def run_inference(self):
        import cv2
        from matplotlib import pyplot as plt
        from os import makedirs
        import time
        from lib.utils import bounding_box_batch

        # rng = np.random.seed(int(time.time()))


        model_ckpt, _ = self._load_ckpt("reg_ckpt",only_model=True)


        dataset, image_transforms = get_dataset(self.config["data"])
        print(f'data keys are "{self.data_keys}"')
        dkeys = self.data_keys + ["img_size", "paired_keypoints"]
        test_dataset = dataset(
            image_transforms,
            data_keys=dkeys,
            mode="test",
            **self.config["data"],
            **self.config["training"],
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["metrics"]["test_batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=0 #self.config["data"]["n_data_workers"]  # self.config["data"]["n_data_workers"],
        )
        n_channels_x = (
            3 * len(test_dataset.joint_model.norm_T)
            if self.config["data"]["inplane_normalize"]
            else 3
        )

        vunet = VunetAlter(n_channels_x=n_channels_x, **self.config["architecture"],**self.config["data"])

        if model_ckpt is not None:
            vunet.load_state_dict(model_ckpt)
        else:
            raise FileNotFoundError("no ckpt found for inference")

        vunet.to(self.all_devices[0])
        n_trainable_params = n_parameters(vunet)
        print(f"Number of trainable params is {n_trainable_params}")
        ssize = self.config["data"]["spatial_size"]
        print(f"Spatial size is {ssize}")

        vunet.eval()

        ssim = compute_ssim(
            vunet,
            self.all_devices,
            data_keys=self.data_keys,
            debug=False,
            **self.config["data"],
            **self.config["training"],
            **self.config["metrics"]
        )

        print(f"SSIM for self reconstruction is: {ssim}")


        tr_imgs = []
        rec_imgs = []
        for i, batch in enumerate(tqdm(test_loader,
                                       total=self.config["metrics"]["max_n_samples"] // test_loader.batch_size)):
            if i * test_loader.batch_size >= self.config["metrics"]["max_n_samples"]:
                break
            if self.parallel:
                imgs = {name: batch[name] for name in dkeys}
            else:
                imgs = {
                    name: batch[name].to(self.device) for name in
                    dkeys
                }

            app_img = imgs["app_img"]
            timg = imgs["pose_img_inplane"] if self.config["data"]["inplane_normalize"] else \
            imgs["pose_img"]
            shape_img = imgs["stickman"]
            kps_rec = imgs["keypoints"]
            kps_tr = imgs["paired_keypoints"]


            sizes = imgs["img_size"]

            kps_rec = kps_rec * self.config["data"]["spatial_size"]
            kps_tr = kps_tr * self.config["data"]["spatial_size"]


            with torch.no_grad():
                rec_img, _, _, _ = vunet(timg, shape_img)
                tr_img = vunet.transfer(app_img, shape_img)

            # crop to bounding boxes
            sizes = torch.full_like(sizes,self.config["data"]["spatial_size"])
            tr_img = bounding_box_batch(kps_rec,tr_img,sizes,self.config["data"]["spatial_size"])
            rec_img = bounding_box_batch(kps_rec,rec_img,sizes,self.config["data"]["spatial_size"])


            tr_img_cp = deepcopy(tr_img)
            rec_img_cp = deepcopy(rec_img)
            rec_imgs.append(rec_img_cp.detach().cpu())
            tr_imgs.append(tr_img_cp.detach().cpu())

            del rec_img
            del timg
            del shape_img
            del app_img
            del tr_img

        tr_imgs = torch.cat(tr_imgs, dim=0)
        rec_imgs = torch.cat(rec_imgs, dim=0)

        tr_dataset = torch.utils.data.TensorDataset(tr_imgs)
        rec_dataset = torch.utils.data.TensorDataset(rec_imgs)

        is_rec, std_rec = inception_score(rec_dataset, self.device, resize=True,
                                          batch_size=self.config["metrics"]["test_batch_size"])
        is_tr, std_tr = inception_score(tr_dataset, self.device,
                                        batch_size=self.config["metrics"]["test_batch_size"],
                                        resize=True)

        print(f"computed IS for self reconstruction on masks: Value is: {is_rec}")
        print(f"computed IS for transfer on masks: Value is: {is_tr}")


        latent_widths = [
            self.config["data"]["spatial_size"] // (2 ** (vunet.n_scales - i))
            for i in range(self.config["architecture"]["n_latent_scales"], 0, -1)
        ]
        regressor = Regressor(
            len(test_dataset.joint_model.kps_to_use) * 2,
            latent_widths=latent_widths,
            **self.config["architecture"]
        ).to(self.device)
        optimizer_regressor = Adam(regressor.parameters(), lr=0.001)
        print(
            "Number of parameters in regressor",
            sum(p.numel() for p in regressor.parameters()),
        )

        vunet.eval()
        loss_reg = []
        n_epoch = 20  # self.config["training"]["end_iteration"] // len(test_loader)
        print(f"Train regressor for {n_epoch} epochs.")
        it = 0

        for epoch in range(n_epoch):
            data_iterator = tqdm(
                enumerate(test_loader), desc=f"Train regressor, epoch: {epoch}"
            )
            for i, batch in data_iterator:
                if self.parallel:
                    imgs = {name: batch[name] for name in test_dataset.datakeys}
                else:
                    imgs = {
                        name: batch[name].to(self.device)
                        for name in test_dataset.datakeys
                    }
                    # app_img = imgs["app_img"]
                target_img = imgs["pose_img"]
                shape_img = imgs["stickman"]
                reg_targets = imgs["keypoints"]
                reg_targets = reg_targets.reshape(self.config["metrics"]["test_batch_size"], -1)

                if not isinstance(test_dataset, Human36mDataset):
                    pose_img = imgs["pose_img_inplane"]
                else:
                    pose_img = target_img

                # apply vunet
                with torch.no_grad():
                    out_img, means, logstds, activations = vunet(
                        pose_img, shape_img
                    )

                preds = regressor(means)
                loss_regressor = torch.mean(
                    torch.norm(preds - reg_targets, dim=1)
                )
                optimizer_regressor.zero_grad()
                loss_regressor.backward()
                optimizer_regressor.step()

                loss_reg.append(loss_regressor.detach().cpu())
                if (i + 1) % 30 == 0 and len(loss_reg) > 100:
                    update = f"Epoch: {epoch}, Regressor loss:{torch.mean(torch.stack(loss_reg[-100:])).item()}:"
                    data_iterator.set_description(update)

                loss_reg[-1] = loss_reg[-1]

        loss_reg = torch.stack(loss_reg, dim=0).numpy()
        x = np.arange(loss_reg.shape[0])

        plt.plot(x, loss_reg)
        plt.xlabel("Train iterations")
        plt.ylabel("Loss")
        plt.title("Loss of regressor from shape latents to pose.")
        plt.ioff()
        plt.savefig(self.dirs["generated"] + "/loss_course_eval.png")
        plt.close()



