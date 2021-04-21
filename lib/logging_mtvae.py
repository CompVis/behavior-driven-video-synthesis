import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2
import umap
from kornia.geometry import transform
import kornia
from scipy.stats import kstest, gaussian_kde
from matplotlib import pyplot as plt
from os import path
from copy import deepcopy
from torch.optim import Adam,SGD

from lib.utils import (
    fig2data,
    text_to_vid,
    scale_img,
    slerp,
    prepare_input
)
from data.base_dataset import BaseDataset
from data.human36m import Human36mDataset
from data.data_conversions_3d import (
    revert_output_format,
    project_onto_image_plane,
    revert_coordinate_space,
)

from lib.logging import prepare_videos,create_video_3d
from models.pose_behavior_rnn import Regressor


def eval_nets(
        net,
        loader,
        device,
        epoch,
        quantitative=True,
        cf_action=None,
        cf_action_beta=None,
        cf_action2=None,
        dim_to_use=51,
        save_dir = None,
        debug=False
):

    net.eval()
    data_iterator = tqdm(loader, desc="Eval", total=len(loader))
    self_recon_eval_av = 0
    recon_eval = nn.MSELoss()

    # collect_b = []
    # collect_ac = []
    # prior_samples = []
    # collect_mu = []
    # collect_pre_stats = []

    # Metrics
    APD = []
    ADE = []
    FDE = []
    ADE_c = []
    FDE_c = []
    CF_cross = []
    CF_action = []
    CF_action_beta = []
    CF_cross_rel = []
    ASD = []
    FSD = []

    CF_cross_L2 = []
    CF_cross_COS = []
    CF_cross_rel_L2 = []
    CF_cross_rel_COS = []

    CF_cross2 = []
    CF_cross2_L2 =[]
    CF_cross2_COS = []
    CF_cross_rel2 = []
    CF_cross2_rel_L2 = []
    CF_cross2_rel_COS = []
    CF_action2 = []


    X_prior = []
    X_cross = []
    X_orig = []
    X_self = []
    X_embed = []
    X_start = []
    X_cross_rel = []
    X_flow = []
    num_samples = 0

    # incrementatest_bsl time step between two frames depends on sequential frame lag
    for batch_nr, batch in enumerate(data_iterator):
        # get data
        kps1 = batch["keypoints"].to(dtype=torch.float32)
        # NOTE: this can be "paired_keypoints" as no label transfer is wished for this dataset,
        # hence, the map_ids of the dataset point to sequences with the same label
        kps2 = batch["paired_keypoints"].to(dtype=torch.float32)
        # kps3, sample_ids_3 = batch["matched_keypoints"]

        actions = batch["action"]

        # assert torch.all(torch.equal(actions, actions[0]))
        # action_name = test_dataset.action_id_to_action[actions[0]]
        action_names = []
        for aid in actions[:, 0].cpu().numpy():
            action_names.append(loader.dataset.action_id_to_action[aid])

        # build inputs
        # input: bs x 1 x n_kps x k
        # x_s, target_s = prepare_input(kps1, device)
        # x_t, target_t = prepare_input(kps2, device)
        # x_related, _ = prepare_input(kps3, device)
        #data_b_s = x_s

        seq_1 = kps1.to(torch.float).to(device)
        seq_2 = kps2.to(torch.float).to(device)

        # actions of related keypoints
        #actions_related = loader.dataset.datadict["action"][sample_ids_3[:, 0].cpu().numpy()]

        dev = seq_1.get_device() if seq_1.get_device() >= 0 else "cpu"

        # eval - reconstr.
        # seq_len = seq_1.size(1)

        with torch.no_grad():
            # self reconstruction
            seq_pred_s, mu, logstd, _ = net(seq_1,seq_2)

            # transfer
            seq_pred_t, *_ = net(seq_1,seq_2,transfer=True)

            target_s = seq_1[:,net.div:]
            # seq_pred_s, c_s, _, b, mu, logstd, pre = net(
            #     data_b_s, x_s, seq_len
            # )
            #
            # seq_pred_mu_s, *_ = net.generate_seq(mu, x_s, seq_len, start_frame=0)
            #
            # # sample new behavior
            # _, _, _, sampled_prior, *_ = net(
            #     data_b_s, x_s, seq_len, sample=True
            # )
            seq_len = seq_pred_s.size(1)



            ## Draw n samples from prior and evaluate below
            if epoch > 99:
                skip = 4
                fsids = [
                    loader.dataset._sample_valid_seq_ids(
                        [ids[-1], batch["sample_ids"].shape[1] - 1]
                    )
                    for ids in batch["sample_ids"][::skip].cpu().numpy()
                ]
                future_seqs = torch.stack(
                    [
                        torch.tensor(
                            loader.dataset._get_keypoints(sids),
                            device=dev,
                        )
                        for sids in fsids
                    ],
                    dim=0,
                )

                n_samples = 50
                seq_samples = []
                for _ in range(n_samples):

                    seq_s, *_ = net(seq_1[::skip],seq_2[::skip],sample_prior=True)
                    # seq_s = torch.cat([seq_s,target_s[::skip,-1].unsqueeze(dim=1)], dim=1)
                    dev = (
                        seq_s.get_device() if seq_s.get_device() >= 0 else "cpu"
                    )
                    seq_s = torch.stack(
                        [
                            torch.tensor(
                                revert_output_format(
                                    s.cpu().numpy(),
                                    loader.dataset.data_mean,
                                    loader.dataset.data_std,
                                    loader.dataset.dim_to_ignore,
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
                                loader.dataset.data_mean,
                                loader.dataset.data_std,
                                loader.dataset.dim_to_ignore,
                            ),
                            device=dev,
                        )
                        for s in future_seqs
                    ],
                    dim=0,
                ).unsqueeze(1)

                seq_samples = seq_samples.reshape(*seq_samples.shape[:3], len(loader.dataset.joint_model.kps_to_use), 3)
                seq_gt = seq_gt.reshape(*seq_gt.shape[:3], len(loader.dataset.joint_model.kps_to_use), 3)[:, :, 1:]

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

                if batch_nr % 10 == 0:
                    update = "APD:{0:.2f}, ASD:{1:.2f}, FSD:{2:.2f}, ADE:{3:.2f}, FDE:{4:.2f}".format(np.mean(APD), np.mean(ASD), np.mean(FSD), np.mean(ADE), np.mean(FDE))
                    data_iterator.set_description(update)

                # labels = batch["action"][:, 0] - 2
                # seq_cross, *_ = net(x_s, x_t, seq_len, sample=False)
                #
                # if cf_action and quantitative:
                #     predict = cf_action(seq_cross)
                #     _, labels_pred = torch.max(predict, 1)
                #     acc_action = (
                #             torch.sum(labels_pred.cpu() == labels).float()
                #             / labels_pred.shape[0]
                #     )
                #     CF_cross.append(acc_action.item())
                #
                #     predict = cf_action(x_s)
                #     _, labels_pred = torch.max(predict, 1)
                #     acc_action = (
                #             torch.sum(labels_pred.cpu() == labels).float()
                #             / labels_pred.shape[0]
                #     )
                #     CF_action.append(acc_action.item())

                if cf_action_beta and quantitative:
                    predict = cf_action_beta(mu)
                    _, labels_pred = torch.max(predict, 1)
                    acc_action_beta = (torch.sum(labels_pred.cpu() == labels).float() / labels_pred.shape[0])
                    CF_action_beta.append(acc_action_beta.item())

            # Evaluate realisticness from prior samples with classifier
            # Evaluate action label and realisticness for cross setting
            labels = batch["action"][:, 0] - 2
            # seq_cross, _, _, _, mu, *_ = net(data_b_s, x_t, seq_len, sample=False)
            # seq_pred_mu_cross, *_ = net.generate_seq(mu, x_t, seq_len, start_frame=0)
            # # mu2 = net.infer_b(seq_cross)
            #
            # labels_related = torch.from_numpy(actions_related).to(device) - 2
            # seq_cross_rel, *_ = net(x_related, x_s, seq_len, sample=False)
            # samples_prior, *_ = net(x_s, target_s, seq_len, sample=True, start_frame=target_s.shape[1] - 1)
            # seq_pred_mu_s, *_ = net.generate_seq(mu, x_s, seq_len, start_frame=0)

            if num_samples < 25000:
                #     ADE_c.append(torch.mean(torch.norm((seq_cross-x_s), dim=2)).item())
                #     FDE_c.append(torch.mean(torch.norm((seq_cross[:, -1]-x_s[:, -1]), dim=1)).item())
                #     X_prior.append(samples_prior.cpu())
                #     X_cross.append(seq_pred_mu_cross.cpu())
                #     X_cross_rel.append(seq_cross_rel.cpu())
                #     X_self.append(seq_pred_mu_s.cpu())
                X_orig.append(seq_1.cpu())
                X_embed.append(mu.cpu())

                num_samples += seq_pred_s.shape[0]
            else:
                break

            # seq_cross = x_related
            # seq_cross_rel = x_related
            if cf_action_beta:
                predict = cf_action_beta(mu)
                _, labels_pred = torch.max(predict, 1)
                acc_action_beta = (torch.sum(labels_pred.cpu() == labels).float() / labels_pred.shape[0])
                CF_action_beta.append(acc_action_beta.item())

            recon_batch_av = recon_eval(seq_pred_s, target_s)
            self_recon_eval_av += recon_batch_av.detach().cpu().numpy()


            if debug and (batch_nr + 1) * seq_pred_s.shape[0] > 1000:
                break

    print("ADE corss task {0:.2f} and FDE cross task {1:.2f}".format(np.mean(ADE_c), np.mean(FDE_c)))
    n_epochs_classifier = 99
    if epoch % n_epochs_classifier == 0:
        ## Train Classifiers on real vs fake task
        X_orig = torch.stack(X_orig, dim=0).reshape(-1, seq_1.shape[1], dim_to_use)
        # X_prior = torch.stack(X_prior, dim=0).reshape(-1, x_s.shape[1], dim_to_use)
        # X_cross = torch.stack(X_cross, dim=0).reshape(-1, x_s.shape[1], dim_to_use)
        # X_cross_rel = torch.stack(X_cross_rel, dim=0).reshape(-1, x_s.shape[1], dim_to_use)
        # X_self = torch.stack(X_self, dim=0).reshape(-1, x_s.shape[1], dim_to_use)
        X_embed = torch.stack(X_embed, dim=0).reshape(-1, net.dim_hidden_b)

        # Define classifiers

        regressor = Regressor(net.dim_hidden_b, dim_to_use).to(device)
        optimizer_regressor = Adam(regressor.parameters(), lr=0.001)

        bs = 256
        iterations = 2000
        epochs = iterations // (num_samples // bs)

        loss1 = []
        loss2 = []
        loss3 = []
        loss_self = []
        loss_cross_rel = []
        loss_regressor = []

        acc1 = []
        acc2 = []
        acc3 = []
        acc_self = []
        acc_cross_rel = []

        data_iterator = tqdm(range(epochs), desc="Train classifier", total=epochs)
        cls_loss = nn.BCEWithLogitsLoss(reduction="mean")
        for _ in data_iterator:

            for i in range(num_samples // bs):
        #
        #         # Select data/batch
        #         x_true = X_orig[i * bs:(i + 1) * bs].to(device)
        #         x_s = X_prior[i * bs:(i + 1) * bs].to(device)
        #         x_c = X_cross[i * bs:(i + 1) * bs].to(device)
        #         x_self = X_self[i * bs:(i + 1) * bs].to(device)

                x_mu = X_embed[i * bs:(i + 1) * bs].to(device)
                # this is the start frame
                x_start = X_orig[i * bs:(i + 1) * bs, net.div-1].to(device)
        #         x_cross_rel = X_cross_rel[i * bs:(i + 1) * bs].to(device)
        #
        #         # Train classifier1 on prior samples
        #         predict = class_real1(x_s)
        #         target = torch.zeros_like(predict)
        #         loss_classifier_gen = cls_loss(predict, target)
        #         acc1.append(torch.mean(nn.Sigmoid()(predict)).item())
        #
        #         predict = class_real1(x_true)
        #         target = torch.ones_like(predict)
        #         loss_classifier_gt = cls_loss(predict, target)
        #
        #         loss_class_real1 = loss_classifier_gen + loss_classifier_gt
        #         loss1.append(loss_class_real1.item())
        #         optimizer_classifier_real1.zero_grad()
        #         loss_class_real1.backward()
        #         optimizer_classifier_real1.step()
        #
        #         # Train classifier2 on cross samples
        #         predict = class_real2(x_c)
        #         target = torch.zeros_like(predict)
        #         loss_classifier_gen = cls_loss(predict, target)
        #         acc2.append(torch.mean(nn.Sigmoid()(predict)).item())
        #
        #         predict = class_real2(x_true)
        #         target = torch.ones_like(predict)
        #         loss_classifier_gt = cls_loss(predict, target)
        #
        #         loss_class_real2 = loss_classifier_gen + loss_classifier_gt
        #         loss2.append(loss_class_real2.item())
        #         optimizer_classifier_real2.zero_grad()
        #         loss_class_real2.backward()
        #         optimizer_classifier_real2.step()
        #
        #         # Train classifier2 on self reconstructions
        #         predict = class_real_self(x_self)
        #         target = torch.zeros_like(predict)
        #         loss_classifier_gen = cls_loss(predict, target)
        #         acc_self.append(torch.mean(nn.Sigmoid()(predict)).item())
        #
        #         predict = class_real_self(x_true)
        #         target = torch.ones_like(predict)
        #         loss_classifier_gt = cls_loss(predict, target)
        #
        #         loss_class_real_self = loss_classifier_gen + loss_classifier_gt
        #         loss_self.append(loss_class_real_self.item())
        #         optimizer_classifier_real_self.zero_grad()
        #         loss_class_real_self.backward()
        #         optimizer_classifier_real_self.step()
        #
        #         ## Train regressor
                predict = regressor(x_mu)
                loss_regressor_ = torch.mean((predict - x_start) ** 2)
                optimizer_regressor.zero_grad()
                loss_regressor_.backward()
                optimizer_regressor.step()
                loss_regressor.append(loss_regressor_.item())
        #
        #         # Train classifier2 on self reconstructions
        #         predict = class_real_cross_rel(x_cross_rel)
        #         target = torch.zeros_like(predict)
        #         loss_classifier_gen = cls_loss(
        #             predict, target)
        #         acc_cross_rel.append(torch.mean(nn.Sigmoid()(predict)).item())
        #
        #         predict = class_real_cross_rel(x_true)
        #         target = torch.ones_like(predict)
        #         loss_classifier_gt = cls_loss(predict, target)
        #
        #         loss_class_cross_rel = loss_classifier_gen + loss_classifier_gt
        #         loss_cross_rel.append(loss_class_real_self.item())
        #         optim_class_cross_rel.zero_grad()
        #         loss_class_cross_rel.backward()
        #         optim_class_cross_rel.step()
        #
        #     update = "Acc Prior:{0:.2f}, Acc Cross:{1:.2f}, Loss_regressor:{2:.2f}".format(acc1[-1], acc2[-1], loss_regressor[-1])
        #     data_iterator.set_description(update)
        #
        # x = np.arange(len(acc1))
        #
        # plt.plot(x, loss_regressor)
        # plt.xlabel('Iterations')
        # plt.ylabel('Loss')
        # plt.title('Loss of regressor trained on embedding')
        # plt.ioff()
        # name = 'Loss of regressor trained on embeddings'
        # wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        # plt.close()
        #
        # plt.plot(x, acc1)
        # plt.xlabel('Iterations')
        # plt.ylabel('Accuracy')
        # plt.title('Accuracy of classifier trained on prior samples')
        # plt.ioff()
        # name = 'Accuracy of classifier trained on prior samples'
        # wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        # plt.close()
        #
        # plt.plot(x, acc2)
        # plt.xlabel('Iterations')
        # plt.ylabel('Accuracy')
        # plt.title('Accuracy of classifier trained on cross task (acc only for cross samples)')
        # plt.ioff()
        # name = 'Accuracy of classifier trained on cross samples (acc only for cross samples)'
        # wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        # plt.close()
        #
        # plt.plot(x, acc_self)
        # plt.xlabel('Iterations')
        # plt.ylabel('Accuracy')
        # plt.title('Accuracy of classifier trained on self recon task')
        # plt.ioff()
        # name = 'Accuracy of classifier trained on self reconstructed sequences'
        # wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        # plt.close()
        #
        # plt.plot(x, acc_cross_rel)
        # plt.xlabel('Iterations')
        # plt.ylabel('Accuracy')
        # plt.title('Accuracy of classifier trained on transfer task with related labels')
        # plt.ioff()
        # name = 'Accuracy of classifier trained on on transfer task with related labels'
        # wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        # plt.close()
        #
        # plt.plot(x, loss1)
        # plt.xlabel('Iterations')
        # plt.ylabel('Loss')
        # name = 'Loss of classifier trained on prior samples'
        # plt.title('Loss of classifier trained on prior samples')
        # plt.ioff()
        # wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        # plt.close()
        #
        # plt.plot(x, loss2)
        # plt.xlabel('Iterations')
        # plt.ylabel('Loss')
        # name = 'Loss of classifier trained on cross samples'
        # plt.title('Loss of classifier traned on cross samples')
        # plt.ioff()
        # wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        # plt.close()
        #
        # plt.plot(x, loss_self)
        # plt.xlabel('Iterations')
        # plt.ylabel('Loss')
        # name = 'Loss of classifier trained on self recon sequences'
        # plt.title('Loss of classifier trained on self recon sequences')
        # plt.ioff()
        # wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        # plt.close()
        #
        # plt.plot(x, loss_cross_rel)
        # plt.xlabel('Iterations')
        # plt.ylabel('Loss')
        # name = 'Loss of classifier trained on transfer task with related labels'
        # plt.title('Loss of classifier trained on transfer task with related labels')
        # plt.ioff()
        # wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        # plt.close()

        # print results:

    if quantitative:
        self_recon_eval_av = self_recon_eval_av / (batch_nr + 1)
        print(f" [eval] self recon (self reconstr.): {self_recon_eval_av}")
        log_dict = {
            "epoch": epoch,
            "self_recon_eval": self_recon_eval_av,
            "APD": np.mean(APD),
            "ADE": np.mean(ADE),
            "FDE": np.mean(FDE),
            "ASD": np.mean(ASD),
            "ADE_C": np.mean(ADE_c),
            "FDE_C": np.mean(FDE_c),
            "FSD": np.mean(FSD),

            # "Classifier action label cross": np.mean(CF_cross),
            # "Classifier action label cross same action": np.mean(CF_cross_rel),
            # "Classifier action label original": np.mean(CF_action),

            "Classifier action beta": np.mean(CF_action_beta),

            # "Classifier distance logits L2": np.mean(CF_cross_L2),
            # "Classifier distance logits COS": np.mean(CF_cross_COS),
            # "Classifier distance logits L2 related": np.mean(CF_cross_rel_L2),
            # "Classifier distance logits COS related": np.mean(CF_cross_rel_COS),

            # "Classifier CHANGES action label cross": np.mean(CF_cross2),
            # "Classifier CHANGES action label cross same action": np.mean(CF_cross_rel2),
            # "Classifier CHANGES action label original": np.mean(CF_action2),

            "Classifier CHANGES distance logits L2": np.mean(CF_cross2_L2),
            "Classifier CHANGES distance logits COS": np.mean(CF_cross2_COS),
            "Classifier CHANGES distance logits L2 related": np.mean(CF_cross2_rel_L2),
            "Classifier CHANGES distance logits COS related": np.mean(CF_cross2_rel_COS),
        }

        if epoch % n_epochs_classifier == 0:
            log_dict.update({
                "Classifier real vs fake acc prior samples": acc1[-1],
                "Classifier real vs fake acc cross samples": acc2[-1],
                "Classifier real vs fake acc self reconstruction": acc_self[-1],
                "Classifier trained on transfer task with related labels": acc_cross_rel[-1],
                "Loss regressor": loss_regressor[-1],

            })

        # log
        if save_dir is None:
            wandb.log(log_dict)


def visualize_transfer3d(
    model: torch.nn.Module,
    loader: DataLoader,
    device,
    name,
    dirs,
    revert_coord_space=True,
    epoch=None,
    logwandb=True,
    n_vid_to_generate=2
):
    # get data
    assert isinstance(loader.dataset, BaseDataset)
    model.eval()
    vids = {}

    sampling = True
    vunet = None

    it = iter(loader)
    # flag indicates if generated keypoints shall be reprojected into the image
    project = (
        "intrinsics" in loader.dataset.datakeys
        and "extrinsics" in loader.dataset.datakeys
        and "intrinsics_paired" in loader.dataset.datakeys
        and "extrinsics_paired" in loader.dataset.datakeys
        and epoch is not None
    )


    kin_tree = loader.dataset.kinematic_tree


    for i in tqdm(
        range(n_vid_to_generate), desc="Generate Videos for logging."
    ):
        batch = next(it)
        print("got data")
        kps1 = batch["keypoints"].to(dtype=torch.float32)
        # kps1_change = batch["kp_change"].to(dtype=torch.float32)
        ids1 = batch["sample_ids"][0].cpu().numpy()
        id1 = ids1[0]
        label_id1 = loader.dataset.datadict["action"][id1]

        kps2 = batch["paired_keypoints"].to(dtype=torch.float32)
        # kps2_change = batch["paired_change"].to(dtype=torch.float32)
        ids2 = batch["paired_sample_ids"][0].cpu().numpy()
        id2 = ids2[0]
        label_id2 = loader.dataset.datadict["action"][id2]

        data1 = kps1.to(device)
        data2 = kps2.to(device)

        data_b_1 = data1
        data_b_2 = data2

        print("preprocessed data")
        with torch.no_grad():
            # last frame of conditioning sequence
            x1_start = data1[:, :model.div]
            x2_start = data2[:, :model.div]
            # task 1: infer self behavior and reconstruct (w/ and w/o target location)

            # reconstruction
            x1_rec, *_ = model(
                data_b_1, data_b_2
            )
            x2_rec, *_ = model(
                data_b_2, data_b_1
            )


            # transfer
            # model transition in the ground truth sequence 2 mapped on the start pose of sequence 1
            x1_b2, *_ = model(
                data_b_2, data_b_1, transfer=True
            )
            # transferred pose 2; startpose2 and behavior 1

            # x2_b1, *_ = model.generate_seq(b1, x2_start, data1.shape[1])
            # model transition in the ground truth sequence 1 mapped on the start pose of sequence 2
            x2_b1, *_ = model(
                data_b_1, data_b_2, transfer=True
            )

            # sample
            if sampling:
                n_samples = 5
                x1_samples = []
                x2_samples = []
                for j in range(n_samples):

                    x1_b_sampled, *_ = model(
                        data_b_1, data_b_1, sample_prior=True
                    )
                    x1_samples.append(
                        torch.cat(
                            [x1_start[:,-1].unsqueeze(dim=1), x1_b_sampled], dim=1
                        )
                        .squeeze(dim=0)
                        .cpu()
                        .numpy()
                    )

                    x2_b_sampled, *_ = model(
                        data_b_2, data_b_2, sample_prior=True
                    )
                    x2_samples.append(
                        torch.cat(
                            [x2_start[:,-1].unsqueeze(dim=1), x2_b_sampled], dim=1
                        )
                        .squeeze(dim=0)
                        .cpu()
                        .numpy()
                    )

        print("ran model")
        # prepare seq1 predictions
        p1_rec = torch.cat([x1_start, x1_rec], dim=1)
        p1_rec = p1_rec.squeeze(dim=0).cpu().numpy()

        # transferred pose 1; startpose1 and behavior 2
        p1_t = torch.cat([x1_start, x1_b2], dim=1)
        p1_t = p1_t.squeeze(dim=0).cpu().numpy()


        # prepare seq2 predictions
        p2_rec = torch.cat([x2_start, x2_rec], dim=1)
        p2_rec = p2_rec.squeeze(dim=0).cpu().numpy()

        # transferred pose 2; startpose2 and behavior 1
        p2_t = torch.cat([x2_start, x2_b1], dim=1)
        p2_t = p2_t.squeeze(dim=0).cpu().numpy()


        # already in right form
        p1_gt = kps1.squeeze(dim=0).cpu().numpy()
        p2_gt = kps2.squeeze(dim=0).cpu().numpy()

        label1 = loader.dataset.action_id_to_action[label_id1]
        label2 = loader.dataset.action_id_to_action[label_id2]
        labels = [
            label1,
            label1,
            label2,
            label1,
            label2,
            label2,
        ]
        # colors = ["b", "r", "g", "g", "r", "b"]
        colors = [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]

        poses_array = [
            p1_gt,
            p1_rec,
            p1_t,
            p2_t,
            p2_rec,
            p2_gt,
        ]
        poses_array = [
            revert_output_format(
                p,
                loader.dataset.data_mean,
                loader.dataset.data_std,
                loader.dataset.dim_to_ignore,
            )
            for p in poses_array
        ]
        if sampling:
            x1_samples_array = prepare_videos(
                x1_samples, loader, kin_tree, revert_coord_space
            )
            x2_samples_array = prepare_videos(
                x2_samples, loader, kin_tree, revert_coord_space
            )

            x1_samples_videos = [
                create_video_3d(
                    p,
                    [0, 1, 1],
                    loader.dataset,
                    "b sampled",
                    use_limits=not revert_coord_space,
                )
                for p in x1_samples_array
            ]
            x2_samples_videos = [
                create_video_3d(
                    p,
                    [0, 1, 1],
                    loader.dataset,
                    "b sampled",
                    use_limits=not revert_coord_space,
                )
                for p in x2_samples_array
            ]

        print("post processed data")

        if revert_coord_space:
            poses_array = [
                revert_coordinate_space(p, np.eye(3), np.zeros(3))
                for p in poses_array
            ]

        # generate 3d keypoints
        # if "angle" in loader.dataset.keypoint_key:
        #     poses_array = [
        #         convert_to_3d(p, kin_tree, swap_yz=revert_coord_space)
        #         for p in poses_array
        #     ]
        # else:
        poses_array = [
            p.reshape(
                p.shape[0], len(loader.dataset.joint_model.kps_to_use), 3
            )
            for p in poses_array
        ]

        # create single images
        plot_data = [
            (p, labels[i], colors[i]) for i, p in enumerate(poses_array)
        ]
        print("create videos----")
        # if loader.dataset.keypoint_key != "keypoints_3d_world":
        videos_single = [
            create_video_3d(
                p[0],
                p[2],
                loader.dataset,
                p[1],
                use_limits=not revert_coord_space,
            )
            for p in plot_data
        ]
        print("----finished video creation")

        # arange
        upper = np.concatenate(videos_single[:3], axis=2)
        lower = np.concatenate(videos_single[3:6], axis=2)
        full_single = np.concatenate([upper, lower], axis=1)
        full_single = np.moveaxis(full_single, [0, 1, 2, 3], [0, 2, 3, 1])

        # compare sampled sequences with ground truth
        if sampling:  # and loader.dataset.keypoint_key != "keypoints_3d_world":
            upper_with_sampled = np.concatenate(
                [videos_single[0][model.div-1:]] + x1_samples_videos, axis=2
            )
            lower_with_sampled = np.concatenate(
                [videos_single[5][model.div-1:]] + x2_samples_videos, axis=2
            )
            full_sampled = np.concatenate(
                [upper_with_sampled, lower_with_sampled], axis=1
            )
            full_sampled = np.moveaxis(full_sampled, [0, 1, 2, 3], [0, 2, 3, 1])

        if project:
            if "image_size" not in loader.dataset.datadict.keys():
                raise TypeError(
                    "Dataset doesn't contain image sizes, not possible to project 3d points onto 2d images"
                )
            print("create 2d images----")


            extrs = batch["extrinsics"].squeeze(dim=0).cpu().numpy()[0]
            intrs = batch["intrinsics"].squeeze(dim=0).cpu().numpy()[0]

            app_img = None

            sizes = [loader.dataset.datadict["image_size"][ids1]] * 3 + [
                loader.dataset.datadict["image_size"][ids2]
            ] * 3

            colors = [
                [[0, 0, 255]],
                [[255, 0, 0]],
                [[0, 255, 0]],
                [[0, 255, 0]],
                [[255, 0, 0]],
                [[0, 0, 255]],
                ids2,
            ]

            texts = [
                f"GT1; behavior: {label1}",
                f"R1; behavior: {label1}",
                f"T1; behavior: {label2}",
                f"T2; behavior: {label1}",
                f"R2; behavior: {label2}",
                f"GT2; behavior: {label2}",
            ]

            projections = [
                project_onto_image_plane(
                    *t,
                    (5, 40),
                    extrs,
                    intrs,
                    loader.dataset,
                    target_size=(256, 256),
                    background_color=255,
                    synth_model=None,
                    app_img=app_img,
                    cond_id=model.div
                )
                for t in zip(poses_array[:6], sizes, colors, texts)
            ]

            projections = [p[0] for p in projections]


            p_upper = np.concatenate(projections[:3], axis=2)
            p_lower = np.concatenate(projections[3:6], axis=2)

            p_complete = np.concatenate([p_upper, p_lower], axis=1)

            savepath = dirs["generated"] if isinstance(dirs,dict) else dirs
            filename = f"projection{i}@epoch{epoch}.mp4"
            savename = path.join(savepath, filename)

            writer = cv2.VideoWriter(
                savename,
                cv2.VideoWriter_fourcc(*"MP4V"),
                12,
                (p_complete.shape[2], p_complete.shape[1]),
            )

            # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

            for frame in p_complete:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)

            writer.release()

            if logwandb:
                p_complete_wandb = np.moveaxis(p_complete,[0,1,2,3],[0,2,3,1])
                vids.update({name + f'2D-Projections transferred and self recon {i}': wandb.Video(p_complete_wandb, fps=10, format="mp4", caption="2D Projections transferred and self recon")})

            if sampling:
                sample_x1_proj = [
                    project_onto_image_plane(
                        *t,
                        (5, 40),
                        extrs,
                        intrs,
                        loader.dataset,
                        target_size=(256, 256),
                        background_color=255,
                        synth_model=vunet,
                        app_img=app_img,
                    )
                    for t in zip(
                        [poses_array[0][model.div-1:]] + x1_samples_array,
                        [loader.dataset.datadict["image_size"][ids1]]
                        * (len(x1_samples_array) + 1),
                        [[[255, 69, 0], [0, 255, 0]]]
                        + [None] * len(x1_samples_array),
                        [f"GT id {id1}"]
                        + [
                            f"sample#{i + 1} | x1"
                            for i in range(len(x1_samples_array))
                        ],
                    )
                ]
                sample_x1_proj = [p[0] for p in sample_x1_proj]
                sample_x2_proj = [
                    project_onto_image_plane(
                        *t,
                        (5, 40),
                        extrs,
                        intrs,
                        loader.dataset,
                        target_size=(256, 256),
                        background_color=255,
                        synth_model=vunet,
                        app_img=app_img,
                    )
                    for t in zip(
                        [poses_array[5][model.div-1:]] + x2_samples_array,
                        [loader.dataset.datadict["image_size"][ids2]]
                        * (len(x1_samples_array) + 1),
                        [[[255, 69, 0], [0, 255, 0]]]
                        + [None] * len(x1_samples_array),
                        [f"GT id {id2}"]
                        + [
                            f"sample#{i + 1} | x2"
                            for i in range(len(x2_samples_array))
                        ],
                    )
                ]

                sample_x2_proj = [p[0] for p in sample_x2_proj]


                samples_upper = np.concatenate(sample_x1_proj, axis=2)
                samples_lower = np.concatenate(sample_x2_proj, axis=2)

                samples_full = np.concatenate(
                    [samples_upper, samples_lower], axis=1
                )


                filename = f"projection{i}_samples@epoch{epoch}.mp4"
                savename = path.join(savepath, filename)
                save_name = savepath + "/frames/" + filename[:-6]
                # breakpoint()
                # import matplotlib.pyplot as plt
                # for i in range(samples_full.shape[0]):
                #     plt.imsave(save_name + 'frame' + str(i) + '.png', samples_full[i])

                writer = cv2.VideoWriter(
                    savename,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    12,
                    (samples_full.shape[2], samples_full.shape[1]),
                )

                # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

                for frame in samples_full:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)

                writer.release()

                if logwandb:
                    samples_wandb = np.moveaxis(samples_full,[0,1,2,3],[0,2,3,1])
                    vids.update({name + f'2D-Projections samples {i}':wandb.Video(samples_wandb,fps=10,format="mp4",caption="2D Projections samples")})
            print("----finished 2d projections")
        # if loader.dataset.keypoint_key != "keypoints_3d_world":
        vids.update(
            {
                name
                + f"Poses  {i}, seq_len": wandb.Video(
                    full_single,
                    fps=10,
                    format="mp4",
                    caption=f"Video #{i} seq_len: {data1.shape[1]}",
                ),
                # name
                # + f"Pose Comparison {i}": wandb.Video(
                #     full_comp,
                #     fps=5,
                #     format="mp4",
                #     caption=f"Comp #{i} seq_len: {data1.shape[1]}",
                # ),
            }
        )
        if sampling:
            vids.update(
                {
                    name
                    + f"Behavior Samples vs GT {i}": wandb.Video(
                        full_sampled,
                        fps=10,
                        format="mp4",
                        caption=f"Behavior Samples vs GT {i}: {data1.shape[1]}",
                    )
                }
            )
    if logwandb:
        wandb.log(vids)