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
    revert_coordinate_space,
    _some_variables,
    Ax3DPose,
    convert_to_3d,
    project_onto_image_plane,
)
from models.pose_behavior_rnn import ResidualBehaviorNet, Classifier, Regressor
from lib.losses import FlowLoss
from models.vunets import VunetAlter



def prepare_videos(raw_poses, loader, kin_tree, revert_coord_space):
    """

    :param raw_poses:
    :param loader:
    :param kin_tree:
    :param revert_coord_space:
    :return:
    """
    if isinstance(raw_poses, list):
        out_poses = [
            revert_output_format(
                p,
                loader.dataset.data_mean,
                loader.dataset.data_std,
                loader.dataset.dim_to_ignore,
            )
            for p in raw_poses
        ]
        if revert_coord_space:
            out_poses = [
                revert_coordinate_space(p, np.eye(3), np.zeros(3))
                for p in out_poses
            ]
        if "angle" not in loader.dataset.keypoint_key:
            out_poses = [
                op.reshape(
                    op.shape[0], len(loader.dataset.joint_model.kps_to_use), 3
                )
                for op in out_poses
            ]
        else:
            out_poses = [
                convert_to_3d(p, kin_tree, swap_yz=revert_coord_space)
                for p in out_poses
            ]

    else:
        out_poses = revert_output_format(
            raw_poses,
            loader.dataset.data_mean,
            loader.dataset.data_std,
            loader.dataset.dim_to_ignore,
        )
        if revert_coord_space:
            out_poses = revert_coordinate_space(
                out_poses, np.eye(3), np.zeros(3)
            )

        if "angle" not in loader.dataset.keypoint_key:
            out_poses = out_poses.reshape(
                out_poses.shape[0],
                len(loader.dataset.joint_model.kps_to_use),
                3,
            )
        else:
            out_poses = convert_to_3d(
                out_poses, kin_tree, swap_yz=revert_coord_space
            )

    return out_poses


def visualize_transfer3d(
    model: torch.nn.Module,
    loader: DataLoader,
    device,
    name,
    dirs,
    revert_coord_space=True,
    epoch=None,
    synth_ckpt=None,
    synth_params=None,
    flow=None,
    logwandb=True,
    n_vid_to_generate=2
):
    # get data
    assert isinstance(loader.dataset, BaseDataset)
    model.eval()
    vids = {}

    sampling = isinstance(model, ResidualBehaviorNet) and model.ib
    vunet = None
    if synth_ckpt is not None and synth_params is not None:
        print("Loading synth model for generation of real images")
        n_channels_x = (
            3 * len(loader.dataset.joint_model.norm_T)
            if synth_params["inplane_normalize"]
            else 3
        )
        vunet = VunetAlter(
            n_channels_x=n_channels_x, **synth_params
        )
        vunet.load_state_dict(synth_ckpt)
        vunet = vunet.to(device)
        vunet.eval()

        assert loader.dataset.complete_datadict is not None

    it = iter(loader)
    # flag indicates if generated keypoints shall be reprojected into the image
    project = (
        "intrinsics" in loader.dataset.datakeys
        and "extrinsics" in loader.dataset.datakeys
        and "intrinsics_paired" in loader.dataset.datakeys
        and "extrinsics_paired" in loader.dataset.datakeys
        and epoch is not None
    )

    if hasattr(loader.dataset, "kinematic_tree"):
        kin_tree = loader.dataset.kinematic_tree
    else:
        parent, offset, rotInd, expmapInd, posInd = _some_variables(
            use_posInd=not revert_coord_space
        )
        kin_tree = {
            "parent": parent,
            "offset": offset,
            "rotInd": rotInd,
            "expmapInd": expmapInd,
            "posInd": {"ids": posInd},
        }


    print(f"project is {project}")

    for i in tqdm(
        range(n_vid_to_generate), desc="Generate Videos for logging."
    ):
        batch = next(it)
        print("got data")
        kps1 = batch["keypoints"].to(dtype=torch.float32)
        # kps1_change = batch["kp_change"].to(dtype=torch.float32)
        ids1 = batch["sample_ids"][0, 1:, ...].cpu().numpy()
        id1 = ids1[0]
        label_id1 = loader.dataset.datadict["action"][id1]

        kps2 = batch["paired_keypoints"].to(dtype=torch.float32)
        # kps2_change = batch["paired_change"].to(dtype=torch.float32)
        ids2 = batch["paired_sample_ids"][0, 1:, ...].cpu().numpy()
        id2 = ids2[0]
        label_id2 = loader.dataset.datadict["action"][id2]
        n_kps = kps1.shape[2]

        data1, _ = prepare_input(kps1,device)
        data2, _ = prepare_input(kps2,device)

        data_b_1 = data1
        data_b_2 = data2

        seq_len = data_b_1.shape[1]

        print("preprocessed data")
        with torch.no_grad():
            x1_start = data1[:, 0]
            x2_start = data2[:, 0]
            # task 1: infer self behavior and reconstruct (w/ and w/o target location)

            if sampling:
                x1_rec, c1_rec, _, _, mu1, *_ = model(
                    data_b_1, data1, len=data_b_1.shape[1]
                )
                x2_rec, c2_rec, _, _, mu2, *_ = model(
                    data_b_2, data2, len=data_b_2.shape[1]
                )

                x1_mu2, *_ = model.generate_seq(
                    mu2, data1, len=data_b_2.shape[1], start_frame=0
                )
                x2_mu1, *_ = model.generate_seq(
                    mu1, data2, len=data_b_1.shape[1], start_frame=0
                )

                x1_mu2 = (
                    torch.cat([x1_start.unsqueeze(dim=1), x1_mu2], dim=1)
                    .squeeze(dim=0)
                    .cpu()
                    .numpy()
                )
                x2_mu1 = (
                    torch.cat([x2_start.unsqueeze(dim=1), x2_mu1], dim=1)
                    .squeeze(dim=0)
                    .cpu()
                    .numpy()
                )

            else:
                x1_rec, c1_rec, *_ = model(
                    data_b_1, data1, len=data_b_1.shape[1]
                )
                x2_rec, c2_rec, *_ = model(
                    data_b_2, data2, len=data_b_2.shape[1]
                )

            # x1_b2, *_ = model.generate_seq(b2, x1_start, data2.shape[1])
            x1_b2, c12t, *_ = model(
                data_b_2, data1, len=data_b_2.shape[1]
            )
            # transferred pose 2; startpose2 and behavior 1

            # x2_b1, *_ = model.generate_seq(b1, x2_start, data1.shape[1])
            x2_b1, c21t, *_ = model(
                data_b_1, data2, len=data_b_1.shape[1]
            )

            # sample
            if sampling:
                n_samples = 40
                x1_samples = []
                x2_samples = []
                if flow is not None:
                    x1_flow_samples = []
                    x2_flow_samples = []
                for j in range(n_samples):

                    x1_b_sampled, _, _, b_dummy, *_ = model(
                        data_b_1, data1, len=data_b_1.shape[1], sample=True
                    )
                    x1_samples.append(
                        torch.cat(
                            [x1_start.unsqueeze(dim=1), x1_b_sampled], dim=1
                        )
                        .squeeze(dim=0)
                        .cpu()
                        .numpy()
                    )

                    x2_b_sampled, *_ = model(
                        data_b_2, data2, len=data_b_2.shape[1], sample=True
                    )
                    x2_samples.append(
                        torch.cat(
                            [x2_start.unsqueeze(dim=1), x2_b_sampled], dim=1
                        )
                        .squeeze(dim=0)
                        .cpu()
                        .numpy()
                    )

                    if flow is not None:

                        gsamples1 = torch.randn_like(b_dummy)
                        b_fs1 = flow.reverse(gsamples1)

                        b_fs1 = b_fs1.squeeze(dim=-1).squeeze(dim=-1)
                        #
                        # b_fs1 = flow.sample(
                        #     shape=b_dummy.shape + (1, 1), device=device
                        # )
                        x1_bfs, *_ = model.generate_seq(
                            b_fs1, data1, len=data1.shape[1], start_frame=0
                        )
                        x1_flow_samples.append(
                            torch.cat(
                                [x1_start.unsqueeze(dim=1), x1_bfs], dim=1
                            )
                            .squeeze(dim=0)
                            .cpu()
                            .numpy()
                        )

                        gsamples2 = torch.randn_like(b_dummy)
                        b_fs2 = flow.reverse(gsamples2)

                        b_fs2 = b_fs2.squeeze(dim=-1).squeeze(dim=-1)
                        # b_fs2 = flow.sample(
                        #     shape=b_dummy.shape + (1, 1), device=device
                        # )
                        x2_bfs, *_ = model.generate_seq(
                            b_fs2, data2, len=data2.shape[1], start_frame=0
                        )
                        x2_flow_samples.append(
                            torch.cat(
                                [x2_start.unsqueeze(dim=1), x2_bfs], dim=1
                            )
                            .squeeze(dim=0)
                            .cpu()
                            .numpy()
                        )

            # evaluate time invariance
            start_frame_id = int(
                np.random.choice(
                    np.arange(
                        int(loader.dataset.seq_length[0] // 2),
                        3 * int(loader.dataset.seq_length[0] // 4) + 1,
                    ),
                    1,
                )
            )
            x1_b2_s10, *_ = model(
                data_b_2,
                data1,
                len=data_b_1.shape[1],
                start_frame=start_frame_id,
            )
            x2_b1_s10, *_ = model(
                data_b_1,
                data2,
                len=data_b_1.shape[1],
                start_frame=start_frame_id,
            )

        print("ran model")
        # prepare seq1 predictions
        p1_rec = torch.cat([x1_start.unsqueeze(dim=1), x1_rec], dim=1)
        p1_rec = p1_rec.squeeze(dim=0).cpu().numpy()

        # transferred pose 1; startpose1 and behavior 2
        p1_t = torch.cat([x1_start.unsqueeze(dim=1), x1_b2], dim=1)
        p1_t = p1_t.squeeze(dim=0).cpu().numpy()

        # transferred pose 1, start_frame 10: startpose1, frame 10, behavior 2
        p1_t_s10 = torch.cat(
            [data1[:, start_frame_id].unsqueeze(dim=1), x1_b2_s10], dim=1
        )
        p1_t_s10 = p1_t_s10.squeeze(dim=0).cpu().numpy()

        # prepare seq2 predictions
        p2_rec = torch.cat([x2_start.unsqueeze(dim=1), x2_rec], dim=1)
        p2_rec = p2_rec.squeeze(dim=0).cpu().numpy()

        # transferred pose 2; startpose2 and behavior 1
        p2_t = torch.cat([x2_start.unsqueeze(dim=1), x2_b1], dim=1)
        p2_t = p2_t.squeeze(dim=0).cpu().numpy()

        # transferred pose 2, start_frame 10: startpose2, frame 10, behavior 1
        p2_t_s10 = torch.cat(
            [data2[:, start_frame_id].unsqueeze(dim=1), x2_b1_s10], dim=1
        )
        p2_t_s10 = p2_t_s10.squeeze(dim=0).cpu().numpy()

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
            label2,
            label1,
        ]
        # colors = ["b", "r", "g", "g", "r", "b"]
        colors = [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
        ]

        poses_array = [
            p1_gt,
            p1_rec,
            p1_t,
            p2_t,
            p2_rec,
            p2_gt,
            p1_t_s10,
            p2_t_s10,
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
            x1_mu2_arr = prepare_videos(
                x1_mu2, loader, kin_tree, revert_coord_space
            )
            x2_mu1_arr = prepare_videos(
                x2_mu1, loader, kin_tree, revert_coord_space
            )

            if flow is not None:
                x1_fs_arr = prepare_videos(
                    x1_flow_samples, loader, kin_tree, revert_coord_space
                )
                x2_fs_arr = prepare_videos(
                    x2_flow_samples, loader, kin_tree, revert_coord_space
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
        if "angle" in loader.dataset.keypoint_key:
            poses_array = [
                convert_to_3d(p, kin_tree, swap_yz=revert_coord_space)
                for p in poses_array
            ]
        else:
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
                [videos_single[0]] + x1_samples_videos, axis=2
            )
            lower_with_sampled = np.concatenate(
                [videos_single[5]] + x2_samples_videos, axis=2
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

            if vunet is not None:
                app_img, extrs, intrs = get_synth_input(loader, vunet)
            else:
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
                    synth_model=vunet,
                    app_img=app_img,
                )
                for t in zip(poses_array[:6], sizes, colors, texts)
            ]

            projections_rgb = [p[1] for p in projections]
            projections = [p[0] for p in projections]

            if sampling:
                x1_mu2_proj, x1_mu2_proj_rgb = project_onto_image_plane(
                    x1_mu2_arr,
                    loader.dataset.datadict["image_size"][ids1],
                    None,
                    f"T1; behavior: {label2}, mu",
                    (5, 40),
                    extrs,
                    intrs,
                    loader.dataset,
                    target_size=(256, 256),
                    background_color=255,
                    synth_model=vunet,
                    app_img=app_img,
                )
                x2_mu1_proj, x2_mu1_proj_rgb = project_onto_image_plane(
                    x2_mu1_arr,
                    loader.dataset.datadict["image_size"][ids1],
                    None,
                    f"T2; behavior: {label1}, mu",
                    (5, 40),
                    extrs,
                    intrs,
                    loader.dataset,
                    target_size=(256, 256),
                    background_color=255,
                    synth_model=vunet,
                    app_img=app_img,
                )

                p_upper = np.concatenate(
                    projections[:3] + [x1_mu2_proj], axis=2
                )
                p_lower = np.concatenate(
                    projections[3:6] + [x2_mu1_proj], axis=2
                )
                if vunet is not None:
                    p_upper_rgb = np.concatenate(
                        projections_rgb[:3] + [x1_mu2_proj_rgb], axis=2
                    )
                    p_lower_rgb = np.concatenate(
                        projections_rgb[3:6] + [x2_mu1_proj_rgb], axis=2
                    )
            else:
                p_upper = np.concatenate(projections[:3], axis=2)
                p_lower = np.concatenate(projections[3:6], axis=2)
                if vunet is not None:
                    p_upper_rgb = np.concatenate(projections_rgb[:3], axis=2)
                    p_lower_rgb = np.concatenate(projections_rgb[3:6], axis=2)

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

            if vunet is not None:
                p_complete_rgb = np.concatenate(
                    [p_upper_rgb, p_lower_rgb], axis=1
                )

                savepath = dirs["generated"] if isinstance(dirs,dict) else dirs
                filename = f"projection_rgb{i}@epoch{epoch}.mp4"
                savename = path.join(savepath, filename)

                writer = cv2.VideoWriter(
                    savename,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    12,
                    (p_complete_rgb.shape[2], p_complete_rgb.shape[1]),
                )

                # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

                for frame in p_complete_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)

                writer.release()

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
                        [poses_array[0]] + x1_samples_array,
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
                sample_x1_proj_rgb = [p[1] for p in sample_x1_proj]
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
                        [poses_array[5]] + x2_samples_array,
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

                sample_x2_proj_rgb = [p[1] for p in sample_x2_proj]
                sample_x2_proj = [p[0] for p in sample_x2_proj]

                if flow is not None:
                    sample_x1_pf = [
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
                            [poses_array[0]] + x1_fs_arr,
                            [loader.dataset.datadict["image_size"][ids1]]
                            * (len(x1_samples_array) + 1),
                            [[[255, 69, 0], [0, 255, 0]]]
                            + [None] * len(x1_samples_array),
                            ["GT | x1"]
                            + [
                                f"flow sample#{i + 1} | x1"
                                for i in range(len(x1_samples_array))
                            ],
                        )
                    ]
                    sample_x1_pf_rgb = [p[1] for p in sample_x1_pf]
                    sample_x1_pf = [p[0] for p in sample_x1_pf]

                    sample_x2_pf = [
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
                            [poses_array[5]] + x2_fs_arr,
                            [loader.dataset.datadict["image_size"][ids1]]
                            * (len(x1_samples_array) + 1),
                            [[[255, 69, 0], [0, 255, 0]]]
                            + [None] * len(x1_samples_array),
                            ["GT | x1"]
                            + [
                                f"flow sample#{i + 1} | x1"
                                for i in range(len(x1_samples_array))
                            ],
                        )
                    ]
                    sample_x2_pf_rgb = [p[1] for p in sample_x2_pf]
                    sample_x2_pf = [p[0] for p in sample_x2_pf]

                samples_upper = np.concatenate(sample_x1_proj, axis=2)
                samples_lower = np.concatenate(sample_x2_proj, axis=2)
                if flow is None:
                    samples_full = np.concatenate(
                        [samples_upper, samples_lower], axis=1
                    )
                else:
                    samples_flow_upper = np.concatenate(sample_x1_pf, axis=2)
                    samples_flow_lower = np.concatenate(sample_x2_pf, axis=2)

                    samples_full = np.concatenate(
                        [
                            samples_upper,
                            samples_flow_upper,
                            samples_lower,
                            samples_flow_lower,
                        ],
                        axis=1,
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
                if vunet is not None:
                    samples_upper_rgb = np.concatenate(
                        sample_x1_proj_rgb, axis=2
                    )
                    samples_lower_rgb = np.concatenate(
                        sample_x2_proj_rgb, axis=2
                    )
                    if flow is None:
                        samples_full_rgb = np.concatenate(
                            [samples_upper_rgb, samples_lower_rgb], axis=1
                        )
                    else:
                        samples_flow_upper_rgb = np.concatenate(
                            sample_x1_pf_rgb, axis=2
                        )
                        samples_flow_lower_rgb = np.concatenate(
                            sample_x2_pf_rgb, axis=2
                        )
                        samples_full_rgb = np.concatenate(
                            [
                                samples_upper_rgb,
                                samples_flow_upper_rgb,
                                samples_lower_rgb,
                                samples_flow_lower_rgb,
                            ],
                            axis=1,
                        )
                    filename = f"projection{i}_rgb_samples@epoch{epoch}.mp4"
                    savename = path.join(savepath, filename)

                    writer = cv2.VideoWriter(
                        savename,
                        cv2.VideoWriter_fourcc(*"MP4V"),
                        12,
                        (samples_full_rgb.shape[2], samples_full_rgb.shape[1]),
                    )

                    # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

                    for frame in samples_full_rgb:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(frame)

                    writer.release()
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


def latent_interpolate(
    model: ResidualBehaviorNet,
    loader: DataLoader,
    device,
    dirs,
    epoch,
    synth_ckpt=None,
    synth_params=None,
    n_vid_to_generate=2
):
    # get data
    assert isinstance(loader.dataset, BaseDataset)
    assert model.ib
    model.eval()

    project = isinstance(loader.dataset, Human36mDataset)

    vunet = None
    if synth_ckpt is not None and synth_params is not None:
        print("Loading synth model for generation of real images")
        n_channels_x = (
            3 * len(loader.dataset.joint_model.norm_T)
            if synth_params["inplane_normalize"]
            else 3
        )
        vunet = VunetAlter(
            n_channels_x=n_channels_x, **synth_params
        )
        vunet.load_state_dict(synth_ckpt)
        vunet = vunet.to(device)
        vunet.eval()

        assert loader.dataset.complete_datadict is not None

    it = iter(loader)
    # flag indicates if generated keypoints shall be reprojected into the image

    kin_tree = loader.dataset.kinematic_tree

    for i in tqdm(
        range(n_vid_to_generate), desc="Generate Videos for logging."
    ):
        batch = next(it)
        print("got data")
        kps1 = batch["keypoints"].to(dtype=torch.float32)
        # kps1_change = batch["kp_change"].to(dtype=torch.float32)
        ids1 = batch["sample_ids"][0, 1:, ...].cpu().numpy()
        id1 = ids1[0]
        label_id1 = loader.dataset.datadict["action"][id1]

        kps2 = batch["paired_keypoints"].to(dtype=torch.float32)
        # kps2_change = batch["paired_change"].to(dtype=torch.float32)
        ids2 = batch["paired_sample_ids"][0, 1:, ...].cpu().numpy()
        id2 = ids2[0]
        label_id2 = loader.dataset.datadict["action"][id2]
        n_kps = kps1.shape[2]

        data1, _ = prepare_input(kps1,device)
        data2, _ = prepare_input(kps2,device)

        data_b_1 = data1
        data_b_2 = data2

        print("preprocessed data")

        if project:
            if vunet is not None:
                app_img, extrs, intrs = get_synth_input(loader, vunet)
            else:
                extrs = batch["extrinsics"].squeeze(dim=0).cpu().numpy()[0]
                intrs = batch["intrinsics"].squeeze(dim=0).cpu().numpy()[0]

                app_img = None

        steps = np.linspace(0, 1, 6)
        with torch.no_grad():
            x1_start = data1[:, 0]
            b2, *_ = model.infer_b(data_b_2, False)
            x2_start = data2[:, 0]
            b1, *_ = model.infer_b(data_b_1, False)

            inter1_to_2_proj = []
            inter2_to_1_proj = []
            inter1_to_2_proj_rgb = []
            inter2_to_1_proj_rgb = []
            for s in steps:
                # interpolate spherically
                dev = b1.get_device() if b1.get_device() >= 0 else "cpu"
                b1_to_2 = torch.tensor(
                    slerp(
                        s,
                        b1.squeeze(dim=0).cpu().numpy(),
                        b2.squeeze(dim=0).cpu().numpy(),
                    ),
                    device=dev,
                ).unsqueeze(dim=0)
                b2_to_1 = torch.tensor(
                    slerp(
                        s,
                        b2.squeeze(dim=0).cpu().numpy(),
                        b1.squeeze(dim=0).cpu().numpy(),
                    ),
                    device=dev,
                ).unsqueeze(dim=0)
                # b1_to_2 = b1 * (1 - s) + b2 * s
                # b2_to_1 = b1 * s + b2 * (1 - s)

                # generate sequences with interpolated b
                p1_to_2, *_ = model.generate_seq(
                    b1_to_2, data1, data1.shape[1], start_frame=0
                )
                p2_to_1, *_ = model.generate_seq(
                    b2_to_1, data2, data2.shape[1], start_frame=0
                )
                p1_to_2 = (
                    torch.cat([x1_start.unsqueeze(dim=1), p1_to_2], dim=1)
                    .squeeze(dim=0)
                    .cpu()
                    .numpy()
                )
                p2_to_1 = (
                    torch.cat([x2_start.unsqueeze(dim=1), p2_to_1], dim=1)
                    .squeeze(dim=0)
                    .cpu()
                    .numpy()
                )

                p1_to_2 = revert_output_format(
                    p1_to_2,
                    loader.dataset.data_mean,
                    loader.dataset.data_std,
                    loader.dataset.dim_to_ignore,
                )
                p2_to_1 = revert_output_format(
                    p2_to_1,
                    loader.dataset.data_mean,
                    loader.dataset.data_std,
                    loader.dataset.dim_to_ignore,
                )
                if "angle" in loader.dataset.keypoint_key:
                    p1_to_2 = convert_to_3d(p1_to_2, kin_tree, swap_yz=False)
                    p2_to_1 = convert_to_3d(p2_to_1, kin_tree, swap_yz=False)

                else:
                    p1_to_2 = p1_to_2.reshape(
                        p1_to_2.shape[0],
                        len(loader.dataset.joint_model.kps_to_use),
                        3,
                    )
                    p2_to_1 = p2_to_1.reshape(
                        p1_to_2.shape[0],
                        len(loader.dataset.joint_model.kps_to_use),
                        3,
                    )


                if project:

                    i_img_2d1, i_img_2d1_rgb = project_onto_image_plane(
                        p1_to_2,
                        loader.dataset.datadict["image_size"][ids1],
                        [[int((1.0 - s) * 255.0), int(s * 255.0), 0]],
                        "",
                        (10, 10),
                        extrs,
                        intrs,
                        loader.dataset,
                        target_size=(256, 256),
                        background_color=255,
                        synth_model=vunet,
                        app_img=app_img,
                    )

                    inter1_to_2_proj.append(i_img_2d1)
                    inter1_to_2_proj_rgb.append(i_img_2d1_rgb)

                    i_img_2d2, i_img_2d2_rgb = project_onto_image_plane(
                        p2_to_1,
                        loader.dataset.datadict["image_size"][ids2],
                        [[int(s * 255.0), int((1.0 - s) * 255.0), 0]],
                        "",
                        (10, 10),
                        extrs,
                        intrs,
                        loader.dataset,
                        target_size=(256, 256),
                        background_color=255,
                        synth_model=vunet,
                        app_img=app_img,
                    )
                    inter2_to_1_proj.append(i_img_2d2)
                    inter2_to_1_proj_rgb.append(i_img_2d2_rgb)


        if project:
            inter1_to_2_proj = np.concatenate(inter1_to_2_proj, axis=2)
            inter2_to_1_proj = np.concatenate(inter2_to_1_proj, axis=2)
            inter_full_proj = np.concatenate(
                [inter1_to_2_proj, inter2_to_1_proj], axis=1
            )
            if vunet is not None:
                inter1_to_2_proj_rgb = np.concatenate(
                    inter1_to_2_proj_rgb, axis=2
                )
                inter2_to_1_proj_rgb = np.concatenate(
                    inter2_to_1_proj_rgb, axis=2
                )
                inter_full_proj_rgb = np.concatenate(
                    [inter1_to_2_proj_rgb, inter2_to_1_proj_rgb], axis=1
                )

        action1 = loader.dataset.action_id_to_action[label_id1]
        action2 = loader.dataset.action_id_to_action[label_id2]

        # text to videos
        savepath = dirs["generated"] if isinstance(dirs,dict) else dirs

        if project:
            inter_full_proj = text_to_vid(
                inter_full_proj,
                f"2d: Interpolation #{i}: 1st: {action1} -> {action2} with x1_start; 2nd vice versa, id1: {int(id1)}; id2: {int(id2)}",
                (5, 40),
            )

            filename = f"interpolations2d{i}-id1{int(id1)}-id2{int(id2)}.mp4"
            savename = path.join(savepath, filename)

            writer = cv2.VideoWriter(
                savename,
                cv2.VideoWriter_fourcc(*"MP4V"),
                12,
                (inter_full_proj.shape[2], inter_full_proj.shape[1]),
            )

            # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

            for frame in inter_full_proj:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)

            writer.release()

            if vunet is not None:
                inter_full_proj_rgb = text_to_vid(
                    inter_full_proj_rgb,
                    f"2d: Interpolation #{i}: 1st: {action1} -> {action2} with x1_start; 2nd vice versa, d1: {int(id1)}; id2: {int(id2)}",
                        (5, 40),
                )

                filename = f"interpolations2d_rgb{i}-id1{int(id1)}-id2{int(id2)}.mp4"
                savename = path.join(savepath, filename)

                writer = cv2.VideoWriter(
                    savename,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    12,
                    (
                        inter_full_proj_rgb.shape[2],
                        inter_full_proj_rgb.shape[1],
                    ),
                )

                # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

                for frame in inter_full_proj_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)

                writer.release()
        # # if loader.dataset.keypoint_key != "keypoints_3d_world":
        # inter_full = text_to_vid(
        #     inter_full,
        #     f"3d: Interpolation #{i}: 1st: {action1} -> {action2} with x1_start; 2nd vice versa",
        #     (5, 70),
        # )
        #
        # # if loader.dataset.keypoint_key != "keypoints_3d_world":
        # filename = f"interpolations3d{i}@epoch{epoch}.mp4"
        # savename = path.join(savepath, filename)
        #
        # writer = cv2.VideoWriter(
        #     savename,
        #     cv2.VideoWriter_fourcc(*"MP4V"),
        #     12,
        #     (inter_full.shape[2], inter_full.shape[1]),
        # )
        #
        # # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)
        #
        # for frame in inter_full:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #     writer.write(frame)
        #
        # writer.release()


def create_video_3d(
    poses,
    color,
    dataset: BaseDataset,
    label=None,
    ticks_visible=True,
    use_limits=False,
    limits=None,
):
    """
    Creates a series of images which can be used as videos, based on an array of poses as output by the net
    :param poses: The poses as 3d keyoints
    :param label: The action label as string
    :return:
    """
    font_size = 0.7
    font_thickness = 2
    imgs_i = []
    fig = plt.figure()
    ax = plt.gca(projection="3d")
    if use_limits:
        if limits is None:
            # limkps = np.reshape(keypoints_3d, (keypoints_3d.shape[0], 32, -1))
            limits = {
                "x": (np.amin(poses[:, :, 0]), np.amax(poses[:, :, 0])),
                "y": (np.amin(poses[:, :, 1]), np.amax(poses[:, :, 1])),
                "z": (np.amin(poses[:, :, 2]), np.amax(poses[:, :, 2])),
            }
            ob = Ax3DPose(
                ax,
                dataset=dataset,
                marker_color=color,
                ticks=ticks_visible,
                limits=limits,
            )
        else:
            ob = Ax3DPose(
                ax,
                dataset=dataset,
                marker_color=color,
                ticks=ticks_visible,
                limits=limits,
            )
    else:
        ob = Ax3DPose(
            ax, dataset=dataset, marker_color=color, ticks=ticks_visible
        )

    # keypoints_3d = poses.reshape((poses.shape[0], -1))

    for t_nr, k in enumerate(poses):

        ob.update(k)
        imsize = fig.get_size_inches() * fig.dpi
        asarr = fig2data(fig, imsize)
        if label is not None:
            cv2.putText(
                asarr,
                "action: " + label,
                (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (0, 0, 0),
                font_thickness,
            )

        imgs_i.append(np.expand_dims(asarr, axis=0))

    plt.close()
    imgs_i = np.concatenate(imgs_i, axis=0)
    return imgs_i


def make_3d_overlay_plot(
    data1, data2, dataset: BaseDataset, caption=None, use_limits=False
):
    poses1, label1, color1 = data1
    poses2, label2, color2 = data2

    if use_limits:
        cat_poses = np.concatenate([poses1, poses2], axis=1)
        limits = {
            "x": (np.amin(cat_poses[:, :, 0]), np.amax(cat_poses[:, :, 0])),
            "y": (np.amin(cat_poses[:, :, 1]), np.amax(cat_poses[:, :, 1])),
            "z": (np.amin(cat_poses[:, :, 2]), np.amax(cat_poses[:, :, 2])),
        }
    else:
        limits = None

    video1 = create_video_3d(
        poses1,
        color1,
        dataset,
        ticks_visible=False,
        use_limits=use_limits,
        limits=limits,
    )
    video2 = create_video_3d(
        poses2,
        color2,
        dataset,
        ticks_visible=False,
        use_limits=use_limits,
        limits=limits,
    )

    colortgt1 = np.asarray([int(c * 255) for c in color1], dtype=video2.dtype)
    colortgt2 = np.asarray([int(c * 255) for c in color2], dtype=video2.dtype)
    # make overlay:
    out_vid = []
    for frame_pair in zip(video1, video2):
        img1 = frame_pair[0]
        img2 = frame_pair[1]

        # second video overlays first where both have colored entries
        img2[
            np.logical_and(
                np.equal(img1, colortgt1).all(axis=-1),
                np.equal(img2, [255, 255, 255]).all(axis=-1),
            )
        ] = 0
        img1[np.equal(img2, colortgt2).all(axis=-1)] = colortgt2

        out_frame = cv2.addWeighted(img1, 1, img2, 1, 0)

        if caption is not None:
            cv2.putText(
                out_frame,
                caption,
                (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        out_vid.append(out_frame)

    out_vid = np.stack(out_vid, axis=0)
    return out_vid


def make_hist(hist, limit, epoch):
    bins_edges = np.arange(0, limit + 1).astype(np.float)
    centroids = (bins_edges[1:] + bins_edges[:-1]) / 2

    hist_, bins_, _ = plt.hist(
        centroids,
        bins=len(hist),
        weights=hist,
        range=(min(bins_edges), max(bins_edges)),
    )
    plt.title(
        f"Averaged training errors of joint changes for sequences of length {limit} over frame number."
    )
    plt.xlabel("Frame number within sequence")
    plt.ylabel("MSE Training Error: Joint Changes")
    wandb.log({"epoch": epoch, f"sequence_{limit}": wandb.Image(plt)})
    plt.close()


def log_umap(data, labels, epoch, name, id_T_label, add_data=None,log_dir=None):
    """
    Logs umap plot with wandb
    :param data:
    :return:
    """

    print("Generate umap plot for plot with title: " + name)

    umap_transform = umap.UMAP()
    if add_data is not None:
        stacked = np.concatenate([data,add_data],axis=0)
    else:
        stacked = data
    transformation = umap_transform.fit(stacked)
    transformed_data = transformation.transform(data)

    # create plot to visualize embedding space
    if add_data is None:
        unique_actions, action_count = np.unique(labels, return_counts=True)

        plt.scatter(
            transformed_data[:, 0],
            transformed_data[:, 1],
            s=5,
            c=labels,
            cmap="Spectral",
        )
        plt.gca().set_aspect("equal", "datalim")
        cb = plt.colorbar(
            boundaries=np.arange(unique_actions.shape[0] + 1) - 0.5,
            values=unique_actions,
        )
        cb.set_ticks(np.arange(unique_actions.shape[0]))
        cb.ax.set_yticklabels([id_T_label[u] for u in unique_actions])
    else:

        d_trans = gaussian_kde(transformed_data.T)(transformed_data.T)
        tr_idx = d_trans.argsort()
        xt, yt, zt = (
            transformed_data[:, 0][tr_idx],
            transformed_data[:, 1][tr_idx],
            d_trans[tr_idx],
        )

        add_transformed = transformation.transform(add_data)

        d_add = gaussian_kde(add_transformed.T)(add_transformed.T)
        ad_idx = d_add.argsort()
        xa, ya, za = (
            add_transformed[:, 0][ad_idx],
            add_transformed[:, 1][ad_idx],
            d_add[ad_idx],
        )

        plt.scatter(
            xt, yt, c=zt, s=1, label="posterior", cmap=plt.get_cmap("autumn"), alpha=.3,rasterized=True
        )
        plt.scatter(
            xa, ya, c=za, s=1, label="sampling distribution", cmap=plt.get_cmap("winter"), alpha=.3,rasterized=True
        )
        # plt.legend()

    # plt.title(name)
    plt.axis('off')
    plt.ioff()
    if log_dir is None:
        wandb.log({"epoch": epoch, name: wandb.Image(plt)})
    else:
        plt.savefig(path.join(log_dir,f"umap-{name}.pdf"),dpi=300)
    plt.close()


def get_synth_input_fix(loader,vunet,id):
    # cameras, cids = np.unique(
    #     loader.dataset.complete_datadict["camera_id"], return_index=True
    # )
    #
    # tcid = int(np.random.choice(cameras, 1))
    # cid = int(cids[cameras == tcid])

    pids = np.unique(loader.dataset.complete_datadict["p_ids"],)
    tpid = np.random.choice(pids,1)


    dev = next(vunet.parameters()).device


    tids = np.nonzero(loader.dataset.complete_datadict["p_ids"] == tpid)[0]
    tid = np.random.choice(tids,1)


    app_img = loader.dataset._get_app_img(
        tid, inplane_norm=vunet.n_channels_x > 3, use_complete_ddict=True
    ).unsqueeze(dim=0)

    app_img = app_img.to(dev)
    app_img = transform.resize(app_img,
                               (vunet.spatial_size, vunet.spatial_size))

    extrs = loader.dataset.complete_datadict["extrinsics_univ"][id]
    intrs = loader.dataset.complete_datadict["intrinsics_univ"][id]

    return app_img, extrs, intrs



def get_synth_input(loader, vunet,all_cameras = False):
    cameras, cids = np.unique(
        loader.dataset.complete_datadict["camera_id"], return_index=True
    )

    if all_cameras:

        dev = next(vunet.parameters()).device
        persons, pids = np.unique(
            loader.dataset.complete_datadict["p_ids"], return_index=True
        )
        tpid = int(np.random.choice(persons, 1,replace=False))
        # pids_cont = [9,9,9,9]
        extrs= []
        intrs = []
        apps = []
        for camera,cid in zip(cameras,cids):
            tid = int(
                np.nonzero(
                    np.logical_and(
                        loader.dataset.complete_datadict["p_ids"] == tpid,
                        loader.dataset.complete_datadict["camera_id"] == camera,
                    )
                )[0][0]
            )

            app_img = loader.dataset._get_app_img(
                tid, inplane_norm=vunet.n_channels_x > 3, use_complete_ddict=True
            ).unsqueeze(dim=0)

            app_img = app_img.to(dev)
            app_img = transform.resize(app_img,
                                       (vunet.spatial_size, vunet.spatial_size))

            extr = loader.dataset.complete_datadict["extrinsics_univ"][cid]
            intr = loader.dataset.complete_datadict["intrinsics_univ"][cid]

            extrs.append(extr)
            intrs.append(intr)
            apps.append(app_img)

        return apps, extrs, intrs
    else:

        tcid = int(np.random.choice(cameras, 1))
        cid = int(cids[cameras == tcid])

        dev = next(vunet.parameters()).device
        persons, pids = np.unique(
            loader.dataset.complete_datadict["p_ids"], return_index=True
        )
        tpid = int(np.random.choice(persons, 1))

        tid = int(
            np.nonzero(
                np.logical_and(
                    loader.dataset.complete_datadict["p_ids"] == tpid,
                    loader.dataset.complete_datadict["camera_id"] == tcid,
                )
            )[0][0]
        )

        app_img = loader.dataset._get_app_img(
            tid, inplane_norm=vunet.n_channels_x > 3, use_complete_ddict=True
        ).unsqueeze(dim=0)

        app_img = app_img.to(dev)
        app_img = transform.resize(app_img, (vunet.spatial_size, vunet.spatial_size))

        extrs = loader.dataset.complete_datadict["extrinsics_univ"][cid]
        intrs = loader.dataset.complete_datadict["intrinsics_univ"][cid]

        return app_img, extrs, intrs


def make_eval_grid(
    model: torch.nn.Module,
    loader: DataLoader,
    device,
    dirs,
    revert_coord_space=True,
    epoch=None,
    synth_ckpt=None,
    synth_params=None,
    write_index=False
):

    # get data
    assert isinstance(loader.dataset, BaseDataset)
    model.eval()

    vunet = None
    if synth_ckpt is not None and synth_params is not None:
        print("Loading synth model for generation of real images")
        n_channels_x = (
            3 * len(loader.dataset.joint_model.norm_T)
            if synth_params["inplane_normalize"]
            else 3
        )
        vunet = VunetAlter(
            n_channels_x=n_channels_x, **synth_params
        )
        vunet.load_state_dict(synth_ckpt)
        vunet = vunet.to(device)
        vunet.eval()

        assert loader.dataset.complete_datadict is not None

    # flag indicates if generated keypoints shall be reprojected into the image
    project = (
        "intrinsics" in loader.dataset.datakeys
        and "extrinsics" in loader.dataset.datakeys
        and "intrinsics_paired" in loader.dataset.datakeys
        and "extrinsics_paired" in loader.dataset.datakeys
        and epoch is not None
    )

    if hasattr(loader.dataset, "kinematic_tree"):
        kin_tree = loader.dataset.kinematic_tree
    else:
        parent, offset, rotInd, expmapInd, posInd = _some_variables(
            use_posInd=not revert_coord_space
        )
        kin_tree = {
            "parent": parent,
            "offset": offset,
            "rotInd": rotInd,
            "expmapInd": expmapInd,
            "posInd": {"ids": posInd},
        }

    print(f"project is {project}")

    # sample start_poses
    unique_actions = np.unique(loader.dataset.datadict["action"])
    np.random.shuffle(unique_actions)
    start_ids = []
    for a in unique_actions[:6]:
        start_ids.append(
            int(
                np.random.choice(
                    np.nonzero(loader.dataset.datadict["action"] == a)[0],
                    size=1,
                )
            )
        )


    startpose_ids = np.random.choice(
        np.nonzero(loader.dataset.datadict["action"])[0], size=5, replace=False
    )
    seq_length = loader.dataset.seq_length[0]

    first_col_rgb = np.concatenate(
        [
            np.full(
                (loader.dataset.seq_length[-1], 256, 256, 3),
                255,
                dtype=np.uint8,
            )
        ]
        + [
            np.stack(
                [
                    cv2.resize(
                        (
                            scale_img(
                                loader.dataset._get_pose_img(
                                    [spi], use_crops=False
                                )
                            )
                            * 255.0
                        )
                        .cpu()
                        .squeeze(dim=0)
                        .permute(1, 2, 0)
                        .numpy()
                        .astype(np.uint8),
                        (256, 256),
                    )
                ]
                * loader.dataset.seq_length[-1],
                axis=0,
            )
            for spi in startpose_ids
        ],
        axis=1,
    )

    cols_rgb = [first_col_rgb]

    cols = []
    first_col = [
        np.zeros((loader.dataset.seq_length[-1], 256, 256, 3), dtype=np.uint8)
    ]

    for nr, i in enumerate(tqdm(start_ids, desc=f"Generating eval grid.")):

        # get behavior sequence
        bids = loader.dataset._sample_valid_seq_ids([i, seq_length])
        b_seq = torch.tensor(
            loader.dataset._get_keypoints(bids), device=device
        ).unsqueeze(dim=0)
        action = loader.dataset.action_id_to_action[
            int(loader.dataset.datadict["action"][i])
        ]
        sizes = loader.dataset.datadict["image_size"][bids]

        gt_b_vid = loader.dataset._get_pose_img(bids, use_crops=False).squeeze(
            dim=0
        )
        gt_b_vid = scale_img(gt_b_vid)
        gt_b_vid = kornia.geometry.transform.resize(gt_b_vid, (256, 256))
        gt_b_vid = (
            (gt_b_vid * 255.0)
            .cpu()
            .permute(0, 2, 3, 1)
            .numpy()
            .astype(np.uint8)
        )
        col_rgb = [gt_b_vid]

        if vunet is not None:
            app_img, extrs, intrs = get_synth_input(loader, vunet)
        else:
            extrs = loader.dataset._get_extrinsic_params(bids[0])
            intrs = loader.dataset._get_intrinsic_params(bids[0])

            app_img = None

        gt_b_vid = prepare_videos(
            b_seq.squeeze().cpu().numpy(), loader, kin_tree, revert_coord_space
        )
        col = [
            project_onto_image_plane(
                gt_b_vid,
                sizes,
                None,
                f"B: {action}; id {bids[0]}",
                (5, 40),
                extrs,
                intrs,
                loader.dataset,
                target_size=(256, 256),
                background_color=255,
            )[0]
        ]

        # sample start pose
        for xid in startpose_ids:
            extrs = loader.dataset._get_extrinsic_params(xid)
            intrs = loader.dataset._get_intrinsic_params(xid)

            xids = loader.dataset._sample_valid_seq_ids([xid, seq_length])
            x_seq = (
                torch.tensor(loader.dataset._get_keypoints(xids))
                .unsqueeze(dim=0)
                .to(device)
            )

            # get appearance

            x_seq, _ = prepare_input(x_seq,device)
            b_seq, _ = prepare_input(b_seq,device)
            x_start = x_seq[:, 0]

            with torch.no_grad():
                seq_len = b_seq.shape[1]

                tr, *_ = model(b_seq, x_seq, len=seq_length)

            tr = torch.cat([x_start.unsqueeze(dim=1), tr], dim=1)
            tr = tr.squeeze(dim=0).cpu().numpy()

            tr_poses = prepare_videos(tr, loader, kin_tree, revert_coord_space)

            sizes = loader.dataset.datadict["image_size"][xids]

            # preparation
            project_2d, project_2d_rgb = project_onto_image_plane(
                tr_poses,
                sizes,
                None,
                f"id_xstart: {xids[0]}",
                (5, 40),
                extrs,
                intrs,
                loader.dataset,
                target_size=(256, 256),
                background_color=255,
                synth_model=vunet,
                app_img=app_img,
            )

            if nr == 0:
                first_col.append(
                    np.stack(
                        [project_2d[0]] * loader.dataset.seq_length[-1], axis=0
                    )
                )

            col.append(project_2d)
            if vunet is not None:
                col_rgb.append(project_2d_rgb)

        if vunet is not None:
            cols_rgb.append(np.concatenate(col_rgb, axis=1))
        cols.append(np.concatenate(col, axis=1))

    min_time = min([c.shape[0] for c in cols])
    first_col = np.concatenate(first_col, axis=1)

    grid = np.concatenate(
        [first_col[:min_time]] + [c[:min_time] for c in cols], axis=2
    )

    if vunet is not None:
        grid_rgb = np.concatenate([c[:min_time] for c in cols_rgb], axis=2)

    savepath = dirs["generated"] if isinstance(dirs,dict) else dirs
    filename = f"grid_12fps@epoch{epoch}.mp4"
    savename = path.join(savepath, filename)

    writer = cv2.VideoWriter(
        savename,
        cv2.VideoWriter_fourcc(*"MP4V"),
        12,
        (grid.shape[2], grid.shape[1]),
    )

    # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

    for frame in grid:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()

    if vunet is not None:
        savepath = dirs["generated"] if isinstance(dirs,dict) else dirs
        filename = f"grid_12fps_rgb@epoch{epoch}.mp4"
        savename = path.join(savepath, filename)

        # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

        writer = cv2.VideoWriter(
            savename,
            cv2.VideoWriter_fourcc(*"MP4V"),
            12,
            (grid_rgb.shape[2], grid_rgb.shape[1]),
        )

        # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

        for frame in grid_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()

        filename = f"grid_25fps@epoch{epoch}.mp4"
        savename = path.join(savepath, filename)

        writer = cv2.VideoWriter(
            savename,
            cv2.VideoWriter_fourcc(*"MP4V"),
            25,
            (grid.shape[2], grid.shape[1]),
        )

        # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

        for frame in grid:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()

        filename = f"grid_25fps_rgb@epoch{epoch}.mp4"
        savename = path.join(savepath, filename)

        writer = cv2.VideoWriter(
            savename,
            cv2.VideoWriter_fourcc(*"MP4V"),
            25,
            (grid_rgb.shape[2], grid_rgb.shape[1]),
        )

        # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

        for frame in grid_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()


def eval_nets(
    net,
    loader,
    device,
    epoch,
    n_max_umap_samples,
    mode="test",
    flow=None,
    quantitative=True,
    cf_action=None,
    cf_action_beta=None,
    cf_action2=None,
    dim_to_use=51,
    save_dir = None,
    debug=False
):

    RED = "\033[91m"
    ENDC = "\033[0m"

    net.eval()
    data_iterator = tqdm(loader, desc="Eval", total=len(loader))
    self_recon_eval_av = 0
    recon_eval = nn.MSELoss()

    collect_b = []
    collect_ac = []
    prior_samples = []
    collect_mu = []
    collect_pre_stats = []

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
    CF_cross_rel2 =[]
    CF_cross2_rel_L2 =[]
    CF_cross2_rel_COS =[]
    CF_action2 = []


    test_flow = flow is not None

    if test_flow:
        flow_loss_eval = FlowLoss()
        avg_flow_loss = {
            "flow_loss": np.asarray([], dtype=np.float),
            "nlogdet_loss": np.asarray([], dtype=np.float),
            "reference_nll_loss": np.asarray([], dtype=np.float),
            "nll_loss": np.asarray([], dtype=np.float),
        }
        collect_flow_samples = []
        collect_gaussian_samples = []

    X_prior = []
    X_cross = []
    X_orig  = []
    X_self = []
    X_embed = []
    X_start = []
    X_cross_rel = []
    X_flow=[]
    num_samples = 0

    # incrementatest_bsl time step between two frames depends on sequential frame lag
    for batch_nr, batch in enumerate(data_iterator):
        # get data
        kps1 = batch["keypoints"].to(dtype=torch.float32)
        # NOTE: this can be "paired_keypoints" as no label transfer is wished for this dataset,
        # hence, the map_ids of the dataset point to sequences with the same label
        kps2 = batch["paired_keypoints"].to(dtype=torch.float32)
        kps3, sample_ids_3 = batch["matched_keypoints"]

        actions = batch["action"]

        # assert torch.all(torch.equal(actions, actions[0]))
        # action_name = test_dataset.action_id_to_action[actions[0]]
        action_names = []
        for aid in actions[:, 0].cpu().numpy():
            action_names.append(loader.dataset.action_id_to_action[aid])

        # build inputs
        # input: bs x 1 x n_kps x k
        x_s, target_s = prepare_input(kps1,device)
        x_t, target_t = prepare_input(kps2,device)
        x_related, _ = prepare_input(kps3,device)
        data_b_s = x_s

        # actions of related keypoints
        actions_related = loader.dataset.datadict["action"][sample_ids_3[:,0].cpu().numpy()]


        dev = data_b_s.get_device() if data_b_s.get_device() >= 0 else "cpu"

        # eval - reconstr.
        seq_len = data_b_s.size(1)

        with torch.no_grad():
            # self reconstruction


            seq_pred_s, c_s, _, b, mu, logstd, pre = net(
                data_b_s, x_s, seq_len
            )

            seq_pred_mu_s, *_ = net.generate_seq(mu,x_s,seq_len,start_frame=0)

            # sample new behavior
            _, _, _, sampled_prior, *_ = net(
                data_b_s, x_s, seq_len, sample=True
            )


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
                    if test_flow:
                        gsamples1 = torch.randn_like(b[::skip])
                        b_fs1 = flow.reverse(gsamples1)

                        b_fs1 = b_fs1.squeeze(dim=-1).squeeze(dim=-1)
                        seq_s, *_ = net.generate_seq(
                            b_fs1, target_s[::skip], len=seq_len, start_frame=target_s.shape[1] - 1
                        )
                    else:
                        seq_s, *_ = net(x_s[::skip], target_s[::skip], seq_len,
                                    sample=True,
                                    start_frame=target_s.shape[1] - 1)
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
                

                seq_samples = seq_samples.reshape(*seq_samples.shape[:3],len(loader.dataset.joint_model.kps_to_use),3)
                seq_gt = seq_gt.reshape(*seq_gt.shape[:3],len(loader.dataset.joint_model.kps_to_use),3)[:, :, 1:]


                # average pairwise distance; average self distance; average final distance
                for samples in seq_samples:
                    dist_APD = 0
                    dist_ASD = 0
                    dist_FSD = 0
                    for seq_q in samples:
                        dist =torch.norm((seq_q - samples).reshape(samples.shape[0], -1), dim=1)
                        dist_APD += torch.sum(dist) / (n_samples - 1)
                        dist = torch.mean(torch.norm((seq_q - samples).reshape(samples.shape[0], seq_len, -1), dim=2), dim=1)
                        dist_ASD += np.sort(dist.cpu().numpy())[1] ## take 2nd value since 1st value is 0 distance with itself
                        dist_f = torch.norm((seq_q[-1] - samples[:, -1]).reshape(samples.shape[0], -1), dim=1)
                        dist_FSD += np.sort(dist_f.cpu().numpy())[1] ## take 2nd value since 1st value is 0 distance with itself

                    APD.append(dist_APD.item() / n_samples)
                    ASD.append(dist_ASD.item() / n_samples)
                    FSD.append(dist_FSD.item() / n_samples)

                # average displacement error
                ADE.append(torch.mean((torch.min(torch.mean(torch.norm((seq_samples - seq_gt).reshape(seq_gt.shape[0], n_samples, seq_len, -1), dim=3), dim=2), dim=1)[0])).item())
                # final displacement error
                FDE.append((torch.mean(torch.min(torch.norm((seq_samples[:, :, -1] - seq_gt[:, :, -1]).reshape(seq_gt.shape[0], n_samples, -1), dim=2), dim=1)[0])).item())

                if batch_nr%10 == 0:
                    update = "APD:{0:.2f}, ASD:{1:.2f}, FSD:{2:.2f}, ADE:{3:.2f}, FDE:{4:.2f}".format(np.mean(APD), np.mean(ASD), np.mean(FSD), np.mean(ADE), np.mean(FDE))
                    data_iterator.set_description(update)

                labels = batch["action"][:, 0] - 2
                seq_cross, *_ = net(x_s, x_t, seq_len, sample=False)

                if cf_action and quantitative:
                    predict = cf_action(seq_cross)
                    _, labels_pred = torch.max(predict, 1)
                    acc_action = (
                        torch.sum(labels_pred.cpu() == labels).float()
                        / labels_pred.shape[0]
                    )
                    CF_cross.append(acc_action.item())
                                    
                    predict = cf_action(x_s)
                    _, labels_pred = torch.max(predict, 1)
                    acc_action = (
                        torch.sum(labels_pred.cpu() == labels).float()
                        / labels_pred.shape[0]
                    )
                    CF_action.append(acc_action.item())

                if cf_action_beta and quantitative:
                    predict = cf_action_beta(mu)
                    _, labels_pred = torch.max(predict, 1)
                    acc_action_beta = (torch.sum(labels_pred.cpu() == labels).float() / labels_pred.shape[0])
                    CF_action_beta.append(acc_action_beta.item())

                
            # Evaluate realisticness from prior samples with classifier
            # Evaluate action label and realisticness for cross setting
            labels = batch["action"][:, 0] - 2
            seq_cross, _, _, _, mu, *_ = net(data_b_s, x_t, seq_len, sample=False)
            seq_pred_mu_cross, *_ = net.generate_seq(mu, x_t, seq_len, start_frame=0)
            # mu2 = net.infer_b(seq_cross)

            labels_related = torch.from_numpy(actions_related).to(device) - 2
            seq_cross_rel, *_ = net(x_related,x_s,seq_len, sample=False)
            samples_prior, *_ = net(x_s, target_s, seq_len, sample=True, start_frame=target_s.shape[1]-1)
            seq_pred_mu_s, *_ = net.generate_seq(mu,x_s,seq_len,start_frame=0)

            if num_samples < 25000:
            #     ADE_c.append(torch.mean(torch.norm((seq_cross-x_s), dim=2)).item())
            #     FDE_c.append(torch.mean(torch.norm((seq_cross[:, -1]-x_s[:, -1]), dim=1)).item())
            #     X_prior.append(samples_prior.cpu())
            #     X_cross.append(seq_pred_mu_cross.cpu())
            #     X_cross_rel.append(seq_cross_rel.cpu())
            #     X_self.append(seq_pred_mu_s.cpu())
            #     X_orig.append(x_s.cpu())
            #     X_embed.append(mu.cpu())

                num_samples += x_s.shape[0]
            else:
                break

            seq_cross = x_related
            seq_cross_rel = x_related

            if cf_action and quantitative:
                
                #CLASSIFIER normalo
                predict = cf_action(seq_cross)
                _, labels_pred = torch.max(predict[0], 1)
                acc_action = (
                    torch.sum(labels_pred.cpu() == labels).float()
                    / labels_pred.shape[0]
                )
                CF_cross.append(acc_action.item())
                predict2 = cf_action(data_b_s)[1]
                CF_cross_L2.append(torch.mean(torch.norm(predict2 - predict[1], dim=1)).item())
                CF_cross_COS.append(torch.mean(torch.nn.CosineSimilarity(dim=1)(predict2,predict[1])).item())

                predict = cf_action(seq_cross_rel)
                _, labels_pred = torch.max(predict[0], 1)
                acc_action = (
                        torch.sum(labels_pred.cpu() == labels_related.cpu()).float()
                        / labels_pred.shape[0]
                )
                CF_cross_rel.append(acc_action.item())
                CF_cross_rel_L2.append(torch.mean(torch.norm(predict2 - predict[1], dim=1)).item())
                CF_cross_rel_COS.append(torch.mean(torch.nn.CosineSimilarity(dim=1)(predict2,predict[1])).item())
                                
                predict = cf_action(x_s)[0]
                _, labels_pred = torch.max(predict, 1)
                acc_action = (
                    torch.sum(labels_pred.cpu() == labels).float()
                    / labels_pred.shape[0]
                )
                CF_action.append(acc_action.item())

                #CLASSIFIER on changes

                predict = cf_action2((seq_cross[:, 1:]-seq_cross[:, :-1]).transpose(1,2))
                _, labels_pred = torch.max(predict[0], 1)
                acc_action = (
                    torch.sum(labels_pred.cpu() == labels).float()
                    / labels_pred.shape[0]
                )
                CF_cross2.append(acc_action.item())
                predict2 = cf_action2((data_b_s[:, 1:]-data_b_s[:, :-1]).transpose(1,2))[1]
                CF_cross2_L2.append(torch.mean(torch.norm(predict2 - predict[1], dim=1)).item())
                CF_cross2_COS.append(torch.mean(torch.nn.CosineSimilarity(dim=1)(predict2,predict[1])).item())

                predict = cf_action2((seq_cross_rel[:, 1:]-seq_cross_rel[:, :-1]).transpose(1,2))
                _, labels_pred = torch.max(predict[0], 1)
                acc_action = (
                        torch.sum(labels_pred.cpu() == labels_related.cpu()).float()
                        / labels_pred.shape[0]
                )
                CF_cross_rel2.append(acc_action.item())
                CF_cross2_rel_L2.append(torch.mean(torch.norm(predict2 - predict[1], dim=1)).item())
                CF_cross2_rel_COS.append(torch.mean(torch.nn.CosineSimilarity(dim=1)(predict2,predict[1])).item())
                                
                predict = cf_action2((x_s[:, 1:]-x_s[:, :-1]).transpose(1,2))
                _, labels_pred = torch.max(predict[0], 1)
                acc_action = (
                    torch.sum(labels_pred.cpu() == labels).float()
                    / labels_pred.shape[0]
                )
                CF_action2.append(acc_action.item())

            if cf_action_beta:
                predict = cf_action_beta(mu)
                _, labels_pred = torch.max(predict, 1)
                acc_action_beta = (torch.sum(labels_pred.cpu() == labels).float() / labels_pred.shape[0])
                CF_action_beta.append(acc_action_beta.item())

            if test_flow:
                # test density estimation

                gauss, logdet = flow(b)
                _, flow_dict = flow_loss_eval(gauss, logdet)
                avg_flow_loss = {
                    key: np.append(avg_flow_loss[key], flow_dict[key])
                    for key in avg_flow_loss
                }

                # test sampling: generate samples in vae latent space
                gsamples = torch.randn_like(b)

                flow_samples = flow.reverse(gsamples)
                if (batch_nr + 1) * seq_pred_s.shape[0] < n_max_umap_samples:
                    collect_flow_samples.append(
                        flow_samples.squeeze(dim=-1)
                            .squeeze(dim=-1)
                            .cpu()
                            .numpy()
                    )
                    collect_gaussian_samples.append(
                        gauss.squeeze(dim=-1).squeeze(dim=-1).cpu().numpy()
                    )

            recon_batch_av = recon_eval(seq_pred_s, target_s)
            self_recon_eval_av += recon_batch_av.detach().cpu().numpy()

            if (batch_nr + 1) * seq_pred_s.shape[0] < n_max_umap_samples:
                # only use last point of sequence as this should encode all the behavior (if it encodes really it)
                collect_b.append(b.cpu().numpy())
                collect_mu.append(mu.cpu().numpy())
                collect_pre_stats.append(pre.cpu().numpy())
                prior_samples.append(sampled_prior.cpu().numpy())
                collect_ac.append(deepcopy(actions[:, -1].cpu().numpy()))

            if debug and (batch_nr + 1) * seq_pred_s.shape[0] > 1000:
                break
            elif (
                not quantitative
                and (batch_nr + 1) * seq_pred_s.shape[0] >= n_max_umap_samples
            ):
                break
            
    
    print("ADE corss task {0:.2f} and FDE cross task {1:.2f}".format(np.mean(ADE_c), np.mean(FDE_c)))
    n_epochs_classifier = 99
    if epoch % n_epochs_classifier == 0:
        ## Train Classifiers on real vs fake task
        X_orig = torch.stack(X_orig, dim=0).reshape(-1, x_s.shape[1], dim_to_use)
        X_prior = torch.stack(X_prior, dim=0).reshape(-1, x_s.shape[1], dim_to_use)
        X_cross = torch.stack(X_cross, dim=0).reshape(-1, x_s.shape[1], dim_to_use)
        X_cross_rel = torch.stack(X_cross_rel, dim=0).reshape(-1, x_s.shape[1], dim_to_use)
        X_self = torch.stack(X_self, dim=0).reshape(-1, x_s.shape[1], dim_to_use)
        X_embed = torch.stack(X_embed, dim=0).reshape(-1, net.dim_hidden_b)
        if test_flow:
            X_flow = torch.stack(X_flow, dim=0).reshape(-1, dim_to_use)

        # Define classifiers
        class_real1 = Classifier(dim_to_use, 1).to(device)
        optimizer_classifier_real1 = SGD(class_real1.parameters(), lr=0.01, momentum=0.9)

        class_real2 = Classifier(dim_to_use, 1).to(device)
        optimizer_classifier_real2 = SGD(class_real2.parameters(), lr=0.01, momentum=0.9)

        if test_flow:
            class_real3 = Classifier(dim_to_use, 1).to(device)
            optimizer_classifier_real3 = SGD(class_real3.parameters(), lr=0.01, momentum=0.9)

        class_real_self = Classifier(dim_to_use, 1).to(device)
        optimizer_classifier_real_self = SGD(class_real_self.parameters(), lr=0.01, momentum=0.9)     

        regressor = Regressor(net.dim_hidden_b, dim_to_use).to(device)
        optimizer_regressor = Adam(regressor.parameters(), lr=0.001)

        class_real_cross_rel = Classifier(dim_to_use, 1).to(device)
        optim_class_cross_rel = SGD(class_real_self.parameters(), lr=0.01, momentum=0.9)    

        bs = 256
        iterations = 2000
        epochs = iterations//(num_samples//bs)

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

            for i in range(num_samples//bs):

                # Select data/batch
                x_true  = X_orig[i*bs:(i+1)*bs].to(device)
                x_s     = X_prior[i*bs:(i+1)*bs].to(device)
                x_c     = X_cross[i*bs:(i+1)*bs].to(device)
                x_self  = X_self[i*bs:(i+1)*bs].to(device)
                x_mu    = X_embed[i*bs:(i+1)*bs].to(device)
                x_start = X_orig[i*bs:(i+1)*bs, 0].to(device) 
                if test_flow:
                    x_f = X_flow[i*bs:(i+1)*bs].to(device) 
                x_cross_rel = X_cross_rel[i*bs:(i+1)*bs].to(device)


                # Train classifier1 on prior samples
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

                # Train classifier2 on cross samples
                predict = class_real2(x_c)
                target = torch.zeros_like(predict)
                loss_classifier_gen = cls_loss(predict, target)
                acc2.append(torch.mean(nn.Sigmoid()(predict)).item())

                predict = class_real2(x_true)
                target = torch.ones_like(predict)
                loss_classifier_gt = cls_loss(predict, target)

                loss_class_real2 = loss_classifier_gen + loss_classifier_gt
                loss2.append(loss_class_real2.item())
                optimizer_classifier_real2.zero_grad()
                loss_class_real2.backward()
                optimizer_classifier_real2.step()

                # Train classifier2 on self reconstructions
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

                if test_flow:
                    predict = class_real3(x_f)
                    target = torch.zeros_like(predict)
                    loss_classifier_gen = cls_loss(predict, target)
                    acc3.append(torch.mean(nn.Sigmoid()(predict)).item())

                    predict = class_real3(x_true)
                    target = torch.ones_like(predict)
                    loss_classifier_gt = cls_loss(predict,target)

                    loss_class_real3 = loss_classifier_gen + loss_classifier_gt
                    loss3.append(loss_class_real3.item())
                    optimizer_classifier_real3.zero_grad()
                    loss_class_real3.backward()
                    optimizer_classifier_real3.step()

                ## Train regressor
                predict = regressor(x_mu)
                loss_regressor_ = torch.mean((predict - x_start)**2)
                optimizer_regressor.zero_grad()
                loss_regressor_.backward()
                optimizer_regressor.step()
                loss_regressor.append(loss_regressor_.item())

                # Train classifier2 on self reconstructions
                predict = class_real_cross_rel(x_cross_rel)
                target = torch.zeros_like(predict)
                loss_classifier_gen = cls_loss(
                predict, target)
                acc_cross_rel.append(torch.mean(nn.Sigmoid()(predict)).item())

                predict = class_real_cross_rel(x_true)
                target = torch.ones_like(predict)
                loss_classifier_gt = cls_loss(predict, target)

                loss_class_cross_rel = loss_classifier_gen + loss_classifier_gt
                loss_cross_rel.append(loss_class_real_self.item())
                optim_class_cross_rel.zero_grad()
                loss_class_cross_rel.backward()
                optim_class_cross_rel.step()


            update = "Acc Prior:{0:.2f}, Acc Cross:{1:.2f}, Loss_regressor:{2:.2f}".format(acc1[-1], acc2[-1], loss_regressor[-1])
            data_iterator.set_description(update)


        x = np.arange(len(acc1))

        plt.plot(x, loss_regressor)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss of regressor trained on embedding')
        plt.ioff()
        name = 'Loss of regressor trained on embeddings'
        wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        plt.close()

        plt.plot(x, acc1)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of classifier trained on prior samples')
        plt.ioff()
        name = 'Accuracy of classifier trained on prior samples'
        wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        plt.close()

        plt.plot(x, acc2)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of classifier trained on cross task (acc only for cross samples)')
        plt.ioff()
        name = 'Accuracy of classifier trained on cross samples (acc only for cross samples)'
        wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        plt.close()

        plt.plot(x, acc_self)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of classifier trained on self recon task')
        plt.ioff()
        name = 'Accuracy of classifier trained on self reconstructed sequences'
        wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        plt.close()

        plt.plot(x, acc_cross_rel)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of classifier trained on transfer task with related labels')
        plt.ioff()
        name = 'Accuracy of classifier trained on on transfer task with related labels'
        wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        plt.close()

        plt.plot(x, loss1)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        name = 'Loss of classifier trained on prior samples'
        plt.title('Loss of classifier trained on prior samples')
        plt.ioff()
        wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        plt.close()

        plt.plot(x, loss2)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        name = 'Loss of classifier trained on cross samples'
        plt.title('Loss of classifier traned on cross samples')
        plt.ioff()
        wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        plt.close()

        if test_flow:
            plt.plot(x, acc3)
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.title('Accuracy of classifier trained on flow samples')
            plt.ioff()
            name = 'Accuracy of classifier trained on flow samples'
            wandb.log({"epoch": epoch, name: wandb.Image(plt)})
            plt.close()

            plt.plot(x, loss3)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            name = 'Loss of classifier trained on flow samples'
            plt.title('Loss of classifier trained on flow samples')
            plt.ioff()
            wandb.log({"epoch": epoch, name: wandb.Image(plt)})
            plt.close()

        plt.plot(x, loss_self)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        name = 'Loss of classifier trained on self recon sequences'
        plt.title('Loss of classifier trained on self recon sequences')
        plt.ioff()
        wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        plt.close()
        
        plt.plot(x, loss_cross_rel)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        name = 'Loss of classifier trained on transfer task with related labels'
        plt.title('Loss of classifier trained on transfer task with related labels')
        plt.ioff()
        wandb.log({"epoch": epoch, name: wandb.Image(plt)})
        plt.close()

    collect_b = np.concatenate(collect_b, axis=0)
    collect_pre_stats = np.concatenate(collect_pre_stats, axis=0)
    collect_mu = np.concatenate(collect_mu, axis=0)
    collect_ac = np.concatenate(collect_ac, axis=0)
    prior_samples = np.concatenate(prior_samples, axis=0)

    if test_flow:
        collect_flow_samples = np.concatenate(collect_flow_samples, axis=0)
        collect_gaussian_samples = np.concatenate(
            collect_gaussian_samples, axis=0
        )

        # conduct ks test for samples as kstest is only applicable to 1d distributions,
        # and a standard normal gaussian is separable, iterate over the single dimensions
        ps = np.asarray([], dtype=np.float)
        for sdim in range(collect_gaussian_samples.shape[1]):
            _, p = kstest(collect_gaussian_samples[:, sdim], "norm")
            ps = np.append(ps, p)

        p_mean = np.mean(ps)
        if save_dir is None:
            wandb.log(
                {
                    f"{mode}-dataset: Mean of p values for ks tests, whether flow sampling dist is gaussian": p_mean
                }
        )

    if epoch % 10 == 0:

        log_umap(
            collect_b,
            collect_ac,
            epoch,
            f"UMAP projection of posterior samples and prior samples for the {mode} data",
            loader.dataset.action_id_to_action,
            add_data=prior_samples,
            log_dir=save_dir
        )
    # if mode == "test":
        log_umap(
            collect_mu,
            collect_ac,
            epoch,
            f"UMAP projection of estimated means for the {mode} data",
            loader.dataset.action_id_to_action,
            log_dir=save_dir
        )
        log_umap(
            collect_pre_stats,
            collect_ac,
            epoch,
            f"UMAP projection of beta-encoder features for the {mode} data",
            loader.dataset.action_id_to_action,
            log_dir=save_dir
        )

    if test_flow:
        try:
            log_umap(
                collect_b,
                collect_ac,
                epoch,
                f"UMAP projection of flow samples and and posterior samples for {mode} data",
                loader.dataset.action_id_to_action,
                add_data=collect_flow_samples,
                log_dir=save_dir
            )
        except Exception as e:
            print(
                RED
                + "Catched Exception in umap... continue without plot..."
            )
            print(e)
            print(ENDC)
        log_umap(
            collect_gaussian_samples,
            collect_ac,
            epoch,
            f"Sampling space of flow on {mode} data, should be gaussian",
            loader.dataset.action_id_to_action,
            log_dir=save_dir
        )

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

            "Classifier action label cross": np.mean(CF_cross),
            "Classifier action label cross same action": np.mean(CF_cross_rel),
            "Classifier action label original": np.mean(CF_action),

            "Classifier action beta": np.mean(CF_action_beta),
            
            "Classifier distance logits L2": np.mean(CF_cross_L2),
            "Classifier distance logits COS": np.mean(CF_cross_COS),
            "Classifier distance logits L2 related": np.mean(CF_cross_rel_L2),
            "Classifier distance logits COS related": np.mean(CF_cross_rel_COS),

            "Classifier CHANGES action label cross": np.mean(CF_cross2),
            "Classifier CHANGES action label cross same action": np.mean(CF_cross_rel2),
            "Classifier CHANGES action label original": np.mean(CF_action2),

            "Classifier CHANGES distance logits L2": np.mean(CF_cross2_L2),
            "Classifier CHANGES distance logits COS": np.mean(CF_cross2_COS),
            "Classifier CHANGES distance logits L2 related": np.mean(CF_cross2_rel_L2),
            "Classifier CHANGES distance logits COS related": np.mean(CF_cross2_rel_COS),
        }
        if test_flow:
            avg_flow_loss = {
                key: np.mean(avg_flow_loss[key]) for key in avg_flow_loss
            }
            log_dict.update(avg_flow_loss)
            if epoch % n_epochs_classifier == 0:
                log_dict.update({
                "Classifier real vs fake acc flow samples": acc3[-1]})

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


def make_enrollment(test_loader,vunet,net,save_dir,device,grid_bids=None,grid_sids=None,n_samples=10,crop=False):
    import time

    seq_length = test_loader.dataset.seq_length[0]
    disc_step = 4
    test_dataset = test_loader.dataset
    unique_name = False
    if grid_bids is None:
        unique_name = True
        print("Randomize plot generation")
        grid_bids = np.random.choice(
            np.arange(test_dataset.datadict["img_paths"].shape[0]), n_samples,
            replace=False)


    for nr, i in enumerate(
            tqdm(grid_bids, desc=f"Generating enrollment plots.")):

        if grid_sids is None:
            startpose_ids = np.random.choice(np.arange(test_dataset.datadict["img_paths"].shape[0]),20,replace=False)
        else:
            startpose_ids = grid_sids


        bids = test_dataset._sample_valid_seq_ids([i, seq_length])
        b_seq = torch.tensor(
            test_dataset._get_keypoints(bids), device=device
        ).unsqueeze(dim=0)
        action = test_dataset.action_id_to_action[
            int(test_dataset.datadict["action"][i])
        ]
        sizes = test_dataset.datadict["image_size"][bids]

        gt_b_vid = test_dataset._get_pose_img(bids,
                                              use_crops=False).squeeze(
            dim=0
        )
        gt_b_vid = scale_img(gt_b_vid)
        gt_b_vid = kornia.resize(gt_b_vid, (256, 256))
        gt_b_vid_rgb_real = (
            (gt_b_vid * 255.0)
                .cpu()
                .permute(0, 2, 3, 1)
                .numpy()
                .astype(np.uint8)
        )

        if vunet is not None:
            app_img, extrs, intrs = get_synth_input(test_loader, vunet)
        else:
            extrs = test_dataset._get_extrinsic_params(bids[0])
            intrs = test_dataset._get_intrinsic_params(bids[0])

            app_img = None

        gt_b_vid = prepare_videos(
            b_seq.squeeze().cpu().numpy(), test_loader,
            test_dataset.kinematic_tree,
            False
        )
        # non rgb
        outs= project_onto_image_plane(
            gt_b_vid,
            sizes,
            [[255, 0, 0], [0, 102, 0]],
            f"B: {action}; id {bids[0]}",
            (5, 40),
            extrs,
            intrs,
            test_dataset,
            target_size=(256, 256),
            background_color=255,
            crop=True,
            synth_model=vunet,
            app_img=app_img,
            yield_mask=True
        )

        gt_b_vid = outs[0]
        gt_b_vid_rgb = outs[1]
        masks=outs[2]

        gt_b_vid_rgb_real = [img[:,m[0]:m[1]] for (img,m) in zip(gt_b_vid_rgb_real[::disc_step],masks[::disc_step])]

        transferred = [list(gt_b_vid[::disc_step])]
        transferred_rgb = [gt_b_vid_rgb_real,list(gt_b_vid_rgb[::disc_step])]
        transferrred_col = [gt_b_vid]
        transferred_col_rgb = [gt_b_vid_rgb]

        for xid in startpose_ids:
            extrs = test_dataset._get_extrinsic_params(xid)
            intrs = test_dataset._get_intrinsic_params(xid)

            xids = test_dataset._sample_valid_seq_ids(
                [xid, seq_length])
            x_seq = (
                torch.tensor(test_dataset._get_keypoints(xids))
                    .unsqueeze(dim=0)
                    .to(device)
            )

            # get appearance

            x_seq, _ = prepare_input(x_seq,device)
            inp_seq, _ = prepare_input(b_seq,device)
            x_start = x_seq[:, 0]

            with torch.no_grad():
                tr, *_ = net(inp_seq, x_seq, len=seq_length)

            tr = torch.cat([x_start.unsqueeze(dim=1), tr], dim=1)
            tr = tr.squeeze(dim=0).cpu().numpy()

            tr_poses = prepare_videos(tr, test_loader,
                                      test_dataset.kinematic_tree,
                                      False)

            sizes = test_dataset.datadict["image_size"][xids]

            # preparation
            project_2d, project_2d_rgb = project_onto_image_plane(
                tr_poses,
                sizes,
                None,
                f"id_xstart: {xids[0]}",
                (5, 40),
                extrs,
                intrs,
                test_dataset,
                target_size=(256, 256),
                background_color=255,
                synth_model=vunet,
                app_img=app_img,
                crop=True
            )
            transferred.append(list(project_2d[::disc_step]))
            transferred_rgb.append(
                list(project_2d_rgb[::disc_step]))
            transferrred_col.append(project_2d)
            transferred_col_rgb.append(project_2d_rgb)



        enrolled_kps = np.concatenate(
            [np.concatenate(vid, axis=1) for vid in transferred],
            axis=0)
        enrolled_kps_rgb = np.concatenate(
            [np.concatenate(vid, axis=1) for vid in
             transferred_rgb], axis=0)

        transferred_vid = np.concatenate(transferrred_col,axis=1)
        transferred_vid_rgb = np.concatenate(transferred_col_rgb, axis=1)

        t = time.strftime("%y-%m-%d-%M-%S")
        if unique_name:
            n = path.join(save_dir, f"enrollment_vid#{nr}-{t}.mp4")
            n_rgb = path.join(save_dir, f"enrollment-rgb_vid#{nr}-{t}.mp4")
        else:
            n = path.join(save_dir, f"enrollment_vid#{nr}.mp4")
            n_rgb = path.join(save_dir, f"enrollment_vid-rgb#{nr}.mp4")

        writer = cv2.VideoWriter(
            n,
            cv2.VideoWriter_fourcc(*"MP4V"),
            12,
            (transferred_vid.shape[2], transferred_vid.shape[1]),
        )

        # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

        for frame in transferred_vid:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()

        writer = cv2.VideoWriter(
            n_rgb,
            cv2.VideoWriter_fourcc(*"MP4V"),
            12,
            (transferred_vid_rgb.shape[2], transferred_vid_rgb.shape[1]),
        )

        # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

        for frame in transferred_vid_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()

        enrolled_kps = cv2.cvtColor(enrolled_kps, cv2.COLOR_RGB2BGR)
        enrolled_kps_rgb = cv2.cvtColor(enrolled_kps_rgb,
                                        cv2.COLOR_RGB2BGR)


        if unique_name:
            n = path.join(save_dir, f"enrollment#{nr}-{t}.png")
            n_rgb = path.join(save_dir, f"enrollment-rgb#{nr}-{t}.png")
        else:
            n = path.join(save_dir, f"enrollment#{nr}.png")
            n_rgb = path.join(save_dir, f"enrollment-rgb#{nr}.png")

        cv2.imwrite(n,
                    enrolled_kps)
        cv2.imwrite(n_rgb,
                    enrolled_kps_rgb)