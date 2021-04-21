import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import numpy as np
import cv2
import kornia
from os import path, makedirs


from lib.logging import prepare_videos, get_synth_input
from data import get_dataset
from data.samplers import SequenceSampler
from data.data_conversions_3d import revert_output_format, convert_to_3d
from experiments.experiment import Experiment
from lib.utils import scale_img, slerp, text_to_vid, prepare_input
from data.data_conversions_3d import project_onto_image_plane
from models.vunets import VunetOrg, VunetAlter


__synth_models__ = {"vunet": VunetOrg, "cvbae_vunet": VunetAlter}


def nearest_neighbours(experiment:Experiment, net, save_dir,xids=None,bids=None):

    dataset, image_transforms = get_dataset(experiment.config["data"])

    t_datakeys = (
        [key for key in experiment.data_keys]
        + ["action"]
        + [
            "sample_ids",
            "extrinsics",
            "intrinsics",
            "extrinsics_paired",
            "intrinsics_paired",
        ]
    )
    test_dataset = dataset(
        image_transforms,
        data_keys=t_datakeys,
        mode="test",
        sequential_frame_lag=dataset.sequential_frame_lag,
        crop_app=False,
        label_transfer=True,
        **experiment.config["data"]
    )

    rand_sampler_test = RandomSampler(data_source=test_dataset)
    seq_sampler_test = SequenceSampler(
        test_dataset, rand_sampler_test, batch_size=1, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, num_workers=0, batch_sampler=seq_sampler_test
    )

    zb_file = path.join(save_dir, "zbs.npy")
    seq_file = path.join(save_dir, "sequences.npy")

    # get embeddings
    #if not path.isfile(zb_file) or not path.isfile(seq_file):
    train_dataset = dataset(
        image_transforms,
        data_keys=experiment.data_keys + ["action"] + ["sample_ids"],
        mode="train",
        label_transfer=True,
        crop_app=True,
        **experiment.config["data"]
    )


    if not path.isfile(zb_file):
        print("compute nearest neighbours in beta space.")
        rand_sampler_train = SequentialSampler(data_source=train_dataset)
        seq_sampler_train = SequenceSampler(
            train_dataset,
            rand_sampler_train,
            batch_size=1024,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset, num_workers=experiment.config["data"]["n_data_workers"] , batch_sampler=seq_sampler_train
        )

        betas_train = None
        for i, batch in enumerate(
            tqdm(train_loader, desc="Generating embeddings on train-set")
        ):

            poses = batch["keypoints"]

            poses_in, _ = prepare_input(poses,experiment.device)

            with torch.no_grad():
                b, mu, logstd, pre = net.infer_b(poses_in, False)

            if betas_train is None:
                betas_train = mu.cpu().numpy()
            else:
                betas_train = np.concatenate(
                    [betas_train, mu.cpu().numpy()], axis=0
                )

        # save embeddings, in case model is re-evaluated
        np.save(zb_file, betas_train)
    else:
        print("load precomputed nearest neighbours in beta space.")
        betas_train = np.load(zb_file)

    # generate sequence file to find nearest neighbour in pose space
    if not path.isfile(seq_file):

        print("compute nearest neighbours in keypoint space.")

        seqs_train = []
        for nr, batch in enumerate(
            tqdm(
                train_loader,
                desc=f"Generate sequences with sequence length {train_dataset.seq_length[-1]} for the entire train dataset.",
            )
        ):
            poses = batch["keypoints"].cpu().numpy()
            # reconvert poses to world frame
            poses = np.stack(
                [
                    revert_output_format(
                        p,
                        train_dataset.data_mean,
                        train_dataset.data_std,
                        train_dataset.dim_to_ignore,
                    ).reshape(
                        p.shape[0], len(train_dataset.joint_model.kps_to_use), 3
                    )
                    for p in poses
                ],
                axis=0,
            )

            seqs_train.append(poses)

        seqs_train = np.concatenate(seqs_train, axis=0)
        np.save(seq_file, seqs_train)
    else:
        print("load precomputed nearest neighbours in keypoint space.")
        seqs_train = np.load(seq_file)

    # compute mean pelvis to align poses there when comparing in pose space

    kp_id = int(
        np.nonzero(
            np.asarray(test_dataset.joint_model.kp_to_joint) == "pelvis"
        )[0]
    )
    mean_pelvis = np.expand_dims(
        np.expand_dims(
            np.mean(seqs_train[:, 0, kp_id], axis=0, keepdims=True), axis=1
        ),
        axis=1,
    )

    seqs_aligned = (
        seqs_train
        + mean_pelvis
        - np.expand_dims(seqs_train[:, :, kp_id], axis=2)
    )

    if xids is None and bids is None:
        bids = np.random.choice(np.arange(len(test_dataset)), 100,
                                replace=False)
        xids = np.random.choice(np.arange(len(test_dataset)),100,replace=False)
        n_examples = 100
    elif bids is None:
        n_examples = len(xids)
        bids = list(np.random.choice(np.arange(len(test_dataset)),n_examples,replace=False))
    elif xids is None:
        n_examples = len(bids)
        xids = list(np.random.choice(np.arange(len(test_dataset)), n_examples,
                                     replace=False))
    else:
        assert len(bids) == len(xids)
        n_examples = len(bids)
    assert n_examples % 5 == 0

    nearest_ns = []
    generated = []
    behavior_s = []
    nearest_poses = []
    posterior_sampled = []
    seq_length = test_dataset.seq_length[0]

    for i, (xid,bid) in enumerate(
        tqdm(
            zip(xids, bids),
            desc="Generating Sequences and get nearest neighbours.",
            total=n_examples,
        )
    ):

        ids_start = test_dataset._sample_valid_seq_ids([xid,seq_length])
        ids_beh = test_dataset._sample_valid_seq_ids([bid,seq_length])

        poses_start = torch.from_numpy(test_dataset._get_keypoints(ids_start)).to(torch.float32).unsqueeze(0)
        poses_beh = torch.from_numpy(test_dataset._get_keypoints(ids_beh)).to(torch.float32).unsqueeze(0)

        id1 = int(ids_start[0])
        id2 = int(ids_beh[0])

        poses_in, _ = prepare_input(poses_start,experiment.device)
        poses_b, _ = prepare_input(poses_beh,experiment.device)

        seq_len = poses_b.shape[1]
        with torch.no_grad():
            xs, _, _, b, mu, logstd, pre = net(poses_b, poses_in, len=seq_len)
            # sample around posterior with same mean
            x_posts = []
            for nr in range(5):
                # sample from posterior with doubled std around inferred mean to show similarity of nearby behaviors
                b, mu, logstd, pre = net.infer_b(poses_b, sample=False)
                # double logstd
                # fixme stds seem to be very small
                logstd = logstd * 0.8
                b_sampled = net.b_enc.reparametrize(mu, logstd)

                x_same, *_ = net.generate_seq(
                    b_sampled, poses_in, len=seq_len, start_frame=0
                )

                # x_same, *_ = net(poses_b, poses_in, len=seq_len)
                x_posts.append(
                    torch.cat([poses_in[:, 0].unsqueeze(dim=1), x_same], dim=1)
                    .squeeze(dim=0)
                    .cpu()
                    .numpy()
                )

        xs = (
            torch.cat([poses_in[:, 0].unsqueeze(dim=1), xs], dim=1)
            .squeeze(dim=0)
            .cpu()
            .numpy()
        )

        xs = revert_output_format(
            xs,
            test_dataset.data_mean,
            test_dataset.data_std,
            test_dataset.dim_to_ignore,
        )
        if "angle" in experiment.keypoint_type:
            xs = convert_to_3d(xs, test_dataset.kinematic_tree, swap_yz=False)
        else:
            xs = xs.reshape(
                xs.shape[0], len(test_dataset.joint_model.kps_to_use), 3
            )

        xb = poses_beh.squeeze(dim=0).cpu().numpy()

        # get nearest neighbour in beta space
        arg_min = np.argmin(
            np.linalg.norm(
                betas_train - mu.squeeze(dim=0).cpu().numpy(), axis=1
            )
        )
        ids = train_dataset._sample_valid_seq_ids([arg_min, seq_len])
        nearest_xs = train_dataset._get_keypoints(ids)

        # get nearest neighbours of behavior sequence in pose space

        xs_aligned = (
            np.expand_dims(xs, axis=0)
            + mean_pelvis
            - np.expand_dims(np.expand_dims(xs[:, kp_id], axis=1), axis=0)
        )

        norms = np.sum(
            np.linalg.norm(xs_aligned - seqs_aligned, axis=-1), axis=(1, 2)
        )
        arg_min_pose = np.argmin(norms)
        nearest_pose = seqs_train[arg_min_pose]

        image_sizes = train_dataset.datadict["image_size"][ids]
        extrs = train_dataset.datadict["extrinsics_univ"][arg_min]
        intrs = train_dataset.datadict["intrinsics_univ"][arg_min]

        nearest_xs = revert_output_format(
            nearest_xs,
            train_dataset.data_mean,
            train_dataset.data_std,
            train_dataset.dim_to_ignore,
        )

        xb = revert_output_format(
            xb,
            test_dataset.data_mean,
            test_dataset.data_std,
            test_dataset.dim_to_ignore,
        )

        # x_posts = [
        #     revert_output_format(
        #         xp,
        #         train_dataset.data_mean,
        #         train_dataset.data_std,
        #         train_dataset.dim_to_ignore,
        #     )
        #     for xp in x_posts
        # ]
        if "angle" in experiment.keypoint_type:
            nearest_xs = convert_to_3d(
                nearest_xs, train_dataset.kinematic_tree, swap_yz=False
            )

            xb = convert_to_3d(xb, test_dataset.kinematic_tree, swap_yz=False)
            nearest_pose = convert_to_3d(
                nearest_pose, train_dataset.kinematic_tree, swap_yz=False
            )
            x_posts = [
                convert_to_3d(xp, train_dataset.kinematic_tree, swap_yz=False)
                for xp in x_posts
            ]
        else:
            nearest_xs = nearest_xs.reshape(
                nearest_xs.shape[0],
                len(train_dataset.joint_model.kps_to_use),
                3,
            )
            xb = xb.reshape(
                xb.shape[0], len(train_dataset.joint_model.kps_to_use), 3
            )
            x_posts = [
                x.reshape(
                    x.shape[0], len(train_dataset.joint_model.kps_to_use), 3
                )
                for x in x_posts
            ]

        video_nearest, _ = project_onto_image_plane(
            nearest_xs,
            image_sizes,
            None,
            "",
            (5, 40),
            extrs,
            intrs,
            train_dataset,
            target_size=(256, 256),
            background_color=255,
            font_size=2,
        )
        nearest_ns.append(video_nearest)

        video_generated, _ = project_onto_image_plane(
            xs,
            image_sizes,
            None,
            "",
            (5, 40),
            extrs,
            intrs,
            train_dataset,
            target_size=(256, 256),
            background_color=255,
            font_size=2,
        )
        generated.append(video_generated)

        video_behavior, _ = project_onto_image_plane(
            xb,
            image_sizes,
            [[255, 0, 0], [0, 102, 0]],
            "",
            (5, 40),
            extrs,
            intrs,
            train_dataset,
            target_size=(256, 256),
            background_color=255,
            font_size=2,
        )
        behavior_s.append(video_behavior)

        video_nearest_pose, _ = project_onto_image_plane(
            nearest_pose,
            image_sizes,
            None,
            "",
            (5, 40),
            extrs,
            intrs,
            train_dataset,
            target_size=(256, 256),
            background_color=255,
            font_size=2,
        )
        nearest_poses.append(video_nearest_pose)

        x_posts = [
            project_onto_image_plane(
                xp,
                image_sizes,
                None,
                "",
                (5, 40),
                extrs,
                intrs,
                train_dataset,
                target_size=(256, 256),
                background_color=255,
                font_size=2,
            )[0]
            for xp in x_posts
        ]

        posterior_sampled.append(np.concatenate(x_posts, axis=2))

    for nr, i in enumerate(range(0, len(nearest_ns), 5)):
        nearests = np.concatenate(nearest_ns[i : i + 5], axis=2)
        generateds = np.concatenate(generated[i : i + 5], axis=2)
        behaviors = np.concatenate(behavior_s[i : i + 5], axis=2)
        seqs_pose = np.concatenate(nearest_poses[i : i + 5], axis=2)
        full = np.concatenate(
            [behaviors, nearests, generateds, seqs_pose], axis=1
        )

        filename = f"generated_and_nearest_{i}.mp4"
        savename = path.join(save_dir, filename)

        writer = cv2.VideoWriter(
            savename,
            cv2.VideoWriter_fourcc(*"MP4V"),
            12,
            (full.shape[2], full.shape[1]),
        )

        for frame in full:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()

        sampled_name = f"posterior_samples_{i}.mp4"
        savename = path.join(save_dir, sampled_name)

        writer = cv2.VideoWriter(
            savename,
            cv2.VideoWriter_fourcc(*"MP4V"),
            12,
            (posterior_sampled[nr].shape[2], posterior_sampled[nr].shape[1]),
        )

        for frame in posterior_sampled[nr]:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()



def make_enrollment_figure(
    test_loader,
    vunet,
    net,
    save_dir,
    device,
    grid_bids=None,
    grid_sids=None,
    n_samples=10,
    crop=False,
):
    import time

    seq_length = test_loader.dataset.seq_length[0]
    disc_step = 6
    test_dataset = test_loader.dataset
    if grid_bids is None:
        print("Randomize plot generation")
        grid_bids = np.random.choice(
            np.arange(test_dataset.datadict["img_paths"].shape[0]),
            n_samples,
            replace=False,
        )

    # if grid_sids is not None and len(grid_sids) < n_samples:
    #     diff = n_samples -len(grid_sids)
    #     additional = list(np.random.choice(np.arange(test_dataset.datadict["keypoints"].shape[0]),diff,replace=False))
    #     grid_sids.extend(additional)

    for nr, i in enumerate(
        tqdm(grid_bids, desc=f"Generating enrollment plots.")
    ):

        if grid_sids is None:
            startpose_ids = np.random.choice(
                np.arange(test_dataset.datadict["img_paths"].shape[0]),
                40,
                replace=False,
            )
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

        gt_b_vid = test_dataset._get_pose_img(bids, use_crops=False).squeeze(
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

        assert vunet is not None
        apps, extrs, intrs = get_synth_input(
            test_loader, vunet, all_cameras=True
        )

        gt_b_vid = prepare_videos(
            b_seq.squeeze().cpu().numpy(),
            test_loader,
            test_dataset.kinematic_tree,
            False,
        )
        # non rgb

        outs = project_onto_image_plane(
            gt_b_vid,
            sizes,
            [[255, 0, 0], [0, 102, 0]],
            f"B: {action}; id {bids[0]}",
            (5, 40),
            extrs[0],
            intrs[0],
            test_dataset,
            target_size=(256, 256),
            background_color=255,
            crop=crop,
            synth_model=vunet,
            app_img=apps[0],
            yield_mask=True,
        )

        gt_b_vid = outs[0]
        gt_b_vid_rgb = outs[1]
        masks = outs[2]

        gt_b_vid_rgb_real = [
            img[:, m[0] : m[1]]
            for (img, m) in zip(
                gt_b_vid_rgb_real[::disc_step], masks[::disc_step]
            )
        ]
        overlay = [
            cv2.addWeighted(gt_stick, 0.5, gt_rgb, 0.5, 0)
            for gt_stick, gt_rgb in zip(
                gt_b_vid[::disc_step], gt_b_vid_rgb_real[::disc_step]
            )
        ]

        # gt_b_vid_rgb_real = [img[:, m[0]:m[1]] for (img, m) in
        #                      zip(gt_b_vid_rgb_real,
        #                          masks)]
        # overlay = [cv2.addWeighted(gt_stick, 0.5, gt_rgb, 0.5, 0) for
        #            gt_stick, gt_rgb in
        #            zip(gt_b_vid, gt_b_vid_rgb_real)]

        for xid in startpose_ids:
            # extrs = test_dataset._get_extrinsic_params(xid)
            # intrs = test_dataset._get_intrinsic_params(xid)

            transferred = [list(gt_b_vid[::disc_step])]
            transferred_rgb = [
                gt_b_vid_rgb_real,
                list(gt_b_vid_rgb[::disc_step]),
            ]

            # transferred = [list(gt_b_vid)]
            # transferred_rgb = [gt_b_vid_rgb_real,
            #                    list(gt_b_vid_rgb)]

            transferrred_col = [gt_b_vid]
            transferred_col_rgb = [gt_b_vid_rgb]
            overlays = []

            xids = test_dataset._sample_valid_seq_ids([xid, seq_length])
            x_seq = (
                torch.tensor(test_dataset._get_keypoints(xids))
                .unsqueeze(dim=0)
                .to(device)
            )

            apps, extrs, intrs = get_synth_input(
                test_loader, vunet, all_cameras=True
            )

            # get appearance
            for ec, ic, app in zip(extrs, intrs, apps):
                sseq, _ = prepare_input(x_seq,device)
                inp_seq, _ = prepare_input(b_seq,device)
                x_start = sseq[:, 0]

                with torch.no_grad():
                    seq_len = inp_seq.shape[1]

                    tr, *_ = net(inp_seq, sseq, len=seq_length)

                tr = torch.cat([x_start.unsqueeze(dim=1), tr], dim=1)
                tr = tr.squeeze(dim=0).cpu().numpy()

                tr_poses = prepare_videos(
                    tr, test_loader, test_dataset.kinematic_tree, False
                )

                sizes = test_dataset.datadict["image_size"][xids]

                # preparation
                project_2d, project_2d_rgb, overlay = project_onto_image_plane(
                    tr_poses,
                    sizes,
                    None,
                    f"id_xstart: {xids[0]}",
                    (5, 40),
                    ec,
                    ic,
                    test_dataset,
                    target_size=(256, 256),
                    background_color=255,
                    synth_model=vunet,
                    app_img=app,
                    crop=crop,
                    overlay=True,
                )

                transferred.append(list(project_2d[::disc_step]))
                transferred_rgb.append(list(project_2d_rgb[::disc_step]))
                transferrred_col.append(project_2d)
                transferred_col_rgb.append(project_2d_rgb)
                overlays.append(list(overlay[::disc_step]))

            enrolled_kps = np.concatenate(
                [np.concatenate(vid, axis=1) for vid in transferred], axis=0
            )
            enrolled_kps_rgb = np.concatenate(
                [np.concatenate(vid, axis=1) for vid in transferred_rgb], axis=0
            )
            enrolled_overlay = np.concatenate(
                [np.concatenate(vid, axis=1) for vid in overlays], axis=0
            )

            transferred_vid = np.concatenate(transferrred_col, axis=1)
            transferred_vid_rgb = np.concatenate(transferred_col_rgb, axis=1)

            # t = time.strftime("%y-%m-%d-%M-%S")
            n = path.join(save_dir, f"enrollment_vid-bid{i}-sid{xid}.mp4")
            n_rgb = path.join(
                save_dir, f"enrollment-rgb_vid-bid{i}-sid{xid}.mp4"
            )

            write_video(transferred_vid,n)
            write_video(transferred_vid_rgb,n_rgb)


            enrolled_kps = cv2.cvtColor(enrolled_kps, cv2.COLOR_RGB2BGR)
            enrolled_kps_rgb = cv2.cvtColor(enrolled_kps_rgb, cv2.COLOR_RGB2BGR)
            enrolled_kps_o = cv2.cvtColor(enrolled_overlay, cv2.COLOR_RGB2BGR)

            n = path.join(save_dir, f"enrollment-bid{i}-sid{xid}.png")
            n_rgb = path.join(save_dir, f"enrollment-rgb-bid{i}-sid{xid}.png")
            n_rgb_o = path.join(
                save_dir, f"enrollment-overlay-bid{i}-sid{xid}.png"
            )

            cv2.imwrite(n, enrolled_kps)
            cv2.imwrite(n_rgb, enrolled_kps_rgb)
            cv2.imwrite(n_rgb_o, enrolled_kps_o)


def latent_interpolate_eval(model, loader: DataLoader, device, dirs, vunet,n_samples=10,sids1=None,sids2=None,crop=False,suppress_text=False):
    # get data
    assert model.ib
    model.eval()

    test_dataset = loader.dataset


    if sids1 is None:
        sids1 = np.random.choice(
            np.arange(test_dataset.datadict["img_paths"].shape[0]),
            n_samples,
            replace=False,
        )
    elif len(sids1) < n_samples:
        sids1 = sids1 + list(np.random.choice(
            np.arange(test_dataset.datadict["img_paths"].shape[0]),
            n_samples-len(sids1),
            replace=False,
        ))

    if sids2 is None:
        sids2 = np.random.choice(
            np.arange(test_dataset.datadict["img_paths"].shape[0]),
            n_samples,
            replace=False,
        )
    elif len(sids2) < n_samples:
        sids2 = sids2 + list(np.random.choice(
            np.arange(test_dataset.datadict["img_paths"].shape[0]),
            n_samples-len(sids2),
            replace=False,
        ))

    savepath = dirs
    seq_length = test_dataset.seq_length[0]
    #it = iter(loader)
    # flag indicates if generated keypoints shall be reprojected into the image

    kin_tree = loader.dataset.kinematic_tree

    for i,(sid1,sid2) in enumerate(tqdm(
        zip(sids1,sids2), desc="Generate latent interpolations."
    )):
        seq_ids1 = test_dataset._sample_valid_seq_ids([sid1,seq_length])
        seq_ids2 = test_dataset._sample_valid_seq_ids([sid2, seq_length])

        kps1 = torch.from_numpy(test_dataset._get_keypoints(seq_ids1)).to(torch.float32).unsqueeze(0)
        # kps1_change = batch["kp_change"].to(dtype=torch.float32)
        #ids1 = batch["sample_ids"][0, 1:, ...].cpu().numpy()
        id1 = sid1
        label_id1 = test_dataset.datadict["action"][id1]

        kps2 = torch.from_numpy(test_dataset._get_keypoints(seq_ids2)).to(torch.float32).unsqueeze(0)
        # kps2_change = batch["paired_change"].to(dtype=torch.float32)
        #ids2 = batch["paired_sample_ids"][0, 1:, ...].cpu().numpy()
        id2 = sid2
        label_id2 = test_dataset.datadict["action"][id2]
        n_kps = kps1.shape[2]

        data1, _ = prepare_input(kps1,device)
        data2, _ = prepare_input(kps2,device)

        data_b_1 = data1
        data_b_2 = data2

        apps, extrs, intrs = get_synth_input(loader, vunet, all_cameras=True)

        cam_savedirs = {cam_nr:
            path.join(savepath, f"camera-{cam_nr}")
            for cam_nr in range(len(extrs))
        }
        [makedirs(d, exist_ok=True) for d in cam_savedirs.values()]

        steps = np.linspace(0, 1, 6)
        with torch.no_grad():
            l = data_b_1.shape[1]

            x1_start = data1[:, 0]
            b2, *_ = model.infer_b(data_b_2, False)
            x2_start = data2[:, 0]
            b1, *_ = model.infer_b(data_b_1, False)

            inter1_to_2_proj = {key:[]for key in cam_savedirs}
            inter2_to_1_proj = {key:[]for key in cam_savedirs}
            inter1_to_2_proj_rgb = {key:[]for key in cam_savedirs}
            inter2_to_1_proj_rgb = {key:[]for key in cam_savedirs}
            inter1_to_2_proj_rgb_overlayed = {key:[]for key in cam_savedirs}
            inter2_to_1_proj_rgb_overlayed = {key:[]for key in cam_savedirs}

            inter1_to_2_proj_linear = {key: [] for key in cam_savedirs}
            inter2_to_1_proj_linear = {key: [] for key in cam_savedirs}
            inter1_to_2_proj_rgb_linear = {key: [] for key in cam_savedirs}
            inter2_to_1_proj_rgb_linear = {key: [] for key in cam_savedirs}
            inter1_to_2_proj_rgb_overlayed_linear = {key: [] for key in cam_savedirs}
            inter2_to_1_proj_rgb_overlayed_linear = {key: [] for key in cam_savedirs}

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

                b1_to_2_linear = b1 * (1 - s) + b2 * s
                b2_to_1_linear = b1 * s + b2 * (1 - s)

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
                    test_dataset.data_mean,
                    test_dataset.data_std,
                    test_dataset.dim_to_ignore,
                )
                p2_to_1 = revert_output_format(
                    p2_to_1,
                    test_dataset.data_mean,
                    test_dataset.data_std,
                    test_dataset.dim_to_ignore,
                )

                ### linear interpolation

                p1_to_2_linear, *_ = model.generate_seq(
                    b1_to_2_linear, data1, data1.shape[1], start_frame=0
                )
                p2_to_1_linear, *_ = model.generate_seq(
                    b2_to_1_linear, data2, data2.shape[1], start_frame=0
                )
                p1_to_2_linear = (
                    torch.cat([x1_start.unsqueeze(dim=1), p1_to_2_linear], dim=1)
                        .squeeze(dim=0)
                        .cpu()
                        .numpy()
                )
                p2_to_1_linear = (
                    torch.cat([x2_start.unsqueeze(dim=1), p2_to_1_linear], dim=1)
                        .squeeze(dim=0)
                        .cpu()
                        .numpy()
                )

                p1_to_2_linear = revert_output_format(
                    p1_to_2_linear,
                    test_dataset.data_mean,
                    test_dataset.data_std,
                    test_dataset.dim_to_ignore,
                )
                p2_to_1_linear = revert_output_format(
                    p2_to_1_linear,
                    test_dataset.data_mean,
                    test_dataset.data_std,
                    test_dataset.dim_to_ignore,
                )





                if "angle" in test_dataset.keypoint_key:
                    p1_to_2 = convert_to_3d(p1_to_2, kin_tree, swap_yz=False)
                    p2_to_1 = convert_to_3d(p2_to_1, kin_tree, swap_yz=False)
                    p1_to_2_linear = convert_to_3d(p1_to_2_linear, kin_tree, swap_yz=False)
                    p2_to_1_linear = convert_to_3d(p2_to_1_linear, kin_tree, swap_yz=False)

                else:
                    p1_to_2 = p1_to_2.reshape(
                        p1_to_2.shape[0],
                        len(test_dataset.joint_model.kps_to_use),
                        3,
                    )
                    p2_to_1 = p2_to_1.reshape(
                        p1_to_2.shape[0],
                        len(test_dataset.joint_model.kps_to_use),
                        3,
                    )
                    p1_to_2_linear = p1_to_2_linear.reshape(
                        p1_to_2_linear.shape[0],
                        len(test_dataset.joint_model.kps_to_use),
                        3,
                    )
                    p2_to_1_linear= p2_to_1_linear.reshape(
                        p1_to_2_linear.shape[0],
                        len(test_dataset.joint_model.kps_to_use),
                        3,
                    )

                for ec, ic, app, cid in zip(extrs, intrs, apps, cam_savedirs):
                    i_img_2d1, i_img_2d1_rgb, i_img_2d1_rgb_overlay = project_onto_image_plane(
                        p1_to_2,
                        test_dataset.datadict["image_size"][seq_ids1],
                        [[int((1.0 - s) * 255.0), int(s * 255.0), 0]],
                        "" if suppress_text else f"id_xstart: {int(id1)}",
                        (5, 40),
                        ec,
                        ic,
                        test_dataset,
                        target_size=(256, 256),
                        background_color=255,
                        synth_model=vunet,
                        app_img=app,
                        crop=crop,
                        overlay=True,
                    )

                    inter1_to_2_proj[cid].append(i_img_2d1)
                    inter1_to_2_proj_rgb[cid].append(i_img_2d1_rgb)
                    inter1_to_2_proj_rgb_overlayed[cid].append(i_img_2d1_rgb_overlay)

                    i_img_2d2, i_img_2d2_rgb, i_img_2d2_rgb_overlay = project_onto_image_plane(
                        p2_to_1,
                        test_dataset.datadict["image_size"][seq_ids2],
                        [[int(s * 255.0), int((1.0 - s) * 255.0), 0]],
                        "",
                        (10, 10),
                        ec,
                        ic,
                        test_dataset,
                        target_size=(256, 256),
                        background_color=255,
                        synth_model=vunet,
                        app_img=app,
                        crop=crop,
                        overlay=True,
                    )
                    inter2_to_1_proj[cid].append(i_img_2d2)
                    inter2_to_1_proj_rgb[cid].append(i_img_2d2_rgb)
                    inter2_to_1_proj_rgb_overlayed[cid].append(i_img_2d2_rgb_overlay)




                    #same for linear interpolated stuff to see which method performs better
                    i_img_2d1_linear, i_img_2d1_rgb_linear, i_img_2d1_rgb_overlay_linear = project_onto_image_plane(
                        p1_to_2_linear,
                        test_dataset.datadict["image_size"][seq_ids1],
                        [[int((1.0 - s) * 255.0), int(s * 255.0), 0]],
                        "" if suppress_text else f"id_xstart: {int(id1)}",
                        (5, 40),
                        ec,
                        ic,
                        test_dataset,
                        target_size=(256, 256),
                        background_color=255,
                        synth_model=vunet,
                        app_img=app,
                        crop=crop,
                        overlay=True,
                    )

                    inter1_to_2_proj_linear[cid].append(i_img_2d1_linear)
                    inter1_to_2_proj_rgb_linear[cid].append(i_img_2d1_rgb_linear)
                    inter1_to_2_proj_rgb_overlayed_linear[cid].append(
                        i_img_2d1_rgb_overlay_linear)

                    i_img_2d2_linear, i_img_2d2_rgb_linear, i_img_2d2_rgb_overlay_linear = project_onto_image_plane(
                        p2_to_1_linear,
                        test_dataset.datadict["image_size"][seq_ids2],
                        [[int(s * 255.0), int((1.0 - s) * 255.0), 0]],
                        "",
                        (10, 10),
                        ec,
                        ic,
                        test_dataset,
                        target_size=(256, 256),
                        background_color=255,
                        synth_model=vunet,
                        app_img=app,
                        crop=crop,
                        overlay=True,
                    )
                    inter2_to_1_proj_linear[cid].append(i_img_2d2_linear)
                    inter2_to_1_proj_rgb_linear[cid].append(i_img_2d2_rgb_linear)
                    inter2_to_1_proj_rgb_overlayed_linear[cid].append(
                        i_img_2d2_rgb_overlay_linear)







        for key in cam_savedirs:

            upper = np.concatenate(inter1_to_2_proj[key], axis=2)
            lower = np.concatenate(inter2_to_1_proj[key][::-1], axis=2)
            inter_full_proj = np.concatenate(
                [upper, lower], axis=1
            )

            upper = np.concatenate(inter1_to_2_proj_rgb[key], axis=2)
            lower = np.concatenate(inter2_to_1_proj_rgb[key][::-1], axis=2)
            inter_full_proj_rgb = np.concatenate(
                [upper, lower], axis=1
            )

            upper = np.concatenate(inter1_to_2_proj_rgb_overlayed[key], axis=2)
            lower = np.concatenate(inter2_to_1_proj_rgb_overlayed[key][::-1], axis=2)
            inter_full_proj_rgb_overlay = np.concatenate(
                [upper, lower], axis=1
            )


            action1 = loader.dataset.action_id_to_action[label_id1]
            action2 = loader.dataset.action_id_to_action[label_id2]

            # text to videos
            if not suppress_text:
                inter_full_proj = text_to_vid(
                inter_full_proj,
                f"2d: Interpolation #{i}: 1st: {action1} -> {action2} with x1_start; 2nd vice versa, id1: {int(id1)}; id2: {int(id2)}",
                (5, 40),
            )

            filename = f"interpolations2d{i}-id1{int(id1)}-id2{int(id2)}.mp4"
            savename = path.join(cam_savedirs[key], filename)


            write_video(inter_full_proj,savename)

            if not suppress_text:
                inter_full_proj_rgb = text_to_vid(
                inter_full_proj_rgb,
                f"2d: Interpolation #{i}: 1st: {action1} -> {action2} with x1_start; 2nd vice versa, d1: {int(id1)}; id2: {int(id2)}",
                (5, 40),
            )

            filename = (
                f"interpolations2d_rgb{i}-id1{int(id1)}-id2{int(id2)}.mp4"
            )
            savename = path.join(cam_savedirs[key], filename)

            write_video(inter_full_proj_rgb,savename)

            if not suppress_text:
                inter_full_proj_rgb_overlay = text_to_vid(inter_full_proj_rgb_overlay,
                    f"2d: Interpolation #{i}: 1st: {action1} -> {action2} with x1_start; 2nd vice versa, d1: {int(id1)}; id2: {int(id2)}",
                    (5, 40),)

            filename = (
                f"interpolations2d_rgb_overlay{i}-id1{int(id1)}-id2{int(id2)}.mp4"
            )
            savename = path.join(cam_savedirs[key], filename)

            write_video(inter_full_proj_rgb_overlay,savename)


            # same for linearly interpolated examples

            upper = np.concatenate(inter1_to_2_proj_linear[key], axis=2)
            lower = np.concatenate(inter2_to_1_proj_linear[key][::-1], axis=2)
            inter_full_proj = np.concatenate(
                [upper, lower], axis=1
            )

            upper = np.concatenate(inter1_to_2_proj_rgb_linear[key], axis=2)
            lower = np.concatenate(inter2_to_1_proj_rgb_linear[key][::-1], axis=2)
            inter_full_proj_rgb = np.concatenate(
                [upper, lower], axis=1
            )

            upper = np.concatenate(inter1_to_2_proj_rgb_overlayed_linear[key], axis=2)
            lower = np.concatenate(inter2_to_1_proj_rgb_overlayed_linear[key][::-1], axis=2)
            inter_full_proj_rgb_overlay = np.concatenate(
                [upper, lower], axis=1
            )

            action1 = loader.dataset.action_id_to_action[label_id1]
            action2 = loader.dataset.action_id_to_action[label_id2]

            # text to videos
            if not suppress_text:
                inter_full_proj = text_to_vid(
                    inter_full_proj,
                    f"2d: Interpolation linear #{i}: 1st: {action1} -> {action2} with x1_start; 2nd vice versa, id1: {int(id1)}; id2: {int(id2)}",
                    (5, 40),
                )

            filename = f"linear_interpolations2d{i}-id1{int(id1)}-id2{int(id2)}.mp4"
            savename = path.join(cam_savedirs[key], filename)

            write_video(inter_full_proj, savename)
            if not suppress_text:
                inter_full_proj_rgb = text_to_vid(
                    inter_full_proj_rgb,
                    f"2d: Interpolation #{i}: 1st: {action1} -> {action2} with x1_start; 2nd vice versa, d1: {int(id1)}; id2: {int(id2)}",
                    (5, 40),
                )

            filename = (
                f"linear_interpolations2d_rgb{i}-id1{int(id1)}-id2{int(id2)}.mp4"
            )
            savename = path.join(cam_savedirs[key], filename)

            write_video(inter_full_proj_rgb, savename)

            inter_full_proj_rgb_overlay = text_to_vid(
                inter_full_proj_rgb_overlay,
                f"2d: Interpolation linear#{i}: 1st: {action1} -> {action2} with x1_start; 2nd vice versa, d1: {int(id1)}; id2: {int(id2)}",
                (5, 40), )

            filename = (
                f"linear_interpolations2d_rgb_overlay{i}-id1{int(id1)}-id2{int(id2)}.mp4"
            )
            savename = path.join(cam_savedirs[key], filename)

            write_video(inter_full_proj_rgb_overlay, savename)




def write_video(data,savename,fps=12):
    writer = cv2.VideoWriter(
        savename,
        cv2.VideoWriter_fourcc(*"MP4V"),
        fps,
        (data.shape[2], data.shape[1]),
    )

    # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

    for frame in data:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()

def make_eval_grid(
    model: torch.nn.Module,
    loader: DataLoader,
    device,
    dirs,
    vunet,
    bids=None,
    sids=None,
    n_samples=6,
    iteration = 0,
    suppress_id=False
):

    # get data
    model.eval()

    test_dataset = loader.dataset
    if bids is None:
        all_bids = np.random.choice(
            np.arange(test_dataset.datadict["img_paths"].shape[0]),
            n_samples,
            replace=False,
        )
    elif len(bids) < n_samples:
        all_bids = bids + list(np.random.choice(
            np.arange(test_dataset.datadict["img_paths"].shape[0]),
            n_samples-len(bids),
            replace=False,
        ))
    else:
        all_bids = bids

    if sids is None:
        all_sids = np.random.choice(
            np.arange(test_dataset.datadict["img_paths"].shape[0]),
            n_samples,
            replace=False,
        )
    elif len(sids) < n_samples:
        all_sids = sids + list(np.random.choice(
            np.arange(test_dataset.datadict["img_paths"].shape[0]),
            n_samples-len(sids),
            replace=False,
        ))
    else:
        all_sids = sids

    kin_tree = loader.dataset.kinematic_tree


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
            for spi in all_sids
        ],
        axis=1,
    )

    cols_rgb = [first_col_rgb]

    cols = []
    first_col = [
        np.zeros((loader.dataset.seq_length[-1], 256, 256, 3), dtype=np.uint8)
    ]

    for nr, i in enumerate(tqdm(all_bids, desc=f"Generating eval grid.")):

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
        gt_b_vid = kornia.resize(gt_b_vid, (256, 256))
        gt_b_vid = (
            (gt_b_vid * 255.0)
            .cpu()
            .permute(0, 2, 3, 1)
            .numpy()
            .astype(np.uint8)
        )
        col_rgb = [gt_b_vid]

        app_img, extrs, intrs = get_synth_input(loader, vunet)

        gt_b_vid = prepare_videos(
            b_seq.squeeze().cpu().numpy(), loader, kin_tree, False
        )
        col = [
            project_onto_image_plane(
                gt_b_vid,
                sizes,
                [[255, 0, 0], [0, 102, 0]],
                "" if suppress_id else f"B: {action}; id {bids[0]}",
                (5, 40),
                extrs,
                intrs,
                loader.dataset,
                target_size=(256, 256),
                background_color=255,
            )[0]
        ]

        # sample start pose
        for xid in all_sids:
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
                l = seq_len


                tr, *_ = model(b_seq, x_seq, len=seq_length)

            tr = torch.cat([x_start.unsqueeze(dim=1), tr], dim=1)
            tr = tr.squeeze(dim=0).cpu().numpy()

            tr_poses = prepare_videos(tr, loader, kin_tree, False)

            sizes = loader.dataset.datadict["image_size"][xids]

            # preparation
            project_2d, project_2d_rgb = project_onto_image_plane(
                tr_poses,
                sizes,
                None,
                "" if suppress_id else f"id_xstart: {xids[0]}",
                (5, 40),
                extrs,
                intrs,
                loader.dataset,
                target_size=(256, 256),
                background_color=255,
                synth_model=vunet,
                app_img=app_img,
            )

            project_2d_sp, _ = project_onto_image_plane(
                tr_poses,
                sizes,
                [[255, 0, 0], [0, 102, 0]],
                "" if suppress_id else f"id_xstart: {xids[0]}",
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
                        [project_2d_sp[0]] * loader.dataset.seq_length[-1], axis=0
                    )
                )

            col.append(project_2d)

            col_rgb.append(project_2d_rgb)


        cols_rgb.append(np.concatenate(col_rgb, axis=1))
        cols.append(np.concatenate(col, axis=1))

    min_time = min([c.shape[0] for c in cols])
    first_col = np.concatenate(first_col, axis=1)

    grid = np.concatenate(
        [first_col[:min_time]] + [c[:min_time] for c in cols], axis=2
    )

    grid_rgb = np.concatenate([c[:min_time] for c in cols_rgb], axis=2)

    savepath = dirs["generated"] if isinstance(dirs,dict) else dirs

    filename = f"grid_suppl@it{iteration}.mp4"
    savename = path.join(savepath, filename)

    write_video(grid,savename)


    filename = f"grid_suppl_rgb@it{iteration}.mp4"
    savename = path.join(savepath, filename)

    write_video(grid_rgb,savename)


def sample_examples(
    model: torch.nn.Module,
    loader: DataLoader,
    device,
    dirs,
    vunet,
    flow,
    n_start_poses=50,
    n_samples=20,
    only_flow=False
):


    discretization = 6

    it = iter(loader)
    # flag indicates if generated keypoints shall be reprojected into the image
    savepath = dirs



    for i in tqdm(
        range(n_start_poses), desc="Generate Videos for logging."
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

        with torch.no_grad():
            x1_start = data1[:, 0]
            x2_start = data2[:, 0]
            # task 1: infer self behavior and reconstruct (w/ and w/o target location)
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

        #
        # # already in right form
        p1_gt = kps1.squeeze(dim=0).cpu().numpy()
        p2_gt = kps2.squeeze(dim=0).cpu().numpy()

        label1 = loader.dataset.action_id_to_action[label_id1]
        label2 = loader.dataset.action_id_to_action[label_id2]

        # if sampling:
        x1_samples_array = prepare_videos(
            x1_samples, loader, loader.dataset.kinematic_tree, False
        )
        x2_samples_array = prepare_videos(
            x2_samples, loader, loader.dataset.kinematic_tree, False
        )
        gt1 = prepare_videos(p1_gt,loader,loader.dataset.kinematic_tree,False)
        gt2 = prepare_videos(p2_gt, loader, loader.dataset.kinematic_tree,
                             False)

        x1_fs_arr = prepare_videos(
            x1_flow_samples, loader, loader.dataset.kinematic_tree, False
        )
        x2_fs_arr = prepare_videos(
            x2_flow_samples, loader, loader.dataset.kinematic_tree, False
        )


        if "image_size" not in loader.dataset.datadict.keys():
            raise TypeError(
                "Dataset doesn't contain image sizes, not possible to project 3d points onto 2d images"
            )


        app_img, extrs, intrs = get_synth_input(loader, vunet)


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
                [gt1] + x1_samples_array,
                [loader.dataset.datadict["image_size"][ids1]]
                * (len(x1_samples_array) + 1),
                [[[255, 0, 0], [0, 102, 0]]]
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
                [gt2] + x2_samples_array,
                [loader.dataset.datadict["image_size"][ids2]]
                * (len(x1_samples_array) + 1),
                [[[255, 0, 0], [0, 102, 0]]]
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
                [gt1] + x1_fs_arr,
                [loader.dataset.datadict["image_size"][ids1]]
                * (len(x1_samples_array) + 1),
                [[[255, 0, 0], [0, 102, 0]]]
                + [None] * len(x1_samples_array),
                ["GT | x1"]
                + [
                    f"flow sample#{i + 1} | x1"
                    for i in range(len(x1_samples_array))
                ],
            )
        ]

        # only make enrollments for flow samples

        sample_x1_pf_rgb = [p[1] for p in sample_x1_pf]
        sample_x1_pf = [p[0] for p in sample_x1_pf]

        enroll_flow_x1 = [list(s[::discretization]) for s in sample_x1_pf]
        enroll_flow_x1_rgb = [list(s[::discretization]) for s in sample_x1_pf_rgb]

        enroll_grid_flow_x1 = np.concatenate([np.concatenate(s,axis=1) for s in enroll_flow_x1],axis=0)
        enroll_grid_flow_x1_rgb = np.concatenate(
            [np.concatenate(s, axis=1) for s in enroll_flow_x1_rgb], axis=0)

        enroll_grid_flow_x1 = cv2.cvtColor(enroll_grid_flow_x1,
                                           cv2.COLOR_RGB2BGR)
        enroll_grid_flow_x1_rgb = cv2.cvtColor(enroll_grid_flow_x1_rgb,
                                               cv2.COLOR_RGB2BGR)

        name_enroll_1 = path.join(savepath, f"enrollment-sid{id1}.png")
        cv2.imwrite(name_enroll_1, enroll_grid_flow_x1)
        name_enroll_1 = path.join(savepath, f"enrollment_rgb-sid{id1}.png")
        cv2.imwrite(name_enroll_1, enroll_grid_flow_x1_rgb)


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
                [gt2] + x2_fs_arr,
                [loader.dataset.datadict["image_size"][ids1]]
                * (len(x1_samples_array) + 1),
                [[[255, 0, 0], [0, 102, 0]]]
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
        enroll_flow_x2 = [list(s[::discretization]) for s in sample_x2_pf]
        enroll_flow_x2_rgb = [list(s[::discretization]) for s in sample_x2_pf_rgb]
        enroll_grid_flow_x2 = np.concatenate(
            [np.concatenate(s, axis=1) for s in enroll_flow_x2], axis=0)
        enroll_grid_flow_x2_rgb = np.concatenate(
            [np.concatenate(s, axis=1) for s in enroll_flow_x2_rgb], axis=0)
        enroll_grid_flow_x2 = cv2.cvtColor(enroll_grid_flow_x2,cv2.COLOR_RGB2BGR)
        enroll_grid_flow_x2_rgb = cv2.cvtColor(enroll_grid_flow_x2_rgb, cv2.COLOR_RGB2BGR)

        name_enroll_2 = path.join(savepath,f"enrollment-sid{id2}.png")
        cv2.imwrite(name_enroll_2,enroll_grid_flow_x2)
        name_enroll_2 = path.join(savepath, f"enrollment_rgb-sid{id2}.png")
        cv2.imwrite(name_enroll_2, enroll_grid_flow_x2_rgb)

        samples_upper = np.concatenate(sample_x1_proj, axis=2)
        samples_lower = np.concatenate(sample_x2_proj, axis=2)

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

        filename = f"samples__sid1{id1}_sid2{id2}.mp4"
        savename = path.join(savepath, filename)
        # save_name = savepath + "/frames/" + filename[:-6]
        # breakpoint()
        # import matplotlib.pyplot as plt
        # for i in range(samples_full.shape[0]):
        #     plt.imsave(save_name + 'frame' + str(i) + '.png', samples_full[i])

        write_video(samples_full,savename)
        samples_upper_rgb = np.concatenate(
            sample_x1_proj_rgb, axis=2
        )
        samples_lower_rgb = np.concatenate(
            sample_x2_proj_rgb, axis=2
        )

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
        filename = f"rgb_samples_sid1{id1}_sid2{id2}.mp4"
        savename = path.join(savepath, filename)

        write_video(samples_full_rgb,savename)



def sample_examples_single(
    model: torch.nn.Module,
    loader: DataLoader,
    device,
    dirs,
    vunet,
    flow,
    start_pose_ids,
    n_samples=20,
    only_flow=False
):


    discretization = 6

    it = iter(loader)
    # flag indicates if generated keypoints shall be reprojected into the image
    savepath = dirs

    seq_length = loader.dataset.seq_length[0]


    for i in tqdm(
        start_pose_ids, desc="Generate Videos for logging."
    ):
        print("got data")
        sids = loader.dataset._sample_valid_seq_ids([i, seq_length])
        kps1 = torch.tensor(
            loader.dataset._get_keypoints(sids), device=device
        ).unsqueeze(dim=0).to(torch.float32)
        # kps1 = batch["keypoints"].to(dtype=torch.float32)
        # kps1_change = batch["kp_change"].to(dtype=torch.float32)
        # ids1 = batch["sample_ids"][0, 1:, ...].cpu().numpy()
        ids1 = sids
        id1 = sids[0]
        label_id1 = loader.dataset.datadict["action"][id1]

        #kps2 = batch["paired_keypoints"].to(dtype=torch.float32)
        kps2 = torch.tensor(
            loader.dataset._get_keypoints(sids,use_map_ids=True), device=device
        ).unsqueeze(dim=0).to(torch.float32)
        # kps2_change = batch["paired_change"].to(dtype=torch.float32)
        #ids2 = batch["paired_sample_ids"][0, 1:, ...].cpu().numpy()
        # id2 = ids2[0]
        ids2 = loader.dataset.datadict["map_ids"][ids1]
        id2 = int(ids2[0])

        label_id2 = loader.dataset.datadict["action"][id2]
        n_kps = kps1.shape[2]

        target_dir1 = path.join(savepath,f"sid_{id1}")
        makedirs(target_dir1,exist_ok=True)
        target_dir2 = path.join(savepath, f"sid_{id2}")
        makedirs(target_dir2, exist_ok=True)


        data1, _ = prepare_input(kps1,device)
        data2, _ = prepare_input(kps2,device)

        data_b_1 = data1
        data_b_2 = data2

        seq_len = data_b_1.shape[1]

        with torch.no_grad():
            x1_start = data1[:, 0]
            x2_start = data2[:, 0]
            # task 1: infer self behavior and reconstruct (w/ and w/o target location)

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

        #
        # # already in right form
        p1_gt = np.stack([kps1.squeeze(dim=0).cpu().numpy()[0]]*x1_flow_samples[0].shape[0],axis=0)
        p2_gt = np.stack([kps2.squeeze(dim=0).cpu().numpy()[0]]*x2_flow_samples[0].shape[0],axis=0)

        label1 = loader.dataset.action_id_to_action[label_id1]
        label2 = loader.dataset.action_id_to_action[label_id2]

        # if sampling:
        x1_samples_array = prepare_videos(
            x1_samples, loader, loader.dataset.kinematic_tree, False
        )
        x2_samples_array = prepare_videos(
            x2_samples, loader, loader.dataset.kinematic_tree, False
        )
        gt1 = prepare_videos(p1_gt,loader,loader.dataset.kinematic_tree,False)
        gt2 = prepare_videos(p2_gt, loader, loader.dataset.kinematic_tree,
                             False)

        x1_fs_arr = prepare_videos(
            x1_flow_samples, loader, loader.dataset.kinematic_tree, False
        )
        x2_fs_arr = prepare_videos(
            x2_flow_samples, loader, loader.dataset.kinematic_tree, False
        )


        if "image_size" not in loader.dataset.datadict.keys():
            raise TypeError(
                "Dataset doesn't contain image sizes, not possible to project 3d points onto 2d images"
            )


        app_img, extrs, intrs = get_synth_input(loader, vunet)

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
                [gt1] + x1_fs_arr,
                [loader.dataset.datadict["image_size"][ids1]]
                * (len(x1_samples_array) + 1),
                [[[255, 0, 0], [0, 102, 0]]]
                + [None] * len(x1_samples_array),
                [""]
                + [
                    f""
                    for i in range(len(x1_samples_array))
                ],
            )
        ]

        # only make enrollments for flow samples

        sample_x1_pf_rgb = [p[1] for p in sample_x1_pf]
        sample_x1_pf = [p[0] for p in sample_x1_pf]

        [write_video(s,path.join(target_dir1,f"{id1}_{snr + 1}.mp4")) for snr,s in enumerate(sample_x1_pf[1:])]
        [write_video(s,path.join(target_dir1, f"rgb_{id1}_{snr + 1}.mp4")) for snr, s
         in enumerate(sample_x1_pf_rgb[1:])]

        write_video(sample_x1_pf[0],path.join(target_dir1,f"{id1}_0.mp4"))
        write_video(sample_x1_pf_rgb[0], path.join(target_dir1, f"rgb_{id1}_0.mp4"))



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
                [gt2] + x2_fs_arr,
                [loader.dataset.datadict["image_size"][ids1]]
                * (len(x1_samples_array) + 1),
                [[[255, 0, 0], [0, 102, 0]]]
                + [None] * len(x1_samples_array),
                [""]
                + [
                    f""
                    for i in range(len(x1_samples_array))
                ],
            )
        ]
        sample_x2_pf_rgb = [p[1] for p in sample_x2_pf]
        sample_x2_pf = [p[0] for p in sample_x2_pf]

        [write_video(s,path.join(target_dir2, f"{id2}_{snr + 1}.mp4")) for snr, s
         in enumerate(sample_x2_pf)]
        [write_video(s,path.join(target_dir2, f"rgb_{id2}_{snr + 1}.mp4")) for snr, s
         in enumerate(sample_x2_pf_rgb)]

        write_video(sample_x2_pf[0], path.join(target_dir2, f"{id2}_0.mp4"))
        write_video(sample_x2_pf_rgb[0], path.join(target_dir2, f"rgb_{id2}_0.mp4"))






